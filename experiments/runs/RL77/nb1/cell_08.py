# [Colab Cell 8] Stage 1 + Stage 2 Inference, Unimodal Re-Ranking, & Top-25 InChIKey14 Deduplication
def assemble_ranked_candidates(
    record: Dict,
    alpha_rerank: float,
    generated_insert_rank: int,
    top_n: int = CFG.TOP_N_SUBMISSION,
) -> List[str]:
    """
    Builds the deduplicated top-25 list for one molecule from the scores stored in `record`.
    Shared by inference and by the validation sweep that fits alpha / insert rank.
      - final = alpha * cos(E^m_+, E^m_i) + (1 - alpha) * cos(E^s, E^m_i) when the beam SMILES is valid
      - the generated structure enters at `generated_insert_rank` (1-based) only if its exact
        mass is within the ppm window of the precursor-derived neutral mass (0 disables it)
      - remaining slots are backfilled from global spectral similarity
    """
    stage1 = record["stage1_scores"]
    stage2 = record["stage2_scores"]
    final = stage1 if stage2 is None else alpha_rerank * stage2 + (1.0 - alpha_rerank) * stage1
    # Optional tiers first (notebook 4 formula priors: 0 = top formula, ...), then mass-consistent candidates,
    # then blended similarity (np.lexsort: last key is primary). Records without "priority" rank as before.
    in_window = np.asarray(record.get("in_window", np.zeros(len(final), dtype=bool)), dtype=bool)
    priority = np.asarray(record.get("priority", np.zeros(len(final), dtype=np.int64)))
    order = np.lexsort((-final, ~in_window, priority))

    seen_ik14, ranked = set(), []
    gen_smi, gen_ik14 = record["gen_smiles"], record["gen_ik14"]
    insert_gen = generated_insert_rank > 0 and record["gen_mass_ok"] and gen_smi is not None
    if insert_gen:
        seen_ik14.add(gen_ik14)  # reserve so a retrieved duplicate cannot take a second slot

    for i in order:
        canon_smi, ik14 = canonicalize_and_inchikey14(record["cand_smiles"][i])
        if canon_smi is not None and ik14 not in seen_ik14:
            seen_ik14.add(ik14)
            ranked.append(canon_smi)
        if len(ranked) >= top_n:
            break
    for smi in record["backfill_smiles"]:
        if len(ranked) >= top_n:
            break
        canon_smi, ik14 = canonicalize_and_inchikey14(smi)
        if canon_smi is not None and ik14 not in seen_ik14:
            seen_ik14.add(ik14)
            ranked.append(canon_smi)

    if insert_gen:
        pos = min(generated_insert_rank - 1, len(ranked))
        ranked.insert(pos, gen_smi)
    return ranked[:top_n]


@torch.no_grad()
def predict_and_rerank_test_molecules(
    model: GLMRPipeline,
    test_loader: DataLoader,
    tokenizer: SMILESTokenizer,
    lib_smiles: List[str],
    lib_masses: np.ndarray,
    lib_embs: torch.Tensor,
    alpha_rerank: float = W("nb1.rerank.alpha"),
    generated_insert_rank: int = W.int("nb1.rerank.generated_insert_rank"),
    device: torch.device = CFG.DEVICE,
    collect_records: Optional[List[Dict]] = None,
) -> pd.DataFrame:
    """
    Executes full GLMR two-stage inference per molecule_id:
      1. Multi-spectrum aggregation -> Stage 1 Pre-Retrieval of K=40 candidates.
      2. Spec-Mol Cross-Fusion + Beam Search (beam=5) -> generates refined SMILES y+.
      3. Encodes y+ via ChemFormerMolecularEncoder -> E^m_+ and re-ranks candidates via
         blended unimodal cosine similarity: alpha * cos(E^m_+, E^m_i) + (1 - alpha) * cos(E^s, E^m_i).
      4. Canonicalizes tautomers (RDKit 2026.03.3), deduplicates strictly by InChIKey14,
         and backfills to 25 unique structures per molecule_id.
    Per-molecule scores are appended to `collect_records` when given (validation sweeps).
    """
    from rdkit.Chem import Descriptors

    model.eval()
    lib_embs_dev = lib_embs.to(device)
    submission_rows = []

    for batch in test_loader:
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                batch[k] = v.to(device)

        # Stage 1: Multi-spectrum pooling & top-K candidate pre-retrieval
        h_s_mol, e_s_mol, _ = model.encode_molecule_spectra(batch)
        h_m_k, topk_indices_batch = retrieve_top_k_candidates(
            e_s_mol,
            batch["precursor_mz"],
            batch["primary_adduct"],
            lib_masses,
            lib_embs,
            top_k=CFG.TOP_K_CANDIDATES,
            ionization_modes=batch["ionization_mode"],
        )

        # Stage 2: Cross-fusion & beam search generation of refined structures
        h_ca = model.cross_fusion(h_s_mol, h_m_k)
        gen_token_seqs = model.mol_decoder.beam_search_decode(
            h_ca, beam_size=CFG.BEAM_SIZE, max_len=W.int("nb1.decode.max_len")
        )

        for b, mol_id in enumerate(batch["molecule_id"]):
            cand_indices = topk_indices_batch[b]
            cand_embs = h_m_k[b]  # (K, D)
            stage1_scores = torch.matmul(cand_embs, e_s_mol[b]).cpu().numpy()  # (K,)

            gen_smiles_raw = tokenizer.decode(gen_token_seqs[b])
            canon_gen_smi, gen_ik14 = canonicalize_and_inchikey14(gen_smiles_raw)

            neutral = neutral_mass_from_precursor(
                float(batch["precursor_mz"][b]), batch["primary_adduct"][b], batch["ionization_mode"][b]
            )
            # If generated SMILES is chemically valid, compute unimodal similarity Eq. (9)
            stage2_scores, gen_mass_ok = None, False
            if canon_gen_smi is not None:
                gen_tok = tokenizer.encode(canon_gen_smi, add_cls=True).unsqueeze(0).to(device)
                _, e_m_plus = model.mol_encoder(gen_tok)  # (1, D)
                stage2_scores = torch.matmul(cand_embs, e_m_plus.squeeze(0)).cpu().numpy()  # (K,)
                gen_mass = Descriptors.ExactMolWt(Chem.MolFromSmiles(canon_gen_smi))
                gen_mass_ok = bool(in_mass_window(np.array([gen_mass]), neutral)[0])

            global_sims = torch.matmul(lib_embs_dev, e_s_mol[b])
            fallback_idx = torch.topk(global_sims, k=min(100, len(lib_smiles))).indices.cpu().tolist()

            record = {
                "molecule_id": str(mol_id),
                "cand_smiles": [lib_smiles[i] for i in cand_indices],
                "stage1_scores": stage1_scores,
                "stage2_scores": stage2_scores,
                "in_window": in_mass_window(lib_masses[cand_indices], neutral),
                "gen_smiles": canon_gen_smi,
                "gen_ik14": gen_ik14,
                "gen_mass_ok": bool(gen_mass_ok),
                "backfill_smiles": [lib_smiles[i] for i in fallback_idx],
            }
            if collect_records is not None:
                collect_records.append(record)

            unique_top25 = assemble_ranked_candidates(record, alpha_rerank, generated_insert_rank)
            submission_rows.append(
                {
                    "molecule_id": mol_id,
                    "smiles": ";".join(unique_top25[: CFG.TOP_N_SUBMISSION]),
                }
            )

    sub_df = pd.DataFrame(submission_rows, columns=["molecule_id", "smiles"])
    return sub_df


def validate_submission_file(sub_df: pd.DataFrame, expected_mol_ids: List[str]) -> None:
    """Automated submission gate checks required by CASMI 2026 rules (format of sample_submission.csv)."""
    assert list(sub_df.columns) == ["molecule_id", "smiles"], "Invalid column headers!"
    assert len(sub_df) == len(expected_mol_ids), "Row count mismatch with test molecule_ids!"
    assert sub_df["molecule_id"].is_unique, "Duplicate molecule_id rows!"
    assert set(sub_df["molecule_id"]) == set(expected_mol_ids), "Missing or unexpected molecule_id!"
    assert not sub_df["smiles"].isna().any(), "Null values found in smiles column!"
    for row in sub_df.itertuples():
        candidates = [c for c in str(row.smiles).split(";") if c]
        assert 1 <= len(candidates) <= 25, f"{row.molecule_id} has {len(candidates)} candidates!"
    print(f"Submission verified: {len(sub_df)} unique molecules, up to 25 deduplicated SMILES each.")
