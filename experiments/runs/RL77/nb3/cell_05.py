# [Colab Ensemble Cell 5] End-to-End Blended Inference & InChIKey14 Deduplication
@torch.no_grad()
def run_blended_casmi26_inference(
    glmr_model: nn.Module,
    siamese_model: SiameseECFP4Retriever,
    fg_head: Mass2SMILESFunctionalGroupHead,
    reranker: CASMI26EnsembleReranker,
    test_loader,
    tokenizer,
    candidate_smiles: List[str],
    candidate_masses: np.ndarray,
    candidate_glmr_embs: torch.Tensor,
    canonicalize_fn,
    retrieve_topk_fn,
    top_k: int = 60,  # Expand pre-retrieval pool to K=60 before ensemble re-ranking
    top_n_sub: int = 25,
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
) -> List[Dict[str, str]]:
    """
    Runs Stage 1 Pre-Retrieval (GLMR + Siamese ECFP4 union), Stage 2 GLMR Cross-Fusion
    Beam Search, Mass2SMILES 71-D Functional Group scoring, and 4-way Z-score/RRF blending.
    """
    glmr_model.eval().to(device)
    siamese_model.eval().to(device)
    fg_head.eval().to(device)

    submission_rows = []

    for batch in test_loader:
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                batch[k] = v.to(device)

        # 1. GLMR Stage 1 Multi-Spectrum Encoding & Top-K Pre-Retrieval
        h_s_mol, e_s_mol, _ = glmr_model.encode_molecule_spectra(batch)
        h_m_k, topk_indices_batch = retrieve_topk_fn(
            e_s_mol,
            batch["precursor_mz"],
            batch["primary_adduct"],
            candidate_masses,
            candidate_glmr_embs,
            top_k=top_k,
        )

        # 2. GLMR Stage 2 Cross-Fusion & Beam Search Generation
        h_ca = glmr_model.cross_fusion(h_s_mol, h_m_k)
        gen_token_seqs = glmr_model.mol_decoder.beam_search_decode(
            h_ca, beam_size=5, max_len=128
        )

        # 3. Mass2SMILES 71-D Functional Group + Heteroatom Prediction from H^s_mol
        fg_71_pred, _ = fg_head(h_s_mol)  # (B, 71)

        # 4. Siamese 2000-Bin Spectrum Encoding & Continuous ECFP4 Prediction
        z_s_batch, fp_logits_batch = siamese_model.encode_spectrum(batch["binned_2000"])
        pred_ecfp4_probs = torch.sigmoid(fp_logits_batch)  # (B, 2048)

        for b, mol_id in enumerate(batch["molecule_id"]):
            cand_indices = topk_indices_batch[b]
            cand_glmr_embs = h_m_k[b]  # (K, 256)

            # Stream A: GLMR Stage 1 Spectral-Molecular Cosine Similarity
            s_glmr_pre = torch.matmul(cand_glmr_embs, e_s_mol[b]).cpu().numpy()

            # Stream B: GLMR Stage 2 Unimodal Generative Cosine Similarity
            raw_gen_smi = tokenizer.decode(gen_token_seqs[b])
            canon_gen_smi, gen_ik14 = canonicalize_fn(raw_gen_smi)
            gen_valid = canon_gen_smi is not None

            if gen_valid:
                gen_tok = tokenizer.encode(canon_gen_smi, add_cls=True).unsqueeze(0).to(device)
                _, e_m_plus = glmr_model.mol_encoder(gen_tok)
                s_glmr_gen = torch.matmul(cand_glmr_embs, e_m_plus.squeeze(0)).cpu().numpy()
            else:
                s_glmr_gen = np.zeros_like(s_glmr_pre)

            # Extract ECFP4 & 71-D FG vectors for the K=60 pre-retrieved candidates
            cand_ecfp4_list, cand_fg71_list, cand_canon_smis, cand_ik14s = [], [], [], []
            for c_idx in cand_indices:
                smi = candidate_smiles[c_idx]
                c_smi, ik14 = canonicalize_fn(smi)
                ecfp4_vec, fg71_vec = smiles_to_ecfp4_and_mass2smiles_71d(c_smi or smi)
                if ecfp4_vec is None:
                    ecfp4_vec = np.zeros(2048, dtype=np.float32)
                    fg71_vec = np.zeros(71, dtype=np.float32)
                cand_ecfp4_list.append(ecfp4_vec)
                cand_fg71_list.append(fg71_vec)
                cand_canon_smis.append(c_smi)
                cand_ik14s.append(ik14)

            cand_ecfp4_t = torch.tensor(np.stack(cand_ecfp4_list), dtype=torch.float32, device=device)
            cand_fg71_t = torch.tensor(np.stack(cand_fg71_list), dtype=torch.float32, device=device)

            # Stream C: 2000-Bin Siamese Cosine + Differentiable ECFP4 Tanimoto Similarity
            z_f_cands = siamese_model.encode_fingerprint(cand_ecfp4_t)  # (K, 256)
            siamese_cos = torch.matmul(z_f_cands, z_s_batch[b])         # (K,)
            ecfp4_tani = SiameseContrastiveTanimotoLoss.continuous_tanimoto_similarity(
                pred_ecfp4_probs[b].unsqueeze(0).expand_as(cand_ecfp4_t),
                cand_ecfp4_t,
            )
            s_siamese_ecfp4 = (0.5 * siamese_cos + 0.5 * ecfp4_tani).cpu().numpy()

            # Stream D: Mass2SMILES 71-D Functional Group + Heteroatom L1 Distance
            s_fg_l1_dist = torch.mean(
                torch.abs(cand_fg71_t - fg_71_pred[b].unsqueeze(0)), dim=-1
            ).cpu().numpy()

            # Blend all 4 streams + Reciprocal Rank Fusion
            blended_scores = reranker.score_candidates_for_molecule(
                s_glmr_gen=s_glmr_gen,
                s_glmr_pre=s_glmr_pre,
                s_siamese_ecfp4=s_siamese_ecfp4,
                s_fg_l1_dist=s_fg_l1_dist,
                gen_smiles_valid=gen_valid,
            )

            ranked_order = np.argsort(-blended_scores)

            # Strict InChIKey14 Deduplication up to 25 candidates
            seen_ik14 = set()
            top25_smiles = []
            for r_i in ranked_order:
                c_smi = cand_canon_smis[r_i]
                ik14 = cand_ik14s[r_i]
                if c_smi is not None and ik14 is not None and ik14 not in seen_ik14:
                    seen_ik14.add(ik14)
                    top25_smiles.append(c_smi)
                if len(top25_smiles) >= top_n_sub:
                    break

            # Append generated de novo structure if valid, distinct, and slots remain
            if gen_valid and gen_ik14 not in seen_ik14 and len(top25_smiles) < top_n_sub:
                seen_ik14.add(gen_ik14)
                top25_smiles.append(canon_gen_smi)

            submission_rows.append(
                {
                    "molecule_id": str(mol_id),
                    "smiles": ";".join(top25_smiles[:top_n_sub]),
                }
            )

    return submission_rows