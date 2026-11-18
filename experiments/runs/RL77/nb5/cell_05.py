# [Colab Q-RL Cell 5] Stage 2.5 Q-learning on the competition data, Q-guided validation, test submission
class QRLCFG:
    RL_MINUTES = float(os.environ.get("RL77_RL_MINUTES", 45))
    N_TRAIN_MOLECULES = int(os.environ.get("RL77_NB5_TRAIN_MOLECULES", 20000))
    N_VAL_MOLECULES = int(os.environ.get("RL77_NB5_VAL_MOLECULES", 200))
    BATCH = 16
    LR = 3e-5
    CENTROID_TOP = 3
    GUARD_WINDOW = 100  # steps in the rollout-reward moving average used by the best-checkpoint guard


_candidate_formula = lru_cache(maxsize=None)(truth_formula)  # RDKit formula of a library SMILES (notebook 4 helper)


def trim_padding(batch: Dict, pad_id: int) -> Dict:
    """Drop all-PAD token columns (SMILES are padded to 256; the median is 39 characters)."""
    for key in ("mol_tokens_cls", "mol_tokens_dec"):
        if key in batch:
            used = int((batch[key] != pad_id).sum(dim=1).max())
            batch[key] = batch[key][:, :used]
    return batch


@torch.no_grad()
def qrl_predict(
    glmr_model, q_decoder, reward_engine, fg_head, mist_filter, tokenizer, loader,
    lib_smiles, lib_masses, lib_embs, collect: Optional[List[Dict]] = None,
    alpha: Optional[float] = None, insert_rank: Optional[int] = None,
) -> Dict[str, List[str]]:
    """Retrieval -> cross-fusion -> Q-guided beams; re-rank candidates with the reward-weighted
    centroid E^m_+ of the top Q-beam structures (PDF, final section); insert the best mass-consistent beam.
    alpha / insert_rank are notebook 5's own Evidently weights (`nb5.rerank.*`): notebook 1 fits its
    knobs for plain beam search, and a value of 0 there would silently switch the Q-centroid off here."""
    alpha = W("nb5.rerank.alpha") if alpha is None else alpha
    insert_rank = W.int("nb5.rerank.generated_insert_rank") if insert_rank is None else insert_rank
    rank_formulas = W.int("nb4.priors.rank_formulas")  # formula-prior ranking tiers adopted from notebook 4 (0 = off)
    lib_embs_dev = lib_embs.to(CFG.DEVICE)
    preds = {}
    for batch in loader:
        batch = {k: (v.to(CFG.DEVICE) if isinstance(v, torch.Tensor) else v) for k, v in batch.items()}
        h_s_mol, e_s_mol, _ = glmr_model.encode_molecule_spectra(batch)
        h_m_k, idx_batch = retrieve_top_k_candidates(
            e_s_mol, batch["precursor_mz"], batch["primary_adduct"], lib_masses, lib_embs,
            top_k=CFG.TOP_K_CANDIDATES, ionization_modes=batch["ionization_mode"],
        )
        h_ca = glmr_model.cross_fusion(h_s_mol, h_m_k)
        fg_71_pred, _ = fg_head(h_s_mol)
        priors = generate_formula_priors_batch(batch, mist_filter, fg_71_pred)
        neutral = np.array([
            neutral_mass_from_precursor(float(batch["precursor_mz"][b]), batch["primary_adduct"][b], batch["ionization_mode"][b])
            for b in range(len(batch["molecule_id"]))
        ], dtype=np.float32)
        rewards: List[List[float]] = []
        q_smis, q_iks = q_guided_beam_search_decode(
            q_decoder, reward_engine, h_ca, neutral, fg_71_pred.cpu().numpy(), priors, reward_lists=rewards,
        )
        for b, mol_id in enumerate(batch["molecule_id"]):
            cand_idx = idx_batch[b]
            cand_embs = h_m_k[b]
            stage2, gen_smi, gen_ik, gen_ok = None, None, None, False
            if q_smis[b]:
                top = q_smis[b][: QRLCFG.CENTROID_TOP]
                toks = torch.stack([tokenizer.encode(s, add_cls=True) for s in top]).to(CFG.DEVICE)
                _, e_top = glmr_model.mol_encoder(toks)
                w = torch.softmax(torch.tensor(rewards[b][: len(top)], device=CFG.DEVICE), dim=0)
                centroid = F.normalize((w.unsqueeze(1) * e_top).sum(dim=0), dim=-1)
                stage2 = torch.matmul(cand_embs, centroid).cpu().numpy()
                gen_smi, gen_ik = q_smis[b][0], q_iks[b][0]
                gen_ok = bool(in_mass_window(np.array([Descriptors.ExactMolWt(Chem.MolFromSmiles(gen_smi))]), float(neutral[b]))[0])
            fallback = torch.topk(torch.matmul(lib_embs_dev, e_s_mol[b]), k=min(100, len(lib_smiles))).indices.cpu().tolist()
            rec = {
                "molecule_id": str(mol_id),
                "cand_smiles": [lib_smiles[i] for i in cand_idx],
                "stage1_scores": torch.matmul(cand_embs, e_s_mol[b]).cpu().numpy(),
                "stage2_scores": stage2,
                "in_window": in_mass_window(lib_masses[cand_idx], float(neutral[b])),
                "gen_smiles": gen_smi, "gen_ik14": gen_ik, "gen_mass_ok": gen_ok,
                "backfill_smiles": [lib_smiles[i] for i in fallback],
                "q_beam_ik14s": list(q_iks[b]),
            }
            if rank_formulas > 0:
                rec["priority"] = formula_priority([_candidate_formula(s) for s in rec["cand_smiles"]], priors[b], rank_formulas)
            if collect is not None:
                collect.append(rec)
            preds[str(mol_id)] = assemble_ranked_candidates(rec, alpha, insert_rank)
    return preds


def run_qrl_pipeline() -> Dict:
    out_dir = ENV.work_dir()
    metrics: Dict = {"weights_version": W.version}
    nb1_dir = ENV.upstream_dir("nb1", "glmr_model.pt")
    lib_dir = ENV.upstream_dir("nb1", "library.parquet")
    split_dir = ENV.upstream_dir("nb1", "val_inchikey14.txt")
    nb3_dir = ENV.upstream_dir("nb3", "mass2smiles_fg_head.pt")

    glmr_model, tokenizer = load_glmr_checkpoint(nb1_dir / "glmr_model.pt", device=CFG.DEVICE)
    glmr_base, _ = load_glmr_checkpoint(nb1_dir / "glmr_model.pt", device=CFG.DEVICE)  # untouched baseline
    lib_smiles, lib_ik14, lib_masses, lib_embs = load_library(lib_dir)
    lib_embs = lib_embs.to(CFG.DEVICE)  # once: retrieve_top_k_candidates() otherwise copies 282 MB to the GPU per step
    fg_head = load_fg_head(nb3_dir)
    retriever = OfflineMassAndFormulaRetriever()
    mist_filter = FastMISTSubformulaFilter(CandidateFormulaBank(retriever.catalog_df["molecular_formula"].dropna().unique().tolist()))
    reward_engine = CASMI26ChemicalRewardEngine(tokenizer, canonicalize_and_inchikey14, cached_ecfp4_fg71)
    for p in glmr_model.parameters():
        p.requires_grad = False
    q_decoder = ChemFormerQDecoderWrapper(glmr_model.mol_decoder).to(CFG.DEVICE)  # fine-tunes glmr_model.mol_decoder in place
    for p in q_decoder.parameters():
        p.requires_grad = True
    trainer = GLMRStage25QLearningTrainer(q_decoder, reward_engine)
    optimizer = torch.optim.AdamW(
        [p for n, p in q_decoder.named_parameters() if not n.startswith("target_")], lr=QRLCFG.LR, weight_decay=1e-4
    )

    # 1. Stage 2.5 Q-learning on notebook 1's training split
    rng = np.random.default_rng(CFG.SEED)
    train_ids = read_split(split_dir, "train")
    train_ids = rng.choice(train_ids, size=min(QRLCFG.N_TRAIN_MOLECULES, len(train_ids)), replace=False).tolist()
    train_loader = make_loader(load_spectra_for_inchikeys(CFG.DATA_DIR, train_ids), tokenizer, train=True, batch_size=QRLCFG.BATCH)
    deadline = ENV.Deadline(QRLCFG.RL_MINUTES)
    log_rows, step = [], 0
    # Best-checkpoint guard: keep the decoder + Q-heads with the highest 100-step moving-average rollout reward
    # and restore it at the end, so late degradation cannot reach validation or the test submission.
    best = {"ma": -float("inf"), "step": None, "state": None}
    while not deadline.expired():
        for batch in train_loader:
            if deadline.expired():
                break
            stats = trainer.train_rl_step(
                trim_padding(batch, tokenizer.pad_id), glmr_model, fg_head, mist_filter, generate_formula_priors_batch,
                retrieve_top_k_candidates, lib_masses, lib_embs, ADDUCT_MASS_SHIFT, optimizer, CFG.DEVICE,
            )
            log_rows.append({"step": step, **stats})
            if step >= QRLCFG.GUARD_WINDOW - 1:
                ma = float(np.mean([r["mean_rollout_reward"] for r in log_rows[-QRLCFG.GUARD_WINDOW:]]))
                if ma > best["ma"]:
                    best = {"ma": ma, "step": step,
                            "state": {k: v.detach().to("cpu", copy=True) for k, v in q_decoder.state_dict().items()}}
            if step % 25 == 0:
                ENV.log(f"RL step {step}: {stats} ({deadline.remaining_min():.1f} min left)")
            step += 1
    pd.DataFrame(log_rows).to_csv(out_dir / "rl_log.csv", index=False)
    final_ma = float(np.mean([r["mean_rollout_reward"] for r in log_rows[-QRLCFG.GUARD_WINDOW:]])) if log_rows else None
    if best["state"] is not None and (final_ma is None or best["ma"] > final_ma):
        q_decoder.load_state_dict(best["state"])  # also restores glmr_model.mol_decoder (shared module)
        ENV.log(f"Restored the step-{best['step']} checkpoint (reward MA {best['ma']:.3f} vs final {final_ma:.3f})")
    torch.save({"state_dict": q_decoder.state_dict(), "weights_version": W.version, "rl_step": best["step"]}, out_dir / "q_decoder.pt")
    metrics["rl"] = {"steps": step, "mean_rollout_reward_last100": final_ma,
                     "best_reward_ma100": best["ma"] if best["state"] is not None else None, "best_step": best["step"],
                     "first_reward_ma100": float(np.mean([r["mean_rollout_reward"] for r in log_rows[:QRLCFG.GUARD_WINDOW]])) if log_rows else None}

    # 2. Validation: notebook 1's pipeline (its adopted re-rank knobs) vs Q-guided beams on the same held-out molecules
    val_ids = read_split(split_dir, "val")
    val_ids = rng.choice(val_ids, size=min(QRLCFG.N_VAL_MOLECULES, len(val_ids)), replace=False).tolist()
    val_df = load_spectra_for_inchikeys(CFG.DATA_DIR, val_ids, timstof_window_only=True)
    gt = ground_truth_ik14(val_df)
    base_sub = predict_and_rerank_test_molecules(
        glmr_base, make_loader(val_df, tokenizer, train=False, batch_size=16), tokenizer, lib_smiles, lib_masses, lib_embs,
    )
    records: List[Dict] = []
    q_preds = qrl_predict(glmr_model, q_decoder, reward_engine, fg_head, mist_filter, tokenizer,
                          make_loader(val_df, tokenizer, train=False, batch_size=8), lib_smiles, lib_masses, lib_embs, records)
    metrics["val"] = {
        "n": len(records),
        "mrr25_nb1_pipeline": compute_casmi_mrr25(predictions_from_submission(base_sub), gt),
        "mrr25_q_guided": compute_casmi_mrr25(q_preds, gt),
        "q_beam_any_valid": float(np.mean([r["gen_smiles"] is not None for r in records])),
        "q_beam_top1_mass_ok": float(np.mean([r["gen_mass_ok"] for r in records])),
        "q_beam_top1_exact": float(np.mean([r["gen_ik14"] == gt[r["molecule_id"]] for r in records])),
        "q_beam_contains_truth": float(np.mean([gt[r["molecule_id"]] in r["q_beam_ik14s"] for r in records])),
    }
    # Fit the Q-centroid blend and Q-beam insert rank (2-fold cross-fit, same grid as notebook 1)
    sweep = sweep_rerank_knobs(records, gt, prior_keys=("nb5.rerank.alpha", "nb5.rerank.generated_insert_rank"))
    metrics["rerank_sweep"] = {k: v for k, v in sweep.items() if k != "grid"}
    ENV.write_json(out_dir / "rerank_grid.json", sweep["grid"])
    ENV.write_json(out_dir / "fitted_weights.json", {
        "notebook": "nb5",
        "weights": {"nb5.rerank.alpha": sweep["best_alpha"],
                    "nb5.rerank.generated_insert_rank": sweep["best_generated_insert_rank"]},
        "evidence": {"metric": "MRR@25", "split": f"{len(records)} held-out timsTOF molecules, 2-fold cross-fit",
                     "prior": sweep["val_mrr25_prior"], "fitted_crossfit": sweep["val_mrr25_crossfit"]},
    })
    (alpha, insert_rank), used_fit = ENV.select_validated(
        (W("nb5.rerank.alpha"), W.int("nb5.rerank.generated_insert_rank")),
        (sweep["best_alpha"], sweep["best_generated_insert_rank"]),
        sweep["val_mrr25_prior"], sweep["val_mrr25_crossfit"],
    )
    metrics["test_rerank"] = {"alpha": alpha, "generated_insert_rank": insert_rank, "source": "fitted" if used_fit else W.version}
    ENV.log(f"Validation: {metrics['val']}; sweep {metrics['rerank_sweep']}; test uses {metrics['test_rerank']}")

    # 3. Test submission
    test_df = pd.read_parquet(CFG.DATA_DIR / "test.parquet")
    test_preds = qrl_predict(glmr_model, q_decoder, reward_engine, fg_head, mist_filter, tokenizer,
                             make_loader(test_df, tokenizer, train=False, batch_size=8), lib_smiles, lib_masses, lib_embs,
                             alpha=alpha, insert_rank=insert_rank)
    sub_df = pd.DataFrame([{"molecule_id": m, "smiles": ";".join(p)} for m, p in test_preds.items()], columns=["molecule_id", "smiles"])
    validate_submission_file(sub_df, test_df["molecule_id"].unique().tolist())
    sub_df.to_csv(out_dir / "submission.csv", index=False)

    ENV.write_json(out_dir / "metrics.json", metrics)
    ENV.write_run_manifest("nb5", out_dir, W.version, {"metrics": metrics})
    ENV.log(f"Notebook 5 outputs written to {out_dir}")
    return metrics


if __name__ == "__main__":
    NB5_METRICS = run_qrl_pipeline()
