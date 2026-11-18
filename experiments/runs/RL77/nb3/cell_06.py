# [Colab Ensemble Cell 6] Train the Siamese ECFP4 and Mass2SMILES heads, fit blend weights out-of-fold, infer test
class EnsCFG:
    N_TRAIN_MOLECULES = int(os.environ.get("RL77_NB3_TRAIN_MOLECULES", 20000))  # subset of notebook 1's train split
    N_VAL_MOLECULES = int(os.environ.get("RL77_NB3_VAL_MOLECULES", 300))       # subset of notebook 1's val split
    SIAMESE_MINUTES = float(os.environ.get("RL77_SIAMESE_MINUTES", 20))
    FG_HEAD_MINUTES = float(os.environ.get("RL77_FG_HEAD_MINUTES", 30))
    BATCH_SIAMESE = 256
    BATCH_FG = 64
    LR = 1e-3


def molecule_features(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str]]:
    """Per inchikey14: composite 2000-bin spectrum, ECFP4 bits, 71-D Mass2SMILES target."""
    specs, fps, fgs, ids = [], [], [], []
    for ik14, grp in df.groupby("inchikey14", sort=False):
        records = grp.head(CFG.MAX_SPECTRA_TRAIN).to_dict("records")
        ecfp4, fg71 = cached_ecfp4_fg71(str(records[0]["normalized_smiles"]))
        if ecfp4 is None:
            continue
        specs.append(aggregate_molecule_2000_bins(records))
        fps.append(ecfp4)
        fgs.append(fg71)
        ids.append(ik14)
    return np.stack(specs).astype(np.float32), np.stack(fps).astype(np.float32), np.stack(fgs).astype(np.float32), ids


def train_siamese(spec: np.ndarray, fp: np.ndarray, deadline: "ENV.Deadline") -> Tuple[SiameseECFP4Retriever, List[float]]:
    model = SiameseECFP4Retriever().to(CFG.DEVICE)
    loss_fn = SiameseContrastiveTanimotoLoss()
    opt = torch.optim.AdamW(model.parameters(), lr=EnsCFG.LR, weight_decay=1e-4)
    spec_t, fp_t = torch.from_numpy(spec), torch.from_numpy(fp)
    history, epoch = [], 0
    while not deadline.expired():
        model.train()
        perm = torch.randperm(len(spec_t))
        total, n = 0.0, 0
        for start in range(0, len(perm) - EnsCFG.BATCH_SIAMESE + 1, EnsCFG.BATCH_SIAMESE):  # drop_last for BatchNorm
            idx = perm[start:start + EnsCFG.BATCH_SIAMESE]
            xs, xf = spec_t[idx].to(CFG.DEVICE), fp_t[idx].to(CFG.DEVICE)
            z_s, z_f, fp_logits = model(xs, xf)
            loss = loss_fn(z_s, z_f, fp_logits, xf)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total, n = total + loss.item(), n + 1
            if deadline.expired():
                break
        history.append(total / max(n, 1))
        ENV.log(f"Siamese epoch {epoch}: loss {history[-1]:.4f} ({deadline.remaining_min():.1f} min left)")
        epoch += 1
    return model.eval(), history


def train_fg_head(glmr_model: GLMRPipeline, loader: DataLoader, deadline: "ENV.Deadline") -> Tuple[Mass2SMILESFunctionalGroupHead, List[float]]:
    """Mass2SMILES head on frozen GLMR multi-spectrum states H^s_mol (targets: log1p 71-D counts from SMILES)."""
    fg_head = Mass2SMILESFunctionalGroupHead().to(CFG.DEVICE)
    opt = torch.optim.AdamW(fg_head.parameters(), lr=EnsCFG.LR, weight_decay=1e-4)
    glmr_model.eval()
    history, epoch = [], 0
    while not deadline.expired():
        fg_head.train()
        total, n = 0.0, 0
        for batch in loader:
            if deadline.expired():
                break
            batch = {k: (v.to(CFG.DEVICE) if isinstance(v, torch.Tensor) else v) for k, v in batch.items()}
            targets = np.stack([cached_ecfp4_fg71(s)[1] if cached_ecfp4_fg71(s)[1] is not None else np.zeros(71, np.float32)
                                for s in batch["smiles"]])
            with torch.no_grad():
                h_s_mol, _, _ = glmr_model.encode_molecule_spectra(batch)
            fg_pred, _ = fg_head(h_s_mol)
            loss = mass2smiles_fg_loss(fg_pred, torch.from_numpy(targets).to(CFG.DEVICE))
            opt.zero_grad()
            loss.backward()
            opt.step()
            total, n = total + loss.item(), n + 1
        history.append(total / max(n, 1))
        ENV.log(f"FG head epoch {epoch}: MAE+MSE {history[-1]:.4f} ({deadline.remaining_min():.1f} min left)")
        epoch += 1
    return fg_head.eval(), history


def ensemble_mrr(records: List[Dict], weights, gt: Dict[str, str]) -> float:
    reranker = CASMI26EnsembleReranker(weights=tuple(weights))
    preds = {r["molecule_id"]: rank_ensemble_record(r, reranker) for r in records}
    return compute_casmi_mrr25(preds, {r["molecule_id"]: gt[r["molecule_id"]] for r in records})


def crossfit_blend_weights(records: List[Dict], gt: Dict[str, str], prior: np.ndarray) -> Dict:
    """2-fold cross-fitting of optimize_weights_oof, then a final fit on all validation records."""
    rng = np.random.default_rng(CFG.SEED)
    order = rng.permutation(len(records))
    folds = [[records[i] for i in order[0::2]], [records[i] for i in order[1::2]]]
    held_total = 0.0
    for fit, held in ((folds[0], folds[1]), (folds[1], folds[0])):
        rr = CASMI26EnsembleReranker(weights=tuple(prior))
        w_fit = rr.optimize_weights_oof(fit, gt)
        held_total += ensemble_mrr(held, w_fit, gt) * len(held)
    final = CASMI26EnsembleReranker(weights=tuple(prior)).optimize_weights_oof(records, gt)
    return {
        "weights": [float(x) for x in final],
        "val_mrr25_prior": ensemble_mrr(records, prior, gt),
        "val_mrr25_crossfit": held_total / len(records),
        "val_mrr25_in_sample": ensemble_mrr(records, final, gt),
    }


def run_ensemble_pipeline() -> Dict:
    out_dir = ENV.work_dir()
    metrics: Dict = {"weights_version": W.version}
    nb1_dir = ENV.upstream_dir("nb1", "glmr_model.pt")
    split_dir = ENV.upstream_dir("nb1", "val_inchikey14.txt")
    glmr_model, tokenizer = load_glmr_checkpoint(nb1_dir / "glmr_model.pt", device=CFG.DEVICE)
    retriever = OfflineMassAndFormulaRetriever()

    def store_retrieve_fn(e_s_mol, precursor_mzs, adducts, candidate_masses, candidate_embs, top_k):
        return retriever.retrieve_top_k_batch(e_s_mol, precursor_mzs, adducts, top_k=top_k)

    # 1. Training molecules: a subset of notebook 1's training split (never the validation blocks)
    rng = np.random.default_rng(CFG.SEED)
    train_ids = read_split(split_dir, "train")
    train_ids = rng.choice(train_ids, size=min(EnsCFG.N_TRAIN_MOLECULES, len(train_ids)), replace=False).tolist()
    train_df = load_spectra_for_inchikeys(CFG.DATA_DIR, train_ids)
    spec, fp, _, ids = molecule_features(train_df)
    ENV.log(f"Ensemble training features: {len(ids):,} molecules")

    # 2. Siamese ECFP4 metric learner and Mass2SMILES functional-group head
    siamese, siamese_hist = train_siamese(spec, fp, ENV.Deadline(EnsCFG.SIAMESE_MINUTES))
    del spec, fp
    fg_head, fg_hist = train_fg_head(
        glmr_model, make_loader(train_df, tokenizer, train=True, batch_size=EnsCFG.BATCH_FG), ENV.Deadline(EnsCFG.FG_HEAD_MINUTES)
    )
    torch.save({"state_dict": siamese.state_dict(), "weights_version": W.version}, out_dir / "siamese_ecfp4.pt")
    torch.save({"state_dict": fg_head.state_dict(), "weights_version": W.version}, out_dir / "mass2smiles_fg_head.pt")
    metrics["train"] = {"siamese_last_loss": siamese_hist[-1] if siamese_hist else None,
                        "fg_head_last_loss": fg_hist[-1] if fg_hist else None,
                        "siamese_epochs": len(siamese_hist), "fg_head_epochs": len(fg_hist)}

    # 3. Validation records on held-out timsTOF molecules -> per-stream and blended MRR@25, cross-fitted weights
    val_ids = read_split(split_dir, "val")
    val_ids = rng.choice(val_ids, size=min(EnsCFG.N_VAL_MOLECULES, len(val_ids)), replace=False).tolist()
    val_df = load_spectra_for_inchikeys(CFG.DATA_DIR, val_ids, timstof_window_only=True)
    records: List[Dict] = []
    run_blended_casmi26_inference(
        glmr_model, siamese, fg_head, CASMI26EnsembleReranker(), make_ensemble_loader(val_df, tokenizer, train=False, batch_size=16),
        tokenizer, retriever.smiles_arr, retriever.masses_arr, None, canonicalize_and_inchikey14, store_retrieve_fn,
        device=CFG.DEVICE, collect_records=records,
    )
    gt = ground_truth_ik14(val_df)
    prior = np.asarray(W.vec("nb3.reranker", ["w_gen", "w_pre", "w_siamese", "w_fg", "w_rrf"]))
    streams = {"gen": 0, "pre": 1, "siamese": 2, "fg": 3, "rrf": 4}
    metrics["val_stream_mrr25"] = {name: ensemble_mrr(records, np.eye(5)[i], gt) for name, i in streams.items()}
    fit = crossfit_blend_weights(records, gt, prior)
    metrics["val_blend"] = {k: v for k, v in fit.items() if k != "weights"}
    metrics["val_blend"]["n"] = len(records)
    ENV.log(f"Validation per-stream MRR@25 {metrics['val_stream_mrr25']}; blend {metrics['val_blend']}")
    ENV.write_json(out_dir / "fitted_weights.json", {
        "notebook": "nb3",
        "weights": {f"nb3.reranker.{n}": w for n, w in zip(["w_gen", "w_pre", "w_siamese", "w_fg", "w_rrf"], fit["weights"])},
        "evidence": {"metric": "MRR@25", "split": f"{len(records)} held-out timsTOF molecules, 2-fold cross-fit",
                     "prior": fit["val_mrr25_prior"], "fitted_crossfit": fit["val_mrr25_crossfit"]},
    })

    # 4. Test inference with the blend `adopt` will activate: the fit only if its cross-fitted MRR@25 beats the prior
    blend, used_fit = ENV.select_validated(prior, np.asarray(fit["weights"]), fit["val_mrr25_prior"], fit["val_mrr25_crossfit"])
    metrics["test_blend"] = {"weights": [float(x) for x in blend], "source": "fitted" if used_fit else W.version}
    ENV.log(f"Test blend ({metrics['test_blend']['source']}): {metrics['test_blend']['weights']}")
    test_df = pd.read_parquet(CFG.DATA_DIR / "test.parquet")
    rows = run_blended_casmi26_inference(
        glmr_model, siamese, fg_head, CASMI26EnsembleReranker(weights=tuple(blend)),
        make_ensemble_loader(test_df, tokenizer, train=False, batch_size=16),
        tokenizer, retriever.smiles_arr, retriever.masses_arr, None, canonicalize_and_inchikey14, store_retrieve_fn,
        device=CFG.DEVICE,
    )
    sub_df = pd.DataFrame(rows, columns=["molecule_id", "smiles"])
    validate_submission_file(sub_df, test_df["molecule_id"].unique().tolist())
    sub_df.to_csv(out_dir / "submission.csv", index=False)

    ENV.write_json(out_dir / "metrics.json", metrics)
    ENV.write_run_manifest("nb3", out_dir, W.version, {"metrics": metrics})
    ENV.log(f"Notebook 3 outputs written to {out_dir}")
    return metrics


if __name__ == "__main__":
    NB3_METRICS = run_ensemble_pipeline()
