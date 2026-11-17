# [Colab Cell 9] End-to-end execution driver on the competition parquet files: train, validate, infer, save outputs
def save_glmr_checkpoint(model: GLMRPipeline, tokenizer: SMILESTokenizer, path: Path, extra: Optional[Dict] = None) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "vocab": tokenizer.vocab,
            "max_smiles_len": tokenizer.max_len,
            "adduct_vocab": ADDUCT_VOCAB,
            "weights_version": W.version,
            **(extra or {}),
        },
        path,
    )
    return path


def load_glmr_checkpoint(path: Path, device: torch.device = CFG.DEVICE) -> Tuple[GLMRPipeline, SMILESTokenizer]:
    """Rebuilds the GLMR pipeline saved by notebook 1 (used by notebooks 2-5)."""
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    tokenizer = SMILESTokenizer(max_len=ckpt.get("max_smiles_len", CFG.MAX_SMILES_LEN))
    if ckpt["vocab"] != tokenizer.vocab:
        raise ValueError("Tokenizer vocabulary differs from the checkpoint; rebuild notebook 1 outputs.")
    if ckpt.get("adduct_vocab", ADDUCT_VOCAB) != ADDUCT_VOCAB:
        raise ValueError("Adduct vocabulary differs from the checkpoint; rebuild notebook 1 outputs.")
    model = GLMRPipeline(tokenizer)
    model.load_state_dict(ckpt["state_dict"])
    return model.to(device).eval(), tokenizer


def save_library(out_dir: Path, lib_smiles: List[str], lib_ik14: List[str], lib_masses: np.ndarray, lib_embs: torch.Tensor) -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"smiles": lib_smiles, "inchikey14": lib_ik14, "exact_mass": lib_masses}).to_parquet(
        out_dir / "library.parquet", index=False
    )
    np.save(out_dir / "lib_embs_fp16.npy", lib_embs.numpy().astype(np.float16))


def load_library(lib_dir: Path) -> Tuple[List[str], List[str], np.ndarray, torch.Tensor]:
    lib = pd.read_parquet(Path(lib_dir) / "library.parquet")
    embs = torch.from_numpy(np.load(Path(lib_dir) / "lib_embs_fp16.npy").astype(np.float32))
    return lib["smiles"].tolist(), lib["inchikey14"].tolist(), lib["exact_mass"].to_numpy(np.float64), embs


def make_loader(df: pd.DataFrame, tokenizer: SMILESTokenizer, train: bool, batch_size: int = CFG.BATCH_SIZE) -> DataLoader:
    ds = MultiSpectrumMoleculeDataset(
        df, tokenizer,
        max_spectra_per_mol=CFG.MAX_SPECTRA_TRAIN if train else CFG.MAX_SPECTRA_INFER,
        is_train=train,
    )
    workers = CFG.NUM_WORKERS if train else 0
    return DataLoader(
        ds, batch_size=batch_size, shuffle=train, drop_last=train,
        num_workers=workers, persistent_workers=workers > 0,
    )


def predictions_from_submission(sub_df: pd.DataFrame) -> Dict[str, List[str]]:
    return {str(r.molecule_id): [s for s in str(r.smiles).split(";") if s] for r in sub_df.itertuples()}


@torch.no_grad()
def evaluate_stage1_retrieval(model, loader, lib_smiles, lib_ik14, lib_masses, lib_embs) -> Dict[str, float]:
    """Stage 1 only (no generation): MRR@25 and recall@K of the true inchikey14 in the K candidates."""
    model.eval()
    preds, gt, hits = {}, {}, []
    lib_embs_dev = lib_embs.to(CFG.DEVICE)
    for batch in loader:
        batch = {k: (v.to(CFG.DEVICE) if isinstance(v, torch.Tensor) else v) for k, v in batch.items()}
        _, e_s_mol, _ = model.encode_molecule_spectra(batch)
        _, idx_batch = retrieve_top_k_candidates(
            e_s_mol, batch["precursor_mz"], batch["primary_adduct"], lib_masses, lib_embs,
            top_k=CFG.TOP_K_CANDIDATES, ionization_modes=batch["ionization_mode"],
        )
        for b, mol_id in enumerate(batch["molecule_id"]):
            idx = idx_batch[b]
            scores = torch.matmul(lib_embs_dev[idx], e_s_mol[b]).cpu().numpy()
            fallback = torch.topk(torch.matmul(lib_embs_dev, e_s_mol[b]), k=min(100, len(lib_smiles))).indices.cpu().tolist()
            rec = {"cand_smiles": [lib_smiles[i] for i in idx], "stage1_scores": scores, "stage2_scores": None,
                   "gen_smiles": None, "gen_ik14": None, "gen_mass_ok": False,
                   "backfill_smiles": [lib_smiles[i] for i in fallback]}
            preds[mol_id] = assemble_ranked_candidates(rec, 0.0, 0)
            gt[mol_id] = batch["inchikey14"][b]
            hits.append(float(batch["inchikey14"][b] in {lib_ik14[i] for i in idx}))
    return {"mrr25": compute_casmi_mrr25(preds, gt), f"recall_at_{CFG.TOP_K_CANDIDATES}": float(np.mean(hits)), "n": len(gt)}


def sweep_rerank_knobs(records: List[Dict], gt: Dict[str, str], seed: int = CFG.SEED) -> Dict:
    """Grid over alpha and generated-structure rank with 2-fold cross-fitting for an honest estimate."""
    alphas = [round(a, 2) for a in np.linspace(0.0, 1.0, 11)]
    ranks = [0, 1, 3, 10, 25]
    grid = [(a, r) for a in alphas for r in ranks]

    def mrr(recs, a, r):
        preds = {rec["molecule_id"]: assemble_ranked_candidates(rec, a, r) for rec in recs}
        return compute_casmi_mrr25(preds, {rec["molecule_id"]: gt[rec["molecule_id"]] for rec in recs})

    rng = np.random.default_rng(seed)
    order = rng.permutation(len(records))
    folds = [[records[i] for i in order[0::2]], [records[i] for i in order[1::2]]]
    cv_scores = []
    for fit, held in ((folds[0], folds[1]), (folds[1], folds[0])):
        best = max(grid, key=lambda g: mrr(fit, *g))
        cv_scores.append(mrr(held, *best) * len(held))
    full_scores = {f"{a}|{r}": mrr(records, a, r) for a, r in grid}
    best_key = max(full_scores, key=full_scores.get)
    best_alpha, best_rank = best_key.split("|")
    prior = (W("nb1.rerank.alpha"), W.int("nb1.rerank.generated_insert_rank"))
    return {
        "best_alpha": float(best_alpha),
        "best_generated_insert_rank": int(best_rank),
        "val_mrr25_prior": mrr(records, *prior),
        "val_mrr25_best_in_sample": full_scores[best_key],
        "val_mrr25_crossfit": float(sum(cv_scores) / len(records)),
        "grid": full_scores,
    }


def run_glmr_pipeline(out_dir: Optional[Path] = None) -> Dict:
    out_dir = Path(out_dir or CFG.OUTPUT_DIR)
    metrics: Dict = {"weights_version": W.version}
    tokenizer = SMILESTokenizer()

    # 1. Leakage-free real data (timsTOF validation blocks never seen in training)
    train_df, val_df, lib_df = load_competition_frames(CFG.DATA_DIR, CFG.N_TRAIN_MOLECULES, CFG.N_VAL_MOLECULES)
    train_df = train_df.dropna(subset=["normalized_smiles"]).reset_index(drop=True)
    (out_dir / "split").mkdir(parents=True, exist_ok=True)
    (out_dir / "split" / "val_inchikey14.txt").write_text("\n".join(sorted(val_df["inchikey14"].unique())) + "\n")
    (out_dir / "split" / "train_inchikey14.txt").write_text("\n".join(sorted(train_df["inchikey14"].unique())) + "\n")
    train_loader = make_loader(train_df, tokenizer, train=True)

    model = GLMRPipeline(tokenizer).to(CFG.DEVICE)
    pretrained = load_pretrained_mol_encoder(model, CFG.MOL_ENCODER_CKPT)

    # 2. Stage 1: dual-path InfoNCE (spectrum <-> molecule)
    s1_params = list(model.spec_encoder.parameters()) + list(model.aggregator.parameters())
    if not pretrained:
        s1_params += list(model.mol_encoder.parameters())
    opt_s1 = torch.optim.AdamW(s1_params, lr=CFG.LR, weight_decay=CFG.WEIGHT_DECAY)
    deadline = ENV.Deadline(CFG.STAGE1_MINUTES)
    history = []
    for epoch in range(CFG.EPOCHS_STAGE1):
        loss = train_stage1_epoch(model, train_loader, opt_s1, freeze_mol_encoder=pretrained, deadline=deadline)
        history.append({"stage": 1, "epoch": epoch, "loss": loss})
        ENV.log(f"Stage 1 epoch {epoch}: InfoNCE {loss:.4f} ({deadline.remaining_min():.1f} min left)")
        if deadline.expired():
            break
    save_glmr_checkpoint(model, tokenizer, CFG.CHECKPOINT_DIR / "glmr_stage1.pt")

    # 3. Retrieval library: every train.parquet structure, embedded by the Stage 1 molecule encoder
    lib_smiles, lib_masses, lib_embs = build_candidate_library_index(model, lib_df["smiles"].tolist(), tokenizer)
    smi_to_ik14 = dict(zip(lib_df["smiles"], lib_df["inchikey14"]))
    lib_ik14 = [smi_to_ik14[s] for s in lib_smiles]
    save_library(out_dir / "library", lib_smiles, lib_ik14, lib_masses, lib_embs)
    ENV.log(f"Library: {len(lib_smiles):,} structures")

    rng = np.random.default_rng(CFG.SEED)
    val_ids = sorted(val_df["inchikey14"].unique())
    eval_ids = set(rng.choice(val_ids, size=min(CFG.N_VAL_EVAL, len(val_ids)), replace=False).tolist())
    eval_df = val_df[val_df["inchikey14"].isin(eval_ids)].reset_index(drop=True)
    # Same molecules as the two-stage evaluation below, so the Stage 1 vs Stage 2 numbers are comparable.
    metrics["val_stage1"] = evaluate_stage1_retrieval(
        model, make_loader(eval_df, tokenizer, train=False, batch_size=32), lib_smiles, lib_ik14, lib_masses, lib_embs
    )
    ENV.log(f"Validation Stage 1 only: {metrics['val_stage1']}")

    # 4. Stage 2: cross-fusion + decoder with frozen encoders
    opt_s2 = torch.optim.AdamW(
        list(model.cross_fusion.parameters()) + list(model.mol_decoder.parameters()),
        lr=CFG.LR, weight_decay=CFG.WEIGHT_DECAY,
    )
    deadline = ENV.Deadline(CFG.STAGE2_MINUTES)
    for epoch in range(CFG.EPOCHS_STAGE2):
        loss = train_stage2_epoch(model, train_loader, lib_masses, lib_embs, opt_s2, tokenizer.pad_id, deadline=deadline)
        history.append({"stage": 2, "epoch": epoch, "loss": loss})
        ENV.log(f"Stage 2 epoch {epoch}: token CE {loss:.4f} ({deadline.remaining_min():.1f} min left)")
        if deadline.expired():
            break
    save_glmr_checkpoint(model, tokenizer, out_dir / "glmr_model.pt")
    pd.DataFrame(history).to_csv(out_dir / "train_log.csv", index=False)

    # 5. Full two-stage validation on held-out timsTOF molecules + re-ranking knob sweep
    records: List[Dict] = []
    val_sub = predict_and_rerank_test_molecules(
        model, make_loader(eval_df, tokenizer, train=False, batch_size=16), tokenizer,
        lib_smiles, lib_masses, lib_embs, collect_records=records,
    )
    gt = {mid: mid for mid in val_sub["molecule_id"]}  # validation groups are keyed by inchikey14
    metrics["val_two_stage_mrr25"] = compute_casmi_mrr25(predictions_from_submission(val_sub), gt)
    metrics["val_generated_valid_rate"] = float(np.mean([r["gen_smiles"] is not None for r in records]))
    metrics["val_generated_mass_ok_rate"] = float(np.mean([r["gen_mass_ok"] for r in records]))
    sweep = sweep_rerank_knobs(records, gt)
    metrics["rerank_sweep"] = {k: v for k, v in sweep.items() if k != "grid"}
    val_sub.to_csv(out_dir / "val_predictions.csv", index=False)
    ENV.write_json(out_dir / "rerank_grid.json", sweep["grid"])
    ENV.write_json(out_dir / "fitted_weights.json", {
        "notebook": "nb1",
        "weights": {
            "nb1.rerank.alpha": sweep["best_alpha"],
            "nb1.rerank.generated_insert_rank": sweep["best_generated_insert_rank"],
        },
        "evidence": {
            "metric": "MRR@25",
            "split": f"{len(records)} held-out timsTOF molecules, 2-fold cross-fit",
            "prior": sweep["val_mrr25_prior"],
            "fitted_crossfit": sweep["val_mrr25_crossfit"],
        },
    })
    ENV.log(f"Validation two-stage MRR@25 {metrics['val_two_stage_mrr25']:.4f}; sweep {metrics['rerank_sweep']}")

    # 6. Test inference with the active (Evidently) weights
    test_df = pd.read_parquet(CFG.DATA_DIR / "test.parquet")
    sub_df = predict_and_rerank_test_molecules(
        model, make_loader(test_df, tokenizer, train=False, batch_size=16), tokenizer,
        lib_smiles, lib_masses, lib_embs,
    )
    validate_submission_file(sub_df, test_df["molecule_id"].unique().tolist())
    sub_df.to_csv(CFG.SUBMISSION_PATH, index=False)

    ENV.write_json(out_dir / "metrics.json", metrics)
    ENV.write_run_manifest("nb1", out_dir, W.version, {"metrics": metrics})
    ENV.log(f"Notebook 1 outputs written to {out_dir}")
    return metrics


if __name__ == "__main__":
    NB1_METRICS = run_glmr_pipeline()
