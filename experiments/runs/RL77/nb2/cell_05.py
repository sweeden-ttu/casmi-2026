# [Colab Companion Cell 5] Build the offline store from the competition data, measure retrieval, save outputs
@torch.no_grad()
def evaluate_store_retrieval(retriever: "OfflineMassAndFormulaRetriever", model, loader, answer_key: Dict[str, str],
                             formula_priors_fn=None) -> Dict[str, float]:
    """Stage 1 retrieval from the offline store on held-out molecules: MRR@25, recall@K, mass-window coverage."""
    model.eval()
    preds, gt, hit_k, in_catalog_window = {}, {}, [], []
    for batch in loader:
        batch = {k: (v.to(CFG.DEVICE) if isinstance(v, torch.Tensor) else v) for k, v in batch.items()}
        _, e_s_mol, _ = model.encode_molecule_spectra(batch)
        priors = formula_priors_fn(batch) if formula_priors_fn else None
        _, ids_batch = retriever.retrieve_top_k_batch(
            e_s_mol, batch["precursor_mz"], batch["primary_adduct"],
            formula_priors_batch=priors, ionization_modes=batch["ionization_mode"],
        )
        e_np = e_s_mol.cpu().numpy().astype(np.float32)
        for b, mol_id in enumerate(batch["molecule_id"]):
            ids = ids_batch[b]
            neutral = neutral_mass_from_precursor(
                float(batch["precursor_mz"][b]), batch["primary_adduct"][b], batch["ionization_mode"][b]
            )
            rec = {
                "cand_smiles": retriever.smiles_arr[ids].tolist(),
                "stage1_scores": np.asarray(retriever.mmap_embs[ids], dtype=np.float32) @ e_np[b],
                "stage2_scores": None,
                "in_window": retriever.in_window(ids, neutral),
                "gen_smiles": None, "gen_ik14": None, "gen_mass_ok": False, "backfill_smiles": [],
            }
            truth = answer_key[str(mol_id)]
            preds[mol_id], gt[mol_id] = assemble_ranked_candidates(rec, 0.0, 0), truth
            hit_k.append(truth in set(retriever.ik14_arr[ids].tolist()))
            lo, hi = retriever.window(neutral, W("nb2.retrieval.ppm_tol"))
            in_catalog_window.append(truth in set(retriever.ik14_arr[lo:hi].tolist()))
    return {
        "mrr25": compute_casmi_mrr25(preds, gt),
        "recall_at_k": float(np.mean(hit_k)),
        "truth_in_mass_window": float(np.mean(in_catalog_window)),
        "n": len(gt),
    }


def run_offline_store_pipeline() -> Dict:
    out_dir = IndexCFG.EXPORT_DIR
    metrics: Dict = {"weights_version": W.version}
    nb1_dir = ENV.upstream_dir("nb1", "glmr_model.pt")
    model, tokenizer = load_glmr_checkpoint(nb1_dir / "glmr_model.pt", device=CFG.DEVICE)

    # 1-2. Catalog (train.parquet tier 1 + COCONUT tier 2), FP16 memmap, FAISS and formula indices
    catalog_df = build_mass_sorted_catalog()
    metrics["catalog"] = {
        "n_structures": int(len(catalog_df)),
        "n_tier1_train": int((catalog_df["source_tier"] == 1).sum()),
        "n_tier2_external": int((catalog_df["source_tier"] == 2).sum()),
        "n_formulas": int(catalog_df["molecular_formula"].nunique()),
    }
    ENV.log(f"Catalog: {metrics['catalog']}")
    encode_catalog_to_memmap_and_faiss(catalog_df, model.mol_encoder, tokenizer, device=CFG.DEVICE)

    # 3. Offline wheels for internet-off submissions
    prepare_and_upload_kaggle_dataset()

    # 4. Retrieval quality on notebook 1's held-out timsTOF molecules (same split everywhere).
    # Every validation molecule is a train.parquet structure (tier 1), so this measures in-library retrieval.
    retriever = OfflineMassAndFormulaRetriever(out_dir)
    split_dir = ENV.upstream_dir("nb1", "val_inchikey14.txt")
    val_df = load_spectra_for_inchikeys(CFG.DATA_DIR, read_split(split_dir, "val"), timstof_window_only=True)
    metrics["val_store_stage1"] = evaluate_store_retrieval(
        retriever, model, make_loader(val_df, tokenizer, train=False, batch_size=32), ground_truth_ik14(val_df)
    )
    ENV.log(f"Validation retrieval from the offline store: {metrics['val_store_stage1']}")

    ENV.write_json(out_dir / "metrics.json", metrics)
    ENV.write_run_manifest("nb2", out_dir, W.version, {"metrics": metrics, "nb1_outputs": str(nb1_dir)})
    ENV.log(f"Notebook 2 outputs written to {out_dir}")
    return metrics


if __name__ == "__main__":
    NB2_METRICS = run_offline_store_pipeline()
