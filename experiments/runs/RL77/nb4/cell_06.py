# [Colab Formula Filter Cell 6] Evaluate the filter on held-out molecules, fit the score weights, priors for the test set
from rdkit.Chem import rdMolDescriptors as _rdmd


def truth_formula(smiles: str) -> Optional[str]:
    mol = Chem.MolFromSmiles(smiles) if isinstance(smiles, str) else None
    return _rdmd.CalcMolFormula(mol) if mol is not None else None


def load_fg_head(nb3_dir: Path) -> "Mass2SMILESFunctionalGroupHead":
    fg_head = Mass2SMILESFunctionalGroupHead()
    fg_head.load_state_dict(torch.load(nb3_dir / "mass2smiles_fg_head.pt", map_location="cpu")["state_dict"])
    return fg_head.to(CFG.DEVICE).eval()


@torch.no_grad()
def collect_formula_records(loader, glmr_model, fg_head, mist_filter) -> List[Dict]:
    records = []
    for batch in loader:
        batch = {k: (v.to(CFG.DEVICE) if isinstance(v, torch.Tensor) else v) for k, v in batch.items()}
        h_s_mol, _, _ = glmr_model.encode_molecule_spectra(batch)
        fg_np = fg_head(h_s_mol)[0].cpu().numpy()
        for b, mol_id in enumerate(batch["molecule_id"]):
            inputs = molecule_formula_inputs(batch, b, mist_filter)
            formulas, comps = mist_filter.score_components(fg_71_pred=fg_np[b], **inputs)
            records.append({
                "molecule_id": str(mol_id),
                "formulas": [str(f) for f in formulas],
                "components": comps,
                "truth": truth_formula(batch["smiles"][b]) if "smiles" in batch else None,
                "neutral_mass": float(inputs["neutral_mass"]),
            })
    return records


def formula_topk_accuracy(records: List[Dict], weights: Dict[str, float], k: int) -> float:
    hits = []
    for r in records:
        if not r["formulas"]:
            hits.append(False)
            continue
        total = sum(weights[key] * r["components"][key] for key in FastMISTSubformulaFilter.SCORE_KEYS)
        top = [r["formulas"][i] for i in np.argsort(-total, kind="stable")[:k]]
        hits.append(r["truth"] in top)
    return float(np.mean(hits)) if hits else 0.0


def fit_score_weights(records: List[Dict], start: Dict[str, float], passes: int = 3) -> Dict[str, float]:
    """Coordinate search on top-1 formula accuracy (top-3 breaks ties)."""
    grid = [0.0, 0.25, 0.5, 0.8, 1.0, 1.2, 1.5, 2.0, 3.0]
    best = dict(start)

    def objective(w):
        return (formula_topk_accuracy(records, w, 1), formula_topk_accuracy(records, w, 3))

    best_score = objective(best)
    for _ in range(passes):
        for key in FastMISTSubformulaFilter.SCORE_KEYS:
            for value in grid:
                trial = {**best, key: value}
                score = objective(trial)
                if score > best_score:
                    best, best_score = trial, score
    return best


def formula_priority(cand_formulas: List[Optional[str]], priors: Optional[List[str]], rank_formulas: int) -> np.ndarray:
    """Ranking tier per candidate: i for the i-th formula prior (i < rank_formulas), rank_formulas otherwise.
    The PDF merged formula hits into the mass-window union, where they changed nothing (Kaggle nb4 v1)."""
    tiers = np.full(len(cand_formulas), rank_formulas, dtype=np.int64)
    for i, form in enumerate((priors or [])[:rank_formulas]):  # tier 0 ranks first; a repeated prior keeps its best tier
        tiers[np.asarray([f == form for f in cand_formulas], dtype=bool) & (tiers == rank_formulas)] = i
    return tiers


@torch.no_grad()
def collect_store_records(retriever, glmr_model, loader, priors_fn=None) -> List[Dict]:
    """Stage 1 retrieval from the offline store; ranking happens later so formula tiers can be compared."""
    records = []
    for batch in loader:
        batch = {k: (v.to(CFG.DEVICE) if isinstance(v, torch.Tensor) else v) for k, v in batch.items()}
        _, e_s_mol, _ = glmr_model.encode_molecule_spectra(batch)
        priors = priors_fn(batch) if priors_fn else None
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
            records.append({
                "molecule_id": str(mol_id),
                "cand_smiles": retriever.smiles_arr[ids].tolist(),
                "cand_formulas": retriever.catalog_df["molecular_formula"].to_numpy()[ids].tolist(),
                "priors": list(priors[b]) if priors else [],
                "stage1_scores": np.asarray(retriever.mmap_embs[ids], dtype=np.float32) @ e_np[b],
                "stage2_scores": None,
                "in_window": retriever.in_window(ids, neutral),
                "gen_smiles": None, "gen_ik14": None, "gen_mass_ok": False, "backfill_smiles": [],
            })
    return records


def rank_store_records(records: List[Dict], rank_formulas: int) -> Dict[str, List[str]]:
    return {
        r["molecule_id"]: assemble_ranked_candidates(
            {**r, "priority": formula_priority(r["cand_formulas"], r["priors"], rank_formulas)}, 0.0, 0
        )
        for r in records
    }


def crossfit_rank_formulas(records: List[Dict], gt: Dict[str, str], options=(0, 1, 2, 3), seed: int = CFG.SEED) -> Dict:
    """Pick how many formula priors become ranking tiers: 2-fold cross-fit over `options` on MRR@25."""
    rr = {k: np.array([compute_casmi_mrr25({m: p}, {m: gt[m]}) for m, p in rank_store_records(records, k).items()])
          for k in options}
    order = np.random.default_rng(seed).permutation(len(records))
    folds = [order[0::2], order[1::2]]
    held = 0.0
    for fit, hold in ((folds[0], folds[1]), (folds[1], folds[0])):
        best = max(options, key=lambda k: (rr[k][fit].mean(), -k))
        held += rr[best][hold].sum()
    best_all = max(options, key=lambda k: (rr[k].mean(), -k))
    prior = W.int("nb4.priors.rank_formulas")
    return {
        "mrr25_by_rank_formulas": {str(k): float(rr[k].mean()) for k in options},
        "best_rank_formulas": int(best_all),
        "val_mrr25_prior": float(rr[prior].mean()) if prior in rr else None,
        "val_mrr25_crossfit": float(held / len(records)),
    }


def run_formula_pipeline() -> Dict:
    out_dir = ENV.work_dir()
    metrics: Dict = {"weights_version": W.version}
    nb1_dir = ENV.upstream_dir("nb1", "glmr_model.pt")
    split_dir = ENV.upstream_dir("nb1", "val_inchikey14.txt")
    nb3_dir = ENV.upstream_dir("nb3", "mass2smiles_fg_head.pt")
    glmr_model, tokenizer = load_glmr_checkpoint(nb1_dir / "glmr_model.pt", device=CFG.DEVICE)
    fg_head = load_fg_head(nb3_dir)
    retriever = OfflineMassAndFormulaRetriever()
    bank = CandidateFormulaBank(retriever.catalog_df["molecular_formula"].dropna().unique().tolist())
    mist_filter = FastMISTSubformulaFilter(bank)
    ENV.log(f"Formula bank: {len(bank.formulas):,} formulas passing Senior / Seven Golden Rules")

    # 1. Formula identification on notebook 1's held-out timsTOF molecules
    val_df = load_spectra_for_inchikeys(CFG.DATA_DIR, read_split(split_dir, "val"), timstof_window_only=True)
    val_loader = make_loader(val_df, tokenizer, train=False, batch_size=32)
    records = [r for r in collect_formula_records(val_loader, glmr_model, fg_head, mist_filter) if r["truth"]]
    prior = dict(mist_filter.score_weights)
    metrics["val_formula"] = {
        "n": len(records),
        "truth_in_candidate_window": float(np.mean([r["truth"] in r["formulas"] for r in records])),
        "top1_prior": formula_topk_accuracy(records, prior, 1),
        "top3_prior": formula_topk_accuracy(records, prior, 3),
    }

    # 2. Cross-fit the four composite weights (fit on one half, score the other), then fit on all
    rng = np.random.default_rng(CFG.SEED)
    order = rng.permutation(len(records))
    folds = [[records[i] for i in order[0::2]], [records[i] for i in order[1::2]]]
    held = sum(formula_topk_accuracy(h, fit_score_weights(f, prior), 1) * len(h) for f, h in ((folds[0], folds[1]), (folds[1], folds[0])))
    fitted = fit_score_weights(records, prior)
    metrics["val_formula"].update({
        "top1_fitted_crossfit": held / len(records),
        "top1_fitted_in_sample": formula_topk_accuracy(records, fitted, 1),
        "top3_fitted_in_sample": formula_topk_accuracy(records, fitted, 3),
    })
    fits = [{  # fitted_weights.json is written after the formula-tier fit below (one evidence block per fit)
        "weights": {f"nb4.score.{k}": v for k, v in fitted.items()},
        "evidence": {"metric": "formula top-1 accuracy", "split": f"{len(records)} held-out timsTOF molecules, 2-fold cross-fit",
                     "prior": metrics["val_formula"]["top1_prior"], "fitted_crossfit": metrics["val_formula"]["top1_fitted_crossfit"]},
    }]
    ENV.log(f"Formula identification: {metrics['val_formula']}")
    # From here on the filter scores with the weights `adopt` will activate (fit only if it wins on held-out halves).
    chosen, used_fit = ENV.select_validated(prior, fitted, metrics["val_formula"]["top1_prior"],
                                            metrics["val_formula"]["top1_fitted_crossfit"])
    mist_filter.score_weights = dict(chosen)
    metrics["test_score_weights"] = {**chosen, "source": "fitted" if used_fit else W.version}

    # 3. Does formula-constrained retrieval help? Same molecules, offline store, Stage 1 ranking, with the
    #    top-1..3 formula priors as ranking tiers (0 = mass window only), cross-fitted like the other knobs.
    def priors_fn(batch):
        return generate_formula_priors_batch(batch, mist_filter, fg_head(glmr_model.encode_molecule_spectra(batch)[0])[0])

    gt = ground_truth_ik14(val_df)
    store_records = collect_store_records(retriever, glmr_model, val_loader, priors_fn)
    tiers = crossfit_rank_formulas(store_records, gt)
    metrics["val_retrieval_mrr25"] = {
        "mass_window_only": compute_casmi_mrr25(rank_store_records(collect_store_records(retriever, glmr_model, val_loader), 0), gt),
        **{f"formula_tiers_{k}": v for k, v in tiers["mrr25_by_rank_formulas"].items()},
    }
    metrics["formula_tier_fit"] = {k: v for k, v in tiers.items() if k != "mrr25_by_rank_formulas"}
    fits.append({
        "weights": {"nb4.priors.rank_formulas": tiers["best_rank_formulas"]},
        "evidence": {"metric": "MRR@25 (offline store, Stage 1)", "split": f"{len(store_records)} held-out timsTOF molecules, 2-fold cross-fit",
                     "prior": tiers["val_mrr25_prior"], "fitted_crossfit": tiers["val_mrr25_crossfit"]},
    })
    rank_formulas, used_fit = ENV.select_validated(W.int("nb4.priors.rank_formulas"), tiers["best_rank_formulas"],
                                                   tiers["val_mrr25_prior"], tiers["val_mrr25_crossfit"])
    metrics["test_rank_formulas"] = {"value": int(rank_formulas), "source": "fitted" if used_fit else W.version}
    ENV.write_json(out_dir / "fitted_weights.json", {"notebook": "nb4", "fits": fits})
    ENV.log(f"Validation retrieval MRR@25: {metrics['val_retrieval_mrr25']}; tier fit {metrics['formula_tier_fit']}; "
            f"test uses rank_formulas={rank_formulas}")

    # 4. Test set: formula priors (written for inspection and reuse) + submission ranked with the chosen tiers
    test_df = pd.read_parquet(CFG.DATA_DIR / "test.parquet")
    test_loader = make_loader(test_df, tokenizer, train=False, batch_size=32)
    test_records = collect_store_records(retriever, glmr_model, test_loader, priors_fn)
    ENV.write_json(out_dir / "test_formula_priors.json", {r["molecule_id"]: r["priors"] for r in test_records})
    preds = rank_store_records(test_records, rank_formulas)
    sub_df = pd.DataFrame([{"molecule_id": m, "smiles": ";".join(p)} for m, p in preds.items()], columns=["molecule_id", "smiles"])
    validate_submission_file(sub_df, test_df["molecule_id"].unique().tolist())
    sub_df.to_csv(out_dir / "submission.csv", index=False)

    ENV.write_json(out_dir / "metrics.json", metrics)
    ENV.write_run_manifest("nb4", out_dir, W.version, {"metrics": metrics})
    ENV.log(f"Notebook 4 outputs written to {out_dir}")
    return metrics


if __name__ == "__main__":
    NB4_METRICS = run_formula_pipeline()
