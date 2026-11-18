# [Colab Formula Filter Cell 5] How to wire formula_priors_batch into the Kaggle inference loop
# The PDF snippet referenced glmr_model / fg_head / mist_filter / offline_retriever as globals
# (NameError when run as a cell); it is wrapped in a function so notebooks 4 and 5 can call it.
@torch.no_grad()
def retrieve_with_formula_priors(
    batch: Dict[str, torch.Tensor],
    glmr_model: nn.Module,
    fg_head: nn.Module,
    mist_filter: FastMISTSubformulaFilter,
    offline_retriever: "OfflineMassAndFormulaRetriever",
    top_k: int = W.int("nb3.retrieval.top_k"),
    ppm_tol: float = W("nb2.retrieval.ppm_tol"),
):
    # 1. Encode multi-spectrum sequence and predict Mass2SMILES 71-D vector first
    h_s_mol, e_s_mol, _ = glmr_model.encode_molecule_spectra(batch)
    fg_71_pred, _ = fg_head(h_s_mol)  # (B, 71)

    # 2. Generate top-3 MIST-CF / SIRIUS candidate molecular formulas per molecule_id
    formula_priors_batch = generate_formula_priors_batch(
        batch=batch,
        subformula_filter=mist_filter,
        fg_71_pred_batch=fg_71_pred,
        ppm_window=W("nb4.priors.ppm_window"),
        top_k_formulas=W.int("nb4.priors.top_k_formulas"),
    )

    # 3. Query OfflineMassAndFormulaRetriever using both Formula Priors + PPM Mass Window
    h_m_k, topk_indices_batch = offline_retriever.retrieve_top_k_batch(
        e_s_batch=e_s_mol,
        precursor_mzs=batch["precursor_mz"],
        adducts=batch["primary_adduct"],
        adduct_mass_shift=ADDUCT_MASS_SHIFT,
        top_k=top_k,
        ppm_tol=ppm_tol,
        formula_priors_batch=formula_priors_batch,
        ionization_modes=batch["ionization_mode"],
    )
    return h_s_mol, e_s_mol, fg_71_pred, formula_priors_batch, h_m_k, topk_indices_batch
