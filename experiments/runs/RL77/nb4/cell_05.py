# 1. Encode multi-spectrum sequence and predict Mass2SMILES 71-D vector first
h_s_mol, e_s_mol, _ = glmr_model.encode_molecule_spectra(batch)
fg_71_pred, _ = fg_head(h_s_mol)  # (B, 71)

# 2. Generate top-3 MIST-CF / SIRIUS candidate molecular formulas per molecule_id
formula_priors_batch = generate_formula_priors_batch(
    batch=batch,
    subformula_filter=mist_filter,
    fg_71_pred_batch=fg_71_pred,
    ppm_window=15.0,
    top_k_formulas=3,
)

# 3. Query OfflineMassAndFormulaRetriever using both Formula Priors + PPM Mass Window
h_m_k, topk_indices_batch = offline_retriever.retrieve_top_k_batch(
    e_s_batch=e_s_mol,
    precursor_mzs=batch["precursor_mz"],
    adducts=batch["primary_adduct"],
    adduct_mass_shift=ADDUCT_MASS_SHIFT,
    top_k=60,
    ppm_tol=15.0,
    formula_priors_batch=formula_priors_batch,
)