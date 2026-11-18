<!-- intro -->
# CASMI 2026 · RL77 Notebook 4 — Precursor + Adduct + Isotope Subformula Filter

*Assembled from **CASMI 2026 Colab Rubric.pdf** (pages 38–48). Python cells are the reviewed `experiments/runs/RL77/nb4/cell_NN.py` files; text cells are taken from the PDF.*

> **User prompt:** Write a fast precursor mass + adduct + isotope subformula filter in Python to generate the top-3 candidate molecular formulas (`formula_priors_batch`) for the `OfflineMassAndFormulaRetriever`.

Predicting the top-3 candidate molecular formulas (`formula_priors_batch`) before candidate retrieval narrows the search space from thousands of nominal-mass isomers down to exact elemental isomers, driving the jump from GLMR's Weight Library (40.35% Top-1) to its Formula Library (43.52% Top-1). This vectorized Python module implements a fast, offline **MIST-CF / SIRIUS-inspired subformula filter** that fuses five physical and neural signals in under 0.8 milliseconds per molecule:

1. **Multi-Spectrum Neutral Mass Consensus:** Pools `precursor_mz` across all spectra and adducts belonging to a `molecule_id` to compute a variance-weighted neutral monoisotopic mass.
2. **Senior's Valence & RDBE Parity Filter:** Enforces closed-shell neutral valence rules (Nitrogen Rule and integer Ring and Double Bond Equivalents).
3. **Diagnostic Neutral Loss Subformula Tree:** Matches observed MS/MS neutral losses (`precursor_mz - ms2_mzs`) against 36 exact-mass diagnostic losses (H₂O, NH₃, CO, SO₃, HPO₃, HCl, hexose, pentose) and penalizes candidate formulas whose elemental stoichiometry cannot form the observed loss.
4. **Raw MS2 Fine-Isotope & Halogen Pattern Detector:** Scans raw fragment pairs separated by 1.9970–1.9980 Da (³⁷Cl, ⁸¹Br, ³⁴S) and 1.00335 Da (¹³C) to score halogen and sulfur presence.
5. **Mass2SMILES 71-D Heteroatom Prior Integration:** Aligns candidate formula heteroatom counts (N, O, S, P, F, Cl, Br, I) directly with the neural predictions from `fg_71_pred[:, 60:68]`.

<!-- before:cell_01 -->
## Cell 1: Exact Atomic Weights, Hill Formula Parser & Diagnostic Neutral Loss Table

<!-- before:cell_02 -->
## Cell 2: Vectorized Formula Catalog Index & De Novo Senior's Rule Enumerator

This class indexes all unique formulas from `candidate_catalog_sorted.parquet` by exact monoisotopic mass and includes a fast vectorized **Seven Golden Rules / Senior's Rule** de novo generator as a fallback for novel structures whose molecular formula does not yet exist in the catalog.

<!-- before:cell_03 -->
## Cell 3: Fast MIST-CF / SIRIUS Multi-Spectrum Subformula Scorer

> **Review fix:** the consensus mass used the shift table (formate/acetate spellings and `[2M+X]` fell back to `[M+H]+`); it now uses `neutral_mass_from_precursor()`. The scorer is split into `score_components()` and `combine()` so validation can fit the composite weights without re-enumerating formulas. Every coefficient (ppm σ, neutral-loss bonus/penalty, isotope bonuses, heteroatom weights, RDBE terms, and the 1.50 / 1.00 / 0.80 / 1.20 composite weights) comes from the Evidently weights.

<!-- before:cell_04 -->
## Cell 4: Batch Hook (`generate_formula_priors_batch`) for `OfflineMassAndFormulaRetriever`

Drop `generate_formula_priors_batch` directly into your inference loop right after `fg_71_pred, _ = fg_head(h_s_mol)` and pass the output into `retriever.retrieve_top_k_batch(..., formula_priors_batch=formula_priors_batch)`.

> **Review fix:** every spectrum was assigned the primary adduct, so a molecule measured as `[M+H]+` and `[M+Na]+` produced a 22 Da disagreement in the consensus mass; per-spectrum adducts are decoded from `batch["adduct"]`. The isotope detector's ratio thresholds assume linear intensities, but the dataset stores √I (a 0.045 ³⁴S ratio became 0.21), so intensities are squared before detection. The synthetic chrysin smoke test is replaced by the real run in Cell 6.

<!-- before:cell_05 -->
## How to Wire `formula_priors_batch` into Your Kaggle Inference Loop

In **Cell 5** of the Ensemble Inference loop (`run_blended_casmi26_inference`), swap the Stage 1 retrieval call so `fg_head(h_s_mol)` runs immediately before `retrieve_top_k_batch`.

> **Review fix:** the snippet used `glmr_model`, `fg_head`, `mist_filter`, and `offline_retriever` as undefined globals (NameError as a cell); the same three steps are wrapped in `retrieve_with_formula_priors()`.

<!-- before:cell_06 -->
## Cell 6: Evaluate on Held-Out Molecules, Fit the Score Weights, Priors for the Test Set

The PDF's verification used five hand-written formulas and a mock batch. This cell runs the filter on notebook 1's held-out timsTOF molecules with the notebook-3 Mass2SMILES head: formula top-1 / top-3 accuracy against the RDKit formula of the true structure, a 2-fold cross-fitted coordinate search over the four composite weights (written to `fitted_weights.json` for `tools/evidently_weights.py adopt`), and Stage 1 retrieval MRR@25 from the offline store with the top-1…3 formula priors as ranking tiers (0 = mass window only), with the number of tiers cross-fitted as `nb4.priors.rank_formulas`. It writes `test_formula_priors.json` and a `submission.csv` ranked with the chosen tiers. Runs on CPU.

> **Review fix (after the nb4 Kaggle run):** `get_candidate_indices` returns the *union* of formula-matched rows and the mass window, and a formula within 15 ppm is already inside that window, so the PDF's "Priority A" changed nothing: MRR@25 with priors equalled the mass-window-only value to 16 digits (0.16405). Formula priors are now ranking tiers (`priority` in `assemble_ranked_candidates`), and notebook 5 applies the adopted tier count.

<!-- end -->
*The test submission scores formulas with the fitted composite weights only if they beat the prior on the held-out halves, and ranks with the fitted number of formula tiers under the same rule (`metrics.json`: `test_score_weights`, `test_rank_formulas`). Notebook 5 recomputes formula priors on the fly with the adopted weights from the Evidently snapshot; `test_formula_priors.json` is kept for inspection.*
