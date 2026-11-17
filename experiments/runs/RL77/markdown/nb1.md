<!-- intro -->
# CASMI 2026 · RL77 Notebook 1 — Two-Stage GLMR (Pre-Retrieval + Spec-Mol Cross-Fusion)

*Assembled from **CASMI 2026 Colab Rubric.pdf** (Gemini conversation, pages 1–21). Python cells are the reviewed `experiments/runs/RL77/nb1/cell_NN.py` files; text cells are taken from the PDF.*

> **User prompt:** Use the @ml-for-chemoinformatics-on-kaggle to create rubric for colab notebooks that will most likely succeed at ranking in the top 10 percentile of the leaderboard for the CASMI 26 challenge on Kaggle

Reaching the top 10 percentile (Bronze medal threshold) in the **Enveda CASMI 2026 - Molecule ID From Mass Spectra** competition requires a public **MRR@25 of at least 0.487 to 0.500+**. Based on the `ml-for-chemoinformatics-on-kaggle` framework, the **GLMR** (Generative Language Model-based Retrieval) architecture, and **Mass2SMILES** spectral-to-structure methodologies, this 100-point rubric evaluates whether a Google Colab training and Kaggle inference workflow is engineered to cross the 0.490+ MRR@25 threshold.

## Summary Scoring Matrix

| Rubric Domain | Max Points | Top 10% Target Benchmark | Primary Failure Mode |
| --- | --- | --- | --- |
| 1. Validation & Metric Alignment | 15 | Local-to-Public gap under 0.03 | Random spectrum split leakage |
| 2. Spectral Curation & Harmonization | 15 | 256-d sinusoidal + neutral losses | Unfiltered noise and adduct drift |
| 3. Formula & Substructure Constraints | 15 | Mass + 60 functional group priors | Unconstrained isomer explosion |
| 4. Multi-Spectrum Hybrid Modeling | 30 | Two-stage Retrieval + Cross-Fusion | Single-spectrum or sparse de novo |
| 5. Post-Processing & Deduplication | 15 | 25 unique InChIKey14 per molecule | Duplicate tautomers or stereoisomers |
| 6. Colab-to-Kaggle Orchestration | 10 | Offline runtime under 9 hours | OOM crash or internet dependency |

The table above weights the six engineering pillars required to bridge the spectral-to-structural modality gap while respecting Kaggle's offline execution limits.

## Readiness Grading Scale

- **90 to 100 Points (Top 10% / Medal Contender):** Fully aligned with 0.490+ MRR@25 public pipelines; combines leakage-free `inchikey14` validation, multi-spectrum aggregation, two-stage GLMR cross-fusion re-ranking, and strict `inchikey14` deduplication.
- **75 to 89 Points (Top 25–30% / Competitive Baseline):** Strong single-stage retrieval or library matching (typical public MRR@25 of 0.450 to 0.480), but missing either context-aware generative re-ranking, multi-spectrum pooling, or tautomer deduplication.
- **Below 75 Points (High Shakeup / Failure Risk):** Vulnerable to severe local-to-public leaderboard divergence (MRR@25 below 0.250) due to random split leakage, unharmonized collision energies, or sparse de novo generation without retrieval priors.

---

> **User prompt:** Generate a complete PyTorch Colab notebook template implementing the two-stage GLMR (Pre-Retrieval + Spec-Mol Cross-Fusion) pipeline and multi-spectrum aggregator for CASMI 2026.

This PyTorch Google Colab notebook template implements the two-stage **GLMR (Generative Language Model-based Retrieval)** architecture paired with a **Multi-Spectrum Attention Aggregator** and **Mass2SMILES** spectral encodings for the **Enveda CASMI 2026** challenge. Each code block corresponds to a sequential Colab cell designed for A100/V100 training and offline Kaggle inference under the 9-hour runtime limit.

<!-- before:cell_01 -->
## Cell 1: Environment Setup, Dependencies & Global Configuration

> **Review fix:** `WEIGHT_decay` was misspelled while Cell 9 read `CFG.WEIGHT_DECAY` (AttributeError). Google Drive paths are replaced by `rl77_env` lookups (competition mount, `/kaggle/working`). Retrieval/beam/temperature values come from the Evidently weights snapshot `W`. The 300/30 epoch counts remain upper bounds; wall-clock budgets (`RL77_STAGE1_MINUTES`, `RL77_STAGE2_MINUTES`) keep the run inside a 12-hour Kaggle session.

<!-- before:cell_02 -->
## Cell 2: RDKit 2026.03.3 Tautomer Canonicalization, `inchikey14` & SMILES Tokenizer

> **Review fix:** The competition files spell formate `[M+CH2O2-H]-` and acetate `[M+C2H4O2-H]-` (the PDF used `[M+FA-H]-` / `[M+Hac-H]-`), and train has ~340k `[2M+X]` multimer spectra. Unknown adducts silently fell back to the `[M+H]+` shift, mis-estimating neutral mass by ~46 Da (formate) or 2× (multimers). `neutral_mass_from_precursor()` now handles n, z, and delta. The tokenizer vocabulary listed `C` twice and lacked `H` and `%`, so `[nH]`, `[C@H]`, and ring closures ≥10 (in >10% of train structures) decoded as empty strings; the vocabulary is now the measured train character set.

<!-- before:cell_03 -->
## Cell 3: Spectral Curation, Electron-Mass Alignment & Neutral Loss Encoding

<!-- before:cell_04 -->
## Cell 4: Leakage-Free `inchikey14` Grouped Dataset & Multi-Spectrum Collation

> **Review fix:** 46% of `train.parquet` `base_peak_intensity` values are null and `max(nan, 1.0)` returns `nan`, which propagated NaN into every encoder pass; non-finite values now default to 1e4. The primary precursor is taken from a singly charged monomer adduct. `load_competition_frames()` applies the same leakage rule as `create_leakage_free_splits()` while streaming the 3 GB parquet through pyarrow filters instead of pandas.

<!-- before:cell_05 -->
## Cell 5: GLMR Architecture & Multi-Spectrum Attention Aggregator

> **Review fix:** The decoder mixed a float causal mask with a boolean padding mask (deprecated in PyTorch). Beam search ran one forward pass per beam; live beams now decode in one batched pass per step (same search, ~beam-size× fewer passes).

<!-- before:cell_06 -->
## Cell 6: Stage 1 Training — Contrastive Pre-Retrieval (Dual-Path InfoNCE Loss)

> **Review fix:** GLMR freezes a *pre-trained* ChemFormer. The template froze a randomly initialized encoder in both stages and never loaded weights, so Stage 1 aligned spectra to random molecule embeddings. The encoder is now frozen only when `RL77_MOL_ENCODER_CKPT` loads; otherwise both towers train jointly.

<!-- before:cell_07 -->
## Cell 7: Stage 2 Training — Spec-Mol Cross-Fusion & Context-Aware Generation

<!-- before:cell_08 -->
## Cell 8: Two-Stage GLMR Inference, Unimodal Re-Ranking & `inchikey14` Deduplication

> **Review fix:** The generated structure was always placed at rank 1 although the comment required a 15 ppm mass match; the mass check is implemented and the insertion rank comes from `W("nb1.rerank.generated_insert_rank")`. Ranking is factored into `assemble_ranked_candidates()` so validation can sweep `alpha` and the insertion rank on the same code path.

<!-- before:cell_09 -->
## Cell 9: End-to-End Orchestration on the Competition Data

The PDF's synthetic smoke test is replaced by the full run: stream `train.parquet` with a leakage-free timsTOF validation split, train Stage 1 and Stage 2 within their time budgets, evaluate MRR@25 on held-out molecules, cross-fit the re-ranking knobs, and infer `test.parquet`. Everything lands in `/kaggle/working`, which `tools/kaggle_pipeline.py` publishes as the `casmi26-rl77-nb1-glmr-outputs` dataset for notebooks 2–5.

| Output | Used by |
| --- | --- |
| `glmr_model.pt`, `checkpoints/glmr_stage1.pt` | notebooks 2, 3, 4, 5 |
| `library/library.parquet`, `library/lib_embs_fp16.npy` | notebooks 3, 5 |
| `split/val_inchikey14.txt`, `split/train_inchikey14.txt` | notebooks 2–5 (same validation molecules everywhere) |
| `metrics.json`, `fitted_weights.json`, `rerank_grid.json` | Evidently weights update |
| `submission.csv` | leaderboard submission |

<!-- end -->
## Key Architectural Highlights Aligned with the Rubric

- **Positive-Mode Electron-Mass Alignment (Cell 3):** Subtracts `0.000548` Da from `enveda-180` positive-mode training spectra so fragment and precursor *m/z* values match the hidden test set calibration.
- **Multi-Spectrum Attention Pooling (Cell 5):** Rather than predicting per spectrum and heuristically merging ranks, `MultiSpectrumMoleculeAggregator` applies cross-spectrum self-attention across all collision energies (`collision_energy_ev`) and `base_peak_intensity` levels for a given `molecule_id` to form both a global vector $\mathbf{E}^s_{\text{mol}}$ and sequence query matrix $\mathbf{H}^s_{\text{mol}}$.
- **Strict `inchikey14` Deduplication (Cell 8):** Every generated and retrieved SMILES candidate is passed through `rdMolStandardize.TautomerEnumerator().Canonicalize` and deduplicated by its 14-character connectivity block (`inchikey14`), ensuring zero wasted rank slots among the 25 predictions per molecule.
