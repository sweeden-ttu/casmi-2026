<!-- intro -->
# CASMI 2026 · RL77 Notebook 3 — Four-Way Ensemble (GLMR + Siamese ECFP4 + Mass2SMILES)

*Assembled from **CASMI 2026 Colab Rubric.pdf** (pages 27–38). Python cells are the reviewed `experiments/runs/RL77/nb3/cell_NN.py` files; text cells are taken from the PDF.*

> **User prompt:** Write the PyTorch module for the 2000-bin Siamese ECFP4 fingerprint retriever and Mass2SMILES 60-functional-group head, and show how to blend their similarity scores with GLMR Stage 2 re-ranking.

To push your **Enveda CASMI 2026** pipeline past the **0.490+ MRR@25** top-10% threshold, blending **GLMR** with the **2000-bin Siamese ECFP4 metric learner** (Carrera et al., BIOSTEC 2026) and the **Mass2SMILES 60-functional-group TCN head** (Elser et al.) introduces two orthogonal structural signals:

1. **Topological Fingerprint Recall (Siamese 2000-Bin ECFP4):** While autoregressive SMILES decoders can suffer from low topological Tanimoto similarity on sparse or noisy spectra, a 3-layer Siamese MLP trained on a dense 2000-bin spectral vector and 2048-bit Morgan (ECFP4) fingerprints achieves a retrieval AUC of 0.9947 and nearly doubles Top-1 Tanimoto fidelity (0.351 vs. 0.189).
2. **Substructural Isomer Discrimination (Mass2SMILES 60-FG + Heteroatoms):** Elser et al. demonstrated that functional group count accuracy does not strictly correlate with global Tanimoto similarity—meaning candidates with tied ChemFormer or ECFP4 cosine scores can be separated by penalizing candidates whose RDKit functional group, heteroatom, or glycoside/sugar counts conflict with the MS/MS fragmentation pattern.

<!-- before:cell_01 -->
## Cell 1: 2000-Bin Spectral Vectorizer, 2048-Bit ECFP4 & 71-D Mass2SMILES Target Extractor

Following **Mass2SMILES** (Figure 1 of Elser et al.), the auxiliary target vector has **71 dimensions**: 60 deterministic RDKit functional group counts (`rdkit.Chem.Fragments`), 8 heteroatom counts (N, O, S, P, F, Cl, Br, I), and 3 sugar/glycoside and ring-system counts.

> **Review fix:** `AllChem.GetMorganFingerprintAsBitVect` is deprecated in current RDKit; the Morgan generator yields the same radius-2 / 2048-bit fingerprint. `aggregate_molecule_2000_bins` weighted spectra by `log10(max(base_peak_intensity, 10))`, and `max(nan, 10)` is `nan` for the 46% of train rows with a null base peak, which turned the whole composite vector into NaN.

<!-- before:cell_02 -->
## Cell 2: 2000-Bin Siamese ECFP4 Metric Learner (`SiameseECFP4Retriever`)

Following Table 2 and Figure 3 of **Carrera et al. (BIOSTEC 2026)**, this module pairs twin 3-layer ReLU MLPs that project the 2000-bin spectrum ($\mathbf{z}_s \in \mathbb{R}^{256}$) and 2048-bit ECFP4 fingerprint ($\mathbf{z}_f \in \mathbb{R}^{256}$) into a shared 256-dimensional cosine embedding space, augmented with a direct 2048-bit continuous Morgan fingerprint decoder head for differentiable Tanimoto scoring.

> **Review fix:** loss margin, temperature, and fingerprint weight come from the Evidently weights (`nb3.siamese_loss.*`).

<!-- before:cell_03 -->
## Cell 3: Mass2SMILES Dilated TCN & 60-Functional-Group Multi-Task Head

Following **Elser et al. (Mass2SMILES Figure 1 & Methods)**, this module processes the sinusoidal spectral peak sequence through a **Temporal Convolutional Network (TCN)** with `kernel_size=8` and 256 filters across exponentially dilated residual blocks, followed by 2 × 2 dense layers predicting the **71-dimensional functional group + heteroatom + sugar vector** and **512-dimensional continuous chemical descriptor**.

<!-- before:cell_04 -->
## Cell 4: Four-Way Ensemble Score Blender & GLMR Stage 2 Re-Ranker

To prevent one model's raw score scale from dominating another, `CASMI26EnsembleReranker` computes four structural similarity scores across the $K = 40$ pre-retrieved candidates per `molecule_id`, applies per-molecule **z-score standardization** combined with **Reciprocal Rank Fusion (RRF)**, and optimizes the blending weights ($\mathbf{w}$) on your local `timsTOF` validation split before running strict `inchikey14` deduplication.

> **Review fix:** the five blend weights, `fg_gamma`, `rrf_k`, and the generative-weight redistribution (0.6 / 0.4) are read from Evidently dataset `casmi-rl77-weights` instead of literals. The out-of-fold optimizer ranks mass-consistent candidates first, matching inference.

> **Review fix (after the nb3 Kaggle run):** a single Nelder–Mead run from the prior stalled on the piecewise-constant MRR@25 surface. One cross-fit half stopped at 0.222 while the Siamese stream alone scored 0.369, so the cross-fitted blend (0.308) underperformed one of its own inputs. The optimizer now scores the prior, uniform and every single-stream vector, and searches from the two best with an explicit simplex step.

<!-- before:cell_05 -->
## Cell 5: Complete Integrated Inference Loop (GLMR + Siamese ECFP4 + Mass2SMILES)

Drop this function into your Kaggle inference notebook to execute all three models in a single pass per batch and produce your deduplicated `submission.csv`.

> **Review fix:** the loop read `batch["binned_2000"]`, which no dataset produced (KeyError on the first batch); `EnsembleMultiSpectrumDataset` adds it. Candidate fingerprints are cached, ranking puts mass-consistent candidates first, and per-candidate stream scores are recorded so Cell 6 can fit the blend out-of-fold. Retrieval uses the notebook-2 offline store (train + COCONUT).

<!-- before:cell_06 -->
## Cell 6: Train the Heads, Fit the Blend Out-of-Fold, Infer the Test Set

The PDF provides the modules but no training or fitting run. This cell trains the Siamese ECFP4 learner and the Mass2SMILES head on notebook 1's training split (never its validation blocks), scores each stream and the blend on held-out timsTOF molecules, cross-fits `optimize_weights_oof` (fit on one half, score the other) so the reported gain is honest, and infers `test.parquet` with the blend `adopt` will activate: the fit if its cross-fitted MRR@25 beats the prior, otherwise the Evidently prior (`metrics.json` → `test_blend`). `fitted_weights.json` feeds `tools/evidently_weights.py adopt`, which applies the same rule. Runs on CPU.

<!-- end -->
## Why This Four-Way Ensemble Lifts MRR@25 Across All Three Novelty Tiers

| Candidate Scenario in `test.parquet` | Primary Scoring Stream Driving Top-1 Rank | Why Single-Model Pipelines Fail |
| --- | --- | --- |
| **Tier 1: In-Library Match** | `s_siamese_ecfp4` + `s_glmr_pre` | Dense 2000-bin vectors capture full fragment ratios without peak-count truncation loss. |
| **Tier 2: Known DB Isomers (PubChem/COCONUT)** | `s_glmr_gen` + `s_fg_match` (Mass2SMILES) | Positional isomers share almost identical ECFP4 bits; 60 functional group & sugar counts break ties. |
| **Tier 3: Novel Scaffolds / Low-Signal Spectra** | `z_rrf` + `s_siamese_ecfp4` fallback | When autoregressive beam search produces an invalid SMILES, RRF seamlessly shifts weight to Siamese + FG streams. |

The table above illustrates how each branch compensates for the others' failure modes while computing re-ranking features only on the $K = 60$ pre-retrieved candidates per molecule, adding less than 15 seconds of total wall-clock overhead across all 400 test molecules. *(PDF claims; the measured per-stream MRR@25 is in `metrics.json`.)*
