# RL77 — CASMI 2026 Colab Rubric notebooks on Kaggle

Five notebooks assembled from **`CASMI 2026 Colab Rubric.pdf`** (a Gemini conversation: a 100-point
top-10% rubric plus five code responses). Each notebook interleaves the reviewed Python cells with
the PDF's text, runs end to end on the real competition data on Kaggle, and publishes its
`/kaggle/working` folder as a Kaggle dataset that the next notebook attaches. Every blend weight,
reward coefficient, and tolerance is read from the Evidently dataset **`casmi-rl77-weights`**.

Competition: [`enveda-CASMI26-molecule-id-mass-spectra`](https://www.kaggle.com/competitions/enveda-CASMI26-molecule-id-mass-spectra)
(metric: MRR@25 on `inchikey14` after RDKit 2026.03.3 tautomer canonicalization; submission
`molecule_id,smiles` with up to 25 `;`-separated SMILES per molecule).

---

## 1. Layout

```
experiments/runs/RL77/
├── CASMI 2026 Colab Rubric.pdf      source document (60 pages)
├── README.md                        this file
├── nb1/ … nb5/cell_NN.py            reviewed Python cells, one file per notebook cell
│                                    (cell_01 = earliest Gemini export; originals are in the PDF)
├── markdown/nbN.md                  text cells from the PDF, split by <!-- intro / before:cell_NN / end -->
├── lib/rl77_env.py                  runtime helpers: input discovery, pinned packages, time budgets, manifests
├── lib/rl77_weights.py              W("key") loader for the Evidently weights snapshot
├── weights/current.json             latest Evidently snapshot (rl77_vNNN.json = each version)
├── notebooks/*.ipynb                assembled notebooks (generated; open these to read the pipeline)
├── tools/build_notebooks.py         assemble notebooks + prerequisites dataset + kernel metadata, pyflakes check
├── tools/evidently_weights.py       publish / pull / adopt / set / record-metric against Evidently
├── tools/kaggle_pipeline.py         push datasets and kernels, wait, fetch outputs, publish outputs, submit
├── tools/standardize_candidates.py  precompute nb2's COCONUT tautomer standardization on local CPUs (optional)
└── build/                           generated (git-ignored): kernel folders, dataset folders, fetched outputs
```

| Notebook | Kaggle kernel | PDF pages | What it does | Accelerator | Output dataset |
| --- | --- | --- | --- | --- | --- |
| 1 | `scottweeden/casmi26-rl77-nb1-glmr` | 1–21 | Two-stage GLMR: Stage 1 InfoNCE retrieval, Stage 2 cross-fusion decoder, re-ranking, submission | GPU T4 | `casmi26-rl77-nb1-glmr-outputs` |
| 2 | `scottweeden/casmi26-rl77-nb2-offline-store` | 21–27 | Candidate catalog (train + COCONUT), FP16 memmap, FAISS, formula index, offline wheels | CPU | `casmi26-rl77-nb2-store-outputs` |
| 3 | `scottweeden/casmi26-rl77-nb3-ensemble` | 27–38 | Siamese ECFP4 + Mass2SMILES heads, 4-way blend fitted out-of-fold, submission | CPU | `casmi26-rl77-nb3-ensemble-outputs` |
| 4 | `scottweeden/casmi26-rl77-nb4-formula-filter` | 38–48 | MIST-CF/SIRIUS-style formula filter, fitted score weights, formula priors, submission | CPU | `casmi26-rl77-nb4-formula-outputs` |
| 5 | `scottweeden/casmi26-rl77-nb5-qrl` | 48–59 | Stage 2.5 soft Q-learning decoder, Q-guided beams, final submission | GPU T4 | `casmi26-rl77-nb5-qrl-outputs` |

Input datasets shared by all kernels: `scottweeden/casmi26-rl77-prereqs` (helper modules, the weights
snapshot, and each notebook as an importable module) and, for notebook 2, `scottweeden/casmi26-rl77-candidates`
(`data/candidates/coconut/coconut_structures.parquet`, 494,774 COCONUT structures). All datasets and kernels are private.

---

## 2. One-time setup

```bash
cd /mnt/LMSTUDIO/CASMI
# Local tools (AGENTS.md: uv + the workspace venv). evidently 0.7.23 needs pandas < 3.
uv pip install --python .venv/bin/python pyarrow "pandas<3" pyflakes "evidently==0.7.23" requests nbformat kaggle
.venv/bin/kaggle config view          # username scottweeden, token in ~/.kaggle/kaggle.json
.venv/bin/kaggle quota                # weekly GPU hours (notebooks 1 and 5 use the T4)
curl -s http://192.168.0.15:8000/api/version   # Evidently UI (API key: ~/.config/evidently/api_key)
```

Accept the competition rules on Kaggle once so kernels can mount the data.

---

## 3. Assemble

```bash
PY=.venv/bin/python; T=experiments/runs/RL77/tools
$PY $T/build_notebooks.py --check
```

For each notebook this writes `notebooks/<slug>.ipynb` with:

1. the PDF intro (user prompt + response) as markdown,
2. a generated **Prerequisites** markdown cell (inputs table, run order, accelerator, packages, weights version),
3. a prerequisites code cell that finds `rl77_env.py` under `/kaggle/input`, installs `rdkit==2026.03.3` and
   `faiss-cpu` if the image lacks them (offline from notebook 2's wheels when attached), loads `W`, and imports the
   upstream notebooks' modules,
4. for every `cell_NN.py`: the PDF's heading and explanation (plus a **Review fix** note where the code changed), then the code,
5. the PDF's closing section.

It also regenerates `build/datasets/casmi26-rl77-prereqs/` (helper modules, `rl77_weights.json` = `weights/current.json`,
and `rl77_nbN_*.py` modules whose drivers run only under `__main__`) and `build/kernels/<slug>/kernel-metadata.json`.
`--check` runs pyflakes on each notebook with its upstream cells inlined; it must print `OK` for all five.

---

## 4. Weights and biases in Evidently

The PDF hard-codes 86 coefficients (GLMR re-rank alpha, ensemble blend, formula posterior weights, reward
shaping, RL hyperparameters, beam settings, tolerances). They live in Evidently project
`01a1d2dd-5ca0-7d05-a07a-135ba489166d` as dataset **`casmi-rl77-weights`**, one dataset per version, one row per value:

| column | meaning |
| --- | --- |
| `name` | key read by the notebooks, e.g. `nb3.reranker.w_gen` |
| `kind` | `weight` (active), `candidate` (a fit that did not beat the prior), `metric` (validation or leaderboard result) |
| `value`, `version`, `source`, `cell`, `note`, `updated_utc` | value, snapshot version, `pdf_prior` / `fitted:nbN`, defining cell, evidence |

Kaggle cannot reach the LAN Evidently server, so the kernels read a snapshot shipped in the prerequisites dataset:

```bash
$PY $T/evidently_weights.py pull        # newest Evidently version -> weights/current.json
$PY $T/evidently_weights.py list        # all versions
$PY $T/evidently_weights.py publish     # push weights/current.json as a new version (after a manual edit)
$PY $T/evidently_weights.py adopt experiments/runs/RL77/build/outputs/nb3   # fold a run into a new version
$PY $T/evidently_weights.py record-metric leaderboard.nb5.public_mrr25 0.000 --note "kernel v1"
$PY $T/evidently_weights.py set nb5.rerank.alpha=0.65 nb5.rerank.generated_insert_rank=1 --cell nb5/cell_05 --source pdf_prior
```

`set` adds or changes active weights by hand in one new version (use it for a new key; fitted values go through `adopt`).

`adopt` reads the run's `fitted_weights.json` and `metrics.json`. Fitted values become **active only if their
2-fold cross-fitted validation score beats the prior on the same molecules**; otherwise they are stored as
`candidate` rows. Metrics are always recorded. Each notebook's own test submission applies the same rule
(`rl77_env.select_validated`), so `submission.csv` matches what `adopt` will activate. What gets fitted:

| Notebook | Keys | Objective (held-out timsTOF molecules from notebook 1's split) |
| --- | --- | --- |
| 1 | `nb1.rerank.alpha`, `nb1.rerank.generated_insert_rank` | MRR@25, grid 11 × 5 |
| 3 | `nb3.reranker.w_gen/w_pre/w_siamese/w_fg/w_rrf` | MRR@25, Nelder–Mead (`optimize_weights_oof`) |
| 4 | `nb4.score.mass/nl/iso/neural` | formula top-1 accuracy, coordinate search |
| 4 | `nb4.priors.rank_formulas` (also used by nb5) | MRR@25 of store retrieval with the top-k formulas as ranking tiers, k ∈ {0,1,2,3} |
| 5 | `nb5.rerank.alpha`, `nb5.rerank.generated_insert_rank` | MRR@25 of the Q-centroid blend, grid 11 × 5 |

Not fitted on purpose: `nb2.retrieval.in_library_boost` (every validation molecule is a `train.parquet` structure, so a
validation sweep always favors the boost while the hidden test also has out-of-library structures), and the RL
reward coefficients (one RL run per setting is too expensive). The initial snapshot `rl77-v001` is
Evidently dataset `01a1eb01-d270-7f4f-9de2-ef495ab69e5a`; the version history is in §10.

After any `adopt`/`pull`, rebuild and version the prerequisites dataset (§5 step 1) so later kernels read the new values.

---

## 5. Test on Kaggle (full runs, in order)

Each notebook runs on the real competition data; there are no synthetic smoke tests. Notebook N needs the output
datasets of notebooks 1…N−1, so run them in order and fix errors from the Kaggle log after each run completes.

```bash
# 1. Inputs (prereqs every time code or weights change; candidates only once)
#    Optional, CPU only: precompute COCONUT standardization (Kaggle's 4 CPUs need ~4.5 h for it)
uv pip install --python .venv/bin/python "rdkit==2026.3.3"
$PY $T/standardize_candidates.py --procs 10        # -> data/candidates/coconut/coconut_standardized.parquet
$PY $T/build_notebooks.py --check
$PY $T/kaggle_pipeline.py push-datasets -m "describe the change"   # add --with-candidates after a new standardization

# 2. For nb in nb1 nb2 nb3 nb4 nb5:
$PY $T/kaggle_pipeline.py push nb1          # refuses if an upstream output dataset is missing
$PY $T/kaggle_pipeline.py wait nb1          # polls; on error downloads output + log and prints the tail
$PY $T/kaggle_pipeline.py fetch nb1         # /kaggle/working + log -> build/outputs/nb1
$PY $T/evidently_weights.py adopt experiments/runs/RL77/build/outputs/nb1
$PY $T/kaggle_pipeline.py publish-output nb1   # build/outputs/nb1 -> scottweeden/casmi26-rl77-nb1-glmr-outputs
$PY $T/build_notebooks.py --check && $PY $T/kaggle_pipeline.py push-datasets -m "weights after nb1"
```

`publish-output` refuses a folder without `run_manifest.json` (the last thing each driver writes), so a failed
run cannot leak half-written outputs into the chain. Kaggle has no cancel command in this CLI version; stop a
running version from the kernel page.

### Outputs per notebook (all in `/kaggle/working`, published as the output dataset)

| Notebook | Files | Consumed by |
| --- | --- | --- |
| 1 | `glmr_model.pt`, `checkpoints/glmr_stage1.pt`, `library/library.parquet` + `lib_embs_fp16.npy`, `split/{train,val}_inchikey14.txt`, `val_predictions.csv`, `rerank_grid.json`, `fitted_weights.json`, `metrics.json`, `train_log.csv`, `submission.csv` | 2, 3, 4, 5 |
| 2 | `candidate_catalog_sorted.parquet`, `chemformer_embs_fp16.mmap`, `chemformer_ivfflat.index`, `formula_offsets.npz`, `wheels/*.whl`, `metrics.json` | 3, 4, 5, offline submission |
| 3 | `siamese_ecfp4.pt`, `mass2smiles_fg_head.pt`, `fitted_weights.json`, `metrics.json`, `submission.csv` | 4, 5 |
| 4 | `test_formula_priors.json`, `fitted_weights.json`, `metrics.json`, `submission.csv` | 5 |
| 5 | `q_decoder.pt`, `rl_log.csv`, `metrics.json`, `submission.csv` | leaderboard |

Every folder also has `run_manifest.json` (device, versions, runtime, weights version, metrics).

---

## 6. Train (budgets and scaling)

A Kaggle session is capped at 12 hours and the account has 30 GPU-hours per week, so the PDF's
"300 epochs + 30 epochs" are upper bounds and wall-clock budgets end each phase. Override with environment
variables (set them in a kernel's first cell or locally):

| Variable | Default | Phase |
| --- | --- | --- |
| `RL77_STAGE1_MINUTES` / `RL77_STAGE2_MINUTES` | 55 / 35 | nb1 GLMR Stage 1 / Stage 2 |
| `RL77_N_TRAIN_MOLECULES`, `RL77_N_VAL_MOLECULES`, `RL77_N_VAL_EVAL` | 40000, 1000, 250 | nb1 split sizes and beam-search evaluation size |
| `RL77_STANDARDIZE_MINUTES` | 70 | nb2 COCONUT tautomer canonicalization (train structures always complete) |
| `RL77_SIAMESE_MINUTES`, `RL77_FG_HEAD_MINUTES`, `RL77_NB3_TRAIN_MOLECULES`, `RL77_NB3_VAL_MOLECULES` | 20, 30, 20000, 300 | nb3 heads and validation |
| `RL77_RL_MINUTES`, `RL77_NB5_TRAIN_MOLECULES`, `RL77_NB5_VAL_MOLECULES` | 45, 20000, 200 | nb5 Q-learning |
| `RL77_MOL_ENCODER_CKPT` | unset | pre-trained ChemFormer encoder; when set, Stage 1 freezes it as GLMR intends |

Longer training: per `AGENTS.md`, heavy GPU training belongs on Colab (`experiments/COLAB.md`). The notebooks run
unchanged there: set `RL77_LIB_DIR=…/build/datasets/casmi26-rl77-prereqs`, `RL77_COMPETITION_DIR` (or
`RL77_INPUT_ROOTS` pointing at a folder of downloaded datasets), `RL77_WORK_DIR`, and raise the budgets. Upload the
resulting folder with `publish-output` so the Kaggle chain picks it up.

---

## 7. Evaluate

All validation numbers come from **notebook 1's leakage-free split**: validation `inchikey14` blocks are sampled from
timsTOF spectra at 157–1159 Da (matching `test.parquet`) and removed from training entirely; notebooks 2–5 reload the
same molecules from `split/`. Validation molecules are in the retrieval library (they are `train.parquet` structures),
so retrieval scores are in-library estimates; the public leaderboard also contains out-of-library molecules.

| `metrics.json` key | Meaning |
| --- | --- |
| nb1 `val_stage1.mrr25`, `val_stage1.recall_at_40` | Stage 1 retrieval only |
| nb1 `val_two_stage_mrr25`, `val_generated_valid_rate`, `val_generated_mass_ok_rate` | full GLMR with beam search |
| nb1 `rerank_sweep.val_mrr25_prior` / `val_mrr25_crossfit` | active alpha vs cross-fitted alpha |
| nb2 `val_store_stage1.{mrr25,recall_at_k,truth_in_mass_window}` | offline store (train + COCONUT) retrieval |
| nb3 `val_stream_mrr25.{gen,pre,siamese,fg,rrf}`, `val_blend.*` | each stream alone, prior blend, cross-fitted blend |
| nb4 `val_formula.{top1_prior,top3_prior,top1_fitted_crossfit,truth_in_candidate_window}`, `val_retrieval_mrr25.{mass_window_only,formula_tiers_k}`, `formula_tier_fit.*` | formula identification; retrieval with k formula tiers |
| nb5 `val.{mrr25_nb1_pipeline,mrr25_q_guided,q_beam_top1_mass_ok,q_beam_top1_exact,q_beam_contains_truth}`, `rerank_sweep.*` | notebook 1's adopted pipeline vs Q-guided on the same molecules; Q-centroid knob fit |
| nb3 `test_blend`, nb4 `test_score_weights` / `test_rank_formulas`, nb5 `test_rerank` | what the test submission used (`fitted` or the weights version) |

`tools/evidently_weights.py adopt` records all of these under `kind=metric` in Evidently.

---

## 8. Submit and score on the leaderboard

A submission is a kernel version's `submission.csv`. Kaggle scored the internet-on nb4 v2 and nb5 v2 versions, so
internet-off is not enforced here; `--offline` (below) still proves a version runs from the bundled wheels alone.

1. Pick the notebook with the best validation MRR@25 (`metrics.json` of each fetched output). On the full catalog that is
   **notebook 3** (four-way blend 0.380 on 300 held-out molecules) ahead of 5 (0.219), 4 (0.190) and 1 (~0.106).
2. Rebuild that kernel with internet off: `$PY $T/build_notebooks.py --check --offline nb3 [nb5]` writes
   `"enable_internet": false` into its `kernel-metadata.json` (the other kernels keep internet on). Every downstream
   kernel attaches `casmi26-rl77-nb2-store-outputs`, so `ensure_packages()` installs RDKit and FAISS from its `wheels/`.
3. Push and wait as in §5, then submit that version (uses one of the day's submissions):

```bash
$PY $T/kaggle_pipeline.py submit nb5 --version <kernel version> -m "RL77 nb5 rl77-vNNN" --yes
$PY $T/kaggle_pipeline.py scores                       # public MRR@25 of your submissions
$PY $T/evidently_weights.py record-metric leaderboard.nb5.public_mrr25 <score> --note "kernel vN, weights rl77-vNNN"
```

Compare the public score with the notebook's validation MRR@25; the rubric's target is a local-to-public gap under
0.03 and a public MRR@25 of at least ~0.49 for the top 10%.

### Leaderboard results (Evidently `leaderboard.*.public_mrr25`)

| Submission | Kernel version | Catalog / knobs | Validation MRR@25 | Public MRR@25 | Gap |
|---|---|---|---|---|---|
| 58445362 (2026-11-16) | `casmi26-b0-db-fragment-baseline` (reference, not RL77) | model-free: formula shortlist, COCONUT+LOTUS, fragment score | — | **0.192** | — |
| 58495366 (2026-11-18 05:00 UTC) | nb4 v2 (Kaggle version 3) | 80k COCONUT, Stage 1 + formula tiers k = 3 | 0.215 | 0.136 | −0.079 |
| 58497621 (2026-11-18 06:13 UTC) | nb5 v2 (Kaggle version 3) | 80k COCONUT, Q-guided α 0.2 + tiers | 0.241 | 0.133 | −0.108 |

The gaps are 3–4× the rubric's 0.03 target, and both RL77 submissions trail the model-free baseline. The validation
molecules' structures are in the training library (tier 1 of the catalog), whereas test structures need not be, so
held-out MRR@25 measures in-library retrieval and overstates the public score.

---

## 9. Code review of the PDF cells

The PDF's code was reviewed against the real data (`train.parquet` 2,539,608 spectra / 275,810 `inchikey14`;
`test.parquet` 1,213 spectra / 400 molecules). Each change is also noted above its cell in the notebook.

| Where | Problem in the PDF code | Fix |
| --- | --- | --- |
| nb1 Cell 1 | `WEIGHT_decay` defined, `CFG.WEIGHT_DECAY` read (AttributeError) | renamed; Drive paths → `rl77_env`; time budgets |
| nb1 Cell 2 | adduct names `[M+FA-H]-`, `[M+Hac-H]-` vs data `[M+CH2O2-H]-`, `[M+C2H4O2-H]-`; ~340k `[2M+X]` spectra; unknown adducts fell back to `[M+H]+` | `ADDUCT_SPECS` (n, z, Δ) + `neutral_mass_from_precursor()` |
| nb1 Cell 2 | tokenizer listed `C` twice and lacked `H`, `%` (32k `[nH]`/charged atoms, ring closures ≥ 10 decoded to empty strings) | vocabulary = measured train character set |
| nb1 Cell 4 | 46% null `base_peak_intensity`; `max(nan, 1.0)` is `nan` → NaN through the encoder | finite guard; primary precursor from a monomer adduct; streamed loader |
| nb1 Cell 5 | float causal mask with bool padding mask (deprecated); one forward pass per beam | bool mask; batched beams |
| nb1 Cell 6 | randomly initialized ChemFormer frozen in both stages (no checkpoint loaded) | freeze only when a pre-trained encoder loads; else train jointly |
| nb1 Cell 7 / nb2 Cell 4 | fewer than K hits in the 15 ppm window → mass-matched candidates discarded for a global top-K (most queries); FAISS `-1` padding indexed the last row | keep in-window first, fill from global; mass-consistent candidates rank first |
| nb1 Cell 8 | generated SMILES always rank 1 despite the documented 15 ppm check | mass check implemented; insertion rank from Evidently |
| nb1 Cell 9, nb4 Cell 4, nb5 Cell 4 | synthetic smoke tests | replaced by full real-data drivers |
| nb2 Cell 1–3 | serial canonicalization of ~770k structures; float32 masses; empty-catalog crash; deprecated `torch.cuda.amp.autocast`; padded 256-token encoding; `matchms` bundled but unused; `pip download` failures ignored | process pool + budget; float64; guard; `torch.autocast`; length-trimmed batches; checked wheel download |
| nb3 Cell 1 | deprecated `GetMorganFingerprintAsBitVect`; NaN composite spectrum from null base peaks | Morgan generator; finite guard |
| nb3 Cell 5 | `batch["binned_2000"]` never produced (KeyError) | `EnsembleMultiSpectrumDataset` |
| nb4 Cell 3–4 | consensus mass used the shift table; every spectrum given the primary adduct; isotope ratio thresholds applied to √I intensities | helper; per-spectrum adducts; squared intensities |
| nb4 Cell 5 | snippet used undefined globals (NameError) | `retrieve_with_formula_priors()` |
| nb2 Cell 4 / nb4 (found by the nb4 run) | formula priors only added rows to the mass-window union, so "Priority A" never changed a ranking | formula tiers via `priority` in `assemble_ranked_candidates`; tier count fitted |
| nb3 Cell 4 (found by the nb3 run) | single Nelder–Mead start stalls on the piecewise-constant MRR surface | multi-start incl. every single-stream vector; explicit simplex step |
| nb5 Cell 1 | `"Cl"`/`"Br"` mass keys never match a character tokenizer (Cl = 12 Da, Br = 0) | per-character increments |
| nb5 Cell 3 | Bellman bootstrap Q(s_{t+1}) computed on the trajectory without BOS (shifted positions) | target network on the full trajectory, positions 1…T |
| nb5 Cell 3 (found by the nb5 run) | actor advantage = immature critic Q − V → policy collapse to invalid SMILES | REINFORCE batch-baseline advantages; best-checkpoint guard in Cell 5 |
| nb5 Cell 5 (found by the nb1 run) | Q-centroid re-ranking reused `nb1.rerank.alpha`, which nb1's cross-fit set to 0 → centroid ignored | own `nb5.rerank.*` keys, cross-fitted in nb5 |
| nb3–nb5 drivers (found by the nb1 run) | test submissions used the prior weights even when the held-out fit won | `select_validated()`: same rule as `adopt` |
| nb1–nb5 validation (external review, verified locally) | answer key `{mid: mid}` used the stored `inchikey14`; 0.3% of held-out molecules store a non-canonical tautomer's key (understated MRR@25 by ≤ 0.003; nb2's 99.7% "truth in mass window" was exactly these 3) | `ground_truth_ik14()` in nb1 Cell 4 |
| all | 86 hard-coded coefficients | read from Evidently `casmi-rl77-weights` through `W("…")` |

The original Gemini cells are reproduced verbatim in the PDF; `git diff` on `nbN/` shows the review once the
originals are committed.

---

## 10. Run log

| Date (UTC) | Step | Result |
| --- | --- | --- |
| 2026-11-17 | Evidently `rl77-v001` published | dataset `01a1eb01-d270-7f4f-9de2-ef495ab69e5a` (86 PDF priors) |
| 2026-11-17 | `casmi26-rl77-prereqs`, `casmi26-rl77-candidates` created | ready |
| 2026-11-17 | nb1 v1 | stopped after data loading to apply the retrieval fix (loaded 40,000 train / 998 val molecules, 275,810 library structures in 0.5 min) |
| 2026-11-17 | nb1 v2 (T4, 96.9 min) | complete. Stage 1: 6 epochs, InfoNCE 3.39 → 1.00; Stage 2: 9 epochs, token CE 1.31 → 0.66. 250 held-out timsTOF molecules: Stage 1 MRR@25 **0.1595**, recall@40 0.648; two-stage with the PDF knobs (α 0.65, generated SMILES at rank 1) 0.1057; beam SMILES valid 92%, within 15 ppm 8.8%. Cross-fit picked α = 0, no insertion (0.1595 held-out). Its `submission.csv` used the v001 knobs (written before this rule existed in nb3–nb5) |
| 2026-11-17 | Evidently `rl77-v002` (`adopt nb1`) | `01a1eb68-cdf8-7f43-b114-1137cbe06e91`: `nb1.rerank.alpha` 0.65 → 0, `nb1.rerank.generated_insert_rank` 1 → 0; 11 nb1 metrics |
| 2026-11-17 | `casmi26-rl77-nb1-glmr-outputs` | published (13 files, 262 MB) |
| 2026-11-17 | Evidently `rl77-v003` (`set`) | `01a1eb70-298c-7b34-92d0-4ead822360ce`: new `nb5.rerank.alpha` 0.65, `nb5.rerank.generated_insert_rank` 1 (nb5 had reused nb1's knobs; α = 0 would have switched the Q-centroid off) |
| 2026-11-17 | nb2 v1 (CPU, 121.6 min) | complete. Tier 1: 273,369 train structures (35 min); the 70-min budget stopped tier 2 after **80,000 of 494,774** COCONUT structures (+73,663). Catalog 347,032 structures / 72,993 formulas, encoded on CPU in 30 min. All 998 held-out molecules: store retrieval MRR@25 **0.1641**, recall@40 0.697, truth in mass window 99.7% |
| 2026-11-17 | Evidently `rl77-v004` (`adopt nb2`) | `01a1ebe7-c942-7a1a-a415-758db9fee56d`: 8 nb2 metrics, no weights fitted |
| 2026-11-17 | nb2 output publish | first attempt rejected *after* the 600 MB upload: dataset slug = kernel slug (`casmi26-rl77-nb2-offline-store`; kernels and datasets share one namespace) and the CLI exited 0. Output renamed `casmi26-rl77-nb2-store-outputs`; `upsert_dataset` now refuses colliding slugs and fails on the server's error text |
| 2026-11-17 | `tools/standardize_candidates.py` | nb2's own `standardize_compound` over all 494,774 COCONUT structures on 10 local CPU processes (RDKit 2026.03.3) → `coconut_standardized.parquet` in the candidates dataset; nb2 v2 uses it |
| 2026-11-17 | nb3 v1 (CPU, 69.5 min, against nb2 v1) | complete. Siamese 100 epochs (loss 0.257), FG head 2 epochs (0.138). 300 held-out molecules, MRR@25 per stream: Siamese **0.369**, RRF 0.235, Stage 1 0.134, generated 0.106, FG 0.094; PDF blend 0.267; cross-fitted blend 0.308 (in-sample 0.390), used for the test submission. A Nelder–Mead fold stalled at 0.222 (below the Siamese stream alone) → multi-start optimizer |
| 2026-11-18 | Evidently `rl77-v005` (`adopt nb3`) | `01a1ec58-2d69-70da-8f6a-f47e2c908e2a`: blend (gen, pre, siamese, fg, rrf) = (0, 0.256, 0.644, 0.029, 0.072); 13 metrics. `casmi26-rl77-nb3-ensemble-outputs` published |
| 2026-11-18 | nb4 v1 push | Kaggle names a **new** kernel after its title, not the metadata `id`: nb4 became `casmi26-rl77-nb4-formula-filter` (now the configured slug) and nb5's title was changed to `CASMI26 RL77 NB5 QRL` so it keeps `casmi26-rl77-nb5-qrl`. `build_notebooks.py` asserts title → slug for every kernel |
| 2026-11-18 | nb4 v1 (CPU, 11.7 min) | complete. 998 held-out molecules: true formula in the candidate window 100%, top-1 **84.4%**, top-3 97.0% (prior weights; cross-fit 84.2% → prior kept). Store retrieval MRR@25 with formula priors = without (0.16405, identical): the PDF's union retrieval had no ranking effect → formula priors became ranking tiers |
| 2026-11-18 | Evidently `rl77-v006` (`adopt nb4`), `rl77-v007` (`set`) | `01a1ec6d-179b-78e5-9314-ad6bff8d13ca`: 13 nb4 metrics, `nb4.score.*` fit stored as candidates. `01a1ec6d-7803-72d4-9931-05d3bc131ee5`: new `nb4.priors.rank_formulas` = 0 |
| 2026-11-18 | nb4 v2 (CPU, 11.5 min, against nb2 v1) | complete. Store retrieval MRR@25 by formula tiers k = 0/1/2/3: 0.1641 / 0.2064 / 0.2137 / **0.2151**; both cross-fit halves chose k = 3 (held-out 0.2151, +31% relative). Submission ranked with k = 3 |
| 2026-11-18 | Evidently `rl77-v008` (`adopt nb4`) | `01a1ec7b-2698-75a6-95e7-6881e5be05ee`: `nb4.priors.rank_formulas` 0 → 3; `nb4.score.*` still prior. `casmi26-rl77-nb4-formula-outputs` published |
| 2026-11-18 | nb5 v1 (T4, 50.8 min, against nb2 v1 / nb3 v1 / nb4 v2) | ran to completion, but **RL collapsed**: 4,885 steps; mean rollout reward −0.28 → best −0.11 (~step 250) → −0.96, with the actor loss drifting to about −10. On 200 held-out molecules the Q-guided beams were 49% valid and 0% within 15 ppm. MRR@25: nb1 pipeline 0.184, Q-guided (prior α 0.65) 0.192, cross-fit α = 0 + formula tiers **0.236** (the gain is the formula tiers, not RL). Fix: Monte Carlo sequence advantages for the actor + best-checkpoint guard |
| 2026-11-18 | Evidently `rl77-v009` (`adopt nb5`) | `01a1ecb5-7e49-76ff-96f6-f8bfdf4e1506`: `nb5.rerank.alpha` 0.65 → 0, insert rank 1 → 0; 16 metrics. `casmi26-rl77-nb5-qrl-outputs` published (v1) |
| 2026-11-18 | nb5 v2 (T4, 51.9 min) | complete, RL healthy: 4,183 steps, rollout reward MA100 +0.79 → best **+1.57** (step 4044, restored by the guard), no all-invalid batches, actor loss −0.09 → −0.17. 200 held-out molecules: Q-beams 100% valid, top-1 within 15 ppm 25.5% (nb1 plain beam 8.8%), exact 0%. MRR@25 nb1 pipeline 0.184 → Q-guided with tiers 0.236 → cross-fit α = 0.2 **0.241** |
| 2026-11-18 | Evidently `rl77-v010` (`adopt nb5`) | `01a1ecea-e50f-7da6-95e2-f46e300fddc5`: `nb5.rerank.alpha` 0 → 0.2; 19 metrics. `casmi26-rl77-nb5-qrl-outputs` v2 published |
| 2026-11-18 | `standardize_candidates.py` finished (10 local CPUs, 327 min) | 483,735 of 494,774 COCONUT structures in the 100–1250 Da window (RDKit 2026.03.3). `casmi26-rl77-candidates` and `casmi26-rl77-prereqs` (weights `rl77-v010`) versioned |
| 2026-11-18 | nb2 v2 (CPU, 122.4 min) | complete. Tier 1 273,369 (32.8 min); tier 2 from the precomputed file in 1 s: +454,554 → **727,923** structures / 138,662 formulas (v1: 347,032), encoded in 85 min. All 998 held-out molecules: store MRR@25 **0.1448** (v1 0.1641), recall@40 0.689, truth in mass window 99.7%. The drop is decoys: held-out structures are already in tier 1 (train.parquet), so validation measures in-library retrieval and COCONUT only adds same-mass competitors; its coverage pays off on novel test structures |
| 2026-11-18 | Evidently `rl77-v011` (`adopt nb2`) | `01a1eda0-dee3-7d58-81fa-24e7a3968dce`: 8 nb2 metrics. `casmi26-rl77-nb2-store-outputs` v2 published; prereqs versioned with v011 |
| 2026-11-18 | nb3 v2 (CPU, 69.1 min, against nb2 v2) | complete. Siamese 121 epochs (loss 0.232), FG head 2 epochs. 300 held-out molecules, MRR@25 per stream: Siamese 0.351, RRF 0.213, Stage 1 0.118, generated 0.098, FG 0.088 (all ~5% below v1: twice the decoys). Blend with the adopted v005 weights **0.380** (now above the best single stream); multi-start cross-fit 0.363 (in-sample 0.388, no fold stalled) → prior kept for the test submission |
| 2026-11-18 | Evidently `rl77-v012` (`adopt nb3`) | `01a1edeb-acfc-7327-ba8a-a37df1ef8c94`: blend weights kept (prior 0.380 > cross-fit 0.363); 13 metrics. `casmi26-rl77-nb3-ensemble-outputs` v2 published |
| 2026-11-18 | nb4 v3 (CPU, 11.4 min, against nb2 v2 / nb3 v2; Kaggle version 4) | complete. Formula bank 109,632; 998 held-out molecules: truth in window 100%, top-1 prior 82.8% → cross-fit **83.3%** (top-3 96.6% → 97.5% in-sample), so the fitted score weights (neural 3.0, mass 1.5, iso 0.25, nl 0.25) ranked the test set. Store MRR@25 by tiers k = 0/1/2/3: 0.1448 / 0.1782 / 0.1878 / **0.1903** (+31%); k = 3 kept |
| 2026-11-18 | Evidently `rl77-v013` (`adopt nb4`) | `01a1edf9-4998-72f6-9885-9926182aafe2`: `nb4.score.{iso,mass,neural,nl}` adopted (first fitted nb4 knobs); `rank_formulas` 3 kept; 20 metrics. `casmi26-rl77-nb4-formula-outputs` v3 published |
| 2026-11-18 | nb5 v3 (T4, 52.7 min, against nb2 v2 / nb3 v2 / nb4 v3; Kaggle version 4) | complete. RL healthy: 3,797 steps, reward MA100 +0.78 → best **+1.55** (step 3700, restored; final 1.45). 200 held-out molecules: Q-beams 100% valid, top-1 within 15 ppm 23.5%, exact 0%. MRR@25 nb1 pipeline 0.184 → Q-guided (α 0.2, tiers k = 3) **0.219**; cross-fit 0.218 (best in-sample α = 0, 0.225) → prior α 0.2 kept |
| 2026-11-18 | Evidently `rl77-v014` (`adopt nb5`) | `01a1ee49-89c9-7ce4-b327-df8b7cf1d48f`: no knob changes; 19 metrics. `casmi26-rl77-nb5-qrl-outputs` v3 published; prereqs versioned with v014 (final chain state) |
| 2026-11-18 | internet-off reruns nb3 (Kaggle version 3) and nb5 (Kaggle version 5) | running; `build_notebooks.py --offline` added |
| 2026-11-18 | Answer-key review (external report) | reported `{mid: mid}` and `formula_priority` bugs were not reproducible (validation `molecule_id` *is* the inchikey14; the runs raised no KeyError; a repeated prior keeps its best tier by design). Found instead: 3 of 998 held-out molecules store a non-canonical tautomer key → `ground_truth_ik14()` in every notebook (takes effect on the next chain run) |
| 2026-11-18 | Evidently `rl77-v015`–`v017` (`record-metric`) | leaderboard scores: nb4 v2 0.136, nb5 v2 0.133, B0 baseline 0.192 (reference) |
