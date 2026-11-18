## Learned User Preferences

- Do not write smoke tests for notebooks; test kernels with full runs uploaded to Kaggle and fix errors after the Kaggle run completes.
- Save each notebook's outputs to a Kaggle dataset so downstream notebooks can attach them as inputs.
- Do not hard-code model weights or blend coefficients; keep them in an Evidently dataset updated with the latest and best estimates (Evidently, not W&B, tracks weights and MRR@25 metrics).
- Record every completed experiment run, including its Kaggle leaderboard score, in Evidently and document the score in the run's README.

## Learned Workspace Facts

- Competition `enveda-CASMI26-molecule-id-mass-spectra` scores MRR@25 on the first 14 InChIKey characters after RDKit 2026.03.3 tautomer canonicalization; submissions are `molecule_id,smiles` with up to 25 `;`-separated SMILES, so pin `rdkit==2026.3.3` wherever InChIKeys are computed.
- RL77 pipeline sources are `experiments/runs/RL77/nbN/cell_NN.py` plus `markdown/nbN.md`; `tools/build_notebooks.py` generates `notebooks/` and `build/` (`--check` runs pyflakes), `tools/kaggle_pipeline.py` pushes, waits, fetches and publishes kernels, and the full workflow is in `experiments/runs/RL77/README.md`.
- RL77 cells read every coefficient via `W("key")` from `weights/current.json`, a snapshot of the Evidently dataset `casmi-rl77-weights` shipped in the prereqs Kaggle dataset because Kaggle cannot reach the LAN Evidently server; change weights with `tools/evidently_weights.py` (`adopt` activates fitted values only when the cross-fit beats the prior, `set` for manual values).
- Evidently 0.7.x in the workspace venv requires `pandas<3`.
- The Kaggle CLI is `.venv/bin/kaggle`; it has no command to cancel a running kernel, and live kernel logs are readable through the Python API `kernels_logs_stream()`.
- Kaggle quirks: kernel and dataset slugs share one namespace, a new kernel's slug is derived from its title rather than the metadata `id`, and the CLI exits 0 on server-side rejections such as "Dataset creation error", so check the output text.
- Kaggle mounts inputs at `/kaggle/input/competitions/<slug>` and `/kaggle/input/datasets/<owner>/<slug>`.
- `kaggle datasets version` returns before Kaggle finishes processing the upload; wait until the new files (size or timestamp) appear in the dataset listing before pushing a kernel that attaches it, or the kernel mounts the previous version.
- Evaluate new models (DreaMS, MIST, DiffMS, DualLGD and similar) as extra ranking streams on RL77's 998-molecule held-out split and the public leaderboard; the P5 harness and `casmi26` score only `data/synthetic`.
- MIST, DiffMS and DualLGD accept only positive-mode spectra and reject `[2M…]` adducts, which excludes about 35% of training rows.
