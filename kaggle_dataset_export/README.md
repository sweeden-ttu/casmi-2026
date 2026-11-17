# CASMI 2026 Source Code Synthetic Data and Models

Self-contained public Kaggle Dataset for **Enveda CASMI 2026 – Molecule ID From Mass Spectra**.

## Contents
- `casmi26/`: Complete Python source package (`config.py`, `chemistry.py`, `validation.py`, `features.py`, `retrieval.py`, `models.py`, `benchmark.py`).
- `synthetic_data/`: Synthetic multi-adduct LC-MS/MS spectra (`train_synthetic.parquet`, `test_synthetic.parquet`) and same-mass isomer candidate shards (`coconut_candidates.parquet`, `pubchem_candidates.parquet`).
- `candidate_pool.parquet` & `casmi_pipeline_models.pkl`: Pre-indexed candidate pool and trained `CASMIPipeline` (`FPNet` + `SpectralLibraryIndex` + `LightGBM`/`GBDT` Reranker).
- `tests/` & `scripts/`: 100-point rubric scorer (`score_notebook_rubric.py`) and end-to-end verification tests.
