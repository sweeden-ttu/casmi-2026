# Enveda CASMI 2026 — Colab GPU Training & Kaggle Inference Pipeline

This repository contains the modular validation, retrieval, neural fingerprint (`FPNet`), spectral library network propagation, and `LightGBM` learning-to-rank codebase for the **Enveda CASMI 2026 – Molecule ID From Mass Spectra** Kaggle competition.

## Directory Structure

```text
/mnt/LMSTUDIO/CASMI/
├── casmi26/
│   ├── config.py          # Auto-detects Local vs. Google Colab vs. Kaggle Notebook paths & constants
│   ├── chemistry.py       # RDKit 2026.03.3 tautomer canonicalization, InChIKey14, 0.000548 Da fix, MRR@25
│   ├── validation.py      # Leak-free inchikey14 GroupKFold & Bruker timsTOF-like validation panel sampler
│   ├── features.py        # Spectral Entropy Similarity, binned m/z + neutral loss vectors, 2215-bit FPs
│   ├── retrieval.py       # O(log N) binary-search CandidatePool (COCONUT 2.0 + PubChem + Train structures)
│   └── models.py          # SpectralLibraryIndex, PyTorch GPU FPNet, and Multi-Spectrum GBDT Reranker
├── notebooks/
│   ├── 01_colab_data_and_gpu_training.ipynb    # Run on Google Colab GPU (mounts Drive, downloads data, trains & exports)
│   └── 02_kaggle_inference_submission.ipynb    # Run offline on Kaggle Notebooks to generate submission.csv
├── scripts/
│   ├── download_datasets.sh                    # Downloads enveda-casmi-2026 & casmi-2026-candidate-database
│   └── publish_kaggle_model_dataset.py         # Bundles kaggle_dataset_export/ + casmi26/ as a Kaggle Dataset
├── kaggle_dataset_export/                      # Ready-to-upload Kaggle Dataset bundle (models + code + metadata)
└── tests/
    └── test_pipeline.py                        # End-to-end synthetic verification & unit test suite
```

## Recommended End-to-End Workflow (Colab GPU -> Kaggle Dataset -> Kaggle Submission)

### Step 1: Download Data & Train on Google Colab GPU
Where is the easiest & fastest place to store the multi-GB competition datasets for Colab?
- **Google Drive (`/content/drive/MyDrive/CASMI2026`) + Colab Local NVMe (`/content/casmi_fast`)**.
- Open [01_colab_data_and_gpu_training.ipynb](file:///mnt/LMSTUDIO/CASMI/notebooks/01_colab_data_and_gpu_training.ipynb) in Google Colab.
- Add `KAGGLE_USERNAME` and `KAGGLE_KEY` to Colab Secrets (or place `kaggle.json` in `/content/drive/MyDrive/CASMI2026/kaggle.json`).
- Cell 4 downloads `enveda-casmi-2026` and `rishavyadav/casmi-2026-candidate-database` directly from Kaggle's servers into your Google Drive (`~100+ MB/s`, zero local bandwidth used) and copies active Parquet shards to `/content/casmi_fast` for fast GPU training.
-*(If you also want a local copy in `/mnt/LMSTUDIO/CASMI/data`, run `bash scripts/download_datasets.sh /mnt/LMSTUDIO/CASMI/data` after placing `~/.kaggle/kaggle.json`.)*

### Step 2: Export Trained Models to a Public/Private Kaggle Dataset
At the end of training, `pipeline.save_checkpoint()` writes `candidate_pool.parquet` and `casmi_pipeline_models.pkl` into `kaggle_dataset_export/`.
Run [publish_kaggle_model_dataset.py](file:///mnt/LMSTUDIO/CASMI/scripts/publish_kaggle_model_dataset.py) (either inside Colab or locally after syncing `kaggle_dataset_export/` from Drive):

```bash
python3 scripts/publish_kaggle_model_dataset.py \
    --export-dir /mnt/LMSTUDIO/CASMI/kaggle_dataset_export \
    --dataset-slug your-username/casmi26-trained-models \
    --publish
```

For subsequent model improvements trained in Colab, push a new dataset version:
```bash
python3 scripts/publish_kaggle_model_dataset.py \
    --export-dir /mnt/LMSTUDIO/CASMI/kaggle_dataset_export \
    --dataset-slug your-username/casmi26-trained-models \
    --publish --update --message "v2: 5-fold FPNet ensemble + COCONUT prior tuning"
```

### Step 3: Score on Kaggle with the Offline Inference Notebook
1. Upload [02_kaggle_inference_submission.ipynb](file:///mnt/LMSTUDIO/CASMI/notebooks/02_kaggle_inference_submission.ipynb) to Kaggle under the **Enveda CASMI 2026** competition.
2. In the right-hand **Input** panel, attach your dataset (`your-username/casmi26-trained-models`).
3. Turn **Internet Off** in Notebook Settings and click **Submit to Competition**.
4. Kaggle automatically swaps `/kaggle/input/enveda-casmi-2026/test.parquet` with the hidden 400-molecule Bruker timsTOF test set, runs multi-spectrum neutral mass triangulation + candidate retrieval + neural fingerprint & GBDT reranking, deduplicates candidates by `RDKit 2026.03.3` `inchikey14`, and scores `/kaggle/working/submission.csv` using `MRR@25`.
