#!/usr/bin/env python3
"""
Creates and publishes/updates a public Kaggle Dataset from `kaggle_dataset_export/`
containing:
1. All Python source files (`casmi26/`, `tests/`, `scripts/`)
2. Synthetic verification & fallback datasets (`data/synthetic/*.parquet`, `train_synthetic.parquet`,
   `test_synthetic.parquet`, `coconut_candidates.parquet`, `pubchem_candidates.parquet`)
3. Trained baseline / Colab GPU model checkpoints (`candidate_pool.parquet`, `casmi_pipeline_models.pkl`)
4. Kaggle `dataset-metadata.json` (`CC0-1.0` public domain license)

Once published, attach that Kaggle Dataset (e.g. `/kaggle/input/casmi26-trained-models`)
to your offline Kaggle Kernel (`notebooks/02_kaggle_inference_submission.ipynb`) so it can
import `casmi26`, load synthetic fallback data or trained weights, and score on the official
Enveda CASMI 2026 Leaderboard with Internet disabled.
"""

from __future__ import annotations
import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def detect_default_kaggle_username() -> str:
    """Detects Kaggle username from env or ~/.kaggle/kaggle.json if present."""
    if os.environ.get("KAGGLE_USERNAME"):
        return os.environ["KAGGLE_USERNAME"].strip()
    k_json = Path.home() / ".kaggle" / "kaggle.json"
    if k_json.exists():
        try:
            data = json.loads(k_json.read_text(encoding="utf-8"))
            if data.get("username"):
                return str(data["username"]).strip()
        except Exception:
            pass
    return "sweeden"


def ensure_synthetic_data_and_baseline_checkpoint(export_dir: Path) -> None:
    """
    Generates synthetic `train_synthetic.parquet`, `test_synthetic.parquet`,
    `coconut_candidates.parquet`, `pubchem_candidates.parquet`, and ensures
    `candidate_pool.parquet` + `casmi_pipeline_models.pkl` exist inside `export_dir`
    so the Kaggle Kernel always has working Python modules, synthetic data, and models.
    """
    from tests.test_pipeline import generate_synthetic_casmi26_data
    from casmi26.validation import assign_leak_free_folds, sample_timstof_like_validation_panel
    from casmi26.retrieval import build_candidate_pool_from_sources
    from casmi26.features import clean_and_normalize_spectrum, smiles_to_fingerprint, spectrum_to_binned_vector
    from casmi26.models import CASMIPipeline, SpectralLibraryIndex, SpectrumToFingerprintModel
    import numpy as np

    synth_dst = export_dir / "synthetic_data"
    synth_dst.mkdir(parents=True, exist_ok=True)

    train_df, cand_df = generate_synthetic_casmi26_data(seed=2026)
    train_df["fold"] = assign_leak_free_folds(train_df, n_splits=5, group_col="inchikey14")
    val_queries_df, val_truth_df = sample_timstof_like_validation_panel(
        train_df[train_df["fold"] == 0].reset_index(drop=True), n_molecules=40
    )

    coconut_path = synth_dst / "coconut_candidates.parquet"
    pubchem_path = synth_dst / "pubchem_candidates.parquet"
    train_synth_path = synth_dst / "train_synthetic.parquet"
    test_synth_path = synth_dst / "test_synthetic.parquet"
    truth_synth_path = synth_dst / "val_truth_synthetic.parquet"

    cand_df[cand_df["is_coconut"]].to_parquet(coconut_path, index=False)
    cand_df[~cand_df["is_coconut"]].to_parquet(pubchem_path, index=False)
    train_df.to_parquet(train_synth_path, index=False)
    val_queries_df.to_parquet(test_synth_path, index=False)
    val_truth_df.to_parquet(truth_synth_path, index=False)

    # Also mirror into REPO_ROOT / data / synthetic
    local_synth = REPO_ROOT / "data" / "synthetic"
    local_synth.mkdir(parents=True, exist_ok=True)
    for src_f in synth_dst.glob("*.parquet"):
        shutil.copy2(src_f, local_synth / src_f.name)

    # Ensure a baseline trained checkpoint exists if not already produced by Colab GPU
    if not (export_dir / "casmi_pipeline_models.pkl").exists() or not (export_dir / "candidate_pool.parquet").exists():
        pool = build_candidate_pool_from_sources(
            train_df=train_df,
            candidate_parquet_paths=[coconut_path, pubchem_path],
        )
        n_tr = len(train_df)
        X_b = np.zeros((n_tr, 1536), dtype=np.float32)
        X_nm = np.zeros(n_tr, dtype=np.float32)
        X_pos = np.zeros(n_tr, dtype=np.float32)
        Y_fp = np.zeros((n_tr, 2215), dtype=np.float32)
        for i, r in enumerate(train_df.itertuples(index=False)):
            cm, cp = clean_and_normalize_spectrum(r.ms2_mzs, r.ms2_normalized_intensities, r.precursor_mz)
            X_b[i] = spectrum_to_binned_vector(cm, cp, r.precursor_mz)
            X_nm[i] = float(r.precursor_mz)
            X_pos[i] = 1.0 if str(r.ionization_mode).startswith("pos") else 0.0
            Y_fp[i] = smiles_to_fingerprint(r.normalized_smiles)

        fp_model = SpectrumToFingerprintModel(use_torch_if_available=False, ridge_alpha=5.0)
        fp_model.fit(X_b, X_nm, X_pos, Y_fp)
        spec_index = SpectralLibraryIndex.build(train_df)
        pipe = CASMIPipeline(pool, spec_index, fp_model, ppm_tolerance=15.0)
        pipe.fit_reranker(val_queries_df, val_truth_df)
        pipe.save_checkpoint(export_dir)


def prepare_dataset_metadata(
    export_dir: Path,
    dataset_slug: str,
    title: str = "CASMI 2026 Source Code Synthetic Data and Models",
) -> Path:
    """
    Stages a complete, self-contained Kaggle Dataset inside `export_dir` including:
    - `casmi26/` (all Python source modules)
    - `tests/` & `scripts/` (verification & rubric utilities)
    - `synthetic_data/` (synthetic MS/MS spectra, COCONUT & PubChem isomer shards, validation truth)
    - `candidate_pool.parquet` & `casmi_pipeline_models.pkl` (trained retrieval + reranking models)
    - `dataset-metadata.json` & `README.md`
    """
    export_dir.mkdir(parents=True, exist_ok=True)

    # 1. Copy Python source folders (`casmi26/`, `tests/`, `scripts/`) into `export_dir`
    for folder_name in ("casmi26", "tests", "scripts"):
        src_dir = REPO_ROOT / folder_name
        dst_dir = export_dir / folder_name
        if src_dir.exists():
            if dst_dir.exists():
                shutil.rmtree(dst_dir)
            shutil.copytree(
                src_dir,
                dst_dir,
                ignore=shutil.ignore_patterns("__pycache__", "*.pyc", ".pytest_cache"),
            )

    # 2. Generate and bundle synthetic data + baseline model checkpoint
    ensure_synthetic_data_and_baseline_checkpoint(export_dir)

    # 3. Write dataset README.md
    readme_text = (
        f"# {title}\n\n"
        "Self-contained public Kaggle Dataset for **Enveda CASMI 2026 – Molecule ID From Mass Spectra**.\n\n"
        "## Contents\n"
        "- `casmi26/`: Complete Python source package (`config.py`, `chemistry.py`, `validation.py`, `features.py`, `retrieval.py`, `models.py`, `benchmark.py`).\n"
        "- `synthetic_data/`: Synthetic multi-adduct LC-MS/MS spectra (`train_synthetic.parquet`, `test_synthetic.parquet`) and same-mass isomer candidate shards (`coconut_candidates.parquet`, `pubchem_candidates.parquet`).\n"
        "- `candidate_pool.parquet` & `casmi_pipeline_models.pkl`: Pre-indexed candidate pool and trained `CASMIPipeline` (`FPNet` + `SpectralLibraryIndex` + `LightGBM`/`GBDT` Reranker).\n"
        "- `tests/` & `scripts/`: 100-point rubric scorer (`score_notebook_rubric.py`) and end-to-end verification tests.\n"
    )
    (export_dir / "README.md").write_text(readme_text, encoding="utf-8")

    # 4. Write Kaggle `dataset-metadata.json`
    metadata = {
        "title": title,
        "id": dataset_slug,
        "licenses": [{"name": "CC0-1.0"}],
    }
    meta_path = export_dir / "dataset-metadata.json"
    meta_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return meta_path


def main() -> None:
    default_user = detect_default_kaggle_username()
    parser = argparse.ArgumentParser(description="Package & publish CASMI 2026 Python src, synthetic data & models as a Kaggle Dataset.")
    parser.add_argument(
        "--export-dir",
        type=Path,
        default=REPO_ROOT / "kaggle_dataset_export",
        help="Directory to stage the complete Kaggle Dataset bundle",
    )
    parser.add_argument(
        "--dataset-slug",
        type=str,
        default=f"{default_user}/casmi26-trained-models",
        help="Kaggle dataset slug in the format <username>/<dataset-slug>",
    )
    parser.add_argument(
        "--title",
        type=str,
        default="CASMI 2026 Source Code Synthetic Data and Models",
        help="Human-readable Kaggle Dataset title (6-50 chars, alphanumeric)",
    )
    parser.add_argument(
        "--publish",
        action="store_true",
        help="Run `kaggle datasets create` or `kaggle datasets version` immediately",
    )
    parser.add_argument(
        "--update",
        action="store_true",
        help="Create a new version of an existing dataset instead of creating from scratch",
    )
    parser.add_argument(
        "--message",
        type=str,
        default="Bundle Python src (casmi26), synthetic data, and trained models",
        help="Version notes when updating the dataset",
    )
    args = parser.parse_args()

    meta_path = prepare_dataset_metadata(args.export_dir, args.dataset_slug, args.title)
    print(f"[OK] Staged complete Kaggle Dataset bundle at: {args.export_dir}")
    print(f"[OK] Wrote metadata ({args.dataset_slug}):     {meta_path}")
    print("Bundled contents:")
    for p in sorted(args.export_dir.iterdir()):
        kind = "DIR " if p.is_dir() else "FILE"
        print(f"  - [{kind}] {p.name}")

    if args.publish:
        if args.update:
            cmd = [
                "kaggle",
                "datasets",
                "version",
                "-p",
                str(args.export_dir),
                "-m",
                args.message,
                "--dir-mode",
                "zip",
            ]
        else:
            cmd = [
                "kaggle",
                "datasets",
                "create",
                "-p",
                str(args.export_dir),
                "--dir-mode",
                "zip",
                "--public",
            ]
        print(f"[RUN] {' '.join(cmd)}")
        subprocess.run(cmd, check=True)
    else:
        print("\nTo publish this public dataset to Kaggle (once KAGGLE_USERNAME & KAGGLE_KEY are set), run:")
        print(f"  kaggle datasets create -p {args.export_dir} --dir-mode zip --public")
        print("Or to push a new version after a Colab GPU training run:")
        print(f"  kaggle datasets version -p {args.export_dir} -m '{args.message}' --dir-mode zip")


if __name__ == "__main__":
    main()

