#!/usr/bin/env python3
"""
Creates and publishes/updates a Kaggle Dataset from the trained model checkpoint
directory (`kaggle_dataset_export/`) produced in Google Colab or locally.

Once published, attach that Kaggle Dataset (e.g., `/kaggle/input/casmi26-trained-models`)
to your offline Kaggle Inference Notebook (`notebooks/02_kaggle_inference_submission.ipynb`)
to score on the official CASMI 2026 Leaderboard.

Usage:
    python3 scripts/publish_kaggle_model_dataset.py \
        --export-dir /mnt/LMSTUDIO/CASMI/kaggle_dataset_export \
        --dataset-slug your-kaggle-username/casmi26-trained-models \
        --message "Colab GPU trained FPNet + LightGBM Reranker v1"
"""

from __future__ import annotations
import argparse
import json
import shutil
import subprocess
from pathlib import Path


def prepare_dataset_metadata(
    export_dir: Path,
    dataset_slug: str,
    title: str = "CASMI 2026 Trained Models & Code",
) -> Path:
    """
    Writes `dataset-metadata.json` and bundles the `casmi26/` Python package inside
    `export_dir` so the offline Kaggle Submission Notebook can `import casmi26` directly
    from `/kaggle/input/<dataset-name>` with zero internet access.
    """
    export_dir.mkdir(parents=True, exist_ok=True)

    # Copy the `casmi26` Python source package into the dataset folder so Kaggle
    # inference notebooks have the exact matching code version offline.
    repo_root = Path(__file__).resolve().parents[1]
    pkg_src = repo_root / "casmi26"
    pkg_dst = export_dir / "casmi26"
    if pkg_src.exists():
        if pkg_dst.exists():
            shutil.rmtree(pkg_dst)
        shutil.copytree(
            pkg_src,
            pkg_dst,
            ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
        )

    metadata = {
        "title": title,
        "id": dataset_slug,
        "licenses": [{"name": "CC0-1.0"}],
    }
    meta_path = export_dir / "dataset-metadata.json"
    meta_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return meta_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Package & publish CASMI 2026 models as a Kaggle Dataset.")
    parser.add_argument(
        "--export-dir",
        type=Path,
        default=Path("/mnt/LMSTUDIO/CASMI/kaggle_dataset_export"),
        help="Directory containing candidate_pool.parquet and casmi_pipeline_models.pkl",
    )
    parser.add_argument(
        "--dataset-slug",
        type=str,
        default="your-username/casmi26-trained-models",
        help="Kaggle dataset slug in the format <username>/<dataset-slug>",
    )
    parser.add_argument(
        "--title",
        type=str,
        default="CASMI 2026 Trained Models and Code",
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
        default="Update trained CASMI 2026 model artifacts from Colab GPU session",
        help="Version notes when updating the dataset",
    )
    args = parser.parse_args()

    meta_path = prepare_dataset_metadata(args.export_dir, args.dataset_slug, args.title)
    print(f"[OK] Prepared Kaggle Dataset bundle at: {args.export_dir}")
    print(f"[OK] Wrote metadata:                    {meta_path}")
    print("Bundled contents:")
    for p in sorted(args.export_dir.iterdir()):
        print(f"  - {p.name}")

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
        print("\nTo publish this bundle to Kaggle from Colab or your terminal, run:")
        print(f"  kaggle datasets create -p {args.export_dir} --dir-mode zip --public")
        print("Or to push a new version after another Colab training run:")
        print(f"  kaggle datasets version -p {args.export_dir} -m '{args.message}' --dir-mode zip")


if __name__ == "__main__":
    main()
