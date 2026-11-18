# [Colab Companion Cell 3] Bundle offline wheels + metadata and push to Kaggle Datasets
import subprocess
import sys

OFFLINE_WHEELS = ["rdkit==2026.03.3", "faiss-cpu"]  # matchms was never imported; pyarrow ships with Kaggle


def prepare_and_upload_kaggle_dataset(
    kaggle_username: str = ENV.KAGGLE_OWNER,
    dataset_slug: str = ENV.OUTPUT_SLUGS["nb2"],
    export_dir: Path = IndexCFG.EXPORT_DIR,
) -> None:
    """
    Downloads pinned wheels for offline installation and generates dataset-metadata.json
    for `kaggle datasets create` or `kaggle datasets version`.
    """
    wheels_dir = export_dir / "wheels"
    wheels_dir.mkdir(parents=True, exist_ok=True)

    # Wheels for this exact interpreter/platform (the Kaggle image that runs the submission)
    result = subprocess.run(
        [sys.executable, "-m", "pip", "download", "--dest", str(wheels_dir), "--only-binary=:all:"] + OFFLINE_WHEELS,
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        print(result.stdout[-2000:], result.stderr[-2000:])
        raise RuntimeError("pip download failed; offline submissions would have no RDKit/FAISS wheels.")
    ENV.log(f"Bundled wheels: {sorted(p.name for p in wheels_dir.glob('*.whl'))}")

    metadata = {
        "title": "CASMI 2026 GLMR Offline Embeddings and Wheels",
        "id": f"{kaggle_username}/{dataset_slug}",
        "licenses": [{"name": "CC-BY-NC-4.0"}],
    }
    meta_path = export_dir / "dataset-metadata.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print(f"Dataset staged at {export_dir}. After this run, publish it from the repository with:")
    print("  .venv/bin/python experiments/runs/RL77/tools/kaggle_pipeline.py fetch nb2")
    print("  .venv/bin/python experiments/runs/RL77/tools/kaggle_pipeline.py publish-output nb2")
