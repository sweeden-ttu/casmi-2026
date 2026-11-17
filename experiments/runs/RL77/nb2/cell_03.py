# [Colab Companion Cell 3] Bundle offline wheels + metadata and push to Kaggle Datasets
import subprocess


def prepare_and_upload_kaggle_dataset(
    kaggle_username: str,
    dataset_slug: str = "casmi26-glmr-offline-store",
    export_dir: Path = IndexCFG.EXPORT_DIR,
) -> None:
    """
    Downloads pinned wheels for offline installation and generates dataset-metadata.json
    for `kaggle datasets create` or `kaggle datasets version`.
    """
    wheels_dir = export_dir / "wheels"
    wheels_dir.mkdir(parents=True, exist_ok=True)

    # Download binary wheels compatible with Kaggle's Python 3.10/3.11 Linux runtime
    packages = ["rdkit==2026.03.3", "matchms", "faiss-cpu", "pyarrow"]
    subprocess.run(
        ["pip", "download", "--dest", str(wheels_dir), "--prefer-binary"] + packages,
        check=False,
    )

    metadata = {
        "title": "CASMI 2026 GLMR Offline Embeddings and Wheels",
        "id": f"{kaggle_username}/{dataset_slug}",
        "licenses": [{"name": "CC-BY-NC-4.0"}],
    }
    meta_path = export_dir / "dataset-metadata.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print(f"Dataset staged at {export_dir}. Run the following CLI command in Colab:")
    print(f"!kaggle datasets create -p {export_dir} --dir-mode zip")