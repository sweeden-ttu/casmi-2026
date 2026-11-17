#!/usr/bin/env bash
# =============================================================================
# Enveda CASMI 2026 - Dataset & Candidate Shard Downloader
# =============================================================================
# Supports two modes:
#   1. Run directly inside Google Colab (RECOMMENDED - fastest download to
#      Google Drive `/content/drive/MyDrive/CASMI2026/data` so your datasets
#      never consume local bandwidth and persist across all Colab GPU sessions).
#   2. Run locally on `/mnt/LMSTUDIO/CASMI/data` if you want a local copy.
#
# Prerequisites:
#   Ensure `~/.kaggle/kaggle.json` exists OR export KAGGLE_USERNAME & KAGGLE_KEY.
# =============================================================================

set -euo pipefail

TARGET_DIR="${1:-/mnt/LMSTUDIO/CASMI/data}"
mkdir -p "${TARGET_DIR}" "${TARGET_DIR}/candidates"

echo "=================================================================="
echo "Enveda CASMI 2026 Dataset Downloader"
echo "Target Directory: ${TARGET_DIR}"
echo "=================================================================="

if ! command -v kaggle >/dev/null 2>&1; then
    echo "[INFO] Installing kaggle CLI..."
    pip install -q kaggle
fi

if [[ ! -f "${HOME}/.kaggle/kaggle.json" ]] && [[ -z "${KAGGLE_USERNAME:-}" ]]; then
    echo "[ERROR] Kaggle credentials not found!"
    echo "Please place your API token at ~/.kaggle/kaggle.json (chmod 600 ~/.kaggle/kaggle.json)"
    echo "or export KAGGLE_USERNAME='...' and KAGGLE_KEY='...' before running this script."
    exit 1
fi

echo "[1/2] Downloading primary competition files (enveda-casmi-2026)..."
kaggle competitions download -c enveda-casmi-2026 -p "${TARGET_DIR}"
for z in "${TARGET_DIR}"/*.zip; do
    if [[ -f "$z" ]]; then
        echo "[INFO] Extracting $z ..."
        unzip -o "$z" -d "${TARGET_DIR}" && rm -f "$z"
    fi
done

echo "[2/2] Downloading CASMI 2026 Candidate Database shards (COCONUT 2.0 + PubChem)..."
kaggle datasets download -d rishavyadav/casmi-2026-candidate-database -p "${TARGET_DIR}/candidates" --unzip

echo "=================================================================="
echo "[DONE] All competition files and candidate shards downloaded to:"
ls -lh "${TARGET_DIR}"
echo "=================================================================="
