"""
Centralized configuration, constants, and path resolution across:
1. Local development (/mnt/LMSTUDIO/CASMI)
2. Google Colab + Google Drive (/content/drive/MyDrive/CASMI2026 & /content/casmi_fast)
3. Kaggle Submission Notebook (/kaggle/input/enveda-casmi-2026 & /kaggle/working)
"""

from __future__ import annotations
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict

# ---------------------------------------------------------------------------
# Fundamental Physical & Mass Spectrometry Constants
# ---------------------------------------------------------------------------

# Mass of an electron in Daltons (Da) - critical for positive/negative mode calibration
ELECTRON_MASS_DA: float = 0.000548579909

# Exact monoisotopic atomic masses (IUPAC / NIST)
MASS_H: float = 1.00782503223
MASS_PROTON: float = MASS_H - ELECTRON_MASS_DA  # 1.00727645232 Da
MASS_C: float = 12.0000000000
MASS_N: float = 14.00307400443
MASS_O: float = 15.99491461957
MASS_NA: float = 22.9897692820
MASS_K: float = 38.9637064864
MASS_CL: float = 34.968852682
MASS_BR: float = 78.9183376

# Common Adduct definitions: (multiplier_on_M, mass_shift_Da, charge_z)
# Neutral Mass M = (precursor_mz * |charge_z| - mass_shift_Da) / multiplier_on_M
ADDUCT_MASS_SHIFTS: Dict[str, tuple[float, float, int]] = {
    # Positive mode adducts
    "[M+H]+": (1.0, MASS_PROTON, 1),
    "[M+H-H2O]+": (1.0, MASS_PROTON - (2 * MASS_H + MASS_O), 1),
    "[M-H2O+H]+": (1.0, MASS_PROTON - (2 * MASS_H + MASS_O), 1),
    "[M+H-2H2O]+": (1.0, MASS_PROTON - 2 * (2 * MASS_H + MASS_O), 1),
    "[M-2H2O+H]+": (1.0, MASS_PROTON - 2 * (2 * MASS_H + MASS_O), 1),
    "[M+Na]+": (1.0, MASS_NA - ELECTRON_MASS_DA, 1),
    "[M+K]+": (1.0, MASS_K - ELECTRON_MASS_DA, 1),
    "[M+NH4]+": (1.0, MASS_N + 4 * MASS_H - ELECTRON_MASS_DA, 1),
    "[M+CH3OH+H]+": (1.0, MASS_C + 4 * MASS_H + MASS_O + MASS_PROTON, 1),
    "[M+ACN+H]+": (1.0, 2 * MASS_C + 3 * MASS_H + MASS_N + MASS_PROTON, 1),
    "[2M+H]+": (2.0, MASS_PROTON, 1),
    "[2M+Na]+": (2.0, MASS_NA - ELECTRON_MASS_DA, 1),
    "[2M+NH4]+": (2.0, MASS_N + 4 * MASS_H - ELECTRON_MASS_DA, 1),
    "[M+2H]2+": (1.0, 2 * MASS_PROTON, 2),
    "[M]+": (1.0, -ELECTRON_MASS_DA, 1),
    # Negative mode adducts
    "[M-H]-": (1.0, -(MASS_H - ELECTRON_MASS_DA), -1),
    "[M-H-H2O]-": (1.0, -(MASS_H - ELECTRON_MASS_DA) - (2 * MASS_H + MASS_O), -1),
    "[M-H2O-H]-": (1.0, -(MASS_H - ELECTRON_MASS_DA) - (2 * MASS_H + MASS_O), -1),
    "[M+Cl]-": (1.0, MASS_CL + ELECTRON_MASS_DA, -1),
    "[M+Br]-": (1.0, MASS_BR + ELECTRON_MASS_DA, -1),
    "[M+FA-H]-": (1.0, MASS_C + 2 * MASS_H + 2 * MASS_O - (MASS_H - ELECTRON_MASS_DA), -1),
    "[M+CH2O2-H]-": (1.0, MASS_C + 2 * MASS_H + 2 * MASS_O - (MASS_H - ELECTRON_MASS_DA), -1),
    "[M+Hac-H]-": (1.0, 2 * MASS_C + 4 * MASS_H + 2 * MASS_O - (MASS_H - ELECTRON_MASS_DA), -1),
    "[M+C2H4O2-H]-": (1.0, 2 * MASS_C + 4 * MASS_H + 2 * MASS_O - (MASS_H - ELECTRON_MASS_DA), -1),
    "[M+CH3COO]-": (1.0, 2 * MASS_C + 3 * MASS_H + 2 * MASS_O + ELECTRON_MASS_DA, -1),
    "[M+HCOO]-": (1.0, MASS_C + MASS_H + 2 * MASS_O + ELECTRON_MASS_DA, -1),
    "[2M-H]-": (2.0, -(MASS_H - ELECTRON_MASS_DA), -1),
    "[2M+Na-2H]-": (2.0, MASS_NA - 2 * MASS_H + ELECTRON_MASS_DA, -1),
    "[2M+CH2O2-H]-": (2.0, MASS_C + 2 * MASS_H + 2 * MASS_O - (MASS_H - ELECTRON_MASS_DA), -1),
    "[M-2H]2-": (1.0, -2 * (MASS_H - ELECTRON_MASS_DA), -2),
    "[M]-": (1.0, ELECTRON_MASS_DA, -1),
}

# Required pinning per CASMI 2026 evaluation rules
REQUIRED_RDKIT_VERSION = "2026.03.3"
TOP_K_CANDIDATES = 25


@dataclass
class EnvironmentPaths:
    """Resolved paths for data, candidate databases, and model checkpoints."""
    env_name: str
    root_dir: Path
    data_dir: Path
    candidate_db_dir: Path
    artifacts_dir: Path
    output_dir: Path
    kaggle_dataset_slug: str = "your-username/casmi26-trained-models"

    @property
    def train_parquet(self) -> Path:
        return self.data_dir / "train.parquet"

    @property
    def test_parquet(self) -> Path:
        return self.data_dir / "test.parquet"

    @property
    def submission_csv(self) -> Path:
        return self.output_dir / "submission.csv"


def detect_environment(
    custom_root: str | Path | None = None,
    kaggle_model_slug: str = "casmi26-trained-models",
) -> EnvironmentPaths:
    """
    Auto-detects whether code is running on:
    - Kaggle Notebook (/kaggle/input/enveda-casmi-2026 or /kaggle/inputs/enveda-casmi-2026)
    - Google Colab (/content/drive/MyDrive/CASMI2026 & /content/casmi_fast)
    - Local Workspace (/mnt/LMSTUDIO/CASMI, including tracked ./kaggle/input and ./drive_mount/MyDrive/CASMI2026 placeholders)
    """
    if custom_root is not None:
        root = Path(custom_root)
        return EnvironmentPaths(
            env_name="custom",
            root_dir=root,
            data_dir=root / "data",
            candidate_db_dir=root / "data" / "candidates",
            artifacts_dir=root / "kaggle_dataset_export",
            output_dir=root,
        )

    # 1. Check Kaggle Notebook runtime (/kaggle/input or /kaggle/inputs)
    for k_base in (Path("/kaggle/input"), Path("/kaggle/inputs")):
        if (k_base / "enveda-casmi-2026").exists():
            return EnvironmentPaths(
                env_name="kaggle",
                root_dir=Path("/kaggle/working"),
                data_dir=k_base / "enveda-casmi-2026",
                candidate_db_dir=k_base / "casmi-2026-candidate-database",
                artifacts_dir=k_base / kaggle_model_slug,
                output_dir=Path("/kaggle/working"),
            )

    # 2. Check Google Colab runtime (Google Drive mount at /content/drive/MyDrive/CASMI2026)
    if Path("/content").exists():
        drive_root = Path("/content/drive/MyDrive/CASMI2026")
        fast_nvme = Path("/content/casmi_fast")
        for d in (
            drive_root / "data",
            drive_root / "candidates" / "coconut",
            drive_root / "candidates" / "pubchem",
            drive_root / "kaggle_dataset_export",
            drive_root / "outputs",
        ):
            d.mkdir(parents=True, exist_ok=True)
        data_dir = fast_nvme if (fast_nvme / "train.parquet").exists() else (drive_root / "data")
        return EnvironmentPaths(
            env_name="colab",
            root_dir=drive_root,
            data_dir=data_dir,
            candidate_db_dir=(
                fast_nvme / "candidates"
                if (fast_nvme / "candidates").exists()
                else (drive_root / "candidates")
            ),
            artifacts_dir=drive_root / "kaggle_dataset_export",
            output_dir=drive_root / "outputs",
        )

    # 3. Default to Local / Cloned Repository Workspace
    local_root = Path(__file__).resolve().parents[1]
    return EnvironmentPaths(
        env_name="local",
        root_dir=local_root,
        data_dir=local_root / "data",
        candidate_db_dir=local_root / "data" / "candidates",
        artifacts_dir=local_root / "kaggle_dataset_export",
        output_dir=local_root / "outputs",
    )
