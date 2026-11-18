"""Runtime helpers shared by the RL77 notebooks on Kaggle, Colab, and a local checkout.

Every notebook imports this module from the ``casmi26-rl77-prereqs`` Kaggle dataset
(or from ``experiments/runs/RL77/lib`` in the repository).  It answers four questions
the Gemini cells hard-coded for Google Drive:

* where the competition parquet files are mounted,
* where an upstream notebook's output dataset is mounted,
* where this notebook writes its outputs, and
* how to install the pinned RDKit / FAISS wheels with or without internet.
"""

from __future__ import annotations

import glob
import importlib.metadata
import importlib.util
import json
import os
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple

COMPETITION_SLUG = "enveda-CASMI26-molecule-id-mass-spectra"
KAGGLE_OWNER = os.environ.get("RL77_KAGGLE_OWNER", "scottweeden")
KAGGLE_INPUT = Path("/kaggle/input")
KAGGLE_WORKING = Path("/kaggle/working")

# Kaggle dataset slugs written by the RL77 pipeline (see tools/kaggle_pipeline.py).
PREREQS_SLUG = "casmi26-rl77-prereqs"
CANDIDATES_SLUG = "casmi26-rl77-candidates"
OUTPUT_SLUGS = {
    "nb1": "casmi26-rl77-nb1-glmr-outputs",
    "nb2": "casmi26-rl77-nb2-store-outputs",
    "nb3": "casmi26-rl77-nb3-ensemble-outputs",
    "nb4": "casmi26-rl77-nb4-formula-outputs",
    "nb5": "casmi26-rl77-nb5-qrl-outputs",
}

# The competition metric canonicalizes with RDKit 2026.03.3 (PEP 440 spelling below).
PINNED_REQUIREMENTS = {
    "rdkit": ("rdkit", "2026.3.3"),
    "faiss": ("faiss-cpu", None),
    "scipy": ("scipy", None),
}

_T0 = time.time()


def on_kaggle() -> bool:
    return KAGGLE_INPUT.is_dir()


def log(message: str) -> None:
    """Print with wall-clock minutes since import, so Kaggle logs show where time went."""
    print(f"[{(time.time() - _T0) / 60.0:7.2f} min] {message}", flush=True)


def input_roots() -> List[Path]:
    roots: List[Path] = []
    if on_kaggle():
        roots.append(KAGGLE_INPUT)
    extra = os.environ.get("RL77_INPUT_ROOTS", "")
    roots.extend(Path(p) for p in extra.split(os.pathsep) if p)
    return [r for r in roots if r.is_dir()]


def _walk_for(filename: str, roots: Iterable[Path], max_depth: int = 6) -> List[Path]:
    hits: List[Path] = []
    for root in roots:
        for dirpath, dirnames, filenames in os.walk(root, followlinks=True):
            depth = len(Path(dirpath).relative_to(root).parts)
            if depth >= max_depth:
                dirnames[:] = []
            if filename in filenames:
                hits.append(Path(dirpath) / filename)
    return hits


def competition_dir() -> Path:
    """Directory holding train.parquet / test.parquet, whichever mount layout Kaggle uses."""
    env = os.environ.get("RL77_COMPETITION_DIR")
    if env and (Path(env) / "test.parquet").exists():
        return Path(env)
    for hit in _walk_for("test.parquet", input_roots()):
        if (hit.parent / "train.parquet").exists() or (hit.parent / "sample_submission.csv").exists():
            return hit.parent
    raise FileNotFoundError(
        f"Competition data not found. Attach `{COMPETITION_SLUG}` as a notebook input "
        "or set RL77_COMPETITION_DIR."
    )


def dataset_dir(slug: str, marker: str, required: bool = True) -> Optional[Path]:
    """Locate an attached dataset (or notebook-output dataset) by slug and a marker file.

    Kaggle has mounted datasets at /kaggle/input/<slug> and at
    /kaggle/input/datasets/<owner>/<slug>; both layouts are searched.
    """
    env_key = "RL77_DIR_" + slug.upper().replace("-", "_")
    env = os.environ.get(env_key)
    if env and (Path(env) / marker).exists():
        return Path(env)
    hits = _walk_for(marker, input_roots())
    for hit in hits:
        if slug in hit.parts:
            return hit.parent
    if required:
        raise FileNotFoundError(
            f"Input `{slug}` (marker {marker}) is not attached. Add the dataset "
            f"{KAGGLE_OWNER}/{slug} to this notebook or set {env_key}."
        )
    return None


def upstream_dir(notebook: str, marker: str, required: bool = True) -> Optional[Path]:
    return dataset_dir(OUTPUT_SLUGS[notebook], marker, required=required)


def work_dir() -> Path:
    if on_kaggle():
        path = KAGGLE_WORKING
    else:
        path = Path(os.environ.get("RL77_WORK_DIR", Path.cwd() / "rl77_work"))
    path.mkdir(parents=True, exist_ok=True)
    return path


def _installed_version(dist: str) -> Optional[str]:
    try:
        return importlib.metadata.version(dist)
    except importlib.metadata.PackageNotFoundError:
        return None


def _wheel_dir() -> Optional[Path]:
    for pattern in ("rdkit-*.whl",):
        for root in input_roots():
            hits = glob.glob(str(root / "**" / pattern), recursive=True)
            if hits:
                return Path(hits[0]).parent
    return None


def ensure_packages() -> Dict[str, Optional[str]]:
    """Install the pinned RDKit and FAISS when the image lacks them.

    Offline (competition submission) installs come from the wheels that notebook 2
    bundles into its output dataset; online runs fall back to PyPI.
    """
    needed: List[str] = []
    for module, (dist, version) in PINNED_REQUIREMENTS.items():
        have = _installed_version(dist) if importlib.util.find_spec(module) else None
        if have is None or (version is not None and have != version):
            needed.append(f"{dist}=={version}" if version else dist)
    if needed:
        cmd = [sys.executable, "-m", "pip", "install", "-q"]
        wheels = _wheel_dir()
        if wheels is not None:
            cmd += ["--no-index", "--find-links", str(wheels)]
        log(f"Installing {needed} ({'offline wheels ' + str(wheels) if wheels else 'PyPI'})")
        result = subprocess.run(cmd + needed, capture_output=True, text=True)
        if result.returncode != 0:
            print(result.stdout[-2000:], result.stderr[-4000:])
            raise RuntimeError(f"pip install failed for {needed}")
        importlib.invalidate_caches()
    versions = {dist: _installed_version(dist) for dist, _ in PINNED_REQUIREMENTS.values()}
    log(f"Package versions: {versions}")
    return versions


class Deadline:
    """Wall-clock budget for a training phase."""

    def __init__(self, minutes: float):
        self.minutes = float(minutes)
        self.t_end = time.time() + 60.0 * self.minutes

    def expired(self) -> bool:
        return time.time() >= self.t_end

    def remaining_min(self) -> float:
        return max(0.0, (self.t_end - time.time()) / 60.0)


def write_json(path: Path, payload: Mapping[str, Any]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=_json_default) + "\n", encoding="utf-8")
    return path


def read_json(path: Path) -> Dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _json_default(obj: Any) -> Any:
    try:
        import numpy as np

        if isinstance(obj, np.generic):
            return obj.item()
        if isinstance(obj, np.ndarray):
            return obj.tolist()
    except ImportError:
        pass
    if isinstance(obj, Path):
        return str(obj)
    raise TypeError(f"Not JSON serializable: {type(obj)}")


def select_validated(prior: Any, fitted: Any, prior_score: Optional[float], fitted_score: Optional[float],
                     min_gain: float = 0.0) -> Tuple[Any, bool]:
    """Pick the values this run uses on the test set with the same rule as `tools/evidently_weights.py adopt`:
    fitted values win only when their cross-fitted held-out score beats the prior's. Returns (values, used_fitted)."""
    used = prior_score is not None and fitted_score is not None and fitted_score > prior_score + min_gain
    return (fitted if used else prior), used


def write_run_manifest(notebook: str, out_dir: Path, weights_version: str, extra: Optional[Mapping[str, Any]] = None) -> Path:
    """Record what produced this output folder; the publish step copies it into the dataset."""
    try:
        import torch

        device = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
        torch_version = torch.__version__
    except ImportError:
        device, torch_version = "unknown", None
    payload: Dict[str, Any] = {
        "notebook": notebook,
        "finished_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "runtime_min": round((time.time() - _T0) / 60.0, 2),
        "weights_version": weights_version,
        "python": platform.python_version(),
        "torch": torch_version,
        "device": device,
        "rdkit": _installed_version("rdkit"),
        "on_kaggle": on_kaggle(),
        "kaggle_kernel_run_type": os.environ.get("KAGGLE_KERNEL_RUN_TYPE"),
    }
    if extra:
        payload.update(dict(extra))
    return write_json(Path(out_dir) / "run_manifest.json", payload)
