"""RL77 blend weights, reward coefficients, and tolerances, read from an Evidently snapshot.

The source of truth is the Evidently dataset ``casmi-rl77-weights`` in the CASMI project
(see ``tools/evidently_weights.py``).  Kaggle kernels cannot reach the LAN Evidently
server, so ``tools/evidently_weights.py pull`` writes the latest dataset version to
``rl77_weights.json`` and the prerequisites Kaggle dataset ships that file.

Notebook code reads every coefficient through ``W("<key>")``.  A missing key raises
instead of silently falling back to a hard-coded number.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

WEIGHTS_FILENAME = "rl77_weights.json"
REPO_CURRENT = Path(__file__).resolve().parents[1] / "weights" / "current.json"


class RL77Weights:
    def __init__(self, payload: Dict[str, Any], source: Path):
        self.payload = payload
        self.source = Path(source)
        self.version: str = str(payload.get("version", "unknown"))
        self.values: Dict[str, float] = {str(k): float(v) for k, v in payload["weights"].items()}
        self.evidently: Dict[str, Any] = dict(payload.get("evidently", {}))

    def __call__(self, key: str) -> float:
        if key not in self.values:
            raise KeyError(f"Weight `{key}` is not in RL77 weights {self.version} ({self.source}).")
        return self.values[key]

    def int(self, key: str) -> int:
        return int(round(self(key)))

    def vec(self, prefix: str, names: Iterable[str]) -> List[float]:
        return [self(f"{prefix}.{name}") for name in names]

    def group(self, prefix: str) -> Dict[str, float]:
        cut = len(prefix) + 1
        return {k[cut:]: v for k, v in self.values.items() if k.startswith(prefix + ".")}

    def describe(self) -> str:
        ds = self.evidently.get("dataset_id", "not published")
        return f"RL77 weights {self.version} ({len(self.values)} values, Evidently dataset {ds}) from {self.source}"


def find_weights_file() -> Path:
    env = os.environ.get("RL77_WEIGHTS_JSON")
    if env:
        return Path(env)
    here = Path(__file__).resolve().parent / WEIGHTS_FILENAME
    if here.exists():
        return here
    if REPO_CURRENT.exists():
        return REPO_CURRENT
    raise FileNotFoundError(
        f"{WEIGHTS_FILENAME} not found next to rl77_weights.py and {REPO_CURRENT} is missing. "
        "Run `python tools/evidently_weights.py pull` and rebuild the prerequisites dataset."
    )


_LOADED: Dict[str, RL77Weights] = {}


def load_rl77_weights(path: Optional[Path] = None) -> RL77Weights:
    """Loads (once per process) the snapshot; upstream modules and the notebook share one object."""
    weight_path = (Path(path) if path else find_weights_file()).resolve()
    key = str(weight_path)
    if key not in _LOADED:
        payload = json.loads(weight_path.read_text(encoding="utf-8"))
        _LOADED[key] = RL77Weights(payload, weight_path)
        print(_LOADED[key].describe())
    return _LOADED[key]
