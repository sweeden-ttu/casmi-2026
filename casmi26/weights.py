"""Active reranker weights shared by scoring and the experiment log.

The numbers in ``v001`` are the blend recorded in
``outputs/evidently/evidently_weights_biases_and_metrics.json``.
Experiment runs read that file and write a new version only when a fit
changes a coefficient. ``current.json`` is the version scoring uses.
"""

from __future__ import annotations

from pathlib import Path
import json
from typing import Dict

REPO_ROOT = Path(__file__).resolve().parents[1]
WEIGHTS_DIR = REPO_ROOT / "experiments" / "weights"
CURRENT_PATH = WEIGHTS_DIR / "current.json"

# Same coefficients as CASMIPipeline.score_candidates before they moved here.
V001_ANALYTICAL: Dict[str, float] = {
    "lib_max_entropy_sim": 4.5,
    "lib_mean_entropy_sim": 2.0,
    "lib_max_cosine_sim": 1.5,
    "fp_tanimoto_sim": 3.0,
    "fp_log_likelihood": 1.2,
    "network_max_tanimoto": 1.8,
    "network_weighted_tanimoto": 1.2,
    "source_prior": 0.35,
    "np_likeness_score": 0.25,
    "mass_error_ppm": -0.08,
}


def analytical_weights(path: Path | None = None) -> Dict[str, float]:
    """Return the active analytical blend. Missing file falls back to v001."""
    weight_path = path or CURRENT_PATH
    if weight_path.is_file():
        payload = json.loads(weight_path.read_text(encoding="utf-8"))
        stored = payload.get("reranker_analytical_weights")
        if isinstance(stored, dict) and stored:
            return {str(k): float(v) for k, v in stored.items()}
    return dict(V001_ANALYTICAL)
