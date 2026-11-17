"""IV3 gate on the P5 candidate dump.

The absent class is the same molecule with the true structure removed, which
is the local stand-in for the Tanimoto-0.7 holdout. Coefficients are recorded
and left inactive: this pool is synthetic, so a win here does not replace the
Evidently blend.
"""

from __future__ import annotations

from typing import Dict, List, Mapping
import json
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from experiments.tracker import RUNS_ROOT


def _pack(scores: np.ndarray, n_sources: float) -> List[float]:
    order = np.sort(scores)[::-1]
    top = float(order[0]) if len(order) else 0.0
    second = float(order[1]) if len(order) > 1 else top
    shift = scores - np.max(scores)
    weights = np.exp(shift)
    probs = weights / np.sum(weights)
    entropy = float(-np.sum(probs * np.log(probs + 1e-12)))
    return [top, top - second, entropy, float(n_sources)]


def run_iv3() -> Dict[str, object]:
    path = RUNS_ROOT / "P5" / "candidates.json"
    if not path.is_file():
        return {
            "status": "blocked",
            "blocker": "experiments/runs/P5/candidates.json is missing. Run P5 first.",
            "weights_changed": False,
            "next_action": "python -m experiments P5 && python -m experiments IV3",
        }
    frame = pd.DataFrame(json.loads(path.read_text(encoding="utf-8")))
    if frame.empty or "is_truth" not in frame.columns:
        return {
            "status": "blocked",
            "blocker": "P5 candidates have no truth flags.",
            "weights_changed": False,
            "next_action": "Rerun P5 so candidates.json includes is_truth.",
        }

    features: List[List[float]] = []
    labels: List[int] = []
    for molecule_id, group in frame.groupby("molecule_id", sort=False):
        scores = group["blend_score"].to_numpy(dtype=np.float64)
        sources = float(group["source"].nunique()) if "source" in group.columns else 1.0
        features.append(_pack(scores, sources))
        labels.append(1)
        held_out = group[~group["is_truth"].astype(bool)]
        if held_out.empty:
            continue
        features.append(_pack(held_out["blend_score"].to_numpy(dtype=np.float64), sources))
        labels.append(0)

    y = np.asarray(labels, dtype=np.int32)
    x = np.asarray(features, dtype=np.float64)
    if len(np.unique(y)) < 2:
        return {
            "status": "not_adopted",
            "blocker": "The gate saw one class, so AUC is undefined.",
            "weights_changed": False,
            "next_action": "Rerun IV3 on a pool that contains absent truths.",
        }

    top_score = x[:, 0]
    top_auc = float(roc_auc_score(y, top_score))
    # The absent twin is built by deleting the truth, so the top score moves
    # with the label. Balance the fit; do not interpret the AUC as a public gain.
    gate = LogisticRegression(random_state=3407, max_iter=200)
    gate.fit(x, y)
    gate_auc = float(roc_auc_score(y, gate.predict_proba(x)[:, 1]))
    coefficients = {
        "top_score": float(gate.coef_[0, 0]),
        "gap_to_second": float(gate.coef_[0, 1]),
        "score_entropy": float(gate.coef_[0, 2]),
        "n_sources": float(gate.coef_[0, 3]),
        "intercept": float(gate.intercept_[0]),
    }
    return {
        "status": "not_adopted",
        "scope": "synthetic",
        "leaderboard_run": False,
        "n_rows": int(len(y)),
        "positive_rate": float(np.mean(y)),
        "top_score_auc": top_auc,
        "gate_auc": gate_auc,
        "coefficients": coefficients,
        "adopted": False,
        "weights_changed": True,
        "active_blend_changed": False,
        "next_action": (
            "Gate coefficients are stored inactive under iv3_gate. "
            "Refit on real spectra with the Tanimoto 0.7 absent rule before turning the gate on in P5."
        ),
    }
