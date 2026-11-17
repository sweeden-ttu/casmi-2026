"""NPComposer experiments.

NPC0 and NPC1 load ralyn/NPComposer and ralyn/NPComposer-v2. Those two names
publish the same COCONUT fine-tune of GP-MoLFormer. NPC2 trains a ranker on
the SMILES that model generates and keeps the ranker only when it beats that
generated order on a held-out set of molecules.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Mapping, Tuple
import hashlib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from casmi26.chemistry import score_mrr_at_25

REPO_ROOT = Path(__file__).resolve().parents[1]
NPC0_WEIGHTS = REPO_ROOT / "artifacts" / "NPC0" / "npcomposer" / "pytorch_model.bin"
NPC1_WEIGHTS = REPO_ROOT / "artifacts" / "NPC1" / "npcomposer-v2" / "pytorch_model.bin"

FEATURE_COLUMNS = ("mass_error_ppm", "np_likeness_score")
# Published NPComposer eval loss after 8 epochs. Generation metrics are measured
# from predictions.json and are the bar NPC2 has to clear.
PUBLISHED_EVAL_LOSS = 0.2718


def _ranked_lists(frame: pd.DataFrame, score_column: str) -> Tuple[list, list]:
    ordered = frame.sort_values(["molecule_id", score_column], ascending=[True, False], kind="mergesort")
    predictions = []
    truths = []
    for molecule_id, group in ordered.groupby("molecule_id", sort=False):
        del molecule_id
        predictions.append(group["inchikey14"].astype(str).tolist())
        truths.append(str(group.loc[group["is_truth"].astype(bool), "inchikey14"].iloc[0]))
    return predictions, truths


def train_to_beat_generations(frame: pd.DataFrame, seed: int = 3407) -> Dict[str, object]:
    """Rerank NPComposer samples. Adopt only if held-out MRR@25 rises.

    ``generation_rank`` is the order the language model emitted. The ranker
    never sees ``is_truth`` as a feature. Fit molecules and eval molecules are
    disjoint.
    """
    required = {"molecule_id", "inchikey14", "generation_rank", "is_truth", *FEATURE_COLUMNS}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"predictions are missing {sorted(missing)}")
    work = frame.copy()
    work["molecule_id"] = work["molecule_id"].astype(str)
    molecules = sorted(work["molecule_id"].unique())
    if len(molecules) < 2:
        return {
            "status": "not_adopted",
            "adopted": False,
            "reason": "Need at least two molecules so the ranker can be scored on a holdout.",
            "weights_changed": False,
        }
    # Stable split from the seed, not from row order inside a molecule.
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(molecules))
    cut = max(1, len(molecules) // 2)
    fit_ids = {molecules[i] for i in order[:cut]}
    eval_ids = {molecules[i] for i in order[cut:]}
    fit = work[work["molecule_id"].isin(fit_ids)]
    eval_frame = work[work["molecule_id"].isin(eval_ids)]
    if fit["is_truth"].nunique() < 2 or eval_frame.empty:
        return {
            "status": "not_adopted",
            "adopted": False,
            "reason": "The holdout did not contain both classes.",
            "weights_changed": False,
        }

    rank_score = -pd.to_numeric(eval_frame["generation_rank"], errors="coerce").fillna(0.0)
    baseline_frame = eval_frame.assign(_baseline=rank_score)
    baseline_mrr, _ = score_mrr_at_25(
        *_ranked_lists(baseline_frame, "_baseline"),
        ground_truth_is_inchikey14=True,
    )

    model = LogisticRegression(random_state=seed, max_iter=200)
    model.fit(fit.loc[:, FEATURE_COLUMNS].to_numpy(dtype=np.float64), fit["is_truth"].astype(int).to_numpy())
    scored = eval_frame.copy()
    scored["_learned"] = model.predict_proba(scored.loc[:, FEATURE_COLUMNS].to_numpy(dtype=np.float64))[:, 1]
    learned_mrr, _ = score_mrr_at_25(
        *_ranked_lists(scored, "_learned"),
        ground_truth_is_inchikey14=True,
    )
    adopted = bool(learned_mrr > baseline_mrr)
    coefficients = {
        name: float(coef) for name, coef in zip(FEATURE_COLUMNS, model.coef_.reshape(-1))
    }
    coefficients["intercept"] = float(model.intercept_[0])
    return {
        "status": "adopted" if adopted else "not_adopted",
        "adopted": adopted,
        "baseline_mrr_at_25": float(baseline_mrr),
        "learned_mrr_at_25": float(learned_mrr),
        "published_eval_loss": PUBLISHED_EVAL_LOSS,
        "coefficients": coefficients if adopted else {},
        "n_fit_molecules": len(fit_ids),
        "n_eval_molecules": len(eval_ids),
        "weights_changed": False,
        "reason": (
            "Held-out MRR@25 beat the NPComposer generation order."
            if adopted
            else "Held-out MRR@25 did not beat the NPComposer generation order. The ranker was not kept."
        ),
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def same_published_weights() -> Dict[str, object]:
    """NPComposer-v2 is a second name for the NPComposer checkpoint when the bytes match."""
    if not NPC0_WEIGHTS.is_file() or not NPC1_WEIGHTS.is_file():
        return {"identical": False}
    if NPC0_WEIGHTS.stat().st_size != NPC1_WEIGHTS.stat().st_size:
        return {"identical": False, "npc0_bytes": NPC0_WEIGHTS.stat().st_size, "npc1_bytes": NPC1_WEIGHTS.stat().st_size}
    npc0_sha = _sha256(NPC0_WEIGHTS)
    npc1_sha = _sha256(NPC1_WEIGHTS)
    return {
        "identical": npc0_sha == npc1_sha,
        "bytes": NPC0_WEIGHTS.stat().st_size,
        "sha256": npc0_sha,
    }


def prediction_baseline(payload: Mapping[str, object]) -> Dict[str, float]:
    """Metrics stored with the generations NPC2 has to beat."""
    metrics = payload.get("metrics")
    if not isinstance(metrics, dict):
        raise ValueError("predictions.json has no metrics object")
    return {str(key): float(value) for key, value in metrics.items()}
