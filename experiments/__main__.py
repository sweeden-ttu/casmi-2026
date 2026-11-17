"""Run the experiment queue and write the resume log.

    /home/sweeden/.conda/envs/kagg/bin/python -m experiments
    /home/sweeden/.conda/envs/kagg/bin/python -m experiments P5 IV3
"""

from __future__ import annotations

from pathlib import Path
import importlib.util
import json
import sys

import torch

from experiments import QUEUE
from experiments.iv3_run import run_iv3
from experiments.p5_run import run_p5
from experiments.publish_weights import publish
from experiments.queue import GPU_REQUIRED, NEXT_COMMAND, REQUIREMENTS
from experiments.tracker import (
    REPO_ROOT,
    append_log,
    load_status,
    render_tracker,
    save_status,
    utc_now,
    write_manifest,
)
from casmi26.weights import CURRENT_PATH, V001_ANALYTICAL, WEIGHTS_DIR

SNAPSHOT = REPO_ROOT / "outputs" / "evidently" / "evidently_weights_biases_and_metrics.json"


def _cuda_available() -> bool:
    return bool(torch.cuda.is_available())


def _rdkit_available() -> bool:
    return importlib.util.find_spec("rdkit") is not None


def load_weights() -> dict:
    WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
    if CURRENT_PATH.is_file():
        return json.loads(CURRENT_PATH.read_text(encoding="utf-8"))
    snapshot = json.loads(SNAPSHOT.read_text(encoding="utf-8"))
    blend = {key: float(value) for key, value in snapshot["reranker_analytical_weights"].items()}
    if blend != V001_ANALYTICAL:
        raise SystemExit("Evidently snapshot blend does not match casmi26.weights.V001_ANALYTICAL")
    payload = {
        "version": "v001",
        "source": "outputs/evidently/evidently_weights_biases_and_metrics.json",
        "reranker_analytical_weights": blend,
        "fp_model_weights_and_biases": snapshot["fp_model_weights_and_biases"],
        "reranker_gbdt_importances": snapshot.get("reranker_gbdt_importances", {}),
        "gbdt_mix": {"gbdt": 0.65, "analytical": 0.35},
        "active_blend_changed": False,
        "inactive": {},
        "notes": "Copied from the Evidently local snapshot. Active coefficients were not retuned.",
    }
    text = json.dumps(payload, indent=2) + "\n"
    (WEIGHTS_DIR / "v001.json").write_text(text, encoding="utf-8")
    CURRENT_PATH.write_text(text, encoding="utf-8")
    return payload


def save_weights(payload: dict) -> None:
    text = json.dumps(payload, indent=2) + "\n"
    version = str(payload["version"])
    (WEIGHTS_DIR / f"{version}.json").write_text(text, encoding="utf-8")
    CURRENT_PATH.write_text(text, encoding="utf-8")


def blocked_record(exp_id: str, cuda: bool) -> dict:
    missing = []
    for relative, purpose in REQUIREMENTS.get(exp_id, []):
        if not (REPO_ROOT / relative).exists():
            missing.append({"path": relative, "purpose": purpose})
    if exp_id in GPU_REQUIRED and not cuda:
        missing.append({"path": "cuda", "purpose": "This run needs a GPU. torch.cuda.is_available() is false."})
    if exp_id in {"FH0", "RR1"} and not _rdkit_available():
        missing.append({"path": "rdkit", "purpose": "Molecule graphs need RDKit. It is not installed in the kagg env."})
    if not missing:
        return {
            "status": "ready",
            "next_action": NEXT_COMMAND.get(exp_id, "Implement the training step and rerun."),
            "weights_changed": False,
        }
    first = missing[0]
    return {
        "status": "blocked",
        "missing": missing,
        "blocker": f"{first['path']}: {first['purpose']}",
        "weights_changed": False,
        "next_action": NEXT_COMMAND.get(exp_id, "Supply the missing artifact and rerun."),
    }


def main(argv: list[str]) -> None:
    selected = argv or list(QUEUE)
    unknown = [item for item in selected if item not in QUEUE]
    if unknown:
        raise SystemExit(f"Unknown experiment id: {', '.join(unknown)}")
    weights = load_weights()
    cuda = _cuda_available()
    status = load_status()
    status["experiments"] = status.get("experiments", {})
    blend = weights["reranker_analytical_weights"]

    for exp_id in selected:
        if exp_id == "P5":
            result = run_p5(blend)
        elif exp_id == "IV3":
            result = run_iv3()
            coefficients = result.get("coefficients")
            if coefficients:
                weights["version"] = "v002"
                weights["inactive"] = dict(weights.get("inactive") or {})
                weights["inactive"]["iv3_gate"] = {
                    "adopted": False,
                    "scope": "synthetic",
                    "coefficients": coefficients,
                    "top_score_auc": result.get("top_score_auc"),
                    "gate_auc": result.get("gate_auc"),
                    "reason": "Fitted on synthetic candidates. The active blend was not changed.",
                }
                weights["active_blend_changed"] = False
                weights["notes"] = "v002 adds the inactive IV3 gate. Scoring still uses the v001 blend."
                save_weights(weights)
        elif exp_id == "IV4":
            result = blocked_record(exp_id, cuda)
            if result["status"] == "blocked":
                result["next_action"] = (
                    "IV4 waits on DN6. P5 already merges a DualLGD frame when "
                    "experiments/runs/DN6/candidates.json is present."
                )
        else:
            result = blocked_record(exp_id, cuda)

        result["exp_id"] = exp_id
        result["utc"] = utc_now()
        result["seed"] = 3407
        result["weight_version"] = weights["version"]
        manifest = write_manifest(exp_id, result)
        result["manifest"] = str(manifest.relative_to(REPO_ROOT))
        append_log(result)
        status["experiments"][exp_id] = {
            "status": result["status"],
            "next_action": result.get("next_action", ""),
            "blocker": result.get("blocker", ""),
            "manifest": result["manifest"],
            "weight_version": weights["version"],
        }
        print(f"{exp_id}: {result['status']}")

    status["updated_utc"] = utc_now()
    status["weight_version"] = weights["version"]
    status["python"] = sys.executable
    status["cuda"] = cuda
    status["rdkit"] = _rdkit_available()
    status["queue"] = list(QUEUE)
    try:
        dataset_id = publish(weights)
        weights["evidently_dataset_id"] = dataset_id
        save_weights(weights)
        status["evidently_weights_dataset_id"] = dataset_id
        append_log({"event": "weights_published", "dataset_id": dataset_id, "version": weights["version"]})
        print(f"weights: {dataset_id}")
    except Exception as exc:
        status["evidently_weights_dataset_id"] = status.get("evidently_weights_dataset_id", "upload_failed")
        append_log({"event": "weights_publish_failed", "error": str(exc)})
        print(f"weights upload failed: {exc}")
    save_status(status)
    render_tracker(status)


if __name__ == "__main__":
    main(sys.argv[1:])
