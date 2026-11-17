"""Append-only run log and the status file other agents resume from."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import json
from typing import Any, Dict, List, Mapping

from experiments import QUEUE

REPO_ROOT = Path(__file__).resolve().parents[1]
EXP_ROOT = REPO_ROOT / "experiments"
STATUS_PATH = EXP_ROOT / "status.json"
LOG_PATH = EXP_ROOT / "log.jsonl"
TRACKER_PATH = EXP_ROOT / "TRACKER.md"
RUNS_ROOT = EXP_ROOT / "runs"


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def append_log(record: Mapping[str, Any]) -> None:
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    line = dict(record)
    line.setdefault("utc", utc_now())
    with LOG_PATH.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(line, sort_keys=True) + "\n")


def write_manifest(exp_id: str, payload: Mapping[str, Any]) -> Path:
    run_dir = RUNS_ROOT / exp_id
    run_dir.mkdir(parents=True, exist_ok=True)
    path = run_dir / "manifest.json"
    path.write_text(json.dumps(dict(payload), indent=2) + "\n", encoding="utf-8")
    return path


def load_status() -> Dict[str, Any]:
    if not STATUS_PATH.is_file():
        return {"experiments": {}}
    return json.loads(STATUS_PATH.read_text(encoding="utf-8"))


def save_status(status: Mapping[str, Any]) -> None:
    STATUS_PATH.write_text(json.dumps(dict(status), indent=2) + "\n", encoding="utf-8")


def render_tracker(status: Mapping[str, Any]) -> None:
    lines: List[str] = [
        "# CASMI experiment tracker",
        "",
        f"Updated {status.get('updated_utc', '')}.",
        "",
        "Resume here. `status.json` is the machine-readable copy of this page, and `log.jsonl` is the append-only history. Rerun with the kagg interpreter from the repository root:",
        "",
        "```bash",
        "/home/sweeden/.conda/envs/kagg/bin/python -m experiments",
        "```",
        "",
        "A finished checkpoint run is `python -m experiments <EXP_ID>` after the artifact in the next-action column is in place. Seed for every run is 3407. Active blend is `experiments/weights/current.json`. Do not retune that blend on `data/synthetic`; those SMILES are not molecules and their inchikey14 values are hashes.",
        "",
        f"Weights version: `{status.get('weight_version', 'v001')}`. Evidently dataset: `{status.get('evidently_weights_dataset_id', 'not uploaded')}`.",
        "",
        "## Queue",
        "",
        "| Order | Experiment | Status | Next action |",
        "| --- | --- | --- | --- |",
    ]
    experiments: Mapping[str, Any] = status.get("experiments", {})
    for index, exp_id in enumerate(QUEUE, start=1):
        row = experiments.get(exp_id, {})
        state = row.get("status", "not_started")
        action = row.get("next_action", "Run python -m experiments " + exp_id)
        lines.append(f"| {index} | {exp_id} | {state} | {action} |")
    lines.extend(
        [
            "",
            "## Environment",
            "",
            f"- Python: `{status.get('python', '')}`",
            f"- CUDA: `{status.get('cuda', False)}`",
            f"- RDKit: `{status.get('rdkit', False)}`",
            "- Local spectra: `data/synthetic` only. `data/raw/train.parquet` is not on disk.",
            "",
        ]
    )
    TRACKER_PATH.write_text("\n".join(lines), encoding="utf-8")
