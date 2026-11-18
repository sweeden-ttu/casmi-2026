"""Keep the RL77 weights in the CASMI Evidently project (dataset name `casmi-rl77-weights`).

Every version is a separate Evidently dataset with one row per value:
  name, kind (weight | metric | candidate), group, value, active, version, source, cell, note, updated_utc

Commands (run with the workspace venv from the repository root):
  .venv/bin/python experiments/runs/RL77/tools/evidently_weights.py publish            # push weights/current.json
  .venv/bin/python experiments/runs/RL77/tools/evidently_weights.py pull               # newest version -> weights/current.json
  .venv/bin/python experiments/runs/RL77/tools/evidently_weights.py adopt <output_dir> # fold a notebook run into a new version
  .venv/bin/python experiments/runs/RL77/tools/evidently_weights.py list
  .venv/bin/python experiments/runs/RL77/tools/evidently_weights.py record-metric leaderboard.nb5.public_mrr25 0.41 --note "v3"
  .venv/bin/python experiments/runs/RL77/tools/evidently_weights.py set nb5.rerank.alpha=0.65 nb5.rerank.generated_insert_rank=1 --cell nb5/cell_05

`adopt` reads fitted_weights.json (one fit, or a "fits" list) and metrics.json from a downloaded notebook output. Fitted
values become active only when their cross-fitted validation score beats the prior on the same
molecules; otherwise they are stored as inactive `candidate` rows. Metrics are always recorded.
"""

from __future__ import annotations

import argparse
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pandas as pd
import requests

RL77 = Path(__file__).resolve().parents[1]
WEIGHTS_DIR = RL77 / "weights"
CURRENT = WEIGHTS_DIR / "current.json"
EVIDENTLY_URL = "http://192.168.0.15:8000"
PROJECT_ID = "01a1d2dd-5ca0-7d05-a07a-135ba489166d"  # same CASMI project as experiments/publish_weights.py
DATASET_NAME = "casmi-rl77-weights"
API_KEY_PATH = Path.home() / ".config" / "evidently" / "api_key"


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def workspace():
    from evidently.ui.workspace import RemoteWorkspace

    key = API_KEY_PATH.read_text(encoding="utf-8").strip()
    login = requests.get(f"{EVIDENTLY_URL}/api/users/login", headers={"X-Evidently-Token": key}, timeout=30)
    login.raise_for_status()
    token = login.text.strip()
    ws = RemoteWorkspace(EVIDENTLY_URL)
    original = ws._prepare_request

    def with_bearer(*args, **kwargs):
        prepared = original(*args, **kwargs)
        prepared.headers["Authorization"] = f"Bearer {token}"
        return prepared

    ws._prepare_request = with_bearer
    return ws


def group_of(name: str) -> str:
    return ".".join(name.split(".")[:2])


def payload_to_frame(payload: Dict[str, Any]) -> pd.DataFrame:
    version, now = payload["version"], utc_now()
    rows: List[Dict[str, Any]] = []
    for name, value in payload["weights"].items():
        meta = payload.get("meta", {}).get(name, {})
        rows.append({"name": name, "kind": "weight", "group": group_of(name), "value": float(value),
                     "active": "yes", "version": version, "source": meta.get("source", ""),
                     "cell": meta.get("cell", ""), "note": meta.get("note", ""), "updated_utc": meta.get("updated_utc", now)})
    for name, cand in payload.get("candidates", {}).items():
        rows.append({"name": name, "kind": "candidate", "group": group_of(name), "value": float(cand["value"]),
                     "active": "no", "version": version, "source": cand.get("source", ""), "cell": "",
                     "note": cand.get("note", ""), "updated_utc": cand.get("updated_utc", now)})
    for name, value in payload.get("metrics", {}).items():
        rows.append({"name": name, "kind": "metric", "group": group_of(name), "value": float(value),
                     "active": "n/a", "version": version, "source": "kaggle run", "cell": "",
                     "note": payload.get("metric_notes", {}).get(name, ""), "updated_utc": now})
    return pd.DataFrame(rows)


def frame_to_payload(frame: pd.DataFrame, dataset_id: str, created: str) -> Dict[str, Any]:
    version = str(frame["version"].iloc[0])
    payload: Dict[str, Any] = {
        "version": version,
        "created_utc": created,
        "evidently": {"project_id": PROJECT_ID, "dataset_name": DATASET_NAME, "dataset_id": dataset_id},
        "weights": {}, "meta": {}, "candidates": {}, "metrics": {}, "metric_notes": {},
    }
    for row in frame.itertuples():
        if row.kind == "weight":
            payload["weights"][row.name] = float(row.value)
            payload["meta"][row.name] = {"source": row.source, "cell": row.cell, "note": row.note,
                                         "active": True, "updated_utc": row.updated_utc}
        elif row.kind == "candidate":
            payload["candidates"][row.name] = {"value": float(row.value), "source": row.source,
                                               "note": row.note, "updated_utc": row.updated_utc}
        elif row.kind == "metric":
            payload["metrics"][row.name] = float(row.value)
            if row.note:
                payload["metric_notes"][row.name] = row.note
    return payload


def publish(payload: Dict[str, Any]) -> str:
    from evidently import Dataset
    from evidently.core.datasets import DataDefinition

    frame = payload_to_frame(payload)
    definition = DataDefinition(
        numerical_columns=["value"],
        categorical_columns=["name", "kind", "group", "active", "version", "source", "cell"],
        text_columns=["note", "updated_utc"],
    )
    description = (
        f"RL77 notebook weights {payload['version']}: blend weights, reward coefficients and tolerances "
        "read by experiments/runs/RL77 notebooks through W(\"...\"). kind=weight rows are active; "
        "kind=candidate rows are fits that did not beat the prior; kind=metric rows are validation results."
    )
    dataset_id = workspace().add_dataset(
        PROJECT_ID, Dataset.from_pandas(frame, data_definition=definition), name=DATASET_NAME, description=description
    )
    return str(dataset_id)


def latest() -> Tuple[str, str]:
    infos = [d for d in workspace().list_datasets(PROJECT_ID).datasets if d.name == DATASET_NAME]
    if not infos:
        raise SystemExit(f"No `{DATASET_NAME}` dataset in project {PROJECT_ID}; run `publish` first.")
    newest = max(infos, key=lambda d: d.created_at)
    return str(newest.id), newest.created_at.isoformat()


def pull() -> Dict[str, Any]:
    import uuid

    dataset_id, created = latest()
    frame = workspace().load_dataset(uuid.UUID(dataset_id)).as_dataframe()
    payload = frame_to_payload(frame, dataset_id, created)
    write_local(payload)
    return payload


def write_local(payload: Dict[str, Any]) -> None:
    text = json.dumps(payload, indent=2) + "\n"
    CURRENT.write_text(text, encoding="utf-8")
    (WEIGHTS_DIR / f"{payload['version'].replace('-', '_')}.json").write_text(text, encoding="utf-8")


def bump(version: str) -> str:
    m = re.match(r"^(.*?)(\d+)$", version)
    if not m:
        return version + "-v002"
    return f"{m.group(1)}{int(m.group(2)) + 1:0{len(m.group(2))}d}"


def flatten_metrics(prefix: str, obj: Any, out: Dict[str, float]) -> None:
    if isinstance(obj, dict):
        for k, v in obj.items():
            flatten_metrics(f"{prefix}.{k}", v, out)
    elif isinstance(obj, (int, float)) and not isinstance(obj, bool):
        out[prefix] = float(obj)


def adopt(output_dir: Path, min_gain: float = 0.0) -> Dict[str, Any]:
    payload = json.loads(CURRENT.read_text(encoding="utf-8"))
    payload.setdefault("candidates", {})
    payload.setdefault("metrics", {})
    payload.setdefault("metric_notes", {})
    manifest = json.loads((output_dir / "run_manifest.json").read_text()) if (output_dir / "run_manifest.json").exists() else {}
    notebook = manifest.get("notebook", output_dir.name)
    run_note = f"{notebook} run finished {manifest.get('finished_utc', '?')} on {manifest.get('device', '?')}"
    now = utc_now()

    fitted_path = output_dir / "fitted_weights.json"
    blob = json.loads(fitted_path.read_text()) if fitted_path.exists() else {}
    # One {"weights", "evidence"} block, or {"fits": [...]} when a notebook fits several groups on different metrics.
    for fitted in blob.get("fits", [blob] if "weights" in blob else []):
        ev = fitted.get("evidence", {})
        prior, fit = ev.get("prior"), ev.get("fitted_crossfit")
        improved = prior is not None and fit is not None and fit > prior + min_gain
        note = f"{ev.get('metric', 'score')} prior {prior} vs cross-fit {fit} on {ev.get('split', '?')}; {run_note}"
        for key, value in fitted["weights"].items():
            if key not in payload["weights"]:
                raise SystemExit(f"{fitted_path}: `{key}` is not an RL77 weight")
            if improved:
                payload["weights"][key] = float(value)
                payload["meta"][key] = {**payload["meta"].get(key, {}), "source": f"fitted:{notebook}",
                                        "note": note, "active": True, "updated_utc": now}
                payload["candidates"].pop(key, None)
            else:
                payload["candidates"][key] = {"value": float(value), "source": f"fitted:{notebook}",
                                              "note": "Not adopted: " + note, "updated_utc": now}
        print(f"{notebook}: {'adopted' if improved else 'kept prior for'} {sorted(fitted['weights'])} ({note})")

    metrics_path = output_dir / "metrics.json"
    if metrics_path.exists():
        flat: Dict[str, float] = {}
        flatten_metrics(notebook, json.loads(metrics_path.read_text()), flat)
        for key, value in flat.items():
            payload["metrics"][key] = value
            payload["metric_notes"][key] = run_note
        print(f"{notebook}: recorded {len(flat)} metrics")

    payload["version"] = bump(payload["version"])
    payload["created_utc"] = now
    payload["evidently"]["dataset_id"] = publish(payload)
    write_local(payload)
    print(f"Published {payload['version']} as Evidently dataset {payload['evidently']['dataset_id']}")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("publish", help="publish weights/current.json as a new Evidently dataset version")
    sub.add_parser("pull", help="write the newest Evidently version to weights/current.json")
    sub.add_parser("list", help="list published versions")
    p_adopt = sub.add_parser("adopt", help="fold a downloaded notebook output folder into a new version")
    p_adopt.add_argument("output_dir", type=Path)
    p_adopt.add_argument("--min-gain", type=float, default=0.0)
    p_metric = sub.add_parser("record-metric", help="record one metric (e.g. a leaderboard score) as a new version")
    p_metric.add_argument("name")
    p_metric.add_argument("value", type=float)
    p_metric.add_argument("--note", default="")
    p_set = sub.add_parser("set", help="add or change one active weight by hand (new version); fitted values go through adopt")
    p_set.add_argument("assignments", nargs="+", metavar="NAME=VALUE")
    p_set.add_argument("--cell", default="")
    p_set.add_argument("--note", default="")
    p_set.add_argument("--source", default="manual")
    args = parser.parse_args()

    if args.cmd == "publish":
        payload = json.loads(CURRENT.read_text(encoding="utf-8"))
        payload.setdefault("evidently", {})["dataset_id"] = publish(payload)
        write_local(payload)
        print(f"Published {payload['version']} as Evidently dataset {payload['evidently']['dataset_id']}")
    elif args.cmd == "pull":
        payload = pull()
        print(f"Pulled {payload['version']} ({len(payload['weights'])} weights, {len(payload['metrics'])} metrics) "
              f"from Evidently dataset {payload['evidently']['dataset_id']}")
    elif args.cmd == "list":
        for d in sorted((d for d in workspace().list_datasets(PROJECT_ID).datasets if d.name == DATASET_NAME),
                        key=lambda d: d.created_at):
            print(d.created_at.isoformat(), d.id, d.row_count, "rows")
    elif args.cmd == "adopt":
        adopt(args.output_dir, args.min_gain)
    elif args.cmd == "record-metric":
        payload = json.loads(CURRENT.read_text(encoding="utf-8"))
        payload.setdefault("metrics", {})[args.name] = args.value
        payload.setdefault("metric_notes", {})[args.name] = args.note or f"recorded {utc_now()}"
        payload["version"] = bump(payload["version"])
        payload["created_utc"] = utc_now()
        payload["evidently"]["dataset_id"] = publish(payload)
        write_local(payload)
        print(f"Recorded {args.name}={args.value} in {payload['version']} ({payload['evidently']['dataset_id']})")
    elif args.cmd == "set":
        payload = json.loads(CURRENT.read_text(encoding="utf-8"))
        changes = []
        for item in args.assignments:
            name, _, raw = item.partition("=")
            if not name or not raw:
                raise SystemExit(f"`{item}`: expected NAME=VALUE")
            changes.append(f"{name}: {payload['weights'].get(name)} -> {float(raw)}")
            payload["weights"][name] = float(raw)
            payload.setdefault("meta", {})[name] = {
                **payload["meta"].get(name, {}), "source": args.source, "active": True, "updated_utc": utc_now(),
                **({"cell": args.cell} if args.cell else {}), **({"note": args.note} if args.note else {}),
            }
            payload.get("candidates", {}).pop(name, None)
        payload["version"] = bump(payload["version"])
        payload["created_utc"] = utc_now()
        payload["evidently"]["dataset_id"] = publish(payload)
        write_local(payload)
        print(f"{payload['version']} ({payload['evidently']['dataset_id']}): " + "; ".join(changes))


if __name__ == "__main__":
    main()
