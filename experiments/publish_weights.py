"""Publish the active blend, and any inactive fits, to the CASMI Evidently project."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List
import json
import requests
import pandas as pd
from evidently import Dataset
from evidently.core.datasets import DataDefinition
from evidently.ui.workspace import RemoteWorkspace

PROJECT_ID = "01a1d2dd-5ca0-7d05-a07a-135ba489166d"
API_KEY_PATH = Path("/home/sweeden/.config/evidently/api_key")


def weight_rows(payload: Dict[str, object]) -> pd.DataFrame:
    version = str(payload["version"])
    rows: List[dict] = []
    for name, value in payload.get("reranker_analytical_weights", {}).items():
        rows.append(
            {
                "name": name,
                "group": "reranker_analytical_weights",
                "value": float(value),
                "active": "yes",
                "version": version,
                "note": "Active blend. Unchanged from the local Evidently snapshot.",
            }
        )
    for name, value in payload.get("fp_model_weights_and_biases", {}).items():
        if isinstance(value, (int, float)):
            rows.append(
                {
                    "name": name,
                    "group": "fp_model_weights_and_biases",
                    "value": float(value),
                    "active": "yes",
                    "version": version,
                    "note": str(payload["fp_model_weights_and_biases"].get("model_type", "")),
                }
            )
    inactive = payload.get("inactive", {})
    if isinstance(inactive, dict):
        gate = inactive.get("iv3_gate", {})
        if isinstance(gate, dict):
            for name, value in gate.get("coefficients", {}).items():
                rows.append(
                    {
                        "name": name,
                        "group": "iv3_gate",
                        "value": float(value),
                        "active": "no",
                        "version": version,
                        "note": "Inactive. Fitted on synthetic candidates and not used for scoring.",
                    }
                )
    return pd.DataFrame(rows)


def publish(payload: Dict[str, object]) -> str:
    frame = weight_rows(payload)
    definition = DataDefinition(
        numerical_columns=["value"],
        categorical_columns=["name", "group", "active", "version"],
        text_columns=["note"],
    )
    key = API_KEY_PATH.read_text(encoding="utf-8").strip()
    login = requests.get(
        "http://192.168.0.15:8000/api/users/login",
        headers={"X-Evidently-Token": key},
        timeout=30,
    )
    login.raise_for_status()
    token = login.text.strip()
    workspace = RemoteWorkspace("http://192.168.0.15:8000")
    original = workspace._prepare_request

    def with_bearer(*args, **kwargs):
        prepared = original(*args, **kwargs)
        prepared.headers["Authorization"] = f"Bearer {token}"
        return prepared

    workspace._prepare_request = with_bearer
    version = str(payload["version"])
    description = (
        f"CASMI blend {version}. Active rows are the reranker coefficients scoring uses. "
        "Inactive rows are fits that were not adopted. "
        "The active blend is the one in outputs/evidently/evidently_weights_biases_and_metrics.json."
    )
    dataset_id = workspace.add_dataset(
        PROJECT_ID,
        Dataset.from_pandas(frame, data_definition=definition),
        name="casmi-weights",
        description=description,
    )
    return str(dataset_id)
