#!/usr/bin/env python3
"""Evidently 0.7.x Python Bridge for the VS Code Evidently Remote Workspace Extension.

Supports:
- Authenticating with a local/remote Evidently workspace (`RemoteWorkspace`)
- Loading datasets by UUID (with CSV download fallback for tracing/virtual datasets)
- Joining two or more datasets with pandas (`inner`, `left`, `right`, `outer`, `cross`, `union`),
  including automatic key normalization across numeric/string column dtypes
- Generating Evidently Reports (`DataSummaryPreset`, `DataDriftPreset`, `RegressionPreset`,
  `ClassificationPreset`, `ValueStats`, `DatasetCorrelations`)
- Publishing the Snapshot Report and (optionally) the joined Dataset back to the remote
  Evidently project linked via `SnapshotLink`
"""

from __future__ import annotations

import io
import json
import math
import os
import sys
import tempfile
import traceback
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import requests


def resolve_api_key(explicit_key: Optional[str] = None) -> str:
    if explicit_key and explicit_key.strip():
        return explicit_key.strip()
    env_key = os.environ.get("EVIDENTLY_TOKEN", "").strip()
    if env_key:
        return env_key
    for candidate in [
        Path.home() / ".config" / "evidently" / "api_key",
        Path.home() / ".config" / "evidently" / "token",
    ]:
        try:
            if candidate.exists():
                val = candidate.read_text(encoding="utf-8").strip()
                if val:
                    return val
        except Exception:
            pass
    return ""


def connect_remote_workspace(server_url: str, api_key: str) -> Tuple[Any, Optional[str], Dict[str, str]]:
    from evidently.ui.workspace import RemoteWorkspace

    base_url = server_url.rstrip("/")
    token: Optional[str] = None
    headers: Dict[str, str] = {}

    if api_key:
        try:
            login = requests.get(
                f"{base_url}/api/users/login",
                headers={"X-Evidently-Token": api_key},
                timeout=20,
            )
            if login.status_code == 200 and login.text.strip():
                token = login.text.strip().strip('"')
                headers["Authorization"] = f"Bearer {token}"
            else:
                headers["X-Evidently-Token"] = api_key
        except Exception:
            headers["X-Evidently-Token"] = api_key

    ws = RemoteWorkspace(base_url, verify=False)
    if headers:
        original_prepare = ws._prepare_request

        def with_auth(*args: Any, **kwargs: Any) -> Any:
            prepared = original_prepare(*args, **kwargs)
            for hk, hv in headers.items():
                prepared.headers[hk] = hv
            return prepared

        ws._prepare_request = with_auth

    return ws, token, headers


def load_dataframe(
    ws: Any,
    server_url: str,
    headers: Dict[str, str],
    dataset_id: Optional[str] = None,
    csv_path: Optional[str] = None,
    csv_text: Optional[str] = None,
    max_rows: Optional[int] = None,
) -> pd.DataFrame:
    df: Optional[pd.DataFrame] = None
    if csv_text is not None:
        df = pd.read_csv(io.StringIO(csv_text))
    elif csv_path is not None:
        df = pd.read_csv(csv_path)
    elif dataset_id:
        try:
            ds_obj = ws.load_dataset(uuid.UUID(str(dataset_id)))
            df = ds_obj.as_dataframe().copy()
        except Exception:
            base_url = server_url.rstrip("/")
            resp = requests.get(
                f"{base_url}/api/datasets/{dataset_id}/download",
                params={"format": "csv"},
                headers=headers,
                timeout=60,
            )
            resp.raise_for_status()
            text = resp.text
            if not text.strip():
                df = pd.DataFrame()
            else:
                df = pd.read_csv(io.StringIO(text))
    else:
        raise ValueError("One of dataset_id, csv_path, or csv_text must be provided.")

    if df is None:
        df = pd.DataFrame()

    if max_rows and max_rows > 0 and len(df) > max_rows:
        df = df.head(max_rows).copy()

    return df


def canonicalize_join_key_series(series: pd.Series, match_mode: str = "normalized") -> pd.Series:
    """Create a string join key series that safely aligns numeric/string representations."""

    def norm_val(v: Any) -> Any:
        if v is None or (isinstance(v, float) and math.isnan(v)) or pd.isna(v):
            return np.nan
        if isinstance(v, (int, np.integer)):
            return str(int(v))
        if isinstance(v, (float, np.floating)):
            if not math.isfinite(float(v)):
                return np.nan
            fval = float(v)
            if fval.is_integer():
                return str(int(fval))
            return f"{fval:g}"
        s = str(v)
        if match_mode == "exact":
            return s if s != "" else np.nan
        s_trim = s.strip()
        if not s_trim or s_trim.lower() in ("null", "none", "nan"):
            return np.nan
        try:
            fval = float(s_trim)
            if math.isfinite(fval):
                if fval.is_integer():
                    return str(int(fval))
                return f"{fval:g}"
        except ValueError:
            pass
        return s_trim.lower()

    return series.map(norm_val)


def _make_unique_col_name(desired: str, existing: set[str]) -> str:
    if desired not in existing:
        existing.add(desired)
        return desired
    idx = 2
    while f"{desired}_{idx}" in existing:
        idx += 1
    final_name = f"{desired}_{idx}"
    existing.add(final_name)
    return final_name


def _prepare_collision_safe_columns(
    left_df: pd.DataFrame,
    right_df: pd.DataFrame,
    left_on: List[str],
    right_on: List[str],
    suffixes: Tuple[str, str],
    is_cross: bool = False,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, str]]:
    """Pre-rename colliding non-shared-key columns so pandas.merge never raises MergeError
    when columns already end with `_left` or `_right`."""
    l_sfx = suffixes[0] if len(suffixes) > 0 else "_left"
    r_sfx = suffixes[1] if len(suffixes) > 1 else "_right"

    shared_key_set = set()
    if not is_cross:
        for lk, rk in zip(left_on, right_on):
            if lk == rk:
                shared_key_set.add(lk)

    right_non_shared = [c for c in right_df.columns if c not in shared_key_set]
    right_collision_set = set(right_non_shared)
    left_collision_set = set(c for c in left_df.columns if c not in shared_key_set)

    existing_output_cols: set[str] = set()
    left_rename: Dict[str, str] = {}
    right_rename: Dict[str, str] = {}
    right_rename_for_shared: Dict[str, str] = {}

    for lc in left_df.columns:
        if lc in shared_key_set:
            out_name = _make_unique_col_name(str(lc), existing_output_cols)
            if out_name != lc:
                left_rename[lc] = out_name
        elif lc in right_collision_set:
            desired = f"{lc}{l_sfx}" if l_sfx else str(lc)
            out_name = _make_unique_col_name(desired, existing_output_cols)
            left_rename[lc] = out_name
        else:
            out_name = _make_unique_col_name(str(lc), existing_output_cols)
            if out_name != lc:
                left_rename[lc] = out_name

    for rc in right_df.columns:
        if rc in shared_key_set:
            temp_r_col = _make_unique_col_name(f"__ev_shared_right_{rc}__", existing_output_cols)
            right_rename[rc] = temp_r_col
            right_rename_for_shared[rc] = temp_r_col
        elif rc in left_collision_set or rc in shared_key_set:
            desired = f"{rc}{r_sfx}" if r_sfx else str(rc)
            out_name = _make_unique_col_name(desired, existing_output_cols)
            right_rename[rc] = out_name
        else:
            out_name = _make_unique_col_name(str(rc), existing_output_cols)
            if out_name != rc:
                right_rename[rc] = out_name

    if left_rename:
        left_df = left_df.rename(columns=left_rename)
    if right_rename:
        right_df = right_df.rename(columns=right_rename)

    return left_df, right_df, right_rename_for_shared


def perform_pandas_join(
    left_df: pd.DataFrame,
    right_df: pd.DataFrame,
    join_type: str,
    left_on: List[str],
    right_on: List[str],
    match_mode: str = "normalized",
    suffixes: Tuple[str, str] = ("_left", "_right"),
    selected_left_columns: Optional[List[str]] = None,
    selected_right_columns: Optional[List[str]] = None,
    drop_null_keys: bool = True,
    max_output_rows: int = 100000,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    join_type = (join_type or "inner").lower()

    if selected_left_columns:
        keep_l = [c for c in left_df.columns if c in selected_left_columns or c in (left_on or [])]
        left_df = left_df[keep_l].copy()
    else:
        left_df = left_df.copy()

    if selected_right_columns:
        keep_r = [c for c in right_df.columns if c in selected_right_columns or c in (right_on or [])]
        right_df = right_df[keep_r].copy()
    else:
        right_df = right_df.copy()

    left_count = int(len(left_df))
    right_count = int(len(right_df))

    if join_type == "union":
        joined = pd.concat([left_df, right_df], ignore_index=True, sort=False)
        truncated = False
        if max_output_rows and len(joined) > max_output_rows:
            joined = joined.head(max_output_rows).copy()
            truncated = True
        diagnostics = {
            "joinType": "union",
            "leftOn": [],
            "rightOn": [],
            "leftRowCount": left_count,
            "rightRowCount": right_count,
            "joinedRowCount": int(len(joined)),
            "truncated": truncated,
            "matchedLeftRows": left_count,
            "unmatchedLeftRows": 0,
            "matchedRightRows": right_count,
            "unmatchedRightRows": 0,
            "distinctLeftKeys": 0,
            "distinctRightKeys": 0,
            "overlappingKeys": 0,
            "keyOverlapPct": 100.0,
        }
        return joined, diagnostics

    if join_type == "cross":
        left_df, right_df, _ = _prepare_collision_safe_columns(
            left_df, right_df, [], [], suffixes, is_cross=True
        )
        joined = left_df.merge(right_df, how="cross")
        truncated = False
        if max_output_rows and len(joined) > max_output_rows:
            joined = joined.head(max_output_rows).copy()
            truncated = True
        diagnostics = {
            "joinType": "cross",
            "leftOn": [],
            "rightOn": [],
            "leftRowCount": left_count,
            "rightRowCount": right_count,
            "joinedRowCount": int(len(joined)),
            "truncated": truncated,
            "matchedLeftRows": left_count if right_count > 0 else 0,
            "unmatchedLeftRows": 0 if right_count > 0 else left_count,
            "matchedRightRows": right_count if left_count > 0 else 0,
            "unmatchedRightRows": 0 if left_count > 0 else right_count,
            "distinctLeftKeys": 0,
            "distinctRightKeys": 0,
            "overlappingKeys": 0,
            "keyOverlapPct": 100.0,
        }
        return joined, diagnostics

    if not left_on or not right_on or len(left_on) != len(right_on):
        raise ValueError("left_on and right_on must be non-empty lists of equal length.")

    for col in left_on:
        if col not in left_df.columns:
            raise ValueError(f"Left join key '{col}' not found in left dataset columns: {list(left_df.columns)}")
    for col in right_on:
        if col not in right_df.columns:
            raise ValueError(f"Right join key '{col}' not found in right dataset columns: {list(right_df.columns)}")

    # Extract canonical join key series BEFORE renaming colliding columns
    left_key_cols = [canonicalize_join_key_series(left_df[lk], match_mode) for lk in left_on]
    right_key_cols = [canonicalize_join_key_series(right_df[rk], match_mode) for rk in right_on]

    left_df, right_df, right_rename_for_shared = _prepare_collision_safe_columns(
        left_df, right_df, left_on, right_on, suffixes, is_cross=False
    )

    # Build temporary canonical join key columns so dtype differences (e.g. int64 vs str vs float) never break merge
    temp_keys: List[str] = []
    for idx, (l_ser, r_ser) in enumerate(zip(left_key_cols, right_key_cols)):
        tk = f"__ev_join_key_{idx}__"
        temp_keys.append(tk)
        left_df[tk] = l_ser
        right_df[tk] = r_ser

    if drop_null_keys:
        # Rows with null keys shouldn't match each other on NaN == NaN
        for idx, tk in enumerate(temp_keys):
            left_null_mask = left_df[tk].isna()
            if left_null_mask.any():
                left_df.loc[left_null_mask, tk] = [
                    f"__LEFT_NULL_{idx}_{i}__" for i in range(int(left_null_mask.sum()))
                ]
            right_null_mask = right_df[tk].isna()
            if right_null_mask.any():
                right_df.loc[right_null_mask, tk] = [
                    f"__RIGHT_NULL_{idx}_{i}__" for i in range(int(right_null_mask.sum()))
                ]

    # Compute key overlap diagnostics before merge
    left_valid_mask = ~left_df[temp_keys[0]].astype(str).str.startswith("__LEFT_NULL_")
    right_valid_mask = ~right_df[temp_keys[0]].astype(str).str.startswith("__RIGHT_NULL_")
    for tk in temp_keys[1:]:
        left_valid_mask &= ~left_df[tk].astype(str).str.startswith("__LEFT_NULL_")
        right_valid_mask &= ~right_df[tk].astype(str).str.startswith("__RIGHT_NULL_")

    left_tuples = set(
        zip(*(left_df.loc[left_valid_mask, tk] for tk in temp_keys))
    ) if left_valid_mask.any() else set()
    right_tuples = set(
        zip(*(right_df.loc[right_valid_mask, tk] for tk in temp_keys))
    ) if right_valid_mask.any() else set()
    overlap_tuples = left_tuples & right_tuples

    if left_count > 0:
        left_key_series = list(zip(*(left_df[tk] for tk in temp_keys)))
        matched_left_rows = sum(1 for t in left_key_series if t in overlap_tuples)
    else:
        matched_left_rows = 0

    if right_count > 0:
        right_key_series = list(zip(*(right_df[tk] for tk in temp_keys)))
        matched_right_rows = sum(1 for t in right_key_series if t in overlap_tuples)
    else:
        matched_right_rows = 0

    how_map = {
        "inner": "inner",
        "left": "left",
        "right": "right",
        "outer": "outer",
    }
    pandas_how = how_map.get(join_type, "inner")

    joined = pd.merge(
        left_df,
        right_df,
        how=pandas_how,
        on=temp_keys,
    )

    # Coalesce shared key columns (e.g. for right/outer joins where left side was NaN)
    for sk, temp_r_col in right_rename_for_shared.items():
        if sk in joined.columns and temp_r_col in joined.columns:
            joined[sk] = joined[sk].where(joined[sk].notna(), joined[temp_r_col])
            joined = joined.drop(columns=[temp_r_col])

    # Drop temporary canonical key columns
    joined = joined.drop(columns=[tk for tk in temp_keys if tk in joined.columns])

    truncated = False
    if max_output_rows and max_output_rows > 0 and len(joined) > max_output_rows:
        joined = joined.head(max_output_rows).copy()
        truncated = True

    denom = max(1, min(len(left_tuples), len(right_tuples)))
    overlap_pct = (
        round((len(overlap_tuples) / denom) * 100.0, 2)
        if left_tuples and right_tuples
        else 0.0
    )

    diagnostics = {
        "joinType": join_type,
        "leftOn": left_on,
        "rightOn": right_on,
        "leftRowCount": left_count,
        "rightRowCount": right_count,
        "joinedRowCount": int(len(joined)),
        "truncated": truncated,
        "matchedLeftRows": int(matched_left_rows),
        "unmatchedLeftRows": int(max(0, left_count - matched_left_rows)),
        "matchedRightRows": int(matched_right_rows),
        "unmatchedRightRows": int(max(0, right_count - matched_right_rows)),
        "distinctLeftKeys": int(len(left_tuples)),
        "distinctRightKeys": int(len(right_tuples)),
        "overlappingKeys": int(len(overlap_tuples)),
        "keyOverlapPct": overlap_pct,
    }
    return joined, diagnostics


def profile_dataframe(df: pd.DataFrame) -> List[Dict[str, Any]]:
    profiles: List[Dict[str, Any]] = []
    total_rows = len(df)
    for col in df.columns:
        s = df[col]
        non_null = int(s.notna().sum())
        missing = int(total_rows - non_null)
        missing_pct = round((missing / total_rows) * 100.0, 2) if total_rows > 0 else 0.0
        unique_count = int(s.nunique(dropna=True))

        is_num = pd.api.types.is_numeric_dtype(s) and not pd.api.types.is_bool_dtype(s)
        is_dt = pd.api.types.is_datetime64_any_dtype(s)
        is_bool = pd.api.types.is_bool_dtype(s)

        inferred_type = "numerical" if is_num else "datetime" if is_dt else "boolean" if is_bool else "string"

        min_v = max_v = mean_v = std_v = None
        if is_num and non_null > 0:
            clean_num = pd.to_numeric(s, errors="coerce").dropna()
            if len(clean_num) > 0:
                min_v = float(clean_num.min())
                max_v = float(clean_num.max())
                mean_v = round(float(clean_num.mean()), 6)
                std_v = round(float(clean_num.std(ddof=1)), 6) if len(clean_num) > 1 else 0.0

        samples = [str(v)[:48] for v in s.dropna().unique()[:4]]
        profiles.append(
            {
                "name": str(col),
                "source": "joined",
                "inferredType": inferred_type,
                "nonNullCount": non_null,
                "missingCount": missing,
                "missingPct": missing_pct,
                "uniqueCount": unique_count,
                "min": min_v if (min_v is not None and math.isfinite(min_v)) else None,
                "max": max_v if (max_v is not None and math.isfinite(max_v)) else None,
                "mean": mean_v if (mean_v is not None and math.isfinite(mean_v)) else None,
                "std": std_v if (std_v is not None and math.isfinite(std_v)) else None,
                "sampleValues": samples,
            }
        )
    return profiles


def dataframe_preview_rows(df: pd.DataFrame, limit: int = 100) -> List[Dict[str, Any]]:
    sub = df.head(limit)
    records: List[Dict[str, Any]] = []
    for _, row in sub.iterrows():
        rec: Dict[str, Any] = {}
        for col in sub.columns:
            val = row[col]
            if val is None or (isinstance(val, float) and math.isnan(val)) or pd.isna(val):
                rec[str(col)] = None
            elif isinstance(val, (np.integer, int)):
                rec[str(col)] = int(val)
            elif isinstance(val, (np.floating, float)):
                f = float(val)
                rec[str(col)] = f if math.isfinite(f) else None
            elif isinstance(val, (np.bool_, bool)):
                rec[str(col)] = bool(val)
            else:
                rec[str(col)] = str(val)
        records.append(rec)
    return records


def _normalize_class_label(v: Any) -> Any:
    """Canonicalize class labels so that missing values stay np.nan (not 'nan')
    and int/float representations of the same integer class (1 vs 1.0) match."""
    if v is None or (isinstance(v, float) and math.isnan(v)) or pd.isna(v):
        return np.nan
    if isinstance(v, (bool, np.bool_)):
        return str(bool(v))
    if isinstance(v, (int, np.integer)):
        return str(int(v))
    if isinstance(v, (float, np.floating)):
        fval = float(v)
        if not math.isfinite(fval):
            return np.nan
        if fval.is_integer():
            return str(int(fval))
        return f"{fval:g}"
    s = str(v).strip()
    if not s or s.lower() in ("nan", "none", "null", "<na>"):
        return np.nan
    try:
        fval = float(s)
        if math.isfinite(fval) and fval.is_integer():
            return str(int(fval))
    except ValueError:
        pass
    return s


def build_evidently_dataset(
    df: pd.DataFrame,
    spec: Dict[str, Any],
) -> Any:
    from evidently import Dataset
    from evidently.core.datasets import (
        BinaryClassification,
        DataDefinition,
        MulticlassClassification,
        Regression,
    )

    work_df = df.copy()
    if len(work_df.columns) == 0:
        work_df["empty_placeholder"] = ["(no columns)"]

    presets = spec.get("presets") or ["DataSummaryPreset"]
    target_col = (spec.get("target_column") or "").strip()
    pred_col = (spec.get("prediction_column") or "").strip()

    regression_defs = None
    classification_defs = None

    if "RegressionPreset" in presets and target_col in work_df.columns and pred_col in work_df.columns:
        work_df[target_col] = pd.to_numeric(work_df[target_col], errors="coerce")
        work_df[pred_col] = pd.to_numeric(work_df[pred_col], errors="coerce")
        regression_defs = [Regression(target=target_col, prediction=pred_col)]

    if "ClassificationPreset" in presets and target_col in work_df.columns and pred_col in work_df.columns:
        work_df[target_col] = work_df[target_col].map(_normalize_class_label)
        work_df[pred_col] = work_df[pred_col].map(_normalize_class_label)
        # Drop rows where target or prediction is missing so Evidently does not treat NaN as a class label
        valid_mask = work_df[target_col].notna() & work_df[pred_col].notna()
        if valid_mask.any() and not valid_mask.all():
            work_df = work_df.loc[valid_mask].reset_index(drop=True)
        unique_labels = sorted(
            set(work_df[target_col].dropna().unique())
            | set(work_df[pred_col].dropna().unique())
        )
        if len(unique_labels) == 2:
            classification_defs = [
                BinaryClassification(
                    target=target_col,
                    prediction_labels=pred_col,
                    pos_label=unique_labels[-1],
                )
            ]
        elif len(unique_labels) >= 1:
            classification_defs = [
                MulticlassClassification(
                    target=target_col,
                    prediction_labels=pred_col,
                )
            ]

    if regression_defs or classification_defs:
        dd = DataDefinition(
            regression=regression_defs,
            classification=classification_defs,
        )
        return Dataset.from_pandas(work_df, data_definition=dd)

    return Dataset.from_pandas(work_df)


def create_report_and_publish(spec: Dict[str, Any]) -> Dict[str, Any]:
    from evidently import Report
    from evidently.metrics import ColumnCount, DatasetCorrelations, RowCount
    from evidently.presets import (
        ClassificationPreset,
        DataDriftPreset,
        DataSummaryPreset,
        RegressionPreset,
        ValueStats,
    )
    from evidently.sdk.models import SnapshotLink

    server_url = spec.get("server_url") or "http://192.168.0.15:8000"
    api_key = resolve_api_key(spec.get("api_key"))
    ws, _, headers = connect_remote_workspace(server_url, api_key)

    max_rows = int(spec.get("max_rows") or 100000)

    left_df = load_dataframe(
        ws,
        server_url,
        headers,
        dataset_id=spec.get("left_dataset_id"),
        csv_path=spec.get("left_csv_path"),
        csv_text=spec.get("left_csv_text"),
        max_rows=max_rows,
    )

    has_right = bool(
        spec.get("right_dataset_id")
        or spec.get("right_csv_path")
        or spec.get("right_csv_text")
    )

    if has_right:
        right_df = load_dataframe(
            ws,
            server_url,
            headers,
            dataset_id=spec.get("right_dataset_id"),
            csv_path=spec.get("right_csv_path"),
            csv_text=spec.get("right_csv_text"),
            max_rows=max_rows,
        )
        suffixes_raw = spec.get("suffixes") or ["_left", "_right"]
        suffixes = (str(suffixes_raw[0]), str(suffixes_raw[1])) if len(suffixes_raw) >= 2 else ("_left", "_right")
        joined_df, diagnostics = perform_pandas_join(
            left_df=left_df,
            right_df=right_df,
            join_type=spec.get("join_type") or "inner",
            left_on=spec.get("left_on") or [],
            right_on=spec.get("right_on") or [],
            match_mode=spec.get("match_mode") or "normalized",
            suffixes=suffixes,
            selected_left_columns=spec.get("selected_left_columns"),
            selected_right_columns=spec.get("selected_right_columns"),
            drop_null_keys=bool(spec.get("drop_null_keys", True)),
            max_output_rows=max_rows,
        )
    else:
        right_df = None
        joined_df = left_df.copy()
        diagnostics = {
            "joinType": "single",
            "leftOn": [],
            "rightOn": [],
            "leftRowCount": int(len(left_df)),
            "rightRowCount": 0,
            "joinedRowCount": int(len(joined_df)),
            "truncated": False,
            "matchedLeftRows": int(len(left_df)),
            "unmatchedLeftRows": 0,
            "matchedRightRows": 0,
            "unmatchedRightRows": 0,
            "distinctLeftKeys": 0,
            "distinctRightKeys": 0,
            "overlappingKeys": 0,
            "keyOverlapPct": 100.0,
        }

    # Optional additional chained joins
    for extra in spec.get("additional_joins") or []:
        extra_df = load_dataframe(
            ws,
            server_url,
            headers,
            dataset_id=extra.get("dataset_id"),
            csv_path=extra.get("csv_path"),
            csv_text=extra.get("csv_text"),
            max_rows=max_rows,
        )
        joined_df, _ = perform_pandas_join(
            left_df=joined_df,
            right_df=extra_df,
            join_type=extra.get("join_type") or "inner",
            left_on=extra.get("left_on") or [],
            right_on=extra.get("right_on") or [],
            match_mode=extra.get("match_mode") or "normalized",
            suffixes=tuple(extra.get("suffixes") or ["", "_extra"]),  # type: ignore[arg-type]
            max_output_rows=max_rows,
        )
        diagnostics["joinedRowCount"] = int(len(joined_df))

    mode = spec.get("mode") or "create_report"
    profiles = profile_dataframe(joined_df)
    preview_rows = dataframe_preview_rows(joined_df, limit=int(spec.get("preview_limit") or 100))

    if mode == "preview_join":
        return {
            "status": "ok",
            "mode": "preview_join",
            "columns": [str(c) for c in joined_df.columns],
            "diagnostics": diagnostics,
            "columnProfiles": profiles,
            "rows": preview_rows,
        }

    if len(joined_df) == 0:
        raise ValueError(
            "The joined table has 0 rows. Adjust your join keys or switch Join Type to 'left' or 'outer' to generate an Evidently report."
        )

    # Build Metric items for the Report
    presets = spec.get("presets") or ["DataSummaryPreset"]
    metric_items: List[Any] = []
    target_col = (spec.get("target_column") or "").strip()
    pred_col = (spec.get("prediction_column") or "").strip()

    if "DataSummaryPreset" in presets:
        metric_items.append(DataSummaryPreset())

    # Resolve reference DataFrame if DataDriftPreset is requested
    ref_df: Optional[pd.DataFrame] = None
    drift_cols: List[str] = []
    ref_source = (spec.get("reference_source") or "").strip()
    suffixes_raw = spec.get("suffixes") or ["_left", "_right"]
    left_sfx = str(suffixes_raw[0]) if len(suffixes_raw) >= 1 else "_left"
    right_sfx = str(suffixes_raw[1]) if len(suffixes_raw) >= 2 else "_right"

    def align_ref_with_joined(raw_ref: pd.DataFrame, side_sfx: str = "") -> Tuple[pd.DataFrame, List[str]]:
        mapped = raw_ref.copy()
        if side_sfx:
            rename_map = {}
            for col in list(mapped.columns):
                if col not in joined_df.columns and f"{col}{side_sfx}" in joined_df.columns:
                    rename_map[col] = f"{col}{side_sfx}"
            if rename_map:
                mapped = mapped.rename(columns=rename_map)

        shared = [
            c
            for c in joined_df.columns
            if c in mapped.columns
            and int(mapped[c].notna().sum()) >= 2
            and int(joined_df[c].notna().sum()) >= 2
            and int(joined_df[c].nunique(dropna=True)) >= 2
            and (
                pd.api.types.is_numeric_dtype(joined_df[c])
                or int(joined_df[c].nunique(dropna=True)) <= 50
            )
        ]
        # Evidently requires reference_data and current_data to have the exact same columns
        aligned = pd.DataFrame(index=mapped.index)
        for col in joined_df.columns:
            if col in mapped.columns and mapped[col].notna().any():
                aligned[col] = mapped[col].astype(joined_df[col].dtype, errors="ignore")
            else:
                non_null_source = joined_df[col].dropna()
                if len(non_null_source) > 0:
                    sampled = non_null_source.sample(
                        n=len(mapped),
                        replace=True,
                        random_state=42,
                    )
                    aligned[col] = sampled.to_numpy()
                else:
                    aligned[col] = np.nan
        return aligned, shared

    if "DataDriftPreset" in presets:
        if ref_source == "split_half" and len(joined_df) >= 4:
            mid = len(joined_df) // 2
            ref_df = joined_df.iloc[:mid].copy()
            joined_df = joined_df.iloc[mid:].copy()
            drift_cols = [
                c
                for c in joined_df.columns
                if int(ref_df[c].notna().sum()) >= 2
                and int(joined_df[c].notna().sum()) >= 2
                and (
                    pd.api.types.is_numeric_dtype(joined_df[c])
                    or int(joined_df[c].nunique(dropna=True)) <= 50
                )
            ]
        elif ref_source == "right" and right_df is not None and len(right_df) > 0:
            ref_df, drift_cols = align_ref_with_joined(right_df, right_sfx)
        elif spec.get("reference_dataset_id"):
            loaded_ref = load_dataframe(
                ws,
                server_url,
                headers,
                dataset_id=spec.get("reference_dataset_id"),
                max_rows=max_rows,
            )
            if len(loaded_ref) > 0:
                ref_df, drift_cols = align_ref_with_joined(loaded_ref, "")
        elif len(left_df) > 0:
            ref_df, drift_cols = align_ref_with_joined(left_df, left_sfx)

        if ref_df is not None and len(ref_df) > 0 and drift_cols:
            metric_items.append(DataDriftPreset(columns=drift_cols))
        else:
            ref_df = None

    if "RegressionPreset" in presets and target_col in joined_df.columns and pred_col in joined_df.columns:
        valid_reg = joined_df[[target_col, pred_col]].apply(pd.to_numeric, errors="coerce").dropna()
        if len(valid_reg) > 0:
            metric_items.append(RegressionPreset())

    if "ClassificationPreset" in presets and target_col in joined_df.columns and pred_col in joined_df.columns:
        valid_cls_mask = (
            joined_df[target_col].map(_normalize_class_label).notna()
            & joined_df[pred_col].map(_normalize_class_label).notna()
        )
        if int(valid_cls_mask.sum()) > 0:
            metric_items.append(ClassificationPreset())

    for col in spec.get("metric_columns") or []:
        if col in joined_df.columns and joined_df[col].notna().any():
            metric_items.append(ValueStats(column=col))

    if spec.get("include_correlations"):
        num_cols = [
            c
            for c in joined_df.columns
            if pd.api.types.is_numeric_dtype(joined_df[c]) and not pd.api.types.is_bool_dtype(joined_df[c])
        ]
        # Ensure all non-empty columns have overlapping non-null rows so scipy chi2_contingency does not fail on disjoint outer joins
        non_empty_cols = [c for c in joined_df.columns if int(joined_df[c].notna().sum()) >= 2]
        has_disjoint_pair = False
        for i in range(len(non_empty_cols)):
            for j in range(i + 1, len(non_empty_cols)):
                c1, c2 = non_empty_cols[i], non_empty_cols[j]
                if int((joined_df[c1].notna() & joined_df[c2].notna()).sum()) < 2:
                    has_disjoint_pair = True
                    break
            if has_disjoint_pair:
                break
        if len(num_cols) >= 2 and not has_disjoint_pair:
            metric_items.append(DatasetCorrelations())

    if not metric_items:
        metric_items = [RowCount(), ColumnCount(), DataSummaryPreset()]

    current_dataset = build_evidently_dataset(joined_df, spec)
    reference_dataset = build_evidently_dataset(ref_df, spec) if (ref_df is not None and len(ref_df) > 0) else None

    report_name = (spec.get("report_name") or "").strip() or (
        f"join-{spec.get('join_type', 'inner')}-{datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M')}"
    )
    tags: List[str] = [str(t).strip() for t in (spec.get("tags") or []) if str(t).strip()]
    if "joined-report" not in tags:
        tags.append("joined-report")

    user_meta = spec.get("metadata") or {}
    metadata: Dict[str, Any] = {
        "created_by": "vscode-evidently-workspace",
        "join_type": str(diagnostics.get("joinType", "inner")),
        "left_dataset_id": str(spec.get("left_dataset_id") or ""),
        "right_dataset_id": str(spec.get("right_dataset_id") or ""),
        "left_on": ",".join(str(x) for x in (spec.get("left_on") or [])),
        "right_on": ",".join(str(x) for x in (spec.get("right_on") or [])),
        "joined_rows": str(len(joined_df)),
        "joined_columns": str(len(joined_df.columns)),
    }
    for mk, mv in user_meta.items():
        if mv is not None:
            metadata[str(mk)] = str(mv)

    include_tests = bool(spec.get("include_tests", False))
    try:
        report = Report(metric_items, metadata=metadata, tags=tags, include_tests=include_tests)
        snapshot = report.run(
            current_data=current_dataset,
            reference_data=reference_dataset,
            name=report_name,
            tags=tags,
            metadata=metadata,
        )
    except Exception:
        # Fallback if a specific preset or reference drift fails on sparse outer-joined columns
        fallback_items = [RowCount(), ColumnCount(), DataSummaryPreset()]
        report = Report(fallback_items, metadata=metadata, tags=tags, include_tests=False)
        snapshot = report.run(
            current_data=current_dataset,
            reference_data=None,
            name=report_name,
            tags=tags,
            metadata=metadata,
        )

    # Save HTML to temp or requested output path
    html_output_path = spec.get("html_output_path")
    if not html_output_path:
        fd, html_output_path = tempfile.mkstemp(prefix="evidently_report_", suffix=".html")
        os.close(fd)
    snapshot.save_html(str(html_output_path))

    snapshot_dict = json.loads(snapshot.json())

    target_project_id = (spec.get("target_project_id") or "").strip()
    publish_to_workspace = bool(spec.get("publish_to_workspace", True)) and bool(target_project_id)
    save_joined_dataset = bool(spec.get("save_joined_dataset", True))

    snapshot_id: Optional[str] = None
    joined_dataset_id: Optional[str] = None
    report_url: Optional[str] = None

    if publish_to_workspace:
        snap_ref = ws.add_run(
            target_project_id,
            snapshot,
            include_data=False,
            name=report_name,
        )
        snapshot_id = str(snap_ref.id)
        report_url = f"{server_url.rstrip('/')}/projects/{target_project_id}/reports/{snapshot_id}"

        if save_joined_dataset:
            joined_dataset_name = (spec.get("joined_dataset_name") or "").strip() or f"{report_name} (joined table)"
            joined_dataset_desc = (spec.get("joined_dataset_description") or "").strip() or (
                f"Joined dataset ({diagnostics.get('joinType')}) on {diagnostics.get('leftOn')} = {diagnostics.get('rightOn')}. "
                f"Rows: {len(joined_df)}, Columns: {len(joined_df.columns)}."
            )
            ds_id = ws.add_dataset(
                project_id=target_project_id,
                dataset=current_dataset,
                name=joined_dataset_name,
                description=joined_dataset_desc,
                link=SnapshotLink(
                    snapshot_id=snap_ref.id,
                    dataset_type="output",
                    dataset_subtype="current",
                ),
            )
            joined_dataset_id = str(ds_id)

    return {
        "status": "ok",
        "mode": "create_report",
        "reportName": report_name,
        "projectId": target_project_id or None,
        "snapshotId": snapshot_id,
        "joinedDatasetId": joined_dataset_id,
        "reportUrl": report_url,
        "htmlPath": str(html_output_path),
        "columns": [str(c) for c in joined_df.columns],
        "diagnostics": diagnostics,
        "columnProfiles": profiles,
        "rows": preview_rows,
        "metricsSummary": snapshot_dict,
    }


def delete_artifacts(spec: Dict[str, Any]) -> Dict[str, Any]:
    server_url = spec.get("server_url") or "http://192.168.0.15:8000"
    api_key = resolve_api_key(spec.get("api_key"))
    _, _, headers = connect_remote_workspace(server_url, api_key)
    base_url = server_url.rstrip("/")

    deleted_snapshots: List[str] = []
    deleted_datasets: List[str] = []

    project_id = spec.get("project_id")
    for sid in spec.get("snapshot_ids") or []:
        if project_id and sid:
            resp = requests.delete(
                f"{base_url}/api/projects/{project_id}/{sid}",
                headers=headers,
                timeout=20,
            )
            if resp.status_code < 300:
                deleted_snapshots.append(str(sid))

    for did in spec.get("dataset_ids") or []:
        if did:
            resp = requests.delete(
                f"{base_url}/api/datasets/{did}",
                headers=headers,
                timeout=20,
            )
            if resp.status_code < 300:
                deleted_datasets.append(str(did))

    return {
        "status": "ok",
        "deletedSnapshots": deleted_snapshots,
        "deletedDatasets": deleted_datasets,
    }


def main() -> None:
    try:
        if len(sys.argv) > 1 and sys.argv[1] != "-":
            raw = Path(sys.argv[1]).read_text(encoding="utf-8")
        else:
            raw = sys.stdin.read()
        spec = json.loads(raw)
        mode = spec.get("mode") or "create_report"
        if mode == "delete_artifacts":
            result = delete_artifacts(spec)
        else:
            result = create_report_and_publish(spec)
        sys.stdout.write(json.dumps(result) + "\n")
        sys.stdout.flush()
    except Exception as exc:
        err_payload = {
            "status": "error",
            "error": str(exc),
            "traceback": traceback.format_exc(),
        }
        sys.stdout.write(json.dumps(err_payload) + "\n")
        sys.stdout.flush()
        sys.exit(1)


if __name__ == "__main__":
    main()
