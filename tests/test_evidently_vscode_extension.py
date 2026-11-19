"""Tests for the VS Code Evidently Remote Workspace extension Python bridge (`python/evidently_bridge.py`)."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import tempfile
import unittest
import pandas as pd

BRIDGE_PATH = (
    Path(__file__).resolve().parents[1]
    / "vscode-evidently-workspace"
    / "python"
    / "evidently_bridge.py"
)


def load_bridge_module():
    spec = importlib.util.spec_from_file_location("evidently_bridge", BRIDGE_PATH)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules["evidently_bridge"] = mod
    spec.loader.exec_module(mod)
    return mod


class TestEvidentlyVscodeExtensionBridge(unittest.TestCase):
    def test_perform_pandas_join_handles_int_float_and_string_key_mismatch(self):
        bridge = load_bridge_module()
        left_df = pd.DataFrame(
            {
                "topic_id": [101, 102, 103],
                "title": ["First", "Second", "Third"],
                "shared": ["L1", "L2", "L3"],
            }
        )
        right_df = pd.DataFrame(
            {
                "topic_id": ["101.0", " 102 ", "999"],
                "votes": [5, 12, 1],
                "shared": ["R1", "R2", "R9"],
            }
        )

        joined, diag = bridge.perform_pandas_join(
            left_df=left_df,
            right_df=right_df,
            join_type="inner",
            left_on=["topic_id"],
            right_on=["topic_id"],
            match_mode="normalized",
            suffixes=("_left", "_right"),
        )
        self.assertEqual(len(joined), 2)
        self.assertEqual(diag["matchedLeftRows"], 2)
        self.assertEqual(diag["unmatchedLeftRows"], 1)
        self.assertIn("topic_id", joined.columns)
        self.assertIn("shared_left", joined.columns)
        self.assertIn("shared_right", joined.columns)

    def test_create_report_local_join_with_regression_and_drift(self):
        bridge = load_bridge_module()
        left_csv = "id,target,feature_a\n1,10.5,1.0\n2,20.0,2.0\n3,30.2,3.0\n4,40.1,4.0\n"
        right_csv = "id,prediction,feature_a\n1,11.0,1.2\n2,19.5,2.1\n3,29.8,2.9\n4,41.0,4.2\n"
        with tempfile.TemporaryDirectory() as tmp_dir:
            out_html = Path(tmp_dir) / "joined_report.html"
            res = bridge.create_report_and_publish(
                {
                    "mode": "create_report",
                    "server_url": "http://192.168.0.15:8000",
                    "left_csv_text": left_csv,
                    "right_csv_text": right_csv,
                    "join_type": "inner",
                    "left_on": ["id"],
                    "right_on": ["id"],
                    "presets": ["DataSummaryPreset", "DataDriftPreset", "RegressionPreset"],
                    "target_column": "target",
                    "prediction_column": "prediction",
                    "metric_columns": ["target", "prediction"],
                    "include_correlations": True,
                    "reference_source": "split_half",
                    "publish_to_workspace": False,
                    "html_output_path": str(out_html),
                }
            )
            self.assertEqual(res["status"], "ok")
            self.assertTrue(out_html.exists())
            self.assertGreater(out_html.stat().st_size, 10000)
            self.assertEqual(res["diagnostics"]["joinedRowCount"], 4)
            self.assertGreater(len(res["metricsSummary"]["metrics"]), 5)

    def test_disjoint_outer_join_does_not_fail_correlations_or_drift(self):
        bridge = load_bridge_module()
        left_csv = "name,left_cat,val\nA,cat1,1.0\nB,cat2,2.0\nC,cat1,3.0\n"
        right_csv = "name,right_cat,val\nX,rc1,10.0\nY,rc2,20.0\nZ,rc1,30.0\n"
        with tempfile.TemporaryDirectory() as tmp_dir:
            out_html = Path(tmp_dir) / "disjoint_outer.html"
            res = bridge.create_report_and_publish(
                {
                    "mode": "create_report",
                    "left_csv_text": left_csv,
                    "right_csv_text": right_csv,
                    "join_type": "outer",
                    "left_on": ["name"],
                    "right_on": ["name"],
                    "presets": ["DataSummaryPreset", "DataDriftPreset"],
                    "include_correlations": True,
                    "reference_source": "left",
                    "publish_to_workspace": False,
                    "html_output_path": str(out_html),
                }
            )
            self.assertEqual(res["status"], "ok")
            self.assertEqual(res["diagnostics"]["joinedRowCount"], 6)
            self.assertTrue(out_html.exists())

    def test_create_report_raises_on_empty_join(self):
        bridge = load_bridge_module()
        with self.assertRaisesRegex(ValueError, "0 rows"):
            bridge.create_report_and_publish(
                {
                    "mode": "create_report",
                    "left_csv_text": "id,val\n1,10\n",
                    "right_csv_text": "id,score\n99,20\n",
                    "join_type": "inner",
                    "left_on": ["id"],
                    "right_on": ["id"],
                    "publish_to_workspace": False,
                }
            )

    def test_perform_pandas_join_handles_pre_existing_suffixed_columns(self):
        bridge = load_bridge_module()
        left_df = pd.DataFrame({"id": [1, 2], "val": [10, 20], "val_left": [100, 200]})
        right_df = pd.DataFrame({"id": [1, 2], "val": [30, 40], "val_right": [300, 400]})
        joined, diag = bridge.perform_pandas_join(
            left_df=left_df,
            right_df=right_df,
            join_type="inner",
            left_on=["id"],
            right_on=["id"],
            suffixes=("_left", "_right"),
        )
        self.assertEqual(len(joined), 2)
        self.assertEqual(diag["joinedRowCount"], 2)
        self.assertFalse(joined.columns.duplicated().any())
        self.assertEqual(len(joined.columns), 5)

    def test_classification_preset_handles_missing_values_and_int_float_mismatch(self):
        bridge = load_bridge_module()
        left_csv = "id,target\n1,1\n2,0\n3,1\n4,\n"
        right_csv = "id,pred\n1,1\n2,0\n99,1\n"
        with tempfile.TemporaryDirectory() as tmp_dir:
            out_html = Path(tmp_dir) / "cls_report.html"
            res = bridge.create_report_and_publish(
                {
                    "mode": "create_report",
                    "left_csv_text": left_csv,
                    "right_csv_text": right_csv,
                    "join_type": "outer",
                    "left_on": ["id"],
                    "right_on": ["id"],
                    "presets": ["ClassificationPreset"],
                    "target_column": "target",
                    "prediction_column": "pred",
                    "publish_to_workspace": False,
                    "html_output_path": str(out_html),
                }
            )
            self.assertEqual(res["status"], "ok")
            self.assertTrue(out_html.exists())
            metrics = {m["metric_name"]: m["value"] for m in res["metricsSummary"]["metrics"]}
            self.assertEqual(metrics.get("Accuracy(classification_name=default)"), 1.0)
            f1_by_label = metrics.get("F1ByLabel(classification_name=default)", {})
            self.assertNotIn("nan", f1_by_label)
            self.assertNotIn("1.0", f1_by_label)
            self.assertEqual(f1_by_label.get("1"), 1.0)
            self.assertEqual(f1_by_label.get("0"), 1.0)


if __name__ == "__main__":
    unittest.main()
