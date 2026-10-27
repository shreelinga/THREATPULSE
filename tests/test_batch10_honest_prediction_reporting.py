"""
=============================================================================
THREATPULSE TEST SUITE: BATCH 10 (PHASE 5: HONEST PREDICTION REPORTING)
=============================================================================
Validates:
  1. Primary Historical Median Baseline definition, metrics, and implementation.
  2. Experimental comparison model tagging (Random Forest & Huber are not proven superior).
  3. Distinction between raw stock returns and benchmark-adjusted abnormal returns.
  4. Honest uncertainty presentation: sample size (N=21) & validation methods (LOOCV, LOGO-CV).
  5. Confidence intervals for MAE distinguished from incident-level prediction intervals.
  6. Unavailable status for incident-level prediction intervals (small sample N=21).
  7. API and Dashboard presentation of dual forecast cards and 8-model scorecard.
  8. Safety & integrity: training_dataset.csv, production joblib, and database preserved.
=============================================================================
"""

import os
import json
import sqlite3
import unittest
import numpy as np
import pandas as pd

from app import app, get_db
from threat_ml_engine import (
    predict_stock_impact,
    load_model_bundle,
    load_model_metrics,
    validate_ml_inputs,
    MODEL_VERSION,
    MODEL_LIMITATION,
    TARGET_DEFINITION
)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class TestBatch10HonestPredictionReporting(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.client = app.test_client()
        cls.metrics_path = os.path.join(BASE_DIR, "model_metrics.json")
        cls.dashboard_path = os.path.join(BASE_DIR, "dashboard.html")
        cls.training_csv_path = os.path.join(BASE_DIR, "training_dataset.csv")
        cls.model_joblib_path = os.path.join(BASE_DIR, "threat_impact_model.joblib")
        cls.db_path = os.path.join(BASE_DIR, "threat_stock_data.db")

    # -------------------------------------------------------------------------
    # 1. Target Definition & Baseline Implementation
    # -------------------------------------------------------------------------
    def test_01_primary_benchmark_metrics_and_target_definition(self):
        """Verify Historical Median Baseline is configured as primary benchmark on 1D abnormal return."""
        with open(self.metrics_path, "r", encoding="utf-8") as f:
            metrics = json.load(f)

        self.assertIn("primary_benchmark", metrics)
        bm = metrics["primary_benchmark"]

        self.assertEqual(bm["model_name"], "Historical Median Baseline")
        self.assertEqual(bm["eval_sample_count"], 21)
        self.assertEqual(bm["median_abnormal_return"], -2.00)
        self.assertEqual(bm["loocv_mae"], 3.66)
        self.assertEqual(bm["logo_mae"], 3.72)
        self.assertEqual(bm["status"], "RECOMMENDED_PRIMARY_BENCHMARK")
        self.assertIn("Benchmark-Adjusted 1-Day Abnormal Return", bm["target_definition"])
        self.assertIn("SPY", bm["market_benchmarks"])
        self.assertIn("^NSEI", bm["market_benchmarks"])

    # -------------------------------------------------------------------------
    # 2. Model Roles: Random Forest & Huber are Experimental, Not Proven Superior
    # -------------------------------------------------------------------------
    def test_02_experimental_model_classification_and_hierarchy(self):
        """Verify Random Forest and Huber are categorized as EXPERIMENTAL and not proven superior."""
        with open(self.metrics_path, "r", encoding="utf-8") as f:
            metrics = json.load(f)

        cand = metrics.get("candidate_evaluation_abnormal_1d", {})
        models = cand.get("models", {})

        self.assertGreaterEqual(len(models), 8)
        self.assertEqual(models["Historical Median Baseline"]["tier"], "RECOMMENDED_BENCHMARK")
        self.assertEqual(models["Random Forest Pipeline"]["tier"], "EXPERIMENTAL")
        self.assertEqual(models["Huber Robust Regression"]["tier"], "EXPERIMENTAL")

        # Baseline LOOCV MAE (3.66%) is strictly lower than RF (4.18%) and Huber (4.65%)
        self.assertLess(models["Historical Median Baseline"]["loocv_mae"], models["Random Forest Pipeline"]["loocv_mae"])
        self.assertLess(models["Historical Median Baseline"]["loocv_mae"], models["Huber Robust Regression"]["loocv_mae"])

        # Notes in metrics confirm not proven superior
        rf_role = models["Random Forest Pipeline"]["role"].lower()
        self.assertIn("does not beat", rf_role)

    # -------------------------------------------------------------------------
    # 3. Raw Return vs. Benchmark-Adjusted Abnormal Return Separation
    # -------------------------------------------------------------------------
    def test_03_never_label_raw_returns_as_abnormal_returns(self):
        """Ensure clear demarcation between raw returns and benchmark-adjusted abnormal returns."""
        # Audited candidate dataset has both raw and benchmark-adjusted columns
        dataset_path = os.path.join(BASE_DIR, "gate2_audit_deliverables", "audited_event_window_table.csv")
        self.assertTrue(os.path.exists(dataset_path), "Audited event window table must exist")

        df = pd.read_csv(dataset_path)
        self.assertEqual(len(df), 21)
        self.assertIn("raw_return_1d", df.columns)
        self.assertIn("benchmark_return_1d", df.columns)
        self.assertIn("abnormal_return_1d", df.columns)

        # Verify abnormal return formula: AR = Stock_Return - Benchmark_Return
        for _, row in df.iterrows():
            expected_ar = round(row["raw_return_1d"] - row["benchmark_return_1d"], 2)
            self.assertAlmostEqual(row["abnormal_return_1d"], expected_ar, delta=0.02)

        # Legacy dataset has raw T-1 to T+2 returns, which must not be labeled abnormal
        with open(self.metrics_path, "r", encoding="utf-8") as f:
            metrics = json.load(f)
        legacy_target = metrics.get("target_definition", "")
        self.assertIn("Stock Price Change", legacy_target)
        self.assertNotIn("Abnormal Return", legacy_target)

    # -------------------------------------------------------------------------
    # 4. Honest Uncertainty & Prediction Interval Presentation
    # -------------------------------------------------------------------------
    def test_04_prediction_interval_unavailable_status(self):
        """Verify that incident-level prediction intervals are marked UNAVAILABLE due to small sample."""
        res = predict_stock_impact(severity_score=8.5, users_affected_millions=10.0, data_sensitivity_score=8.0)
        self.assertEqual(res["status"], "SUCCESS")

        # Incident prediction interval must be explicitly unavailable
        self.assertIn("incident_prediction_interval", res)
        pi = res["incident_prediction_interval"]
        self.assertEqual(pi["status"], "UNAVAILABLE")
        self.assertIsNone(pi["interval"])
        self.assertIn("21", pi.get("note", ""))

        # Confidence interval distinction
        eval_meta = res.get("evaluation_metadata", {})
        self.assertEqual(eval_meta.get("sample_size"), 21)
        self.assertIn("bootstrap", eval_meta.get("confidence_interval_distinction", "").lower())
        self.assertIn("average mae", eval_meta.get("confidence_interval_distinction", "").lower())

        # Descriptive tree spread must state it is NOT a calibrated interval
        spread = res.get("tree_output_spread", {})
        self.assertIn("not a calibrated confidence interval", spread.get("basis", "").lower())

    # -------------------------------------------------------------------------
    # 5. Model Inference Endpoint Returns Dual Benchmark and Experimental Model
    # -------------------------------------------------------------------------
    def test_05_predict_endpoint_dual_benchmark_and_experimental_payload(self):
        """Verify /api/ml/predict returns primary_benchmark alongside experimental_model."""
        payload = {
            "severity_score": 8.5,
            "users_affected_millions": 5.0,
            "data_sensitivity_score": 8.0
        }
        res = self.client.post("/api/ml/predict", json=payload)
        self.assertEqual(res.status_code, 200)
        data = res.get_json()

        # Primary Benchmark
        self.assertIn("primary_benchmark", data)
        self.assertEqual(data["primary_benchmark"]["median_abnormal_return"], -2.00)
        self.assertEqual(data["primary_benchmark"]["loocv_mae"], 3.66)

        # Experimental Model
        self.assertIn("experimental_model", data)
        self.assertEqual(data["experimental_model"]["role"], "EXPERIMENTAL_COMPARISON")
        self.assertIn("predicted_change_percent", data["experimental_model"])
        self.assertEqual(data["experimental_model"]["status"], "NOT_PROVEN_SUPERIOR")

        # Incident interval status
        self.assertEqual(data["incident_prediction_interval"]["status"], "UNAVAILABLE")

    # -------------------------------------------------------------------------
    # 6. Unavailable Status for Invalid / Out-of-Scope Inputs
    # -------------------------------------------------------------------------
    def test_06_unavailable_status_for_out_of_scope_inputs(self):
        """Verify invalid/out-of-bounds inputs return UNAVAILABLE with reason."""
        invalid_cases = [
            {"severity_score": 0.5, "users_affected_millions": 5.0, "data_sensitivity_score": 8.0},
            {"severity_score": 11.0, "users_affected_millions": 5.0, "data_sensitivity_score": 8.0},
            {"severity_score": 8.5, "users_affected_millions": -1.0, "data_sensitivity_score": 8.0},
            {"severity_score": 8.5, "users_affected_millions": 5.0, "data_sensitivity_score": 12.0},
            {"severity_score": "high", "users_affected_millions": 5.0, "data_sensitivity_score": 8.0}
        ]
        for case in invalid_cases:
            res = self.client.post("/api/ml/predict", json=case)
            self.assertEqual(res.status_code, 400)
            data = res.get_json()
            self.assertEqual(data["status"], "UNAVAILABLE")
            self.assertIn("error", data)
            self.assertIsNone(data["predicted_change_percent"])

    # -------------------------------------------------------------------------
    # 7. Dashboard HTML Verification
    # -------------------------------------------------------------------------
    def test_07_dashboard_html_contains_phase5_reporting_elements(self):
        """Verify dashboard.html renders Historical Median benchmark, N=21 sample size, and honest disclosures."""
        with open(self.dashboard_path, "r", encoding="utf-8") as f:
            html = f.read()

        # Dual cards
        self.assertIn("HISTORICAL MEDIAN", html)
        self.assertIn("PRIMARY BENCHMARK", html)
        self.assertIn("RANDOM FOREST", html)
        self.assertIn("EXPERIMENTAL MODEL", html)

        # Metadata disclosures
        self.assertIn("Benchmark-Adjusted 1-Day Abnormal Return", html)
        self.assertIn("N = 21 Audited Incidents", html)
        self.assertIn("LOOCV &amp; LOGO-CV", html)
        self.assertIn("ml-sim-pred-interval", html)

        # Honest disclosure box
        self.assertIn("Honest Uncertainty Disclosure", html)
        self.assertIn("Bootstrap confidence intervals apply to the sample average MAE", html)
        self.assertIn("not to individual incident predictions", html)

        # Scorecard table
        self.assertIn("95% Bootstrap CI for MAE", html)
        self.assertIn("LOGO-CV MAE (%)", html)
        self.assertIn("vs. Median Benchmark", html)
        self.assertIn("Legacy Pre-Production Reference Table", html)

    # -------------------------------------------------------------------------
    # 8. Data and Artifact Safety: Production Model & Datasets Preserved
    # -------------------------------------------------------------------------
    def test_08_production_artifacts_and_database_unmodified(self):
        """Verify training_dataset.csv, threat_impact_model.joblib, and database records are preserved."""
        # Check training_dataset.csv remains exactly 14 historical records
        train_df = pd.read_csv(self.training_csv_path)
        self.assertEqual(len(train_df), 14, "training_dataset.csv must remain strictly 14 historical rows")

        # Check threat_impact_model.joblib is preserved and loads v2.2.0
        bundle = load_model_bundle()
        self.assertIsNotNone(bundle)
        self.assertEqual(bundle["model_version"], MODEL_VERSION)

        # Check database records
        conn = get_db()
        c = conn.cursor()
        c.execute("SELECT COUNT(*) FROM threats")
        threat_count = c.fetchone()[0]
        self.assertGreaterEqual(threat_count, 34)

        c.execute("SELECT COUNT(*) FROM event_studies")
        event_count = c.fetchone()[0]
        self.assertEqual(event_count, 14)

    # -------------------------------------------------------------------------
    # 9. Experimental Huber Model Prediction Payload & Bounds
    # -------------------------------------------------------------------------
    def test_09_experimental_huber_payload_and_bounds(self):
        """Verify Huber Robust Regression is reported as an experimental comparison model."""
        # Valid prediction
        res = self.client.post("/api/ml/predict", json={
            "severity_score": 8.5,
            "users_affected_millions": 5.0,
            "data_sensitivity_score": 8.0
        })
        self.assertEqual(res.status_code, 200)
        data = res.get_json()

        self.assertIn("experimental_huber", data)
        huber = data["experimental_huber"]
        self.assertIsNotNone(huber)
        self.assertEqual(huber["model_name"], "Huber Robust Regression")
        self.assertEqual(huber["role"], "EXPERIMENTAL_COMPARISON")
        self.assertEqual(huber["status"], "NOT_PROVEN_SUPERIOR")
        self.assertEqual(huber["loocv_mae"], 4.65)
        self.assertEqual(huber["logo_mae"], 4.69)
        self.assertIsInstance(huber["predicted_change_percent"], float)
        self.assertIn("formula", huber)

        # Baseline still beats Huber
        self.assertLess(data["primary_benchmark"]["loocv_mae"], huber["loocv_mae"])

        # Invalid prediction: experimental_huber should be None
        res_inv = self.client.post("/api/ml/predict", json={
            "severity_score": 15.0,
            "users_affected_millions": 5.0,
            "data_sensitivity_score": 8.0
        })
        self.assertEqual(res_inv.status_code, 400)
        data_inv = res_inv.get_json()
        self.assertIsNone(data_inv["experimental_huber"])

    # -------------------------------------------------------------------------
    # 10. Clear Demarcation of Raw Return vs. Benchmark-Adjusted Abnormal Return
    # -------------------------------------------------------------------------
    def test_10_target_labeling_and_raw_vs_abnormal_separation(self):
        """Ensure dashboard and engine strictly separate observed raw returns from abnormal returns."""
        with open(self.dashboard_path, "r", encoding="utf-8") as f:
            html = f.read()

        # Tab 4 correlation header must state Observed Raw Stock Return
        self.assertIn("Observed Raw Stock Return", html)
        self.assertIn("This is an observed raw market return, NOT a model prediction or abnormal return", html)

        # Tab 7 event studies table must distinguish raw from abnormal returns
        self.assertIn("Observed Raw Stock Return (1d/3d/5d)", html)
        self.assertIn("Benchmark-Adjusted Abnormal (1D)", html)
        self.assertIn("AR = Raw Stock Return - Benchmark Return", html)

        # Primary benchmark and evaluation metadata must specify benchmark-adjusted abnormal return
        res = predict_stock_impact(severity_score=7.5, users_affected_millions=2.0, data_sensitivity_score=7.0)
        self.assertIn("Benchmark-Adjusted", res["primary_benchmark"]["target_definition"])
        self.assertIn("Benchmark-Adjusted", res["evaluation_metadata"]["target_definition"])
        # Production model legacy target definition is for raw price reaction, never mislabeled as abnormal return
        self.assertNotIn("Abnormal Return", res["target_definition"])

    # -------------------------------------------------------------------------
    # 11. Honest Uncertainty, Sample Size Warnings & Disclaimers
    # -------------------------------------------------------------------------
    def test_11_honest_uncertainty_sample_size_warnings(self):
        """Verify sample size warnings, CI vs. PI distinction, and lack of fabricated intervals."""
        res = predict_stock_impact(severity_score=9.0, users_affected_millions=20.0, data_sensitivity_score=9.0)
        eval_meta = res.get("evaluation_metadata", {})

        # Sample size warning
        self.assertIn("sample_size_warning", eval_meta)
        warning = eval_meta["sample_size_warning"].lower()
        self.assertIn("21", warning)
        self.assertIn("uncertain", warning)

        # No fabricated prediction interval
        pi = res.get("incident_prediction_interval", {})
        self.assertEqual(pi.get("status"), "UNAVAILABLE")
        self.assertIsNone(pi.get("interval"))
        self.assertNotIn("±", str(pi.get("interval")))

        # Tree spread explicitly declares it is NOT a calibrated confidence interval
        spread = res.get("tree_output_spread", {})
        self.assertIn("not a calibrated confidence interval", spread.get("basis", "").lower())

    # -------------------------------------------------------------------------
    # 12. Dashboard Phase 5 UI Elements, Tooltips & Responsiveness
    # -------------------------------------------------------------------------
    def test_12_dashboard_phase5_ui_elements_and_tooltips(self):
        """Verify dashboard includes Huber card, tooltips, responsive grid, and scorecard columns."""
        with open(self.dashboard_path, "r", encoding="utf-8") as f:
            html = f.read()

        # 3-Card forecast display elements
        self.assertIn("forecast-cards-grid", html)
        self.assertIn("HUBER REGRESSION", html)
        self.assertIn("ml-huber-val", html)
        self.assertIn("Baseline-First Evaluation", html)

        # Tooltips CSS & elements
        self.assertIn(".tooltip-wrap", html)
        self.assertIn(".tooltip-trigger", html)
        self.assertIn(".tooltip-content", html)

        # Transparent scorecard headers
        self.assertIn("Target &amp; Horizon", html)
        self.assertIn("LOOCV MAE (%)", html)
        self.assertIn("95% Bootstrap CI for MAE", html)
        self.assertIn("LOGO-CV MAE (%)", html)
        self.assertIn("LOOCV RMSE (%)", html)
        self.assertIn("LOGO-CV RMSE (%)", html)
        self.assertIn("Acc. (Dir / Bal)", html)
        self.assertIn("Sample &amp; Validation", html)
        self.assertIn("Version &amp; Date", html)
        self.assertIn("vs. Median Benchmark", html)

    # -------------------------------------------------------------------------
    # 13. Missing Data & Graceful Handling
    # -------------------------------------------------------------------------
    def test_13_missing_data_and_unavailable_metrics_handling(self):
        """Verify engine and endpoints handle missing or incomplete inputs gracefully."""
        # Missing keys in payload
        res = self.client.post("/api/ml/predict", json={})
        self.assertEqual(res.status_code, 400)
        data = res.get_json()
        self.assertEqual(data["status"], "UNAVAILABLE")
        self.assertTrue("cannot be None" in data["error"] or "Missing" in data["error"])

        # None values
        res_none = self.client.post("/api/ml/predict", json={
            "severity_score": None,
            "users_affected_millions": 5.0,
            "data_sensitivity_score": 8.0
        })
        self.assertEqual(res_none.status_code, 400)
        self.assertEqual(res_none.get_json()["status"], "UNAVAILABLE")

        # Validation utility directly
        valid, err, cleaned = validate_ml_inputs("NaN", 5.0, 8.0)
        self.assertFalse(valid)
        self.assertIsNotNone(err)
        self.assertIsNone(cleaned)


if __name__ == "__main__":
    unittest.main()
