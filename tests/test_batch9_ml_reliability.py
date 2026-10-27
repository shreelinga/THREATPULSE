"""
tests/test_batch9_ml_reliability.py

Automated test suite for Phase 9: THREATPULSE ML Reliability & Leakage-Safe Validation.
Tests:
  1. Dataset target reconstruction integrity (validates raw reaction formula and absence of market leakage).
  2. Training/inference preprocessing parity (Pipeline packaging with log1p).
  3. Feature schema and column order validation.
  4. Model artifact compatibility, metadata schema, and version integrity.
  5. Reproducible inference for identical inputs.
  6. Strict input validation and boundary checks (zero users, large users, negative users, out-of-range severity).
  7. Constant fallback detection (non-constant, feature-responsive predictions; no -8.48% lock).
  8. Honest uncertainty and dispersion labeling (tree spread descriptive dispersion, confidence=None).
  9. Baseline comparison on identical folds (LOOCV metrics integrity).
  10. Grouped leakage prevention (LOGO-CV grouped by company).
  11. Prediction API endpoint (/api/ml/predict) round-trip, serialization, and error responses.
  12. Safe prediction tracking in isolated test database without mutating production DB.
"""

import unittest
import os
import json
import math
import sqlite3
import hashlib
import tempfile
import pandas as pd
import numpy as np
import joblib
from sklearn.pipeline import Pipeline

from app import app
from train_model import (
    run_staged_model_evaluation,
    evaluate_leave_one_group_out,
    MODEL_OUTPUT_PATH,
    METRICS_JSON_PATH,
    DATASET_PATH,
    FEATURE_COLS,
    TARGET_COL,
    MODEL_VERSION,
    TARGET_DEFINITION,
    PREDICTION_HORIZON
)
from threat_ml_engine import (
    predict_stock_impact,
    load_model_bundle,
    validate_ml_inputs,
    MODEL_LIMITATION
)
from prediction_tracking import record_prediction
from database_setup import create_database


class TestMLReliability(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.client = app.test_client()
        cls.bundle = load_model_bundle()
        if os.path.exists(DATASET_PATH):
            cls.df = pd.read_csv(DATASET_PATH)
        else:
            cls.df = None

    # -------------------------------------------------------------------------
    # 1. Dataset Target Reconstruction & Integrity
    # -------------------------------------------------------------------------
    def test_01_training_dataset_target_reconstruction_integrity(self):
        """Verify target is mathematically raw stock price change over disclosure window."""
        self.assertIsNotNone(self.df, "training_dataset.csv must exist")
        self.assertEqual(len(self.df), 14, "Curated historical dataset must contain exactly 14 rows")

        required_cols = [
            "company", "ticker", "breach_date", "stock_price_before",
            "stock_price_after", "price_change_percent", "users_affected_millions",
            "severity_score", "data_sensitivity_score"
        ]
        for col in required_cols:
            self.assertIn(col, self.df.columns, f"Dataset missing required column: {col}")

        # Check mathematical reconstruction: (after - before) / before * 100 == price_change_percent
        for idx, row in self.df.iterrows():
            before = float(row["stock_price_before"])
            after = float(row["stock_price_after"])
            recorded = float(row["price_change_percent"])
            calculated = ((after - before) / before) * 100.0

            self.assertAlmostEqual(
                recorded, calculated, delta=0.01,
                msg=f"Row {idx} ({row['company']}): recorded {recorded}% does not match formula {calculated:.4f}%"
            )

        # Confirm target is raw price return, NOT market-model abnormal return
        self.assertEqual(TARGET_COL, "price_change_percent")

    # -------------------------------------------------------------------------
    # 2. Training/Inference Preprocessing Parity
    # -------------------------------------------------------------------------
    def test_02_pipeline_architecture_and_preprocessing_parity(self):
        """Verify that preprocessing is bundled into a scikit-learn Pipeline object."""
        self.assertIsNotNone(self.bundle)
        pipeline = self.bundle.get("pipeline")
        self.assertIsInstance(pipeline, Pipeline, "Saved model must be a sklearn.pipeline.Pipeline")

        step_names = [name for name, _ in pipeline.steps]
        self.assertIn("preprocessor", step_names, "Pipeline must include 'preprocessor' step")
        self.assertIn("regressor", step_names, "Pipeline must include 'regressor' step")

        # Test inference through pipeline matches step-by-step transformation
        test_input_df = pd.DataFrame([{
            "severity_score": 8.0,
            "users_affected_millions": 25.0,
            "data_sensitivity_score": 7.0
        }])

        pipeline_pred = pipeline.predict(test_input_df)[0]
        preprocessor = pipeline.named_steps["preprocessor"]
        regressor = pipeline.named_steps["regressor"]
        transformed = preprocessor.transform(test_input_df)
        manual_pred = regressor.predict(transformed)[0]

        self.assertAlmostEqual(pipeline_pred, manual_pred, places=6,
                               msg="Pipeline end-to-end prediction must equal sequential preprocessor + regressor")

        # Verify log1p transformation was applied to users_affected_millions
        # Feature order in transformer: severity (passthrough), users (log1p), sensitivity (passthrough)
        expected_log_users = np.log1p(25.0)
        self.assertAlmostEqual(transformed[0][1], expected_log_users, places=4,
                               msg="preprocessor must apply log1p to users_affected_millions")

    # -------------------------------------------------------------------------
    # 3. Feature Schema and Column Order
    # -------------------------------------------------------------------------
    def test_03_feature_schema_and_order_validation(self):
        """Verify feature names and ordering are strict and immutable."""
        expected_features = ["severity_score", "users_affected_millions", "data_sensitivity_score"]
        self.assertEqual(FEATURE_COLS, expected_features)
        self.assertEqual(self.bundle.get("feature_names"), expected_features)

        # Test predict_stock_impact uses correct columns regardless of arg order
        res = predict_stock_impact(
            data_sensitivity_score=6.0,
            severity_score=7.0,
            users_affected_millions=12.0
        )
        self.assertEqual(res["status"], "SUCCESS")
        self.assertIsInstance(res["predicted_change_percent"], float)

    # -------------------------------------------------------------------------
    # 4. Model Artifact Metadata Compatibility
    # -------------------------------------------------------------------------
    def test_04_model_artifact_metadata_compatibility(self):
        """Verify model bundle and metrics JSON carry complete traceability metadata."""
        self.assertTrue(os.path.exists(METRICS_JSON_PATH))
        with open(METRICS_JSON_PATH, "r", encoding="utf-8") as f:
            metrics = json.load(f)

        self.assertEqual(metrics["model_version"], MODEL_VERSION)
        self.assertEqual(metrics["sample_count"], 14)
        self.assertIn("target_definition", metrics)
        self.assertIn("prediction_horizon", metrics)
        self.assertIn("dataset_sha256", metrics)
        self.assertIn("staged_comparison", metrics)
        self.assertIn("limitations", metrics)

        # Bundle metadata
        self.assertEqual(self.bundle.get("model_version"), MODEL_VERSION)
        self.assertEqual(self.bundle.get("target_definition"), TARGET_DEFINITION)
        self.assertEqual(self.bundle.get("prediction_horizon"), PREDICTION_HORIZON)

    # -------------------------------------------------------------------------
    # 5. Reproducible Inference for Identical Inputs
    # -------------------------------------------------------------------------
    def test_05_reproducible_inference_for_identical_inputs(self):
        """Verify that identical inputs produce identical predictions without state leakage."""
        pred1 = predict_stock_impact(severity_score=8.5, users_affected_millions=50.0, data_sensitivity_score=9.0)
        pred2 = predict_stock_impact(severity_score=8.5, users_affected_millions=50.0, data_sensitivity_score=9.0)

        self.assertEqual(pred1["predicted_change_percent"], pred2["predicted_change_percent"])
        self.assertEqual(pred1["predicted_direction"], pred2["predicted_direction"])
        self.assertEqual(pred1["expected_range_percent"], pred2["expected_range_percent"])

    # -------------------------------------------------------------------------
    # 6. Strict Input Validation & Boundary Testing
    # -------------------------------------------------------------------------
    def test_06_strict_input_validation_and_boundary_testing(self):
        """Test mild, severe, zero, negative, out-of-bounds, and non-numeric inputs."""
        # Mild incident
        p_mild = predict_stock_impact(severity_score=2.0, users_affected_millions=0.001, data_sensitivity_score=2.0)
        self.assertEqual(p_mild["status"], "SUCCESS")

        # Catastrophic incident
        p_severe = predict_stock_impact(severity_score=10.0, users_affected_millions=200.0, data_sensitivity_score=10.0)
        self.assertEqual(p_severe["status"], "SUCCESS")
        self.assertLess(p_severe["predicted_change_percent"], p_mild["predicted_change_percent"])

        # Zero affected users (should be allowed: log1p(0) = 0)
        p_zero = predict_stock_impact(severity_score=5.0, users_affected_millions=0.0, data_sensitivity_score=5.0)
        self.assertEqual(p_zero["status"], "SUCCESS")

        # Extremely large affected users (500M)
        p_huge = predict_stock_impact(severity_score=8.0, users_affected_millions=500.0, data_sensitivity_score=8.0)
        self.assertEqual(p_huge["status"], "SUCCESS")
        self.assertTrue(math.isfinite(p_huge["predicted_change_percent"]))

        # Prohibited: Negative users
        p_neg_users = predict_stock_impact(severity_score=5.0, users_affected_millions=-5.0, data_sensitivity_score=5.0)
        self.assertEqual(p_neg_users["status"], "UNAVAILABLE")
        self.assertIsNone(p_neg_users["predicted_change_percent"])

        # Prohibited: Severity out of range (< 1.0 or > 10.0)
        p_low_sev = predict_stock_impact(severity_score=0.5, users_affected_millions=1.0, data_sensitivity_score=5.0)
        self.assertEqual(p_low_sev["status"], "UNAVAILABLE")

        p_high_sev = predict_stock_impact(severity_score=10.5, users_affected_millions=1.0, data_sensitivity_score=5.0)
        self.assertEqual(p_high_sev["status"], "UNAVAILABLE")

        # Prohibited: Sensitivity out of range
        p_high_sens = predict_stock_impact(severity_score=5.0, users_affected_millions=1.0, data_sensitivity_score=11.0)
        self.assertEqual(p_high_sens["status"], "UNAVAILABLE")

        # Prohibited: Non-numeric / malformed
        p_str = predict_stock_impact(severity_score="critical", users_affected_millions="many", data_sensitivity_score=5.0)
        self.assertEqual(p_str["status"], "UNAVAILABLE")

        # Prohibited: NaN / Infinite
        p_nan = predict_stock_impact(severity_score=float("nan"), users_affected_millions=1.0, data_sensitivity_score=5.0)
        self.assertEqual(p_nan["status"], "UNAVAILABLE")

    # -------------------------------------------------------------------------
    # 7. Constant Fallback Detection
    # -------------------------------------------------------------------------
    def test_07_constant_fallback_detection(self):
        """Verify model predictions respond dynamically to features and never lock to -8.48%."""
        predictions = []
        test_cases = [
            (3.0, 0.05, 3.0),
            (6.0, 1.0, 5.0),
            (8.0, 30.0, 8.0),
            (10.0, 150.0, 10.0),
        ]
        for sev, users, sens in test_cases:
            res = predict_stock_impact(severity_score=sev, users_affected_millions=users, data_sensitivity_score=sens)
            self.assertEqual(res["status"], "SUCCESS")
            predictions.append(res["predicted_change_percent"])

        # Ensure not all predictions are identical
        unique_preds = set(predictions)
        self.assertGreater(len(unique_preds), 1, "Predictions must vary with input features")

        # Ensure none are stuck at the old constant -8.48%
        for p in predictions:
            self.assertNotEqual(p, -8.48, "Prediction must not be stuck at historical -8.48%")

    # -------------------------------------------------------------------------
    # 8. Honest Uncertainty and Dispersion Labeling
    # -------------------------------------------------------------------------
    def test_08_honest_uncertainty_and_dispersion_labeling(self):
        """Verify uncertainty is explicitly descriptive dispersion, not a confidence interval."""
        res = predict_stock_impact(severity_score=8.0, users_affected_millions=20.0, data_sensitivity_score=7.0)
        self.assertEqual(res["status"], "SUCCESS")

        # Confidence must be None, with explicit rationale
        self.assertIsNone(res["confidence"], "Confidence must be None for small n=14 sample")
        self.assertIn("unavailable: small sample", res.get("confidence_basis", "").lower())

        # Expected range must be documented as tree dispersion
        self.assertIn("tree_output_spread", res)
        spread = res["tree_output_spread"]
        self.assertIn("dispersion", spread.get("basis", "").lower())
        self.assertIn("not a calibrated confidence interval", spread.get("basis", "").lower())

        # Model limitations must disclose sample size and directional accuracy reality
        limitation = res.get("model_limitation", "").lower()
        self.assertTrue("14" in limitation or "n=14" in limitation)
        self.assertIn("85.7%", limitation)

    # -------------------------------------------------------------------------
    # 9. Baseline Comparison on Identical Folds
    # -------------------------------------------------------------------------
    def test_09_baseline_comparison_identical_folds(self):
        """Verify LOOCV baseline comparisons execute on identical 14 folds."""
        def get_model_hash():
            if not os.path.exists(MODEL_OUTPUT_PATH):
                return None
            with open(MODEL_OUTPUT_PATH, "rb") as f:
                return hashlib.sha256(f.read()).hexdigest()

        model_hash_before = get_model_hash()

        staged_results = run_staged_model_evaluation(save_artifacts=False)

        if model_hash_before:
            model_hash_after = get_model_hash()
            self.assertEqual(
                model_hash_before,
                model_hash_after,
                "BLK-01: Production model artifact was overwritten during test_09!"
            )

        self.assertIn("baseline_zero", staged_results)
        self.assertIn("baseline_mean", staged_results)
        self.assertIn("baseline_median", staged_results)
        self.assertIn("candidate_ridge", staged_results)
        self.assertIn("candidate_huber", staged_results)
        self.assertIn("candidate_decision_tree", staged_results)
        self.assertIn("candidate_rf", staged_results)

        # In-fold Historical Mean baseline achieves 85.7% directional accuracy
        # because 12 of 14 historical incidents are negative drops
        self.assertAlmostEqual(staged_results["baseline_mean"]["directional_accuracy"], 0.8571, delta=0.01)

        # Verify Random Forest MAE is within expected range (~6.73%)
        rf_mae = staged_results["candidate_rf"]["mae"]
        self.assertAlmostEqual(rf_mae, 6.73, delta=0.5)

    # -------------------------------------------------------------------------
    # 10. Grouped Leakage Prevention (LOGO-CV)
    # -------------------------------------------------------------------------
    def test_10_grouped_leakage_prevention_logo_cv(self):
        """Verify Leave-One-Company-Out cross validation avoids intra-company leakage."""
        if self.df is None:
            self.skipTest("training_dataset.csv not available")

        logo_results = evaluate_leave_one_group_out(self.df)
        self.assertIn("mae", logo_results)
        self.assertIn("rmse", logo_results)
        self.assertIn("directional_accuracy", logo_results)
        self.assertGreater(logo_results["mae"], 0.0)
        self.assertGreater(logo_results["rmse"], 0.0)

    # -------------------------------------------------------------------------
    # 11. Prediction API Endpoint Round-Trip & Error Responses
    # -------------------------------------------------------------------------
    def test_11_api_predict_endpoint_roundtrip_and_serialization(self):
        """Test Flask /api/ml/predict with valid, edge-case, and malformed requests."""
        # Valid mild payload
        res = self.client.post("/api/ml/predict", json={
            "severity_score": 4.0,
            "users_affected_millions": 0.5,
            "data_sensitivity_score": 4.0
        })
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertEqual(data["status"], "SUCCESS")
        self.assertIn("predicted_change_percent", data)
        self.assertIn("tree_output_spread", data)

        # Invalid payload: negative users -> 400 Bad Request
        res_neg = self.client.post("/api/ml/predict", json={
            "severity_score": 5.0,
            "users_affected_millions": -10.0,
            "data_sensitivity_score": 5.0
        })
        self.assertEqual(res_neg.status_code, 400)
        data_neg = res_neg.get_json()
        self.assertEqual(data_neg["status"], "UNAVAILABLE")
        self.assertIn("error", data_neg)

        # Invalid payload: out of range severity -> 400 Bad Request
        res_sev = self.client.post("/api/ml/predict", json={
            "severity_score": 15.0,
            "users_affected_millions": 1.0,
            "data_sensitivity_score": 5.0
        })
        self.assertEqual(res_sev.status_code, 400)

        # Empty / non-json body -> 400
        res_empty = self.client.post("/api/ml/predict", data="")
        self.assertEqual(res_empty.status_code, 400)

    # -------------------------------------------------------------------------
    # 12. Safe Prediction Tracking in Isolated Test Database
    # -------------------------------------------------------------------------
    def test_12_prediction_tracking_isolation(self):
        """Verify prediction tracking works in an isolated DB without altering live tables."""
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tmp:
            tmp_db_path = tmp.name

        try:
            # Create schema in isolated test DB
            conn = sqlite3.connect(tmp_db_path)
            cursor = conn.cursor()
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS predictions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    threat_id INTEGER NOT NULL,
                    ticker TEXT NOT NULL,
                    prediction_timestamp TEXT NOT NULL,
                    predicted_direction TEXT NOT NULL,
                    predicted_impact_percent REAL NOT NULL,
                    confidence REAL,
                    actual_1d_return REAL,
                    actual_3d_return REAL,
                    actual_5d_return REAL,
                    status TEXT NOT NULL DEFAULT 'PENDING'
                        CHECK (status IN ('PENDING', 'EVALUATED')),
                    evaluated_at TEXT,
                    model_version TEXT DEFAULT 'v1.0.0',
                    target_definition TEXT DEFAULT 'Post-Disclosure Stock Price Change (% from T-1 Close to T+2 Close)',
                    review_needed INTEGER DEFAULT 0
                )
            """)
            conn.commit()
            conn.close()

            # Record a prediction in isolated DB
            pred_id = record_prediction(
                threat_id=1,
                ticker="TEST.US",
                prediction={
                    "predicted_change_percent": -5.2,
                    "predicted_direction": "DOWN",
                    "confidence": None
                },
                model_version=MODEL_VERSION,
                target_definition=TARGET_DEFINITION,
                db_name=tmp_db_path
            )
            self.assertIsNotNone(pred_id)

            # Query back from isolated DB
            conn = sqlite3.connect(tmp_db_path)
            cursor = conn.cursor()
            cursor.execute("SELECT ticker, predicted_impact_percent, predicted_direction, status, model_version FROM predictions WHERE id = ?", (pred_id,))
            row = cursor.fetchone()
            conn.close()

            self.assertIsNotNone(row)
            self.assertEqual(row[0], "TEST.US")
            self.assertEqual(row[1], -5.2)
            self.assertEqual(row[2], "DOWN")
            self.assertEqual(row[3], "PENDING")
            self.assertEqual(row[4], MODEL_VERSION)

        finally:
            if os.path.exists(tmp_db_path):
                os.remove(tmp_db_path)


if __name__ == "__main__":
    unittest.main()
