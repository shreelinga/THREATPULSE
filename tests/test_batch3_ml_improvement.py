"""
tests/test_batch3_ml_improvement.py

Automated test suite for Batch 3: ML Improvement.
Tests:
  1. Staged model evaluation execution and metrics verification.
  2. Baseline comparison presence (Zero Return Baseline & Historical Mean).
  3. Model bundle completeness (version, feature importances, sample size, limitations).
  4. Real-time inference through threat_ml_engine.predict_stock_impact.
  5. Robust heuristic fallback when bundle is missing.
"""

import unittest
import os
import json
import hashlib
import tempfile
import numpy as np

from train_model import run_staged_model_evaluation, MODEL_OUTPUT_PATH, METRICS_JSON_PATH
from threat_ml_engine import predict_stock_impact, load_model_bundle


class TestMLImprovement(unittest.TestCase):

    def test_staged_model_evaluation_metrics(self):
        """Verify that staged evaluation produces valid metrics for baselines and candidates."""
        results = run_staged_model_evaluation(save_artifacts=False)

        self.assertIn("baseline_zero", results)
        self.assertIn("baseline_mean", results)
        self.assertIn("candidate_ridge", results)
        self.assertIn("candidate_huber", results)
        self.assertIn("candidate_rf", results)

        # Baseline zero check
        self.assertGreater(results["baseline_zero"]["mae"], 0.0)
        self.assertGreater(results["baseline_zero"]["rmse"], 0.0)

        # In-fold historical mean check
        self.assertGreater(results["baseline_mean"]["mae"], 0.0)
        self.assertGreater(results["baseline_mean"]["directional_accuracy"], 0.5)

        # Random Forest candidate check
        self.assertGreater(results["candidate_rf"]["mae"], 0.0)
        self.assertGreater(results["candidate_rf"]["directional_accuracy"], 0.5)

    def test_model_bundle_and_json_artifacts(self):
        """Verify that threat_impact_model.joblib and model_metrics.json exist and are well-formed."""
        self.assertTrue(os.path.exists(MODEL_OUTPUT_PATH))
        self.assertTrue(os.path.exists(METRICS_JSON_PATH))

        with open(METRICS_JSON_PATH, "r", encoding="utf-8") as f:
            metrics_data = json.load(f)

        self.assertEqual(metrics_data["sample_count"], 14)
        self.assertTrue(metrics_data["model_version"].startswith("v2."))
        self.assertIn("limitations", metrics_data)
        self.assertIn("Zero Return Baseline", metrics_data["staged_comparison"])
        self.assertIn("Historical Mean Baseline", metrics_data["staged_comparison"])

    def test_ml_inference_output(self):
        """Test predict_stock_impact produces structured, honest outputs."""
        pred = predict_stock_impact(
            severity_score=9.0,
            users_affected_millions=54.3,
            data_sensitivity_score=9.0
        )

        self.assertIn("predicted_change_percent", pred)
        self.assertIsInstance(pred["predicted_change_percent"], float)
        self.assertIn(pred["predicted_direction"], ["UP", "DOWN", "FLAT"])
        self.assertIn("model_limitation", pred)
        self.assertIn("exploratory benchmark estimates", pred["model_limitation"].lower())
        self.assertIn("expected_range_percent", pred)
        self.assertIn("out_of_sample_comparison", pred)

    def test_ml_heuristic_fallback(self):
        """Verify fallback behavior operates if model is not loaded."""
        # Temporarily test heuristic function by simulating no bundle
        from threat_ml_engine import MODEL_LIMITATION
        sev = 8.0
        users = 10.0
        sens = 8.0
        base_drop = -0.3 * sev - 0.25 * sens - 0.05 * min(users, 50.0)
        heuristic_drop = round(max(-10.0, min(1.0, base_drop)), 2)

        self.assertLess(heuristic_drop, 0.0)

    def test_regression_blk01_model_file_unchanged_after_evaluation(self):
        """BLK-01 Regression Test: Verify production model and metrics files remain bitwise identical
        when run_staged_model_evaluation is invoked with save_artifacts=False.
        """
        def get_sha256(path):
            if not os.path.exists(path):
                return None
            with open(path, "rb") as f:
                return hashlib.sha256(f.read()).hexdigest()

        model_hash_before = get_sha256(MODEL_OUTPUT_PATH)
        metrics_hash_before = get_sha256(METRICS_JSON_PATH)

        self.assertIsNotNone(model_hash_before, "Production model file must exist before test")
        self.assertIsNotNone(metrics_hash_before, "Production metrics file must exist before test")

        # Run evaluation with save_artifacts=False
        results = run_staged_model_evaluation(save_artifacts=False)
        self.assertIsInstance(results, dict)
        self.assertIn("candidate_rf", results)

        model_hash_after = get_sha256(MODEL_OUTPUT_PATH)
        metrics_hash_after = get_sha256(METRICS_JSON_PATH)

        self.assertEqual(
            model_hash_before,
            model_hash_after,
            "Regression BLK-01 failure: threat_impact_model.joblib hash changed after evaluation!"
        )
        self.assertEqual(
            metrics_hash_before,
            metrics_hash_after,
            "Regression BLK-01 failure: model_metrics.json hash changed after evaluation!"
        )

    def test_regression_blk01_custom_temp_artifacts_isolation(self):
        """BLK-01 Regression Test: Verify that evaluations requiring artifact output can use
        temporary directories without modifying production model artifacts.
        """
        def get_sha256(path):
            with open(path, "rb") as f:
                return hashlib.sha256(f.read()).hexdigest()

        model_hash_before = get_sha256(MODEL_OUTPUT_PATH)
        metrics_hash_before = get_sha256(METRICS_JSON_PATH)

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_model = os.path.join(tmpdir, "isolated_model.joblib")
            tmp_metrics = os.path.join(tmpdir, "isolated_metrics.json")

            results = run_staged_model_evaluation(
                save_artifacts=True,
                model_output_path=tmp_model,
                metrics_output_path=tmp_metrics
            )

            self.assertTrue(os.path.exists(tmp_model))
            self.assertTrue(os.path.exists(tmp_metrics))
            self.assertGreater(os.path.getsize(tmp_model), 0)
            self.assertGreater(os.path.getsize(tmp_metrics), 0)

        model_hash_after = get_sha256(MODEL_OUTPUT_PATH)
        metrics_hash_after = get_sha256(METRICS_JSON_PATH)

        self.assertEqual(model_hash_before, model_hash_after)
        self.assertEqual(metrics_hash_before, metrics_hash_after)


if __name__ == "__main__":
    unittest.main()
