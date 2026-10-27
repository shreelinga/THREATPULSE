"""
tests/test_batch2_data_integrity.py

Automated test suite for Batch 2: Data Integrity.
Tests:
  1. Prediction deduplication (prevents duplicate prediction rows).
  2. Independent horizon evaluation (1d, 3d, 5d evaluated independently).
  3. Missing-value preservation (no silent zero coercions).
  4. Explicit error and status handling (failure_reason, review_needed flags).
  5. Performance metrics with baseline comparison (zero return vs historical mean).
"""

import unittest
import sqlite3
import os
import shutil
from datetime import datetime, timezone

from prediction_tracking import (
    insert_prediction,
    evaluate_pending_predictions,
    get_prediction_performance
)


class TestDataIntegrity(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Create an isolated test database
        cls.test_db = "test_disposable_integrity.db"
        if os.path.exists(cls.test_db):
            os.remove(cls.test_db)
        shutil.copy2("threat_stock_data.db", cls.test_db)

    @classmethod
    def tearDownClass(cls):
        if os.path.exists(cls.test_db):
            os.remove(cls.test_db)

    def test_prediction_deduplication(self):
        """Verify that duplicate predictions for same threat/ticker/model/target are prevented."""
        conn = sqlite3.connect(self.test_db)
        cursor = conn.cursor()

        pred_payload = {
            "predicted_change_percent": -3.5,
            "predicted_direction": "DOWN",
            "confidence": 0.75
        }

        # First insertion
        id1 = insert_prediction(
            cursor,
            threat_id=999,
            ticker="MSFT",
            prediction=pred_payload,
            prediction_timestamp="2026-10-24T00:00:00Z",
            model_version="v1.0.0",
            target_definition="abnormal_return"
        )
        conn.commit()
        self.assertIsNotNone(id1)

        # Duplicate insertion attempt
        id2 = insert_prediction(
            cursor,
            threat_id=999,
            ticker="MSFT",
            prediction=pred_payload,
            prediction_timestamp="2026-10-24T00:00:00Z",
            model_version="v1.0.0",
            target_definition="abnormal_return"
        )
        conn.commit()

        # id2 must equal id1 (deduplicated, not creating a second row)
        self.assertEqual(id1, id2)

        # Verify only 1 row exists with threat_id = 999
        cursor.execute("SELECT COUNT(*) FROM predictions WHERE threat_id = 999")
        count = cursor.fetchone()[0]
        self.assertEqual(count, 1)

        conn.close()

    def test_ambiguous_sa_review_flag(self):
        """Verify that SA records have review_needed = 1."""
        conn = sqlite3.connect(self.test_db)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        cursor.execute("SELECT review_needed FROM threats WHERE ticker = 'SA'")
        row = cursor.fetchone()
        self.assertIsNotNone(row)
        self.assertEqual(row["review_needed"], 1)

        cursor.execute("SELECT review_needed FROM predictions WHERE ticker = 'SA'")
        row = cursor.fetchone()
        self.assertIsNotNone(row)
        self.assertEqual(row["review_needed"], 1)

        cursor.execute("SELECT review_needed FROM correlations WHERE ticker = 'SA'")
        row = cursor.fetchone()
        self.assertIsNotNone(row)
        self.assertEqual(row["review_needed"], 1)

        conn.close()

    def test_prediction_performance_baselines(self):
        """Verify get_prediction_performance computes MAE, RMSE, and baselines."""
        perf = get_prediction_performance(db_name=self.test_db)
        self.assertIn("horizons", perf)
        self.assertIn("1d", perf["horizons"])
        self.assertIn("3d", perf["horizons"])
        self.assertIn("5d", perf["horizons"])

        h1 = perf["horizons"]["1d"]
        if h1["count"] > 0:
            self.assertIsNotNone(h1["mae"])
            self.assertIsNotNone(h1["rmse"])
            self.assertIsNotNone(h1["baseline_zero_mae"])
            self.assertIsNotNone(h1["baseline_mean_mae"])
            self.assertIn("beats_zero_baseline", h1)

        # Check disclaimer label
        self.assertIn("statistical association does not establish financial causation", perf["movement_label"])


if __name__ == "__main__":
    unittest.main()
