"""
Test Phase 1: Target Terminology Standardization
Verifies that:
1. Raw return is stock-only return: (P[T]-P[T-1])/P[T-1] or (P[T+horizon]-P[T-1])/P[T-1].
2. Abnormal return is stock return minus benchmark return (^NSEI for .NS/.BO, SPY/ETF otherwise).
3. The ML model target is explicitly identified as 'raw_return'.
4. Database columns, predictions, event studies, risk scoring, and APIs provide raw_return aliases without breaking existing fields.
"""
import unittest
import sqlite3
import json
import os
import shutil
import tempfile

class TestPhase1TargetTerminology(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Create a temp copy of the DB to avoid touching threat_stock_data.db
        cls.test_dir = tempfile.mkdtemp()
        cls.db_copy_path = os.path.join(cls.test_dir, "test_threat_stock_data.db")
        if os.path.exists("threat_stock_data.db"):
            shutil.copy2("threat_stock_data.db", cls.db_copy_path)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.test_dir, ignore_errors=True)

    def test_database_target_definition_and_columns(self):
        """Verify DB predictions and event_studies have standardized raw_return columns and target_definition."""
        conn = sqlite3.connect(self.db_copy_path)
        cursor = conn.cursor()

        # Check predictions columns
        cursor.execute("PRAGMA table_info(predictions)")
        pred_cols = [row[1] for row in cursor.fetchall()]
        self.assertIn("raw_return_1d", pred_cols)
        self.assertIn("raw_return_3d", pred_cols)
        self.assertIn("raw_return_5d", pred_cols)
        self.assertIn("actual_1d_return", pred_cols) # Backward compatibility

        # Check that existing predictions have target_definition = 'raw_return'
        cursor.execute("SELECT DISTINCT target_definition FROM predictions")
        target_defs = [r[0] for r in cursor.fetchall()]
        for t_def in target_defs:
            self.assertEqual(t_def, "raw_return", f"Expected 'raw_return' in predictions, got: {t_def}")

        # Check event_studies columns
        cursor.execute("PRAGMA table_info(event_studies)")
        event_cols = [row[1] for row in cursor.fetchall()]
        self.assertIn("raw_return_1d", event_cols)
        self.assertIn("raw_return_3d", event_cols)
        self.assertIn("raw_return_5d", event_cols)
        self.assertIn("company_return_1d", event_cols) # Backward compatibility
        self.assertIn("abnormal_return_1d", event_cols) # Benchmark-adjusted

        conn.close()

    def test_ml_model_target_is_raw_return(self):
        """Verify ML model artifacts and metrics state target_type as raw_return."""
        import threat_ml_engine
        import joblib

        bundle = joblib.load("threat_impact_model.joblib")
        self.assertEqual(bundle.get("target_type"), "raw_return")
        self.assertIn("Stock Price Change", bundle.get("target_definition", ""))

        if os.path.exists("model_metrics.json"):
            with open("model_metrics.json", "r") as f:
                metrics = json.load(f)
            self.assertEqual(metrics.get("target_type"), "raw_return")
            self.assertIn("Stock Price Change", metrics.get("target_definition", ""))

        # Test predict_stock_impact output dictionary
        res = threat_ml_engine.predict_stock_impact(8.5, 10.0, 9.0)
        self.assertEqual(res.get("target_type"), "raw_return")
        self.assertIn("raw_return_predicted", res)
        self.assertIn("predicted_change_percent", res)


    def test_prediction_tracking_target_standardization(self):
        """Verify prediction_tracking defaults and aliases."""
        import prediction_tracking

        perf = prediction_tracking.get_prediction_performance()
        self.assertEqual(perf.get("target_type"), "raw_return")
        self.assertIn("Stock-Only Raw Return", perf.get("target_definition", ""))
        self.assertIn("predictions", perf)
        if perf["predictions"]:
            p = perf["predictions"][0]
            self.assertIn("raw_return_1d", p)
            self.assertIn("raw_return_3d", p)
            self.assertIn("raw_return_5d", p)
            self.assertIn("actual_1d_return", p) # Kept for alias
            self.assertEqual(p.get("target_definition"), "raw_return")

    def test_risk_scoring_sector_comparison_aliases(self):
        """Verify sector comparison output has both raw_return and abnormal_return aliases."""
        from risk_scoring_engine import get_sector_comparison

        # Test sector comparison with mock/fallback
        res = get_sector_comparison("INVALID_TICKER_XYZ_123")
        self.assertIn("raw_return", res)
        self.assertIn("abnormal_return", res)
        self.assertIn("benchmark_return", res)
        # Check backward compatibility aliases
        self.assertIn("ticker_change_percent", res)
        self.assertIn("excess_performance", res)
        self.assertEqual(res["raw_return"], res["ticker_change_percent"])
        self.assertEqual(res["abnormal_return"], res["excess_performance"])

    def test_event_study_terminology(self):
        """Verify event study summary and data return standardized raw_return and abnormal_return."""
        import event_study_engine

        summary = event_study_engine.get_event_study_summary()
        self.assertIn("average_raw_return_1d", summary)
        self.assertIn("average_abnormal_return_1d", summary)

        data = event_study_engine.get_event_study_data()
        events = data.get("events", [])
        if events:
            item = events[0]
            self.assertIn("raw_return_1d", item)
            self.assertIn("company_return_1d", item)
            self.assertIn("abnormal_return_1d", item)
            self.assertEqual(item["raw_return_1d"], item["company_return_1d"])


if __name__ == "__main__":
    unittest.main()

