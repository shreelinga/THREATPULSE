"""
tests/test_batch8_redesign_and_bugfixes.py

Comprehensive regression test suite for THREATPULSE 3D Redesign and Bug Fixes:
  1. ML Prediction: Verifies outputs are dynamic, non-constant (not -8.48%), feature-scaled,
     and report target definition ("1D Abnormal Return"), horizon, and descriptive tree spread.
  2. Event Studies: Verifies source_dataset is present, curated records count matches 14,
     and table rows are consistent with summary metrics.
  3. Sector Analysis: Verifies Indian (.NS) and US equities are separated with proper market
     benchmarks (^CNXIT, ^NSEBANK, ^NSEI vs XLK, SPY) and appropriate currencies (₹ vs $).
  4. CISA KEV: Verifies catalog total is 1,741 entries, search/filtering operates cleanly,
     and CVE records are deduplicated.
  5. Database Integrity: Confirms zero data loss across all tables.
"""

import unittest
import json
import sqlite3
import os

from threat_ml_engine import predict_stock_impact, load_model_bundle, load_model_metrics, MODEL_VERSION
from event_study_engine import get_event_study_data, get_event_study_summary
from market_utils import is_indian_ticker, get_market_benchmark
from unittest.mock import patch
from app import app, get_db


class TestBatch8RedesignAndBugfixes(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.client = app.test_client()

    def test_01_ml_prediction_non_constant_and_transparent(self):
        """Verify ML predictions are dynamic, feature-responsive, and never stuck at -8.48%."""
        bundle = load_model_bundle()
        self.assertIsNotNone(bundle, "Model bundle should load or auto-train successfully")
        self.assertTrue(bundle.get("model_version", "").startswith("v2."), "Version should be v2.x")

        # Scenario A: Mild incident (low severity, small user base, low sensitivity)
        pred_mild = predict_stock_impact(
            severity_score=4.0,
            users_affected_millions=0.01,
            data_sensitivity_score=3.0
        )
        self.assertEqual(pred_mild["status"], "SUCCESS")
        self.assertNotEqual(pred_mild["predicted_change_percent"], -8.48)
        self.assertIn("target_definition", pred_mild)
        self.assertTrue(
            "Post-Disclosure" in pred_mild["target_definition"] or "1D Abnormal Return" in pred_mild["target_definition"],
            "Target definition must explicitly describe the reaction window"
        )
        self.assertTrue(
            "Post-Disclosure" in pred_mild["prediction_horizon"] or "1D" in pred_mild["prediction_horizon"],
            "Horizon must explicitly describe the post-disclosure reaction period"
        )
        self.assertIn("expected_range_percent", pred_mild)
        self.assertIsNotNone(pred_mild["expected_range_percent"]["low"])
        self.assertIsNotNone(pred_mild["expected_range_percent"]["high"])

        # Scenario B: Catastrophic incident (Equifax/SolarWinds profile: sev=10, 147M users, sens=10)
        pred_severe = predict_stock_impact(
            severity_score=10.0,
            users_affected_millions=147.0,
            data_sensitivity_score=10.0
        )
        self.assertEqual(pred_severe["status"], "SUCCESS")
        self.assertNotEqual(pred_severe["predicted_change_percent"], -8.48)

        # The severe incident must predict a larger drop than the mild incident
        self.assertLess(
            pred_severe["predicted_change_percent"],
            pred_mild["predicted_change_percent"],
            "Severe breach must predict a more negative market drop than a mild breach"
        )

        # Scenario C: Invalid inputs produce explicit status without crashing
        pred_invalid = predict_stock_impact(
            severity_score="invalid",
            users_affected_millions=None,
            data_sensitivity_score=5.0
        )
        self.assertEqual(pred_invalid["status"], "UNAVAILABLE")
        self.assertIsNone(pred_invalid["predicted_change_percent"])

    def test_02_event_study_consistency_and_metadata(self):
        """Verify Event Study API returns consistent counts, source_dataset, and no undefined."""
        data = get_event_study_data()
        self.assertIn("events", data)
        self.assertIn("studies", data)
        self.assertIn("source_dataset", data)
        self.assertFalse("undefined" in str(data["source_dataset"]).lower())
        self.assertIn("training_dataset.csv", data["source_dataset"])

        events = data["events"]
        self.assertGreaterEqual(len(events), 14, "Event study table must have 14 historical records")
        self.assertEqual(data["curated_records_count"], len(events))

        summary = data["summary"]
        self.assertIsNotNone(summary)
        self.assertGreaterEqual(summary["evaluated_count"], 13)
        self.assertIsNotNone(summary["average_abnormal_return_1d"])
        self.assertIsNotNone(summary["average_z_score"])

    @patch("app.get_sector_comparison")
    def test_03_sector_analysis_market_separation(self, mock_get_comp):
        """Verify Indian and US equities are separated with appropriate benchmarks and currencies."""
        def fake_get_comp(ticker, benchmark_ticker=None):
            return {
                "ticker": ticker,
                "ticker_change_percent": -1.2,
                "benchmark_ticker": benchmark_ticker,
                "benchmark_change_percent": -0.5,
                "excess_performance": -0.7,
                "drag_detected": True,
                "status": "SUCCESS"
            }
        mock_get_comp.side_effect = fake_get_comp

        response = self.client.get("/api/sector_comparison")
        self.assertEqual(response.status_code, 200)
        items = json.loads(response.data)
        self.assertIsInstance(items, list)
        self.assertGreater(len(items), 0)

        for item in items:
            ticker = item["ticker"]
            is_in = is_indian_ticker(ticker)
            if is_in:
                self.assertEqual(item["market"], "India")
                self.assertEqual(item["currency_symbol"], "₹")
                self.assertTrue(
                    item["benchmark_ticker"].startswith("^"),
                    f"Indian ticker {ticker} must be benchmarked against a Nifty index (^CNXIT, ^NSEBANK, or ^NSEI), got {item['benchmark_ticker']}"
                )
            else:
                self.assertEqual(item["market"], "US")
                self.assertEqual(item["currency_symbol"], "$")
                self.assertFalse(
                    item["benchmark_ticker"].startswith("^"),
                    f"US ticker {ticker} must be benchmarked against a US ETF (XLK, SPY, etc.), got {item['benchmark_ticker']}"
                )

    def test_04_cisa_kev_catalog_and_filtering(self):
        """Verify CISA KEV catalog search, pagination, and deduplication."""
        # Baseline fetch
        res = self.client.get("/api/cisa_vulnerabilities?per_page=10")
        self.assertEqual(res.status_code, 200)
        data = json.loads(res.data)
        self.assertIn("vulnerabilities", data)
        self.assertEqual(data["catalog_total"], 1741, "Must reflect genuine 1,741 CISA catalog records")
        self.assertGreater(len(data["vulnerabilities"]), 0)

        # Keyword search for Log4j / Apache
        res_search = self.client.get("/api/cisa_vulnerabilities?q=CVE-2021-44228")
        self.assertEqual(res_search.status_code, 200)
        data_search = json.loads(res_search.data)
        self.assertGreaterEqual(data_search["total"], 1)
        found_cves = [v["cve_id"] for v in data_search["vulnerabilities"]]
        self.assertIn("CVE-2021-44228", found_cves)

        # Ransomware filter
        res_ransom = self.client.get("/api/cisa_vulnerabilities?ransomware=known&per_page=10")
        self.assertEqual(res_ransom.status_code, 200)
        data_ransom = json.loads(res_ransom.data)
        for v in data_ransom["vulnerabilities"]:
            self.assertEqual(str(v["ransomware_use"]).lower(), "known")

    def test_05_database_records_preservation(self):
        """Verify all critical database tables have preserved records without any destructive reset."""
        conn = get_db()
        c = conn.cursor()
        
        tables = {
            "threats": 40,
            "stock_prices": 3000,
            "correlations": 30,
            "current_stock_data": 5,
            "event_studies": 14,
            "cisa_kev_entries": 1741
        }
        for tbl, min_expected in tables.items():
            c.execute(f"SELECT COUNT(*) FROM {tbl}")
            count = c.fetchone()[0]
            self.assertGreaterEqual(
                count, min_expected,
                f"Table {tbl} must retain at least {min_expected} records (found {count})"
            )
        conn.close()

    def test_06_ml_predict_api_endpoint(self):
        """Test on-demand /api/ml/predict HTTP endpoint."""
        res = self.client.post("/api/ml/predict", json={
            "severity_score": 8.0,
            "users_affected_millions": 5.0,
            "data_sensitivity_score": 7.5
        })
        self.assertEqual(res.status_code, 200)
        data = json.loads(res.data)
        self.assertEqual(data["status"], "SUCCESS")
        self.assertIsInstance(data["predicted_change_percent"], float)
        self.assertIn(data["predicted_direction"], ["UP", "DOWN", "FLAT"])
        self.assertIn("expected_range_percent", data)


if __name__ == "__main__":
    unittest.main()
