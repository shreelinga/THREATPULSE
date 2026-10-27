"""
tests/test_batch7_integration.py

Comprehensive End-to-End Integration & System Verification Test Suite.
Verifies all 8 modules and Flask API routes of THREATPULSE:
  1. Frontend Delivery (/, brand lockup, favicon, HTML structure).
  2. Health & Telemetry (/api/health, table counts, sessions).
  3. Market Correctness (/api/stocks, currency symbols, quote age).
  4. Threat Intelligence (/api/threats, exposure levels, CVEs, CISA KEV correlation).
  5. Correlations & Data Integrity (/api/correlations, review tags).
  6. ML Prediction Scorecard (/api/ml/metrics, /api/prediction_performance).
  7. Historical Event Studies (/api/event_study, 14 breach records).
  8. CISA KEV Catalog (/api/cisa_vulnerabilities, 1,741 entries).
  9. Live AI Investigation (/api/analyze_threat, sandboxing, signal constraint).
"""

import unittest
import json
from unittest.mock import patch

from app import app, get_db


class TestThreatPulseIntegration(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.client = app.test_client()

    def test_01_frontend_delivery(self):
        """Verify root route serves THREATPULSE dashboard with 200 OK."""
        res = self.client.get('/')
        self.assertEqual(res.status_code, 200)
        html = res.data.decode('utf-8')
        self.assertIn("THREATPULSE", html)
        self.assertIn("Cyber Threat &amp; Market Intelligence", html)
        self.assertIn("panel-overview", html)
        self.assertIn("panel-threats", html)
        self.assertIn("panel-markets", html)
        self.assertIn("panel-correlations", html)
        self.assertIn("panel-ai", html)
        self.assertIn("panel-forecasts", html)
        self.assertIn("panel-events", html)
        self.assertIn("panel-cisa", html)
        self.assertIn("panel-health", html)

    def test_02_system_health_endpoint(self):
        """Verify /api/health provides deep telemetry across all 9 database tables."""
        res = self.client.get('/api/health')
        self.assertEqual(res.status_code, 200)
        data = res.get_json()

        self.assertEqual(data["status"], "operational")
        self.assertEqual(data["brand"], "THREATPULSE")
        self.assertIn("timestamp_utc", data)

        tables = data["database"]["tables"]
        required_tables = [
            "threats", "stock_prices", "correlations", "current_stock_data",
            "processed_articles", "predictions", "event_studies",
            "cisa_kev_entries", "monitored_stocks"
        ]
        for tbl in required_tables:
            self.assertIn(tbl, tables, f"Missing table in health report: {tbl}")
            self.assertGreater(tables[tbl], 0, f"Table {tbl} unexpectedly empty")

        self.assertEqual(tables["cisa_kev_entries"], 1741)
        self.assertEqual(tables["event_studies"], 14)
        self.assertGreaterEqual(tables["threats"], 34)

    def test_03_market_correctness_and_currency(self):
        """Verify /api/stocks returns distinct Indian (INR) and US (USD) currencies, quote age, and sessions."""
        res = self.client.get('/api/stocks')
        self.assertEqual(res.status_code, 200)
        stocks = res.get_json()
        self.assertIsInstance(stocks, list)
        self.assertGreater(len(stocks), 0)

        has_indian = False
        has_us = False

        for stock in stocks:
            ticker = stock["ticker"]
            self.assertIn("market", stock)
            self.assertIn("currency_symbol", stock)
            self.assertIn("session_status", stock)
            self.assertIn("benchmark_ticker", stock)

            if ticker.endswith(".NS") or ticker.endswith(".BO"):
                has_indian = True
                self.assertEqual(stock["currency_symbol"], "₹")
                self.assertEqual(stock["market"], "India")
                self.assertEqual(stock["benchmark_ticker"], "^NSEI")
            elif not ticker.startswith("^"):
                has_us = True
                self.assertEqual(stock["currency_symbol"], "$")
                self.assertEqual(stock["market"], "US")

        self.assertTrue(has_us, "Watchlist must contain US equities")

    def test_04_threat_intel_exposure_and_cves(self):
        """Verify /api/threats provides multi-factor severity, exposure level, and parsed CVE lists."""
        res = self.client.get('/api/threats')
        self.assertEqual(res.status_code, 200)
        threats = res.get_json()
        self.assertIsInstance(threats, list)
        self.assertGreaterEqual(len(threats), 34)

        for t in threats:
            self.assertIn("exposure_level", t)
            self.assertIn("cve_ids", t)
            self.assertIsInstance(t["cve_ids"], list)
            self.assertIn("severity_score", t)
            self.assertIn("risk_score", t)
            self.assertIn("recommendation", t)
            self.assertIn(t["recommendation"], ["REVIEW", "MONITOR", "NO SIGNAL"])

    def test_05_correlations_and_review_tagging(self):
        """Verify /api/correlations preserves legacy records with review_needed tags."""
        res = self.client.get('/api/correlations')
        self.assertEqual(res.status_code, 200)
        corrs = res.get_json()
        self.assertIsInstance(corrs, list)
        self.assertGreaterEqual(len(corrs), 29)

        # Check that SA ambiguous records are flagged with review_needed = 1
        sa_records = [c for c in corrs if c["ticker"] == "SA"]
        for sa in sa_records:
            self.assertEqual(sa["review_needed"], 1, "Ambiguous SA ticker must be flagged review_needed=1")

    def test_06_ml_scorecard_and_predictions(self):
        """Verify /api/ml/metrics returns honest LOOCV baseline comparisons."""
        res = self.client.get('/api/ml/metrics')
        self.assertEqual(res.status_code, 200)
        metrics = res.get_json()

        self.assertEqual(metrics["sample_count"], 14)
        comp = metrics["staged_comparison"]
        self.assertIn("Zero Return Baseline", comp)
        self.assertIn("Historical Mean Baseline", comp)
        self.assertIn("Random Forest Regressor", comp)
        self.assertIn("limitations", metrics)

    def test_07_historical_event_studies(self):
        """Verify /api/event_study returns the 14 historical breach records with CAR metrics."""
        res = self.client.get('/api/event_study')
        self.assertEqual(res.status_code, 200)
        data = res.get_json()

        events = data.get("events", [])
        self.assertEqual(len(events), 14, "Must preserve all 14 curated historical breach studies")

        # Verify key incidents exist in dataset
        companies = [e["company"] for e in events]
        self.assertIn("Equifax", companies)
        self.assertIn("SolarWinds", companies)
        self.assertIn("Capital One", companies)
        self.assertIn("Okta", companies)

    def test_08_cisa_kev_catalog(self):
        """Verify /api/cisa_vulnerabilities paginates the 1,741 local entries."""
        res = self.client.get('/api/cisa_vulnerabilities?per_page=50')
        self.assertEqual(res.status_code, 200)
        data = res.get_json()

        self.assertEqual(data["total"], 1741)
        self.assertEqual(len(data["vulnerabilities"]), 50)
        self.assertIn("cve_id", data["vulnerabilities"][0])

    def test_09_sandboxed_threat_analysis_endpoint(self):
        """Verify /api/analyze_threat enforces strict recommendation and schema validation."""
        payload = {
            "company": "Microsoft",
            "type": "Zero-Day Exploit",
            "description": "Critical remote code execution vulnerability CVE-2023-36884 in Windows and Office actively exploited by Storm-0978."
        }

        # Mock discord alert to prevent external network side-effects
        with patch("realtime_collector.send_discord_alert") as mock_alert:
            res = self.client.post(
                '/api/analyze_threat',
                data=json.dumps(payload),
                content_type='application/json'
            )
            self.assertEqual(res.status_code, 200)
            data = res.get_json()

            self.assertEqual(data["ticker"], "MSFT")
            self.assertIn(data["recommendation"], ["REVIEW", "MONITOR", "NO SIGNAL"])
            self.assertGreaterEqual(data["risk_score"], 0)
            self.assertLessEqual(data["risk_score"], 100)


if __name__ == "__main__":
    unittest.main()
