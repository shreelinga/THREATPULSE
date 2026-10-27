"""
tests/test_batch1_market_correctness.py

Automated test suite for Batch 1: Market Correctness.
Tests:
  1. Indian and US broad and sector benchmark mapping.
  2. Missing benchmark history handling (ensures no silent zero substitution).
  3. Currency formatting (INR: ₹, USD: $, None: --).
  4. Timezone-aware timestamps and quote age calculation.
  5. Ambiguous ticker resolution safeguards (e.g. SA rejection vs valid short tickers).
  6. Event study engine backward compatibility (14 US rows intact).
"""

import unittest
import sqlite3
import os
from datetime import datetime, timezone

from market_utils import (
    get_market_benchmark,
    is_indian_ticker,
    format_currency,
    format_market_timestamp,
    calculate_quote_age_minutes,
    get_market_session_status,
    validate_ticker_query
)
from risk_scoring_engine import resolve_ticker, get_sector_comparison
from event_study_engine import _calculate_event


class TestMarketCorrectness(unittest.TestCase):

    def test_indian_benchmark_resolution(self):
        """Test Indian IT, Banking, Auto, and Broad Market Benchmarks."""
        # IT
        infy = get_market_benchmark("INFY.NS")
        self.assertEqual(infy["market"], "INDIA")
        self.assertEqual(infy["broad_benchmark"], "^NSEI")
        self.assertEqual(infy["sector_benchmark"], "^CNXIT")
        self.assertEqual(infy["currency"], "INR")
        self.assertEqual(infy["currency_symbol"], "₹")
        self.assertEqual(infy["timezone"], "Asia/Kolkata")

        # Banking
        hdfc = get_market_benchmark("HDFCBANK.NS")
        self.assertEqual(hdfc["broad_benchmark"], "^NSEI")
        self.assertEqual(hdfc["sector_benchmark"], "^NSEBANK")

        # Auto
        tata = get_market_benchmark("TATAMOTORS.NS")
        self.assertEqual(tata["broad_benchmark"], "^NSEI")
        self.assertEqual(tata["sector_benchmark"], "^CNXAUTO")

        # Conglomerate / Broad Fallback
        rel = get_market_benchmark("RELIANCE.NS")
        self.assertEqual(rel["broad_benchmark"], "^NSEI")
        self.assertEqual(rel["sector_benchmark"], "^NSEI")

    def test_us_benchmark_resolution(self):
        """Test US Sector Benchmarks (XLK, XLF, XLC, XLY) and broad SPY."""
        # Tech
        msft = get_market_benchmark("MSFT")
        self.assertEqual(msft["market"], "US")
        self.assertEqual(msft["broad_benchmark"], "SPY")
        self.assertEqual(msft["sector_benchmark"], "XLK")
        self.assertEqual(msft["currency_symbol"], "$")
        self.assertEqual(msft["timezone"], "America/New_York")

        # Financials - Must not be forced to XLK
        jpm = get_market_benchmark("JPM")
        self.assertEqual(jpm["sector_benchmark"], "XLF")

        # Communication
        goog = get_market_benchmark("GOOGL")
        self.assertEqual(goog["sector_benchmark"], "XLC")

        # Consumer Discretionary
        amzn = get_market_benchmark("AMZN")
        self.assertEqual(amzn["sector_benchmark"], "XLY")

    def test_currency_formatting(self):
        """Test consistent currency formatting across INR and USD."""
        self.assertEqual(format_currency(1234.56, "INFY.NS"), "₹1,234.56")
        self.assertEqual(format_currency(150.25, "AAPL"), "$150.25")
        self.assertEqual(format_currency(5000.0, currency_code="INR"), "₹5,000.00")
        self.assertEqual(format_currency(5000.0, currency_code="USD"), "$5,000.00")
        # Missing values must never format as zero
        self.assertEqual(format_currency(None, "INFY.NS"), "--")
        self.assertEqual(format_currency(None), "--")

    def test_missing_data_does_not_return_zero(self):
        """Verify that get_sector_comparison returns None and status UNAVAILABLE for invalid tickers."""
        res = get_sector_comparison("INVALID_TICKER_XYZ_123")
        self.assertIsNone(res["ticker_change_percent"])
        self.assertIsNone(res["excess_performance"])
        self.assertIn(res["status"], ["UNAVAILABLE", "ERROR"])

    def test_ambiguous_short_ticker_safeguards(self):
        """Verify that 1-2 letter acronyms like 'SA' are flagged and not mapped to arbitrary stocks."""
        is_valid_sa, reason_sa = validate_ticker_query("SA")
        self.assertFalse(is_valid_sa)
        self.assertIn("Ambiguous token", reason_sa)

        # resolve_ticker should return resolved=False and review_needed=True for 'Sa'
        res_sa = resolve_ticker("Sa")
        self.assertFalse(res_sa["resolved"])
        self.assertTrue(res_sa.get("review_needed"))

        # Valid short tickers must still pass
        is_valid_v, _ = validate_ticker_query("V")
        self.assertTrue(is_valid_v)

        is_valid_ba, _ = validate_ticker_query("BA")
        self.assertTrue(is_valid_ba)

    def test_timezone_aware_timestamp_and_quote_age(self):
        """Test timezone-aware formatting and quote age calculation."""
        dt = datetime(2026, 10, 24, 10, 0, 0, tzinfo=timezone.utc)
        ist_str = format_market_timestamp(dt, market="INDIA")
        self.assertIn("IST", ist_str)

        et_str = format_market_timestamp(dt, market="US")
        self.assertIn("ET", et_str)

        # Quote age
        age = calculate_quote_age_minutes(ist_str, market="INDIA")
        self.assertIsNotNone(age)
        self.assertGreaterEqual(age, 0.0)

        # Session status
        status_ind = get_market_session_status("INDIA")
        self.assertIn("status", status_ind)
        self.assertIn("session", status_ind)

    def test_event_study_backward_compatibility(self):
        """Verify all 14 historical event studies exist and retain evaluated metrics."""
        db_path = "threat_stock_data.db"
        self.assertTrue(os.path.exists(db_path))
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        cursor.execute("SELECT COUNT(*) FROM event_studies")
        count = cursor.fetchone()[0]
        self.assertEqual(count, 14, f"Expected 14 historical event studies, found {count}")

        cursor.execute("SELECT ticker, abnormal_return_1d, cumulative_abnormal_return_5d, status FROM event_studies")
        rows = cursor.fetchall()
        evaluated_count = sum(1 for r in rows if r["status"] == "EVALUATED")
        self.assertGreaterEqual(evaluated_count, 10, "Historical evaluated event studies must be preserved")

        # Verify historical SA records are preserved in database
        cursor.execute("SELECT COUNT(*) FROM threats WHERE ticker = 'SA'")
        sa_threats = cursor.fetchone()[0]
        self.assertEqual(sa_threats, 1, "Historical SA threat record must be preserved")

        cursor.execute("SELECT COUNT(*) FROM correlations WHERE ticker = 'SA'")
        sa_corr = cursor.fetchone()[0]
        self.assertEqual(sa_corr, 1, "Historical SA correlation record must be preserved")

        cursor.execute("SELECT COUNT(*) FROM predictions WHERE ticker = 'SA'")
        sa_pred = cursor.fetchone()[0]
        self.assertEqual(sa_pred, 1, "Historical SA prediction record must be preserved")

        conn.close()


if __name__ == "__main__":
    unittest.main()
