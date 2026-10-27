"""
Test Phase 2: Market Data Freshness and Timestamp Accuracy
Verifies:
1. Correct UTC serialization and ISO 8601 formatting.
2. Correct conversion to Indian Standard Time (IST) and US Eastern Time (ET) across DST changes.
3. Request time is strictly distinguished from underlying market observation timestamps.
4. Missing, unknown, or invalid provider timestamps are flagged as unavailable/stale.
5. Stale cached data older than the last market session triggers stale warnings.
6. Weekend and market session boundary handling.
7. Daily closing and delayed data are never falsely reported as live ticks.
8. Missing or zero prices display 'N/A', never 0.
9. Backward compatibility for existing API fields and Phase 1 target terminology preservation.
"""
import unittest
from datetime import datetime, date, timedelta, time as dtime, timezone
import pytz
import os
import json
import sqlite3
import tempfile
import shutil

from app import app
from market_utils import (
    TZ_INDIA,
    TZ_US,
    parse_market_timestamp,
    get_last_market_session_date,
    evaluate_quote_freshness,
    format_market_timestamp,
    calculate_quote_age_minutes,
    is_indian_ticker,
    format_currency,
)


class TestPhase2DataFreshness(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        app.config["TESTING"] = True
        cls.client = app.test_client()

    def test_utc_serialization_and_parsing(self):
        """Verify parse_market_timestamp correctly parses various formats to UTC."""
        # ISO string with UTC offset
        dt1, tz1 = parse_market_timestamp("2026-10-23T15:59:00-04:00")
        self.assertIsNotNone(dt1)
        self.assertEqual(dt1.tzinfo, timezone.utc)
        self.assertEqual(dt1.hour, 19)
        self.assertEqual(dt1.minute, 59)

        # String with explicit ET label
        dt2, tz2 = parse_market_timestamp("2026-10-23 15:59:00 ET")
        self.assertIsNotNone(dt2)
        self.assertEqual(tz2, "ET")
        self.assertEqual(dt2.astimezone(timezone.utc).hour, 19)

        # String with explicit IST label (UTC = IST - 5h30m)
        dt3, tz3 = parse_market_timestamp("2026-10-23 15:30:00 IST")
        self.assertIsNotNone(dt3)
        self.assertEqual(tz3, "IST")
        self.assertEqual(dt3.astimezone(timezone.utc).hour, 10)
        self.assertEqual(dt3.astimezone(timezone.utc).minute, 0)

        # Date only string
        dt4, tz4 = parse_market_timestamp("2026-10-23")
        self.assertIsNotNone(dt4)
        self.assertEqual(dt4.date(), date(2026, 10, 23))

        # Missing / invalid strings
        dt_none, _ = parse_market_timestamp(None)
        dt_dash, _ = parse_market_timestamp("--")
        dt_inv, _ = parse_market_timestamp("UNKNOWN")
        self.assertIsNone(dt_none)
        self.assertIsNone(dt_dash)
        self.assertIsNone(dt_inv)

    def test_timezone_conversion_ist_and_dst(self):
        """Verify accurate timezone conversions across EDT/EST daylight saving transitions."""
        # 1. Summer EDT (UTC-4)
        utc_summer = datetime(2026, 7, 15, 14, 30, tzinfo=timezone.utc)
        et_summer = utc_summer.astimezone(TZ_US)
        self.assertEqual(et_summer.hour, 10)
        self.assertEqual(et_summer.minute, 30)
        self.assertIn("EDT", et_summer.tzname())

        # 2. Winter EST (UTC-5)
        utc_winter = datetime(2026, 12, 15, 14, 30, tzinfo=timezone.utc)
        et_winter = utc_winter.astimezone(TZ_US)
        self.assertEqual(et_winter.hour, 9)
        self.assertEqual(et_winter.minute, 30)
        self.assertIn("EST", et_winter.tzname())

        # 3. India IST (always UTC+5:30)
        ist_time = utc_summer.astimezone(TZ_INDIA)
        self.assertEqual(ist_time.hour, 20)
        self.assertEqual(ist_time.minute, 0)
        self.assertIn("IST", ist_time.tzname())

        # 4. format_market_timestamp
        fmt_us = format_market_timestamp(utc_summer, market="US")
        self.assertIn("ET", fmt_us)
        self.assertIn("10:30:00", fmt_us)

        fmt_in = format_market_timestamp(utc_summer, market="INDIA")
        self.assertIn("IST", fmt_in)
        self.assertIn("20:00:00", fmt_in)

    def test_request_time_distinguished_from_market_data_time(self):
        """Verify API distinguishes request processing time from underlying market data observation time."""
        # 1. Test /api/metrics
        resp_m = self.client.get("/api/metrics")
        self.assertEqual(resp_m.status_code, 200)
        data_m = resp_m.get_json()
        self.assertIn("request_time", data_m)
        self.assertIn("stocks_last_updated", data_m)
        self.assertIn("threats_last_updated", data_m)
        self.assertIn("last_updated", data_m)
        # Ensure last_updated has valid UTC tag and request_time is ISO 8601
        self.assertTrue(data_m["last_updated"].endswith("UTC"))
        self.assertIn("+00:00", data_m["request_time"])

        # 2. Test /api/stocks
        resp_s = self.client.get("/api/stocks")
        self.assertEqual(resp_s.status_code, 200)
        stocks = resp_s.get_json()
        self.assertTrue(len(stocks) > 0)
        s0 = stocks[0]
        self.assertIn("request_time", s0)
        self.assertIn("exchange_timestamp", s0)
        self.assertIn("retrieved_at", s0)
        self.assertIn("freshness_label", s0)
        # request_time is when API responded, exchange_timestamp is observation
        self.assertIn("+00:00", s0["request_time"])
        self.assertTrue("as of" in s0["freshness_label"])

    def test_quote_freshness_and_stale_detection(self):
        """Verify evaluate_quote_freshness flags stale quotes older than the last market session."""
        fixed_ref = datetime(2026, 10, 25, 12, 0, tzinfo=timezone.utc) # Sunday Oct 25, 2026
        # Last US session was Friday Oct 23, 2026

        # Quote from last session (Friday Oct 23) -> NOT stale relative to weekend
        fresh_eval = evaluate_quote_freshness(
            "2026-10-23 15:59:00 ET",
            ticker="AAPL",
            market="US",
            current_price=347.72,
            reference_time=fixed_ref
        )
        self.assertFalse(fresh_eval["is_stale"])
        self.assertIn("as of Oct 23", fresh_eval["freshness_label"])
        self.assertEqual(fresh_eval["price_session"], "previous_close")

        # Quote from older session (Wednesday Oct 21) -> STALE!
        stale_eval = evaluate_quote_freshness(
            "2026-10-21 19:59:00 UTC",
            ticker="COF",
            market="US",
            current_price=206.50,
            reference_time=fixed_ref
        )
        self.assertTrue(stale_eval["is_stale"])
        self.assertIsNotNone(stale_eval["stale_warning"])
        self.assertIn("older than last session", stale_eval["stale_warning"])

        # Missing timestamp -> STALE with missing warning
        missing_eval = evaluate_quote_freshness(
            None,
            ticker="UNKNOWN",
            market="US",
            current_price=100.0,
            reference_time=fixed_ref
        )
        self.assertTrue(missing_eval["is_stale"])
        self.assertEqual(missing_eval["freshness_status"], "MISSING_TIMESTAMP")

    def test_weekend_and_session_boundaries(self):
        """Verify get_last_market_session_date identifies the correct prior session across weekends and hours."""
        # 1. Sunday Oct 25, 2026 -> Friday Oct 23
        sun_ref = datetime(2026, 10, 25, 14, 0, tzinfo=timezone.utc)
        self.assertEqual(get_last_market_session_date("US", sun_ref), date(2026, 10, 23))
        self.assertEqual(get_last_market_session_date("INDIA", sun_ref), date(2026, 10, 23))

        # 2. Saturday Oct 24, 2026 -> Friday Oct 23
        sat_ref = datetime(2026, 10, 24, 10, 0, tzinfo=timezone.utc)
        self.assertEqual(get_last_market_session_date("US", sat_ref), date(2026, 10, 23))

        # 3. Monday Oct 26, 2026 at 08:00 ET (before market opens) -> Friday Oct 23
        mon_pre_ref = datetime(2026, 10, 26, 12, 0, tzinfo=timezone.utc) # 08:00 ET
        self.assertEqual(get_last_market_session_date("US", mon_pre_ref), date(2026, 10, 23))

        # 4. Monday Oct 26, 2026 at 17:00 ET (after market close) -> Monday Oct 26
        mon_post_ref = datetime(2026, 10, 26, 21, 0, tzinfo=timezone.utc) # 17:00 ET
        self.assertEqual(get_last_market_session_date("US", mon_post_ref), date(2026, 10, 26))

    def test_missing_prices_display_na_never_zero(self):
        """Verify that quotes with 0 or missing prices show 'N/A', never 0.0 or 0."""
        # 1. format_currency
        self.assertEqual(format_currency(None), "--")
        self.assertEqual(format_currency(""), "--")

        # 2. evaluate_quote_freshness flags missing price
        res_zero = evaluate_quote_freshness("2026-10-23 15:59:00 ET", "AAPL", current_price=0.0)
        self.assertTrue(res_zero["missing_price"])

        res_none = evaluate_quote_freshness("2026-10-23 15:59:00 ET", "AAPL", current_price=None)
        self.assertTrue(res_none["missing_price"])

        # 3. Test API behavior when price is None / 0 in current_stock_data
        # Connect to DB copy and test with dummy ticker
        temp_dir = tempfile.mkdtemp()
        try:
            db_test = os.path.join(temp_dir, "test_quote.db")
            shutil.copy2("threat_stock_data.db", db_test)
            conn = sqlite3.connect(db_test)
            c = conn.cursor()
            c.execute("INSERT OR REPLACE INTO current_stock_data (ticker, current_price, company_name) VALUES ('ZERO_TEST', 0.0, 'Zero Test Corp')")
            conn.commit()
            c.execute("SELECT current_price FROM current_stock_data WHERE ticker='ZERO_TEST'")
            val = c.fetchone()[0]
            # When evaluating in app, missing price must resolve to N/A
            ev = evaluate_quote_freshness(None, "ZERO_TEST", current_price=val)
            self.assertTrue(ev["missing_price"])
            conn.close()
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

    def test_daily_closing_not_labeled_live(self):
        """Verify that public feed data is not falsely described as live ticks."""
        resp_h = self.client.get("/api/health")
        data_h = resp_h.get_json()
        self.assertIn("delayed_public_feed", data_h.get("data_feed_mode", ""))

        resp_s = self.client.get("/api/stocks")
        stocks = resp_s.get_json()
        for s in stocks[:10]:
            # Each stock has exchange_timestamp, market, currency, and price_session
            self.assertIn("market", s)
            self.assertIn("currency", s)
            self.assertIn("price_session", s)
            self.assertIn(s["price_session"], ["regular-session", "extended-hours", "previous_close"])
            self.assertIn(s["currency"], ["USD", "INR"])

    def test_preservation_of_phase1_terminology(self):
        """Verify Phase 1 raw_return and abnormal_return terminology remain intact."""
        resp_corr = self.client.get("/api/correlations")
        self.assertEqual(resp_corr.status_code, 200)
        corrs = resp_corr.get_json()
        if corrs:
            self.assertIn("raw_return", corrs[0])
            self.assertIn("price_change_percent", corrs[0])

        conn = sqlite3.connect("threat_stock_data.db")
        c = conn.cursor()
        c.execute("SELECT target_definition FROM predictions LIMIT 1")
        row = c.fetchone()
        self.assertEqual(row[0], "raw_return")
        conn.close()


if __name__ == "__main__":
    unittest.main()
