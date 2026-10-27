"""
realtime_collector.py

Real-time Cyber Threat & Stock Market Collector.

Fetches live stock quotes, monitors cyber threat events,
computes ML-driven risk scores, logs correlations in SQLite,
and sends Discord alerts when configured thresholds are met.
"""

import time
import math
import argparse
import logging
import os
import json
import sqlite3
import threading
from datetime import datetime, timezone
from typing import List, Dict, Any

from fetch_stocks import fetch_stock_data
from threat_analyzer_agent import analyze_threat_with_ai
from threat_ml_engine import predict_stock_impact
from risk_scoring_engine import (
    calculate_risk_score,
    resolve_ticker,
    get_sector_comparison
)
from alerts import send_discord_alert
from prediction_tracking import (
    insert_prediction,
    evaluate_pending_predictions
)


from stock_universe import DEFAULT_WATCHLIST

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_NAME = os.path.join(BASE_DIR, "threat_stock_data.db")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
LOG_DIR = os.path.join(BASE_DIR, "logs")
LOG_PATH = os.path.join(LOG_DIR, "realtime_collector.log")

os.makedirs(LOG_DIR, exist_ok=True)

logger = logging.getLogger("realtime_collector")

if not logger.handlers:
    logger.setLevel(logging.INFO)

    handler = logging.FileHandler(
        LOG_PATH,
        encoding="utf-8"
    )

    handler.setFormatter(
        logging.Formatter(
            "%(asctime)s %(levelname)s %(message)s"
        )
    )

    logger.addHandler(handler)


_scheduler_lock = threading.Lock()
_scheduler_stop_event = threading.Event()
_scheduler_thread = None
_scheduler_started_at = None
_scheduler_interval_minutes = None


def get_db_connection():
    return sqlite3.connect(
        DB_NAME,
        timeout=30.0
    )


def update_monitored_stocks(
    tickers: List[str] = DEFAULT_WATCHLIST
) -> Dict[str, Dict[str, Any]]:

    print(
        f"\n[{datetime.now().strftime('%H:%M:%S')}] "
        f"Polling live market quotes for: {', '.join(tickers)}"
    )

    market_snapshot = {}

    for ticker in tickers:

        data = fetch_stock_data(ticker)

        if not data:
            print(
                f"  [WARN] Could not fetch data for {ticker}"
            )
            continue
        try:
            current_price = float(
                data.get("current_price", data.get("price"))
            )

            if not math.isfinite(current_price) or current_price <= 0:
                raise ValueError("Invalid current price")

            previous_close = data.get("previous_close")
            if previous_close is not None:
                previous_close = float(previous_close)
                if not math.isfinite(previous_close) or previous_close <= 0:
                    previous_close = None

            change_percent = data.get("change_percent")
            if change_percent is not None:
                change_percent = float(change_percent)
                if not math.isfinite(change_percent):
                    change_percent = None

            timestamp = data.get("timestamp") or "UNKNOWN"
            company_name = data.get("company", ticker)

        except (TypeError, ValueError, OverflowError):
            print(f"  [WARN] Invalid price data for {ticker}; skipping update")
            continue

        if previous_close is None or change_percent is None:
            print(
                f"  [WARN] {ticker}: previous close or percentage change "
                "is unavailable"
            )
            continue
        exchange_timestamp = data.get("exchange_timestamp", timestamp)
        market_name = data.get("market", "NSE" if ticker.endswith(".NS") else "NYSE")
        currency = data.get("currency", "INR" if ticker.endswith(".NS") else "USD")
        price_session = data.get("price_session", "previous_close")
        retrieved_at = data.get("retrieved_at", datetime.now(timezone.utc).isoformat())

        market_snapshot[ticker] = {
            "ticker": ticker,
            "current_price": float(current_price),
            "previous_close": float(previous_close),
            "change_percent": float(change_percent),
            "timestamp": timestamp,
            "company_name": company_name,
            "exchange_timestamp": exchange_timestamp,
            "market": market_name,
            "currency": currency,
            "price_session": price_session,
            "retrieved_at": retrieved_at
        }

        try:
            conn = get_db_connection()
            cursor = conn.cursor()

            cursor.execute(
                """
                INSERT INTO current_stock_data (
                    ticker,
                    current_price,
                    previous_close,
                    change_percent,
                    timestamp,
                    company_name,
                    exchange_timestamp,
                    market,
                    currency,
                    price_session,
                    retrieved_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(ticker)
                DO UPDATE SET
                    current_price = excluded.current_price,
                    previous_close = excluded.previous_close,
                    change_percent = excluded.change_percent,
                    timestamp = excluded.timestamp,
                    company_name = excluded.company_name,
                    exchange_timestamp = excluded.exchange_timestamp,
                    market = excluded.market,
                    currency = excluded.currency,
                    price_session = excluded.price_session,
                    retrieved_at = excluded.retrieved_at
                """,
                (
                    ticker,
                    float(current_price),
                    float(previous_close),
                    float(change_percent),
                    timestamp,
                    company_name,
                    exchange_timestamp,
                    market_name,
                    currency,
                    price_session,
                    retrieved_at
                )
            )


            cursor.execute(
                """
                INSERT INTO stock_prices (
                    ticker,
                    price,
                    date,
                    volume
                )
                VALUES (?, ?, ?, ?)
                """,
                (
                    ticker,
                    float(current_price),
                    datetime.now().strftime("%Y-%m-%d"),
                    0
                )
            )

            conn.commit()
            conn.close()

            print(
                f"  [UPDATED] {ticker} "
                f"${current_price:.2f} "
                f"({change_percent:+.2f}%) "
                f"| {timestamp}"
            )

        except Exception as e:
            print(
                f"  [DB ERROR] {ticker}: {e}"
            )

    return market_snapshot


def ingest_and_evaluate_threat(
    company: str,
    threat_type: str,
    description: str,
    threat_date: str = None
) -> Dict[str, Any]:

    if not threat_date:
        threat_date = datetime.now().strftime(
            "%Y-%m-%d"
        )

    ticker_info = resolve_ticker(company)

    if not ticker_info.get("resolved") or ticker_info.get("ticker") == "UNKNOWN":
        print(f"\n>>> INGESTING THREAT REJECTED: '{company}' could not be resolved to a validated public equity.")
        return {
            "threat_id": None,
            "ticker": "UNKNOWN",
            "company": ticker_info.get("name", company),
            "status": "INSUFFICIENT EVIDENCE",
            "risk_score": None,
            "tier": "INSUFFICIENT EVIDENCE",
            "recommendation": "NO SIGNAL",
            "predicted_drop": None,
            "prediction_id": None,
            "predicted_direction": None,
            "expected_range_percent": None,
            "confidence": None,
            "confidence_basis": None,
            "model_limitation": "No validated public company ticker could be resolved. Threat rejected with INSUFFICIENT EVIDENCE.",
            "current_change": None,
            "sector_benchmark": None,
            "sector_change": None,
            "sector_excess_performance": None,
            "drag_detected": False,
            "business_impact": None,
            "evidence_quality": None,
            "exploitation_status": "NOT_REPORTED",
            "causal_attribution_established": False,
            "alert_dispatched": False,
            "error": "INSUFFICIENT EVIDENCE: Company could not be resolved to a validated public equity ticker."
        }

    ticker = ticker_info["ticker"]
    company_name = ticker_info["name"]

    benchmark_ticker = ticker_info.get(
        "benchmark",
        "SPY"
    )

    print(
        f"\n>>> INGESTING THREAT: "
        f"{company_name} (${ticker}) - "
        f"{threat_type}"
    )

    ai_analysis = analyze_threat_with_ai({
        "company": company_name,
        "type": threat_type,
        "description": description
    })

    severity = float(
        ai_analysis.get(
            "severity_score",
            7.0
        )
    )

    sensitivity = float(
        ai_analysis.get(
            "data_sensitivity_score",
            7.0
        )
    )

    users_affected = float(
        ai_analysis.get(
            "users_affected_millions",
            0.0
        )
    )

    business_impact = float(
        ai_analysis.get(
            "business_impact",
            0.0
        )
    )

    evidence_quality = float(
        ai_analysis.get(
            "evidence_quality",
            0.0
        )
    )

    exploitation_status = ai_analysis.get(
        "exploitation_status",
        "NOT_REPORTED"
    )

    exposure_level = ai_analysis.get(
        "exposure_level",
        "UNKNOWN"
    )

    cve_ids_list = ai_analysis.get("cve_ids", [])
    if isinstance(cve_ids_list, list):
        cve_ids_str = json.dumps(cve_ids_list)
    else:
        cve_ids_str = "[]"

    affected_prods_list = ai_analysis.get("affected_products", [])
    if isinstance(affected_prods_list, list):
        affected_prods_str = json.dumps(affected_prods_list)
    else:
        affected_prods_str = "[]"

    summary = ai_analysis.get(
        "incident_summary",
        ai_analysis.get("executive_summary", description)
    )

    analysis_source = ai_analysis.get(
        "provider",
        ai_analysis.get("analysis_source", "Rule-Based Heuristic")
    )

    ml_pred = predict_stock_impact(
        severity_score=severity,
        users_affected_millions=users_affected,
        data_sensitivity_score=sensitivity
    )

    predicted_drop = ml_pred.get(
        "predicted_change_percent",
        -1.5
    )

    live_stock = fetch_stock_data(ticker)

    market_data_available = False
    curr_change = 0.0
    curr_price = None
    prev_close = None

    if live_stock:
        try:
            quote_price = float(
                live_stock.get("current_price", live_stock.get("price"))
            )
            quote_change = float(live_stock.get("change_percent"))
            quote_previous = float(live_stock.get("previous_close"))

            if all(math.isfinite(value) for value in (quote_price, quote_change, quote_previous)) and quote_price > 0 and quote_previous > 0:
                curr_change = quote_change
                curr_price = quote_price
                prev_close = quote_previous
                market_data_available = True
        except (TypeError, ValueError, OverflowError):
            pass

    if not market_data_available:
        print(f"  [WARN] Live quote unavailable or invalid for {ticker}; market movement will be treated as unavailable")

    sector_comparison = get_sector_comparison(
        ticker,
        benchmark_ticker
    )

    sector_change = sector_comparison.get(
        "benchmark_change_percent",
        0.0
    )

    sector_excess_performance = sector_comparison.get(
        "excess_performance",
        0.0
    )

    drag_detected = sector_comparison.get(
        "drag_detected",
        False
    )

    print(
        f"  Sector Benchmark: "
        f"{benchmark_ticker} "
        f"({sector_change:+.2f}%)"
    )

    print(
        f"  Relative Performance: "
        f"{sector_excess_performance:+.2f}%"
    )

    print(
        f"  Market Drag Detected: "
        f"{drag_detected}"
    )

    risk_assessment = calculate_risk_score(
        severity_score=severity,
        data_sensitivity_score=sensitivity,
        predicted_price_drop=predicted_drop,
        current_stock_change=curr_change,
        users_affected_millions=users_affected,
        sector_excess_performance=sector_excess_performance,
        business_impact=business_impact,
        evidence_quality=evidence_quality,
        exploitation_status=exploitation_status
    )

    risk_score = risk_assessment["risk_score"]
    recommendation = risk_assessment["recommendation"]
    tier = risk_assessment["tier"]

    print(
        f"  Risk Assessment: "
        f"{risk_score}/100 | "
        f"{tier} | "
        f"{recommendation}"
    )

    conn = get_db_connection()
    cursor = conn.cursor()

    cursor.execute(
        """
        INSERT INTO threats (
            company,
            ticker,
            threat_type,
            severity,
            severity_score,
            users_affected_millions,
            data_sensitivity_score,
            threat_date,
            description,
            risk_score,
            recommendation,
            analysis_source,
            business_impact,
            exploitation_status,
            evidence_quality,
            exposure_level,
            cve_ids,
            affected_products
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            company_name,
            ticker,
            threat_type,
            tier,
            severity,
            users_affected,
            sensitivity,
            threat_date,
            summary,
            risk_score,
            recommendation,
            analysis_source,
            business_impact,
            exploitation_status,
            evidence_quality,
            exposure_level,
            cve_ids_str,
            affected_prods_str
        )
    )

    threat_id = cursor.lastrowid

    prediction_timestamp = datetime.now(
        timezone.utc
    ).isoformat()

    prediction_id = insert_prediction(
        cursor,
        threat_id,
        ticker,
        ml_pred,
        prediction_timestamp=prediction_timestamp
    )

    cursor.execute(
        """
        INSERT INTO correlations (
            threat_id,
            ticker,
            threat_date,
            stock_price_before,
            stock_price_after,
            price_change_percent,
            risk_score,
            recommendation
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            threat_id,
            ticker,
            threat_date,
            prev_close,
            curr_price,
            curr_change if market_data_available else None,
            risk_score,
            recommendation
        )
    )

    conn.commit()
    conn.close()

    print(
        f"  [SAVED] Threat ID #{threat_id} "
        f"for ${ticker} | "
        f"Risk: {risk_score}/100 "
        f"({tier}) | "
        f"Signal: {recommendation}"
    )

    is_drop_significant = market_data_available and curr_change < -1.0
    is_risk_high = risk_score >= 50

    if is_drop_significant or is_risk_high:

        alert_reason = []

        if is_drop_significant:
            alert_reason.append(
                f"Stock movement "
                f"{curr_change:+.2f}%"
            )

        if drag_detected:
            alert_reason.append(
                f"Relative sector underperformance "
                f"{sector_excess_performance:+.2f}%"
            )

        if is_risk_high:
            alert_reason.append(
                f"Risk score {risk_score}/100"
            )

        expected_range = (
            ml_pred.get("expected_range_percent")
            or {}
        )

        if (
            expected_range.get("low") is not None
            and expected_range.get("high") is not None
        ):
            range_summary = (
                f"{expected_range['low']:+.2f}% to "
                f"{expected_range['high']:+.2f}% "
                f"({expected_range.get('basis', 'model range')})"
            )
        else:
            range_summary = (
                "Unavailable; model point estimate "
                "is not a certainty"
            )

        alert_summary = (
            f"**{threat_type.upper()}**: "
            f"{summary}\n"
            f"• Market Movement: "
            f"`{curr_change:+.2f}%`\n" if market_data_available else
            "• Market Movement: `N/A`\n"
            f"• Sector Benchmark "
            f"({benchmark_ticker}): "
            f"`{sector_change:+.2f}%`\n"
            f"• Relative Performance: "
            f"`{sector_excess_performance:+.2f}%`\n"
            f"• ML Direction Estimate: "
            f"`{ml_pred.get('predicted_direction', 'N/A')}`\n"
            f"• ML Expected Range: "
            f"`{range_summary}`\n"
            f"• Model Limitation: "
            f"{ml_pred.get('model_limitation', 'Uncalibrated estimate')}\n"
            f"• Evidence Status: "
            f"**Causal attribution not established**\n"
            f"• System Signal: "
            f"**{recommendation}**\n"
            f"• Trigger: "
            f"{', '.join(alert_reason)}"
        )

        print(
            f"  [ALERT TRIGGERED] "
            f"Dispatching Discord Alert "
            f"for ${ticker}..."
        )

        send_discord_alert(
            ticker=ticker,
            risk_score=risk_score,
            recommendation=recommendation,
            threat_summary=alert_summary
        )

    else:

        if market_data_available:
            print(
                f"  [INFO] Alert threshold not breached "
                f"(Change: {curr_change:+.2f}%, Risk: {risk_score}/100)"
            )
        else:
            print(
                f"  [INFO] Alert threshold not breached "
                f"(Market change: N/A, Risk: {risk_score}/100)"
            )

    return {
        "threat_id": threat_id,
        "ticker": ticker,
        "company": company_name,
        "risk_score": risk_score,
        "tier": tier,
        "recommendation": recommendation,
        "predicted_drop": predicted_drop,
        "prediction_id": prediction_id,
        "predicted_direction":
            ml_pred.get("predicted_direction"),
        "expected_range_percent":
            ml_pred.get("expected_range_percent"),
        "confidence":
            ml_pred.get("confidence"),
        "confidence_basis":
            ml_pred.get("confidence_basis"),
        "model_limitation":
            ml_pred.get("model_limitation"),
        "current_change": curr_change if market_data_available else None,
        "market_data_available": market_data_available,
        "sector_benchmark": benchmark_ticker,
        "sector_change": sector_change,
        "sector_excess_performance":
            sector_excess_performance,
        "drag_detected": drag_detected,
        "business_impact": business_impact,
        "evidence_quality": evidence_quality,
        "exploitation_status": exploitation_status,
        "causal_attribution_established": False,
        "alert_dispatched":
            is_drop_significant or is_risk_high
    }


def run_single_monitoring_cycle(
    article_limit: int = 1
) -> Dict[str, Any]:

    print("=" * 65)

    print(
        f"STARTING MONITORING CYCLE: "
        f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
    )

    print("=" * 65)

    cycle = {
        "started_at":
            datetime.now(timezone.utc).isoformat(),
        "stock_updates": 0,
        "feed_candidates": 0,
        "ingested": None,
        "prediction_checks": 0,
        "errors": []
    }

    try:

        market_snapshot = update_monitored_stocks(
            DEFAULT_WATCHLIST
        )

        cycle["stock_updates"] = len(
            market_snapshot
        )

    except Exception as error:

        logger.exception(
            "Market quote phase failed"
        )

        cycle["errors"].append(
            f"market quotes: {error}"
        )

    try:

        from threat_collector import (
            ingest_threats,
            scan_feeds
        )

        threats = scan_feeds()

        cycle["feed_candidates"] = len(
            threats
        )

        cycle["ingested"] = ingest_threats(
            threats,
            limit=article_limit
        )

    except Exception as error:

        logger.exception(
            "RSS collection or ingestion phase failed"
        )

        cycle["errors"].append(
            f"RSS pipeline: {error}"
        )

    try:

        evaluation = evaluate_pending_predictions()

        cycle["prediction_checks"] = (
            evaluation["checked"]
        )

        print(
            f"[PREDICTIONS] Checked "
            f"{evaluation['checked']} pending forecasts; "
            f"updated "
            f"{len(evaluation['updated'])} "
            f"from available yfinance data."
        )

    except Exception as error:

        logger.exception(
            "Prediction evaluation phase failed"
        )

        cycle["errors"].append(
            f"prediction evaluation: {error}"
        )

    try:

        conn = get_db_connection()
        cursor = conn.cursor()

        cursor.execute(
            """
            SELECT
                ticker,
                current_price,
                previous_close,
                change_percent
            FROM current_stock_data
            WHERE change_percent < -1.0
            """
        )

        dropping_stocks = cursor.fetchall()
        conn.close()

    except Exception as error:

        logger.exception(
            "Market review phase failed"
        )

        cycle["errors"].append(
            f"market review: {error}"
        )

        dropping_stocks = []

    if dropping_stocks:

        print(
            f"\n[ATTENTION] "
            f"Found {len(dropping_stocks)} stocks "
            f"with >1% drop:"
        )

        for ticker, price, prev, chg in dropping_stocks:

            print(
                f"  - ${ticker}: "
                f"${price:.2f} "
                f"({chg:+.2f}%)"
            )

    cycle["completed_at"] = datetime.now(
        timezone.utc
    ).isoformat()

    if cycle["errors"]:

        logger.warning(
            "Cycle completed with errors: %s",
            cycle
        )

        print(
            f"Cycle completed with "
            f"{len(cycle['errors'])} "
            f"recoverable error(s).\n"
        )

    else:

        logger.info(
            "Cycle completed: %s",
            cycle
        )

        print(
            f"Cycle completed successfully at "
            f"{datetime.now().strftime('%H:%M:%S')}.\n"
        )

    return cycle


def _background_scheduler_worker(
    interval_minutes: int,
    article_limit: int
):

    while not _scheduler_stop_event.is_set():

        try:

            run_single_monitoring_cycle(
                article_limit=article_limit
            )

        except Exception:

            logger.exception(
                "Unexpected scheduler cycle failure"
            )

        if _scheduler_stop_event.wait(
            interval_minutes * 60
        ):
            break


def start_background_scheduler(
    interval_minutes: int = 30,
    article_limit: int = 1
) -> Dict[str, Any]:

    global _scheduler_thread
    global _scheduler_started_at
    global _scheduler_interval_minutes

    interval_minutes = max(
        30,
        min(60, int(interval_minutes))
    )

    article_limit = max(
        0,
        int(article_limit)
    )

    with _scheduler_lock:

        if (
            _scheduler_thread
            and _scheduler_thread.is_alive()
        ):
            return get_scheduler_status()

        _scheduler_stop_event.clear()

        _scheduler_interval_minutes = (
            interval_minutes
        )

        _scheduler_started_at = (
            datetime.now(timezone.utc).isoformat()
        )

        _scheduler_thread = threading.Thread(
            target=_background_scheduler_worker,
            args=(
                interval_minutes,
                article_limit
            ),
            name="cyber-threat-collector",
            daemon=True
        )

        _scheduler_thread.start()

        logger.info(
            "Background scheduler started "
            "interval=%s article_limit=%s",
            interval_minutes,
            article_limit
        )

    return get_scheduler_status()


def stop_background_scheduler() -> Dict[str, Any]:

    _scheduler_stop_event.set()

    logger.info(
        "Background scheduler stop requested"
    )

    return get_scheduler_status()


def get_scheduler_status() -> Dict[str, Any]:

    active = bool(
        _scheduler_thread
        and _scheduler_thread.is_alive()
    )

    return {
        "status":
            "RUNNING" if active else "STOPPED",
        "interval_minutes":
            _scheduler_interval_minutes,
        "started_at":
            _scheduler_started_at,
        "log_path":
            LOG_PATH
    }


def start_scheduler(
    interval_minutes: int = 30
):

    print("=" * 65)

    print(
        "AI CYBER THREAT INTELLIGENCE "
        "REAL-TIME MONITOR"
    )

    print(
        f"Interval: Every "
        f"{interval_minutes} minutes"
    )

    print(
        "Press Ctrl+C to terminate collector."
    )

    print("=" * 65)

    start_background_scheduler(
        interval_minutes=interval_minutes
    )

    try:

        while (
            get_scheduler_status()["status"]
            == "RUNNING"
        ):
            time.sleep(1)

    except KeyboardInterrupt:

        stop_background_scheduler()


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description=(
            "Real-time Threat & Stock "
            "Data Collector"
        )
    )

    parser.add_argument(
        "--once",
        action="store_true",
        help=(
            "Run a single monitoring cycle "
            "and exit immediately."
        )
    )

    parser.add_argument(
        "--interval",
        type=int,
        default=30,
        help=(
            "Minutes between monitoring sweeps "
            "(default: 30)."
        )
    )

    parser.add_argument(
        "--simulate-threat",
        action="store_true",
        help=(
            "Simulates an active threat "
            "ingestion cycle."
        )
    )

    args = parser.parse_args()

    if args.simulate_threat:

        ingest_and_evaluate_threat(
            company="Microsoft",
            threat_type="Zero-Day Exploit",
            description=(
                "Active zero-day vulnerability detected "
                "in Azure cloud storage service exposing "
                "enterprise configuration secrets."
            )
        )

    elif args.once:

        run_single_monitoring_cycle()

    else:

        start_scheduler(
            interval_minutes=args.interval
        )