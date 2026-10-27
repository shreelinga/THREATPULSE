
"""
app.py
Flask Web Backend and REST API for the Cyber Threat Intelligence Platform.
Serves live data from SQLite (threat_stock_data.db) to the frontend dashboard,
provides real-time threat analysis endpoints, and coordinates Discord alerting.
"""

import os
import json
import sqlite3
from datetime import datetime, timezone
from flask import Flask, jsonify, request, send_from_directory, render_template

from fetch_stocks import fetch_stock_data
from stock_universe import STOCK_UNIVERSE
from market_utils import (
    is_indian_ticker,
    format_currency,
    calculate_quote_age_minutes,
    get_market_session_status,
    get_market_benchmark,
    format_market_timestamp,
    parse_market_timestamp,
    get_last_market_session_date,
    evaluate_quote_freshness,
)

from risk_scoring_engine import resolve_ticker, calculate_risk_score, get_sector_comparison, COMPANY_TICKER_MAP
from realtime_collector import (
    DEFAULT_WATCHLIST,
    get_scheduler_status,
    ingest_and_evaluate_threat,
    start_background_scheduler,
    stop_background_scheduler,
    update_monitored_stocks,
)
from alerts import send_discord_alert
from database_setup import create_database
from prediction_tracking import get_prediction_performance, evaluate_pending_predictions
from event_study_engine import get_event_study_data, run_event_study
from threat_ml_engine import predict_stock_impact, load_model_bundle, load_model_metrics

app = Flask(__name__, static_folder=".", static_url_path="")
DB_NAME = "threat_stock_data.db"

@app.route("/api/stock_search", methods=["GET"])
def search_stocks():
    query = request.args.get("q", "").strip().lower()

    if not query:
        return jsonify([])

    results = []
    for ticker, company in STOCK_UNIVERSE.items():
        if query in ticker.lower() or query in company.lower():
            results.append({
                "ticker": ticker,
                "company_name": company
            })

        if len(results) >= 20:
            break

    return jsonify(results)



@app.route("/api/stocks/add", methods=["POST"])
def add_stock():
    data = request.get_json(silent=True) or {}
    ticker = str(data.get("ticker", "")).strip().upper()

    if ticker not in STOCK_UNIVERSE:
        return jsonify({
            "success": False,
            "error": "Please select a stock from the predefined stock list."
        }), 400

    company_name = STOCK_UNIVERSE[ticker]
    market = "India" if ticker.endswith(".NS") else "US"

    try:
        conn = get_db()
        conn.execute(
            """
            INSERT OR IGNORE INTO monitored_stocks
            (ticker, company_name, market)
            VALUES (?, ?, ?)
            """,
            (ticker, company_name, market)
        )
        conn.commit()
        conn.close()

        result = update_monitored_stocks([ticker])
        stock = result.get(ticker)

        if not stock:
            return jsonify({
                "success": False,
                "error": f"{ticker} was saved, but market data could not be fetched. Try refreshing quotes."
            }), 502

        return jsonify({
            "success": True,
            "stock": stock
        })

    except Exception:
        app.logger.exception("Failed to add stock %s", ticker)
        return jsonify({
            "success": False,
            "error": "Stock update failed. Check the application logs."
        }), 500



def get_db():
    conn = sqlite3.connect(DB_NAME)
    conn.row_factory = sqlite3.Row
    return conn


@app.route("/api/cisa_vulnerabilities", methods=["GET"])
def get_cisa_vulnerabilities():
    query = request.args.get("q", "").strip()
    ransomware_filter = request.args.get("ransomware", "").strip().lower()
    vendor_filter = request.args.get("vendor", "").strip()
    page = request.args.get("page", default=1, type=int)
    per_page = request.args.get("per_page", default=100, type=int)

    page = max(1, page)
    per_page = max(1, min(per_page, 200))
    offset = (page - 1) * per_page

    conn = get_db()
    cursor = conn.cursor()

    try:
        # Get total catalog size and last sync timestamp
        cursor.execute("SELECT COUNT(*), MAX(fetched_at) FROM cisa_kev_entries")
        row = cursor.fetchone()
        catalog_total = row[0] if row else 0
        last_sync = row[1] if row and row[1] else None

        # Build dynamic filter query
        where_clauses = []
        params = []

        if query:
            where_clauses.append("(cve_id LIKE ? OR vendor_project LIKE ? OR product LIKE ? OR vulnerability_name LIKE ? OR short_description LIKE ?)")
            q_param = f"%{query}%"
            params.extend([q_param] * 5)

        if ransomware_filter in ("known", "yes", "true"):
            where_clauses.append("LOWER(ransomware_use) = 'known'")
        elif ransomware_filter in ("unknown", "no", "false"):
            where_clauses.append("(ransomware_use IS NULL OR LOWER(ransomware_use) != 'known')")

        if vendor_filter:
            where_clauses.append("LOWER(vendor_project) LIKE ?")
            params.append(f"%{vendor_filter.lower()}%")

        where_sql = f"WHERE {' AND '.join(where_clauses)}" if where_clauses else ""

        # Filtered count
        cursor.execute(f"SELECT COUNT(*) FROM cisa_kev_entries {where_sql}", params)
        filtered_count = cursor.fetchone()[0]

        # Fetch records
        cursor.execute(f"""
            SELECT cve_id, vendor_project, product, vulnerability_name,
                   date_added, short_description, required_action,
                   due_date, ransomware_use, source_url, fetched_at
            FROM cisa_kev_entries
            {where_sql}
            ORDER BY date_added DESC, cve_id DESC
            LIMIT ? OFFSET ?
        """, params + [per_page, offset])

        rows = [dict(r) for r in cursor.fetchall()]

        return jsonify({
            "total": filtered_count,
            "catalog_total": catalog_total,
            "last_sync_time": last_sync,
            "page": page,
            "per_page": per_page,
            "total_pages": (filtered_count + per_page - 1) // per_page if filtered_count else 1,
            "returned": len(rows),
            "vulnerabilities": rows
        })
    except sqlite3.Error as error:
        return jsonify({"error": str(error)}), 500
    finally:
        conn.close()


# Apply additive SQLite migrations before serving routes that depend on them.
create_database()


@app.route("/")
def index():
    """Serves the main dashboard page."""
    return send_from_directory(".", "dashboard.html")


@app.route("/api/metrics", methods=["GET"])
def get_metrics():
    """Returns aggregated high-level KPIs."""
    conn = get_db()
    c = conn.cursor()

    c.execute("SELECT COUNT(*) as cnt FROM threats")
    total_threats = c.fetchone()["cnt"]

    c.execute("SELECT COUNT(*) as cnt FROM current_stock_data")
    monitored_stocks = c.fetchone()["cnt"]

    c.execute("SELECT MIN(change_percent) as max_drop FROM current_stock_data")
    max_drop_row = c.fetchone()
    max_drop = max_drop_row["max_drop"] if max_drop_row["max_drop"] is not None else 0.0

    c.execute("SELECT COUNT(*) as cnt FROM threats WHERE risk_score >= 70")
    critical_alerts = c.fetchone()["cnt"]

    c.execute("SELECT AVG(risk_score) as avg_risk FROM threats")
    avg_risk_row = c.fetchone()
    avg_risk = round(avg_risk_row["avg_risk"], 1) if avg_risk_row["avg_risk"] is not None else 0.0

    c.execute("SELECT MAX(created_at) as latest_threat FROM threats")
    latest_threat_row = c.fetchone()
    latest_threat_time = latest_threat_row["latest_threat"] if latest_threat_row else None

    c.execute("SELECT MAX(timestamp) as latest_stock FROM current_stock_data")
    latest_stock_row = c.fetchone()
    latest_stock_time = latest_stock_row["latest_stock"] if latest_stock_row else None

    conn.close()

    now_utc = datetime.now(timezone.utc)
    request_time_iso = now_utc.isoformat()
    request_time_utc_str = now_utc.strftime("%Y-%m-%d %H:%M:%S UTC")

    return jsonify({
        "total_threats": total_threats,
        "monitored_stocks": monitored_stocks,
        "max_drop_percent": round(max_drop, 2) if max_drop is not None else 0.0,
        "critical_alerts": critical_alerts,
        "average_risk_score": avg_risk,
        "request_time": request_time_iso,
        "threats_last_updated": latest_threat_time,
        "stocks_last_updated": latest_stock_time,
        "last_updated": request_time_utc_str  # Preserved for backward compatibility, now timezone-aware UTC
    })



@app.route("/api/threats", methods=["GET"])
def get_threats():
    """Returns detected threats ordered by date with evidence metadata."""
    conn = get_db()
    c = conn.cursor()
    c.execute("""
        SELECT id, company, ticker, threat_type, severity, severity_score,
               users_affected_millions, data_sensitivity_score,
               threat_date, description, risk_score, recommendation, analysis_source,
               business_impact, exploitation_status, evidence_quality,
               exposure_level, cve_ids, affected_products, review_needed, created_at
        FROM threats
        ORDER BY id DESC
        LIMIT 50
    """)
    rows = []
    for r in c.fetchall():
        row = dict(r)
        # Parse JSON lists safely
        if isinstance(row.get("cve_ids"), str):
            try:
                row["cve_ids"] = json.loads(row["cve_ids"])
            except Exception:
                row["cve_ids"] = []
        elif not row.get("cve_ids"):
            row["cve_ids"] = []

        if isinstance(row.get("affected_products"), str):
            try:
                row["affected_products"] = json.loads(row["affected_products"])
            except Exception:
                row["affected_products"] = []
        elif not row.get("affected_products"):
            row["affected_products"] = []

        rows.append(row)
    conn.close()
    return jsonify(rows)


@app.route("/api/stocks", methods=["GET"])
def get_stocks():
    """Returns current stock snapshot enriched with exchange timestamps, market metadata, and quote freshness."""
    conn = get_db()
    c = conn.cursor()
    c.execute("""
        SELECT ticker, company_name, current_price, previous_close, change_percent,
               timestamp, exchange_timestamp, market, currency, price_session, retrieved_at
        FROM current_stock_data
        ORDER BY change_percent ASC
    """)
    rows = []
    now_utc = datetime.now(timezone.utc)
    req_time = now_utc.isoformat()

    for r in c.fetchall():
        row = dict(r)
        t = row["ticker"]
        is_in = is_indian_ticker(t)
        price_val = row.get("current_price")

        freshness = evaluate_quote_freshness(
            row.get("exchange_timestamp") or row.get("timestamp"),
            ticker=t,
            market="INDIA" if is_in else "US",
            current_price=price_val,
            reference_time=now_utc
        )

        # Missing prices show N/A, never 0
        if freshness["missing_price"]:
            row["current_price"] = None
            row["formatted_price"] = "N/A"
        else:
            row["formatted_price"] = format_currency(price_val, t)

        row["market"] = row.get("market") or freshness["market"]
        row["market_region"] = freshness["market_region"]
        row["currency"] = row.get("currency") or freshness["currency"]
        row["currency_symbol"] = freshness["currency_symbol"]
        row["price_session"] = row.get("price_session") or freshness["price_session"]
        row["exchange_timestamp"] = row.get("exchange_timestamp") or row.get("timestamp") or freshness["observation_time_local"]
        row["quote_age_minutes"] = freshness["age_minutes"]
        row["is_stale"] = freshness["is_stale"]
        row["stale_warning"] = freshness.get("stale_warning")
        row["freshness_label"] = freshness["freshness_label"]
        row["observation_time_utc"] = freshness["observation_time_utc"]
        row["retrieved_at"] = row.get("retrieved_at") or row.get("timestamp") or req_time
        row["request_time"] = req_time

        sess = get_market_session_status(t)
        row["session_status"] = sess.get("status", "CLOSED") if isinstance(sess, dict) else str(sess)
        row["market_session"] = sess
        bm_info = get_market_benchmark(t)
        row["benchmark_ticker"] = bm_info.get("broad_benchmark", "^NSEI" if is_in else "SPY") if isinstance(bm_info, dict) else bm_info
        rows.append(row)
    conn.close()
    return jsonify(rows)



@app.route("/api/correlations", methods=["GET"])
def get_correlations():
    """Returns correlation records between threats and stock drops with review tags."""
    conn = get_db()
    c = conn.cursor()
    c.execute("""
        SELECT c.id, c.threat_id, c.ticker, c.threat_date,
               c.stock_price_before, c.stock_price_after, c.price_change_percent,
               c.risk_score, c.recommendation, c.review_needed, c.created_at,
               t.company, t.threat_type, t.description, t.exposure_level
        FROM correlations c
        LEFT JOIN threats t ON c.threat_id = t.id
        ORDER BY c.id DESC
        LIMIT 30
    """)
    rows = []
    for r in c.fetchall():
        d = dict(r)
        d["raw_return"] = d.get("price_change_percent")
        rows.append(d)
    conn.close()
    return jsonify(rows)


@app.route("/api/sector_comparison", methods=["GET"])
def get_sector_data():
    """
    Returns verified market and sector benchmark comparisons for monitored equities.
    Separates Indian (.NS/.BO) and US equities strictly:
      - Indian IT -> ^CNXIT (Nifty IT), Indian Banking -> ^NSEBANK (Nifty Bank), Broad -> ^NSEI (Nifty 50)
      - US IT -> XLK (Tech SPDR), Comm -> XLC, Consumer -> XLY, Financials -> XLF, Broad -> SPY
    Never compares Indian equities against US-only benchmarks.
    """
    # 1. Fetch currently monitored tickers from database
    conn = get_db()
    c = conn.cursor()
    c.execute("SELECT ticker, company_name FROM current_stock_data ORDER BY ticker ASC")
    db_rows = c.fetchall()
    conn.close()

    active_tickers = {r["ticker"]: r["company_name"] for r in db_rows}
    # Ensure standard cross-market comparison core is always available
    core_targets = [
        "MSFT", "AAPL", "GOOGL", "NVDA", "CRWD",
        "TCS.NS", "INFY.NS", "WIPRO.NS", "HDFCBANK.NS", "RELIANCE.NS"
    ]
    for t in core_targets:
        if t not in active_tickers and t in STOCK_UNIVERSE:
            active_tickers[t] = STOCK_UNIVERSE[t]

    results = []
    for ticker, company_name in active_tickers.items():
        is_in = is_indian_ticker(ticker)
        bm_info = get_market_benchmark(ticker)
        sector_bm = bm_info.get("sector_benchmark", "^NSEI" if is_in else "SPY")
        broad_bm = bm_info.get("broad_benchmark", "^NSEI" if is_in else "SPY")
        
        info = COMPANY_TICKER_MAP.get(ticker.upper(), {})
        sector_name = info.get("sector", "Information Technology" if (".NS" in ticker and "CNXIT" in sector_bm) else "Technology / Financial")

        now_iso = datetime.now(timezone.utc).isoformat()
        comp_res = get_sector_comparison(ticker, benchmark_ticker=sector_bm)
        comp_res["market"] = "India" if is_in else "US"
        comp_res["exchange"] = "NSE" if is_in else "NYSE"
        comp_res["market_region"] = "India" if is_in else "US"
        comp_res["currency"] = "INR" if is_in else "USD"
        comp_res["currency_symbol"] = "₹" if is_in else "$"
        comp_res["sector_name"] = sector_name
        comp_res["broad_benchmark"] = broad_bm
        comp_res["retrieved_at"] = now_iso
        comp_res["request_time"] = now_iso
        results.append(comp_res)


    return jsonify(results)


@app.route("/api/ml/predict", methods=["POST"])
def predict_ml_impact_endpoint():
    """
    On-demand inference endpoint for testing custom threat metrics against
    the trained LOOCV Random Forest Regressor and baseline models.
    """
    data = request.get_json(silent=True)
    if data is None or not isinstance(data, dict):
        return jsonify({
            "status": "UNAVAILABLE",
            "error": "Request body must be a valid JSON object with severity_score, users_affected_millions, and data_sensitivity_score",
            "predicted_change_percent": None
        }), 400

    try:
        raw_sev = data.get("severity_score")
        raw_users = data.get("users_affected_millions")
        raw_sens = data.get("data_sensitivity_score")
        severity = float(raw_sev) if raw_sev is not None else None
        users = float(raw_users) if raw_users is not None else None
        sensitivity = float(raw_sens) if raw_sens is not None else None
    except (ValueError, TypeError) as err:
        return jsonify({
            "status": "UNAVAILABLE",
            "error": f"Invalid numerical inputs: {err}",
            "predicted_change_percent": None
        }), 400

    pred = predict_stock_impact(
        severity_score=severity,
        users_affected_millions=users,
        data_sensitivity_score=sensitivity
    )
    status_code = 400 if pred.get("status") == "UNAVAILABLE" else 200
    return jsonify(pred), status_code



@app.route("/api/ml/metrics", methods=["GET"])
def get_ml_metrics():
    """Returns staged LOOCV evaluation scorecard and baseline comparisons."""
    metrics_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model_metrics.json")
    if os.path.exists(metrics_path):
        try:
            with open(metrics_path, "r", encoding="utf-8") as f:
                return jsonify(json.load(f))
        except Exception as e:
            return jsonify({"error": str(e)}), 500
    return jsonify({"error": "model_metrics.json not found"}), 404


@app.route("/api/health", methods=["GET"])
def get_health_status():
    """Returns deep operational telemetry and database health."""
    conn = get_db()
    c = conn.cursor()
    tables = [
        "threats", "stock_prices", "correlations", "current_stock_data",
        "processed_articles", "predictions", "event_studies",
        "cisa_kev_entries", "monitored_stocks"
    ]
    row_counts = {}
    for tbl in tables:
        try:
            c.execute(f"SELECT COUNT(*) as cnt FROM {tbl}")
            row_counts[tbl] = c.fetchone()["cnt"]
        except Exception:
            row_counts[tbl] = 0

    c.execute("SELECT MAX(timestamp) as latest_ts FROM current_stock_data")
    latest_ts_row = c.fetchone()
    latest_ts = latest_ts_row["latest_ts"] if latest_ts_row else None
    conn.close()

    now_utc = datetime.now(timezone.utc)
    try:
        import zoneinfo
        ist_zone = zoneinfo.ZoneInfo("Asia/Kolkata")
        now_ist = now_utc.astimezone(ist_zone)
        now_ist_str = now_ist.strftime("%Y-%m-%d %H:%M:%S IST")
    except Exception:
        now_ist_str = "Unavailable"

    return jsonify({
        "status": "operational",
        "brand": "THREATPULSE",
        "version": "2.2.0-commercial",
        "request_time": now_utc.isoformat(),
        "server_time_utc": now_utc.strftime("%Y-%m-%d %H:%M:%S UTC"),
        "server_time_ist": now_ist_str,
        "timestamp_utc": now_utc.strftime("%Y-%m-%d %H:%M:%S UTC"),  # Preserved backward-compatible alias
        "timestamp_ist": now_ist_str,  # Preserved backward-compatible alias
        "data_feed_mode": "delayed_public_feed (yfinance, ~15m delay during market hours, previous close off-hours)",
        "database": {

            "status": "healthy",
            "tables": row_counts
        },
        "ai_agent": {
            "provider": "Google Gemini",
            "preferred_model": "gemini-3.8-flash",
            "fallback_model": "gemini-2.5-flash",
            "status": "active"
        },
        "ml_engine": {
            "status": "loaded",
            "model_file": "threat_impact_model.joblib",
            "evaluation": "LOOCV (14 historical incidents)"
        },
        "market_sessions": {
            "US": get_market_session_status("US"),
            "India": get_market_session_status("INDIA"),
            "latest_quote_timestamp": latest_ts
        },
        "scheduler": get_scheduler_status()
    })


@app.route("/api/prediction_performance", methods=["GET"])
def prediction_performance_endpoint():
    """Return tracked predictions and metrics from completed evaluations."""
    return jsonify(get_prediction_performance())


@app.route("/api/predictions/evaluate", methods=["POST"])
def evaluate_predictions_endpoint():
    """On-demand evaluation against real yfinance daily closes."""
    result = evaluate_pending_predictions()
    return jsonify({**result, "performance": get_prediction_performance()})


@app.route("/api/event_study", methods=["GET"])
def event_study_data_endpoint():
    """Return stored studies and evaluated-only aggregate statistics."""
    return jsonify(get_event_study_data())


@app.route("/api/event_study/run", methods=["POST"])
def run_event_study_endpoint():
    """Evaluate the curated historical breach dataset against real market bars."""
    return jsonify(run_event_study())


@app.route("/api/analyze_threat", methods=["POST"])
def analyze_threat_endpoint():
    """
    Submits a threat description or news advisory to run real-time
    Gemini AI extraction, ML impact prediction, risk scoring, DB storage,
    and automatic Discord alert dispatch.
    """
    data = request.get_json() or {}
    company = data.get("company", "").strip()
    threat_type = data.get("threat_type", data.get("type", "Data Breach")).strip()
    description = data.get("description", "").strip()

    if not company or not description:
        return jsonify({"error": "Company and Description fields are required."}), 400

    result = ingest_and_evaluate_threat(
        company=company,
        threat_type=threat_type,
        description=description
    )

    return jsonify(result)


@app.route("/api/refresh_stocks", methods=["POST"])
def refresh_stocks_endpoint():
    """Triggers an immediate live market fetch for watchlist tickers."""
    market_data = update_monitored_stocks(DEFAULT_WATCHLIST)
    return jsonify({
        "status": "success",
        "updated_tickers": list(market_data.keys()),
        "count": len(market_data)
    })


@app.route("/api/collector/status", methods=["GET"])
def collector_status_endpoint():
    """Return the in-process automatic collector's current state."""
    return jsonify(get_scheduler_status())


@app.route("/api/collector/start", methods=["POST"])
def collector_start_endpoint():
    """Start the persistent RSS-to-alert collection scheduler."""
    data = request.get_json(silent=True) or {}
    return jsonify(start_background_scheduler(
        interval_minutes=data.get("interval_minutes", 30),
        article_limit=data.get("article_limit", 1),
    ))


@app.route("/api/collector/stop", methods=["POST"])
def collector_stop_endpoint():
    """Request a graceful stop after the active collector phase."""
    return jsonify(stop_background_scheduler())


@app.route("/api/test_alert", methods=["POST"])
def test_alert_endpoint():
    """Dispatches a test card to verify Discord webhook."""
    data = request.get_json() or {}
    ticker = data.get("ticker", "MSFT")
    risk_score = int(data.get("risk_score", 85))
    recommendation = data.get("recommendation", "REVIEW")
    summary = data.get("summary", "Test diagnostic alert issued from Cyber Threat Platform Web Dashboard.")

    send_discord_alert(
        ticker=ticker,
        risk_score=risk_score,
        recommendation=recommendation,
        threat_summary=summary
    )
    return jsonify({"status": "Alert sent to Discord webhook successfully!"})


if __name__ == "__main__":
    print("=" * 65)
    print("STARTING CYBER THREAT STOCK INTEL DASHBOARD SERVER")
    print("Serving UI at: http://127.0.0.1:5000")
    print("=" * 65)

    start_background_scheduler(interval_minutes=30)

    app.run(host="127.0.0.1", port=5000, debug=False)
