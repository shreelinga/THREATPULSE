import csv
import os
import sqlite3
from datetime import datetime, timedelta, timezone

import pandas as pd


from market_utils import get_market_benchmark, is_indian_ticker

DB_PATH = "threat_stock_data.db"
DATASET_PATH = "training_dataset.csv"
DEFAULT_US_MARKET_TICKER = "SPY"
DEFAULT_INDIA_MARKET_TICKER = "^NSEI"
HORIZONS = (1, 3, 5)
MIN_PRE_EVENT_RETURNS = 60



def _connect():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def _table_columns(conn, table_name):
    return {row["name"] for row in conn.execute(f"PRAGMA table_info({table_name})")}


def _get_value(row, *names, default=None):
    for name in names:
        value = row.get(name)
        if value is not None and str(value).strip():
            return value
    return default


def _parse_date(value):
    if not value:
        return None
    try:
        return pd.Timestamp(str(value)).date()
    except (ValueError, TypeError):
        return None


def _seed_curated_events(conn):
    if not os.path.exists(DATASET_PATH):
        return 0

    inserted = 0
    with open(DATASET_PATH, "r", encoding="utf-8-sig", newline="") as file:
        for row in csv.DictReader(file):
            company = _get_value(row, "company", "company_name")
            ticker = _get_value(row, "ticker", "stock_ticker")
            event_date = _parse_date(
                _get_value(row, "event_date", "breach_date", "incident_date")
            )
            threat_type = _get_value(
                row, "threat_type", "incident_type", "threat", default="Unknown"
            )
            if not company or not ticker or not event_date:
                continue

            try:
                severity_score = float(
                    _get_value(row, "severity_score", "risk_score", default=0)
                )
            except (ValueError, TypeError):
                severity_score = 0.0

            severity_tier = _get_value(
                row, "severity_tier", "risk_tier", default="MEDIUM"
            )
            sector_ticker = _get_value(row, "sector_ticker", "benchmark_ticker")
            source_key = f"{ticker}|{event_date.isoformat()}|{threat_type}"

            cursor = conn.execute(
                """
                INSERT OR IGNORE INTO event_studies
                (source_key, company, ticker, threat_type, event_date,
                 severity_score, severity_tier, sector_ticker, status)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'PENDING')
                """,
                (source_key, company, ticker, threat_type, event_date.isoformat(),
                 severity_score, severity_tier, sector_ticker),
            )
            inserted += max(cursor.rowcount, 0)

    conn.commit()
    return inserted


def _normalise_prices(frame):
    if frame is None or frame.empty or "Close" not in frame.columns:
        return pd.Series(dtype=float)
    series = frame["Close"].copy()
    series.index = pd.to_datetime(series.index).date
    series = pd.to_numeric(series, errors="coerce")
    series = series[~series.index.duplicated(keep="last")]
    series = series.dropna()
    return series[series > 0].sort_index()


def _fetch_prices(ticker, start_date, end_date, cache):
    if not ticker:
        return pd.Series(dtype=float)
    ticker = str(ticker).strip().upper()
    if ticker not in cache:
        try:
            days = (end_date - start_date).days
            frame = fetch_price_window(ticker, start_date.isoformat(), days)
            cache[ticker] = _normalise_prices(frame)
        except Exception:
            cache[ticker] = pd.Series(dtype=float)
    return cache[ticker]


def _calculate_event(row, history_fetcher=None):
    event_date = _parse_date(row["event_date"])
    if event_date is None:
        return None, "invalid event date"

    company_ticker = str(row["ticker"]).strip().upper()
    bm_info = get_market_benchmark(company_ticker)
    market_ticker = bm_info["broad_benchmark"]

    sector_value = row["sector_ticker"]
    if sector_value and str(sector_value).strip():
        sector_ticker = str(sector_value).strip().upper()
    else:
        # Fallback to mapped sector benchmark if different from market benchmark
        candidate_sec = bm_info["sector_benchmark"]
        sector_ticker = candidate_sec if candidate_sec != market_ticker else None

    start_date = event_date - timedelta(days=180)
    end_date = event_date + timedelta(days=30)
    cache = {}

    if history_fetcher is None:
        def fetcher(ticker):
            return _fetch_prices(ticker, start_date, end_date, cache)
    else:
        def fetcher(ticker):
            try:
                frame = history_fetcher(
                    ticker, start_date.isoformat(), (end_date - start_date).days
                )
                return _normalise_prices(frame)
            except Exception:
                return pd.Series(dtype=float)

    company = fetcher(company_ticker)
    market = fetcher(market_ticker)
    if company.empty:
        return None, "company price history unavailable"
    if market.empty:
        return None, f"market benchmark history unavailable ({market_ticker})"

    sector = fetcher(sector_ticker) if sector_ticker else pd.Series(dtype=float)
    sector_available = not sector.empty
    benchmark_type = "sector" if sector_available else "market_fallback"
    benchmark_ticker = sector_ticker if sector_available else market_ticker
    benchmark = sector if sector_available else market


    aligned_dates = sorted(set(company.index) & set(benchmark.index) & set(market.index))
    if not aligned_dates:
        return None, "company and benchmark have no common trading dates"

    before_event = [day for day in aligned_dates if day < event_date]
    if not before_event:
        return None, "no common trading close before event date"

    base_date = before_event[-1]
    base_position = aligned_dates.index(base_date)
    post_event_dates = aligned_dates[base_position + 1:]
    if len(post_event_dates) < max(HORIZONS):
        return None, "fewer than five aligned post-event trading sessions"

    pre_dates = aligned_dates[:base_position + 1]
    if len(pre_dates) < MIN_PRE_EVENT_RETURNS + 1:
        return None, "fewer than 60 pre-event trading returns for volatility"

    company_returns = company.pct_change()
    benchmark_returns = benchmark.pct_change()
    pre_abnormal_returns = (
        company_returns.reindex(pre_dates) - benchmark_returns.reindex(pre_dates)
    ).dropna().tail(MIN_PRE_EVENT_RETURNS)
    if len(pre_abnormal_returns) < MIN_PRE_EVENT_RETURNS:
        return None, "insufficient aligned pre-event abnormal returns for volatility"

    volatility = float(pre_abnormal_returns.std(ddof=1))
    if pd.isna(volatility) or volatility <= 0:
        return None, "pre-event abnormal-return volatility unavailable or zero"

    base_company = float(company.loc[base_date])
    base_benchmark = float(benchmark.loc[base_date])
    base_market = float(market.loc[base_date])
    if min(base_company, base_benchmark, base_market) <= 0:
        return None, "invalid base closing price"

    results = {
        "company_return_1d": None, "company_return_3d": None, "company_return_5d": None,
        "raw_return_1d": None, "raw_return_3d": None, "raw_return_5d": None,
        "sector_return_1d": None, "sector_return_3d": None, "sector_return_5d": None,
        "market_return_1d": None, "market_return_3d": None, "market_return_5d": None,
        "abnormal_return_1d": None, "cumulative_abnormal_return_1d": None,
        "cumulative_abnormal_return_3d": None, "cumulative_abnormal_return_5d": None,
        "z_score": None, "z_score_label": None, "benchmark_type": benchmark_type,
        "benchmark_ticker": benchmark_ticker, "base_date": base_date.isoformat(),
    }

    abnormal_daily = []
    previous_day = base_date
    for index, day in enumerate(post_event_dates, start=1):
        daily_company_return = float(company.loc[day]) / float(company.loc[previous_day]) - 1
        daily_benchmark_return = float(benchmark.loc[day]) / float(benchmark.loc[previous_day]) - 1
        abnormal_daily.append(daily_company_return - daily_benchmark_return)

        if index in HORIZONS:
            company_return = float(company.loc[day]) / base_company - 1
            market_return = float(market.loc[day]) / base_market - 1
            # Standardized: raw_return = stock only, abnormal_return = stock minus benchmark
            results[f"company_return_{index}d"] = company_return * 100
            results[f"raw_return_{index}d"] = company_return * 100
            results[f"market_return_{index}d"] = market_return * 100

            if sector_available:
                sector_return = float(sector.loc[day]) / float(sector.loc[base_date]) - 1
                results[f"sector_return_{index}d"] = sector_return * 100

            results[f"cumulative_abnormal_return_{index}d"] = sum(abnormal_daily[:index]) * 100
            if index == 1:
                results["abnormal_return_1d"] = abnormal_daily[0] * 100
        previous_day = day

    z_score = abnormal_daily[0] / volatility
    results["z_score"] = z_score
    results["z_score_label"] = (
        "notable movement" if abs(z_score) >= 2 else "within normal range"
    )
    return results, None


def _evaluate_row(conn, row, columns):
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    result, error = _calculate_event(row)

    if error:
        conn.execute(
            "UPDATE event_studies SET last_error = ?, last_attempt_at = ? WHERE id = ?",
            (error, now, row["id"]),
        )
        return False

    updates = {key: value for key, value in result.items() if key in columns}
    updates.update({
        "status": "EVALUATED",
        "last_error": None,
        "last_attempt_at": now,
        "evaluated_at": now,
    })
    assignments = ", ".join(f'"{key}" = ?' for key in updates)
    conn.execute(
        f"UPDATE event_studies SET {assignments} WHERE id = ?",
        list(updates.values()) + [row["id"]],
    )
    return True


def run_event_study():
    conn = _connect()
    try:
        inserted = _seed_curated_events(conn)
        pending = conn.execute(
            "SELECT * FROM event_studies WHERE status = 'PENDING' ORDER BY event_date"
        ).fetchall()
        columns = _table_columns(conn, "event_studies")
        evaluated = 0

        for row in pending:
            try:
                if _evaluate_row(conn, row, columns):
                    evaluated += 1
            except Exception as exc:
                conn.execute(
                    "UPDATE event_studies SET last_error = ?, last_attempt_at = ? WHERE id = ?",
                    (str(exc), datetime.now(timezone.utc).isoformat(timespec="seconds"), row["id"]),
                )


        conn.commit()
        return {
            "inserted": inserted,
            "attempted": len(pending),
            "evaluated": evaluated,
            "pending": len(pending) - evaluated,
            "summary": get_event_study_summary(conn),
        }
    finally:
        conn.close()


def get_event_study_summary(conn=None):
    should_close = conn is None
    if conn is None:
        conn = _connect()
    try:
        rows = conn.execute(
            """
            SELECT company_return_1d, abnormal_return_1d, z_score, severity_tier
            FROM event_studies WHERE status = 'EVALUATED'
            """
        ).fetchall()
        raw_values = [
            float(row["company_return_1d"])
            for row in rows if row["company_return_1d"] is not None
        ]
        abnormal_values = [
            float(row["abnormal_return_1d"])
            for row in rows if row["abnormal_return_1d"] is not None
        ]
        z_values = [
            float(row["z_score"])
            for row in rows if row["z_score"] is not None
        ]
        severity_counts = {"CRITICAL": 0, "HIGH": 0, "MEDIUM": 0, "LOW": 0}
        for row in rows:
            tier = str(row["severity_tier"] or "").upper()
            if tier in severity_counts:
                severity_counts[tier] += 1

        return {
            "evaluated_count": len(rows),
            "average_raw_return_1d": (
                sum(raw_values) / len(raw_values) if raw_values else None
            ),
            "average_abnormal_return_1d": (
                sum(abnormal_values) / len(abnormal_values) if abnormal_values else None
            ),
            "average_z_score": sum(z_values) / len(z_values) if z_values else None,
            "count_by_severity_tier": severity_counts,
            "severity_tier_bands": {
                "CRITICAL": "9–10", "HIGH": "7–<9", "MEDIUM": "4–<7", "LOW": "<4"
            },
            "movement_label": "Observed relative movement (raw stock return vs benchmark abnormal return); no incident causation is inferred.",
        }
    finally:
        if should_close:
            conn.close()


def get_event_study_data():
    conn = _connect()
    try:
        rows = conn.execute(
            "SELECT * FROM event_studies ORDER BY event_date DESC, id DESC"
        ).fetchall()
        
        # Proactively seed if empty to ensure table is never unexpectedly blank
        if not rows and os.path.exists(DATASET_PATH):
            _seed_curated_events(conn)
            rows = conn.execute(
                "SELECT * FROM event_studies ORDER BY event_date DESC, id DESC"
            ).fetchall()

        studies_list = []
        for r in rows:
            item = dict(r)
            # Ensure standardized raw return aliases (stock only) and abnormal return (stock - benchmark)
            item["raw_return_1d"] = item.get("raw_return_1d") if item.get("raw_return_1d") is not None else item.get("company_return_1d")
            item["raw_return_3d"] = item.get("raw_return_3d") if item.get("raw_return_3d") is not None else item.get("company_return_3d")
            item["raw_return_5d"] = item.get("raw_return_5d") if item.get("raw_return_5d") is not None else item.get("company_return_5d")
            studies_list.append(item)

        summary = get_event_study_summary(conn)
        
        return {
            "source_dataset": "training_dataset.csv (14 Curated Major Cyber Incidents)",
            "curated_records_count": len(studies_list),
            "evaluated_count": summary.get("evaluated_count", 0),
            "events": studies_list,
            "studies": studies_list,
            "summary": summary,
        }
    finally:
        conn.close()

