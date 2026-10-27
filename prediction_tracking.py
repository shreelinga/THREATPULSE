
"""Persistence, real-market evaluation, and metrics for ML predictions."""

import math
import re
import sqlite3
from pathlib import Path
from datetime import datetime, timedelta, timezone

import yfinance as yf


BASE_DIR = Path(__file__).resolve().parent
DB_NAME = str(BASE_DIR / "threat_stock_data.db")
HORIZONS = (1, 3, 5)


def insert_prediction(
    cursor,
    threat_id,
    ticker,
    prediction,
    prediction_timestamp=None,
    model_version="v1.0.0",
    target_definition=None
):
    """Insert a prediction with strict deduplication and validation.
    Target definition standard: 'raw_return' (stock only) vs 'abnormal_return' (benchmark-adjusted).
    """
    ticker = str(ticker or "").strip().upper()
    if not re.fullmatch(r"[A-Z][A-Z0-9.\-]{0,14}", ticker) or ticker == "UNKNOWN":
        raise ValueError(f"Refusing prediction for invalid ticker: {ticker!r}")

    if target_definition is None:
        target_definition = (
            prediction.get("target_definition")
            if isinstance(prediction, dict)
            else "raw_return"
        ) or "raw_return"

    # Check for duplicate prediction (same threat_id, ticker, model_version, target_definition)
    cursor.execute(
        """SELECT id FROM predictions
           WHERE threat_id = ? AND ticker = ? AND model_version = ? AND target_definition = ?""",
        (threat_id, ticker, model_version, target_definition)
    )
    existing = cursor.fetchone()
    if existing:
        return existing[0]

    impact = float(prediction["predicted_change_percent"])
    if not math.isfinite(impact):
        raise ValueError("Prediction impact must be a finite number")

    direction = prediction.get("predicted_direction") or (
        "UP" if impact > 0 else "DOWN" if impact < 0 else "FLAT"
    )
    direction = str(direction).upper()
    if direction not in {"UP", "DOWN", "FLAT"}:
        raise ValueError(f"Invalid predicted direction: {direction!r}")

    confidence = prediction.get("confidence")
    if confidence is not None:
        confidence = float(confidence)
        if not math.isfinite(confidence):
            confidence = None

    timestamp = prediction_timestamp or datetime.now(timezone.utc).isoformat()
    review_needed = 1 if ticker == "SA" else 0

    try:
        cursor.execute(
            """INSERT INTO predictions (
                   threat_id, ticker, prediction_timestamp, predicted_direction,
                   predicted_impact_percent, confidence, status,
                   model_version, target_definition, review_needed
               ) VALUES (?, ?, ?, ?, ?, ?, 'PENDING', ?, ?, ?)""",
            (
                threat_id,
                ticker,
                timestamp,
                direction,
                impact,
                confidence,
                model_version,
                target_definition,
                review_needed
            ),
        )
        return cursor.lastrowid
    except sqlite3.IntegrityError:
        cursor.execute(
            """SELECT id FROM predictions
               WHERE threat_id = ? AND ticker = ? AND model_version = ? AND target_definition = ?""",
            (threat_id, ticker, model_version, target_definition)
        )
        row = cursor.fetchone()
        return row[0] if row else None


def record_prediction(
    threat_id,
    ticker,
    prediction,
    prediction_timestamp=None,
    model_version="v1.0.0",
    target_definition=None,
    db_name=DB_NAME,
):
    """Store one model output without modifying the associated threat row."""
    if target_definition is None:
        target_definition = (
            prediction.get("target_definition")
            if isinstance(prediction, dict)
            else "raw_return"
        ) or "raw_return"
    if "Post-Disclosure" in str(target_definition):
        target_definition = "raw_return"
    conn = sqlite3.connect(db_name, timeout=30)
    try:
        prediction_id = insert_prediction(
            conn.cursor(),
            threat_id,
            ticker,
            prediction,
            prediction_timestamp,
            model_version,
            target_definition,
        )
        conn.commit()
        return prediction_id
    finally:
        conn.close()


def _market_returns(ticker, prediction_timestamp, history_fetcher=None):
    """Calculate returns from real daily bars for available trading horizons."""
    ticker = str(ticker or "").strip().upper()
    if not re.fullmatch(r"[A-Z][A-Z0-9.\-]{0,14}", ticker) or ticker == "UNKNOWN":
        return {}

    fetch = history_fetcher or (
        lambda symbol, start, end: yf.Ticker(symbol).history(
            start=start,
            end=end,
            interval="1d",
            auto_adjust=False,
        )
    )

    stamp = datetime.fromisoformat(
        prediction_timestamp.replace("Z", "+00:00")
    )
    prediction_date = stamp.date()
    start = (prediction_date - timedelta(days=14)).isoformat()
    end = (
        datetime.now(timezone.utc).date() + timedelta(days=2)
    ).isoformat()

    prices = fetch(ticker, start, end)
    if prices is None or prices.empty or "Close" not in prices:
        return {}

    closes = prices["Close"].dropna()
    dated = []

    for index, close in closes.items():
        bar_date = (
            index.date()
            if hasattr(index, "date")
            else datetime.fromisoformat(str(index)[:10]).date()
        )
        close = float(close)

        if close > 0:
            dated.append((bar_date, close))

    dated.sort(key=lambda item: item[0])

    prior = [item for item in dated if item[0] < prediction_date]
    following = [item for item in dated if item[0] > prediction_date]

    if not prior:
        return {}

    baseline = prior[-1][1]
    result = {}

    for horizon in HORIZONS:
        if len(following) >= horizon:
            result[horizon] = round(
                (following[horizon - 1][1] / baseline - 1) * 100,
                6,
            )

    return result


def evaluate_pending_predictions(
    prediction_id=None,
    db_name=DB_NAME,
    history_fetcher=None,
):
    """Record available horizons independently; retain pending status until all horizons complete."""
    conn = sqlite3.connect(db_name, timeout=30)
    conn.row_factory = sqlite3.Row

    try:
        sql = "SELECT * FROM predictions WHERE status = 'PENDING'"
        params = ()

        if prediction_id is not None:
            sql += " AND id = ?"
            params = (prediction_id,)

        rows = conn.execute(sql + " ORDER BY id", params).fetchall()
        updated = []

        for row in rows:
            try:
                returns = _market_returns(
                    row["ticker"],
                    row["prediction_timestamp"],
                    history_fetcher,
                )
            except Exception as exc:
                conn.execute(
                    "UPDATE predictions SET failure_reason = ? WHERE id = ?",
                    (str(exc), row["id"])
                )
                continue

            values = {
                horizon: (
                    returns[horizon]
                    if returns.get(horizon) is not None
                    else row[f"actual_{horizon}d_return"]
                )
                for horizon in HORIZONS
            }

            evaluated = all(
                values[horizon] is not None
                for horizon in HORIZONS
            )

            # Save to both legacy actual_Xd_return and standardized raw_return_Xd columns
            columns = {col[1] for col in conn.execute("PRAGMA table_info(predictions)").fetchall()}
            if "raw_return_1d" in columns:
                conn.execute(
                    """UPDATE predictions
                       SET actual_1d_return = ?,
                           actual_3d_return = ?,
                           actual_5d_return = ?,
                           raw_return_1d = ?,
                           raw_return_3d = ?,
                           raw_return_5d = ?,
                           status = ?,
                           evaluated_at = ?
                       WHERE id = ?""",
                    (
                        values[1],
                        values[3],
                        values[5],
                        values[1],
                        values[3],
                        values[5],
                        "EVALUATED" if evaluated else "PENDING",
                        (
                            datetime.now(timezone.utc).isoformat()
                            if evaluated
                            else None
                        ),
                        row["id"],
                    ),
                )
            else:
                conn.execute(
                    """UPDATE predictions
                       SET actual_1d_return = ?,
                           actual_3d_return = ?,
                           actual_5d_return = ?,
                           status = ?,
                           evaluated_at = ?
                       WHERE id = ?""",
                    (
                        values[1],
                        values[3],
                        values[5],
                        "EVALUATED" if evaluated else "PENDING",
                        (
                            datetime.now(timezone.utc).isoformat()
                            if evaluated
                            else None
                        ),
                        row["id"],
                    ),
                )
            updated.append(row["id"])

        conn.commit()

        return {
            "checked": len(rows),
            "updated": updated,
        }
    finally:
        conn.close()


def get_prediction_performance(db_name=DB_NAME, limit=50):
    """Calculate horizon metrics independently with MAE, RMSE, and baseline comparisons."""
    conn = sqlite3.connect(db_name, timeout=30)
    conn.row_factory = sqlite3.Row

    try:
        counts = conn.execute(
            """SELECT COUNT(*) AS total,
                      SUM(CASE WHEN status = 'EVALUATED' THEN 1 ELSE 0 END)
                          AS completed,
                      SUM(CASE WHEN status = 'PENDING' THEN 1 ELSE 0 END)
                          AS pending
               FROM predictions"""
        ).fetchone()

        horizons = {}

        for horizon in HORIZONS:
            actual_col = f"actual_{horizon}d_return"

            rows = conn.execute(
                f"""SELECT predicted_direction,
                           predicted_impact_percent,
                           {actual_col} AS actual
                    FROM predictions
                    WHERE {actual_col} IS NOT NULL"""
            ).fetchall()

            count = len(rows)
            if count > 0:
                hits = sum(
                    (
                        "UP"
                        if row["actual"] > 0
                        else "DOWN"
                        if row["actual"] < 0
                        else "FLAT"
                    ) == row["predicted_direction"]
                    for row in rows
                )
                errors = [row["predicted_impact_percent"] - row["actual"] for row in rows]
                actuals = [row["actual"] for row in rows]
                mae = sum(abs(e) for e in errors) / count
                rmse = math.sqrt(sum(e ** 2 for e in errors) / count)

                # Baseline 1: Zero-return model (predicting 0 raw stock return)
                base_zero_mae = sum(abs(a) for a in actuals) / count
                base_zero_rmse = math.sqrt(sum(a ** 2 for a in actuals) / count)

                # Baseline 2: Historical mean model
                hist_mean = sum(actuals) / count
                base_mean_mae = sum(abs(a - hist_mean) for a in actuals) / count
                base_mean_rmse = math.sqrt(sum((a - hist_mean) ** 2 for a in actuals) / count)

                horizons[f"{horizon}d"] = {
                    "count": count,
                    "target_type": "raw_return",
                    "direction_accuracy": round(hits / count, 4),
                    "mae": round(mae, 4),
                    "rmse": round(rmse, 4),
                    "baseline_zero_mae": round(base_zero_mae, 4),
                    "baseline_zero_rmse": round(base_zero_rmse, 4),
                    "baseline_mean_mae": round(base_mean_mae, 4),
                    "baseline_mean_rmse": round(base_mean_rmse, 4),
                    "beats_zero_baseline": mae < base_zero_mae,
                }
            else:
                horizons[f"{horizon}d"] = {
                    "count": 0,
                    "target_type": "raw_return",
                    "direction_accuracy": None,
                    "mae": None,
                    "rmse": None,
                    "baseline_zero_mae": None,
                    "baseline_zero_rmse": None,
                    "baseline_mean_mae": None,
                    "baseline_mean_rmse": None,
                    "beats_zero_baseline": False,
                }

        predictions = conn.execute(
            """SELECT id, threat_id, ticker, prediction_timestamp,
                      predicted_direction, predicted_impact_percent,
                      confidence, actual_1d_return, actual_3d_return,
                      actual_5d_return, status, evaluated_at,
                      model_version, target_definition, review_needed
               FROM predictions
               ORDER BY id DESC
               LIMIT ?""",
            (limit,),
        ).fetchall()

        pred_dicts = []
        for row in predictions:
            d = dict(row)
            # Standardized raw return aliases (stock only, not benchmark-adjusted)
            d["raw_return_1d"] = d.get("actual_1d_return")
            d["raw_return_3d"] = d.get("actual_3d_return")
            d["raw_return_5d"] = d.get("actual_5d_return")
            d["target_type"] = "raw_return"
            if "Post-Disclosure" in str(d.get("target_definition", "")):
                d["target_definition"] = "raw_return"
            pred_dicts.append(d)

        one_day = horizons["1d"]

        return {
            "prediction_count": counts["total"] or 0,
            "completed_count": counts["completed"] or 0,
            "pending_count": counts["pending"] or 0,
            "target_type": "raw_return",
            "target_definition": "Stock-Only Raw Return ((P[T]-P[T-1])/P[T-1])*100",
            "direction_accuracy": one_day["direction_accuracy"],
            "mae": one_day["mae"],
            "rmse": one_day["rmse"],
            "horizons": horizons,
            "predictions": pred_dicts,
            "movement_label": (
                "Observed raw stock movement; statistical association does not establish financial causation."
            ),
        }
    finally:
        conn.close()

