"""
research/market_event_study.py

Event-Study and Financial Labeling Engine for THREATPULSE Research.
Implements:
1. Exact trading-session alignment across US (NYSE/NASDAQ) and Indian (NSE/BSE) market calendars.
2. Market-Model Parameter Estimation via OLS over an unpolluted pre-event estimation window [-120, -21].
3. Calculation of:
   - Raw Stock Returns (Stock-only percentage change)
   - Benchmark Returns (Broad market index)
   - Benchmark-Adjusted Returns (Constant Beta = 1.0)
   - Market-Model Abnormal Returns (Risk-adjusted alpha & beta)
   - Multi-day Cumulative Abnormal Returns (CAR)
   - Standardized Abnormal Return Z-Scores
"""

from datetime import datetime, date, timedelta, time as dtime
from typing import Dict, Any, Optional, Tuple, List
import numpy as np
import pandas as pd
import pytz
import yfinance as yf
from scipy import stats


US_MARKET_TZ = pytz.timezone("America/New_York")
INDIA_MARKET_TZ = pytz.timezone("Asia/Kolkata")

# Market Regular Hours
US_OPEN = dtime(9, 30)
US_CLOSE = dtime(16, 0)
INDIA_OPEN = dtime(9, 15)
INDIA_CLOSE = dtime(15, 30)


def align_disclosure_to_trading_session(
    disclosure_dt_utc: datetime,
    exchange: str = "NASDAQ"
) -> Dict[str, Any]:
    """
    Maps an exact UTC disclosure timestamp to its market session and determines
    the first trading day eligible to react (effective T0).

    Rules:
    - Pre-Market: Released before market open -> T0 is today.
    - Regular Trading Hours: Released during market session -> T0 is today (intraday shock).
    - Post-Market: Released after market close -> T0 is the next trading session.
    - Weekend/Holiday: Exchange closed -> T0 is the next trading session.
    """
    is_india = exchange.upper() in {"NSE", "BSE"} or ".NS" in exchange or ".BO" in exchange
    tz = INDIA_MARKET_TZ if is_india else US_MARKET_TZ
    m_open = INDIA_OPEN if is_india else US_OPEN
    m_close = INDIA_CLOSE if is_india else US_CLOSE

    local_dt = disclosure_dt_utc.astimezone(tz)
    local_time = local_dt.time()
    weekday = local_dt.weekday() # 0=Mon, ..., 6=Sun

    if weekday >= 5: # Saturday or Sunday
        session = "WEEKEND_HOLIDAY"
        days_ahead = 7 - weekday # Next Monday
        effective_date = (local_dt + timedelta(days=days_ahead)).date()
    elif local_time < m_open:
        session = "PRE_MARKET"
        effective_date = local_dt.date()
    elif local_time <= m_close:
        session = "REGULAR_HOURS"
        effective_date = local_dt.date()
    else: # local_time > m_close
        session = "POST_MARKET"
        if weekday == 4: # Friday after-hours -> Monday
            effective_date = (local_dt + timedelta(days=3)).date()
        else:
            effective_date = (local_dt + timedelta(days=1)).date()

    return {
        "session_classification": session,
        "local_timestamp": local_dt.isoformat(),
        "effective_t0_trading_date": effective_date.isoformat(),
        "exchange_timezone": tz.zone
    }


def fetch_aligned_price_series(
    ticker: str,
    benchmark_ticker: str,
    target_date: date,
    pre_event_days: int = 250,
    post_event_days: int = 30,
    use_adjusted_prices: bool = True
) -> Tuple[Optional[pd.DataFrame], Optional[str]]:
    """
    Fetches daily closing prices for both company and benchmark from Yahoo Finance,
    aligning them onto mutual trading sessions.
    Uses split/dividend adjusted prices by default to eliminate artificial corporate action jumps.
    """
    start_dt = target_date - timedelta(days=pre_event_days)
    end_dt = target_date + timedelta(days=post_event_days)

    try:
        stock_df = yf.Ticker(ticker).history(start=start_dt.isoformat(), end=end_dt.isoformat(), auto_adjust=False)
        bm_df = yf.Ticker(benchmark_ticker).history(start=start_dt.isoformat(), end=end_dt.isoformat(), auto_adjust=False)

        if stock_df.empty or "Close" not in stock_df.columns:
            return None, f"No price history returned for stock {ticker} (may be delisted or inactive)"
        if bm_df.empty or "Close" not in bm_df.columns:
            return None, f"No price history returned for benchmark {benchmark_ticker}"

        # Clean dates to timezone-naive dates
        price_col = "Adj Close" if (use_adjusted_prices and "Adj Close" in stock_df.columns) else "Close"
        bm_price_col = "Adj Close" if (use_adjusted_prices and "Adj Close" in bm_df.columns) else "Close"

        stock_close = stock_df[price_col].dropna()
        bm_close = bm_df[bm_price_col].dropna()

        stock_close.index = pd.to_datetime(stock_close.index).date
        bm_close.index = pd.to_datetime(bm_close.index).date

        aligned = pd.DataFrame({
            "stock_close": stock_close,
            "bm_close": bm_close,
            "stock_unadj_close": stock_df["Close"].dropna() if "Close" in stock_df.columns else stock_close,
            "bm_unadj_close": bm_df["Close"].dropna() if "Close" in bm_df.columns else bm_close
        }).dropna()

        aligned = aligned[(aligned["stock_close"] > 0) & (aligned["bm_close"] > 0)].sort_index()

        if len(aligned) < 60:
            return None, f"Insufficient mutual trading sessions: {len(aligned)} < 60"

        return aligned, None
    except Exception as exc:
        return None, f"Error fetching price series: {exc}"


def compute_market_model_labels(
    ticker: str,
    benchmark_ticker: str,
    effective_t0_date: date,
    history_df: Optional[pd.DataFrame] = None,
    use_adjusted_prices: bool = True
) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    """
    Calculates explicit raw returns, benchmark-adjusted returns, and Market-Model abnormal returns.

    Estimation Window: [-120, -21] trading sessions prior to effective T0.
    Event Window: [-1, +2] trading sessions around effective T0.

    Explicit Holding Periods:
    - Previous close to T0 close: R[T-1 -> T0] (Immediate reaction)
    - T0 close to T+1 close: R[T0 -> T+1] (Post-disclosure follow-through)
    - Previous close to T+1 close: R[T-1 -> T+1] (Cumulative 2-day holding period)
    - Previous close to T+2 close: R[T-1 -> T+2] (Cumulative 3-day holding period)

    Model: R_it = alpha_i + beta_i * R_mt + e_it
    Abnormal Return: AR_it = R_it - (alpha_hat + beta_hat * R_mt)
    Cumulative Abnormal Return: CAR[0, 1] = AR_T0 + AR_T1
    """
    if history_df is None:
        df, err = fetch_aligned_price_series(
            ticker, benchmark_ticker, effective_t0_date, use_adjusted_prices=use_adjusted_prices
        )
        if err:
            return None, err
    else:
        df = history_df.copy()

    trading_dates = list(df.index)

    # Locate T0 index
    eligible_t0 = [d for d in trading_dates if d >= effective_t0_date]
    if not eligible_t0:
        return None, f"No trading session found on or after {effective_t0_date}"

    t0_date = eligible_t0[0]
    t0_idx = trading_dates.index(t0_date)

    if t0_idx < 1:
        return None, "No pre-event trading session available for baseline T-1"

    # Identify key event dates
    t_minus_1_idx = t0_idx - 1
    t_minus_1_date = trading_dates[t_minus_1_idx]

    post_sessions = trading_dates[t0_idx:]
    if len(post_sessions) < 3:
        return None, "Fewer than 3 post-event sessions available for T+1 and T+2"

    t_plus_1_date = post_sessions[1]
    t_plus_2_date = post_sessions[2]

    # Monotonicity assertion
    if not (t_minus_1_date < t0_date < t_plus_1_date < t_plus_2_date):
        return None, f"Trading session monotonicity violated: {t_minus_1_date} < {t0_date} < {t_plus_1_date} < {t_plus_2_date}"

    # Prices
    p_stock_tm1 = float(df.loc[t_minus_1_date, "stock_close"])
    p_stock_t0 = float(df.loc[t0_date, "stock_close"])
    p_stock_tp1 = float(df.loc[t_plus_1_date, "stock_close"])
    p_stock_tp2 = float(df.loc[t_plus_2_date, "stock_close"])

    p_bm_tm1 = float(df.loc[t_minus_1_date, "bm_close"])
    p_bm_t0 = float(df.loc[t0_date, "bm_close"])
    p_bm_tp1 = float(df.loc[t_plus_1_date, "bm_close"])
    p_bm_tp2 = float(df.loc[t_plus_2_date, "bm_close"])

    # 1. Explicit Raw Stock Returns
    raw_tm1_to_t0 = ((p_stock_t0 - p_stock_tm1) / p_stock_tm1) * 100.0
    raw_t0_to_t1 = ((p_stock_tp1 - p_stock_t0) / p_stock_t0) * 100.0
    raw_tm1_to_t1 = ((p_stock_tp1 - p_stock_tm1) / p_stock_tm1) * 100.0
    raw_tm1_to_t2 = ((p_stock_tp2 - p_stock_tm1) / p_stock_tm1) * 100.0

    # 2. Explicit Benchmark Returns
    bm_tm1_to_t0 = ((p_bm_t0 - p_bm_tm1) / p_bm_tm1) * 100.0
    bm_t0_to_t1 = ((p_bm_tp1 - p_bm_t0) / p_bm_t0) * 100.0
    bm_tm1_to_t1 = ((p_bm_tp1 - p_bm_tm1) / p_bm_tm1) * 100.0
    bm_tm1_to_t2 = ((p_bm_tp2 - p_bm_tm1) / p_bm_tm1) * 100.0

    # 3. Explicit Benchmark-Adjusted Returns (Constant Beta = 1.0)
    abn_tm1_to_t0 = raw_tm1_to_t0 - bm_tm1_to_t0
    abn_t0_to_t1 = raw_t0_to_t1 - bm_t0_to_t1
    abn_tm1_to_t1 = raw_tm1_to_t1 - bm_tm1_to_t1
    abn_tm1_to_t2 = raw_tm1_to_t2 - bm_tm1_to_t2

    # 4. Market-Model Estimation Window [-120, -21]
    # Slicing includes index t0_idx - 21 and excludes t0_idx - 20 (zero event-window leakage)
    est_start_idx = max(0, t0_idx - 120)
    est_end_idx = max(0, t0_idx - 20)

    market_model_result = {
        "alpha": None,
        "beta": None,
        "r2": None,
        "residual_std": None,
        "ar_t0": None,
        "ar_t1": None,
        "car01": None,
        "z_score": None,
        "n_obs": 0
    }

    if est_end_idx - est_start_idx >= 30:
        est_dates = trading_dates[est_start_idx:est_end_idx]
        est_df = df.loc[est_dates].copy()

        # Calculate daily percentage returns in estimation window
        stock_ret_est = est_df["stock_close"].pct_change().dropna().values
        bm_ret_est = est_df["bm_close"].pct_change().dropna().values

        if len(stock_ret_est) >= 30 and len(bm_ret_est) >= 30:
            reg = stats.linregress(bm_ret_est, stock_ret_est)
            alpha_daily = float(reg.intercept)
            beta_val = float(reg.slope)
            r2_val = float(reg.rvalue ** 2)

            residuals = stock_ret_est - (alpha_daily + beta_val * bm_ret_est)
            residual_std_daily = float(np.std(residuals, ddof=2))

            # Expected return on T0:
            expected_stock_ret_t0 = alpha_daily + beta_val * (bm_tm1_to_t0 / 100.0)
            ar_t0 = raw_tm1_to_t0 - (expected_stock_ret_t0 * 100.0)

            # Expected return on T+1:
            expected_stock_ret_t1 = alpha_daily + beta_val * (bm_t0_to_t1 / 100.0)
            ar_t1 = raw_t0_to_t1 - (expected_stock_ret_t1 * 100.0)

            car01 = ar_t0 + ar_t1
            z_score = (ar_t0 / (residual_std_daily * 100.0)) if residual_std_daily > 0 else 0.0

            market_model_result = {
                "alpha": round(alpha_daily * 100.0, 4), # in %
                "beta": round(beta_val, 4),
                "r2": round(r2_val, 4),
                "residual_std": round(residual_std_daily * 100.0, 4), # in %
                "ar_t0": round(ar_t0, 4),
                "ar_t1": round(ar_t1, 4),
                "car01": round(car01, 4),
                "z_score": round(z_score, 2),
                "n_obs": len(stock_ret_est)
            }

    return {
        "t_minus_1_date": t_minus_1_date.isoformat(),
        "t0_date": t0_date.isoformat(),
        "t_plus_1_date": t_plus_1_date.isoformat(),
        "t_plus_2_date": t_plus_2_date.isoformat(),
        "p_stock_t_minus_1": round(p_stock_tm1, 2),
        "p_stock_t0": round(p_stock_t0, 2),
        "p_stock_t_plus_1": round(p_stock_tp1, 2),
        "p_stock_t_plus_2": round(p_stock_tp2, 2),
        "p_bm_t_minus_1": round(p_bm_tm1, 2),
        "p_bm_t0": round(p_bm_t0, 2),
        "p_bm_t_plus_1": round(p_bm_tp1, 2),
        "p_bm_t_plus_2": round(p_bm_tp2, 2),
        # Explicit fields
        "raw_return_tm1_to_t0": round(raw_tm1_to_t0, 4),
        "raw_return_t0_to_t1": round(raw_t0_to_t1, 4),
        "raw_return_tm1_to_t1": round(raw_tm1_to_t1, 4),
        "raw_return_tm1_to_t2": round(raw_tm1_to_t2, 4),
        "benchmark_return_tm1_to_t0": round(bm_tm1_to_t0, 4),
        "benchmark_return_t0_to_t1": round(bm_t0_to_t1, 4),
        "benchmark_return_tm1_to_t1": round(bm_tm1_to_t1, 4),
        "benchmark_return_tm1_to_t2": round(bm_tm1_to_t2, 4),
        "abnormal_return_tm1_to_t0": round(abn_tm1_to_t0, 4),
        "abnormal_return_t0_to_t1": round(abn_t0_to_t1, 4),
        "abnormal_return_tm1_to_t1": round(abn_tm1_to_t1, 4),
        "abnormal_return_tm1_to_t2": round(abn_tm1_to_t2, 4),
        # Aliases
        "raw_return_1d": round(raw_tm1_to_t0, 4),
        "raw_return_2d": round(raw_tm1_to_t1, 4),
        "raw_return_3d": round(raw_tm1_to_t2, 4),
        "benchmark_return_1d": round(bm_tm1_to_t0, 4),
        "benchmark_return_2d": round(bm_tm1_to_t1, 4),
        "abnormal_return_1d": round(abn_tm1_to_t0, 4),
        "abnormal_return_2d": round(abn_tm1_to_t1, 4),
        "market_model": market_model_result
    }, None
