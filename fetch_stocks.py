import sys
import sqlite3
from datetime import datetime, timezone

if hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass

import yfinance as yf
from stock_universe import STOCK_UNIVERSE, DEFAULT_WATCHLIST
from market_utils import get_market_benchmark, format_currency, format_market_timestamp

# Monitored core watchlists for rapid polling
CORE_US_WATCHLIST = [
    "META",
    "TMUS",
    "GOOGL",
    "MAR",
    "UBER",
    "AAPL",
    "COF",
    "MSFT"
]

CORE_INDIA_WATCHLIST = [
    "RELIANCE.NS",
    "TCS.NS",
    "HDFCBANK.NS",
    "INFY.NS",
    "ICICIBANK.NS",
    "BHARTIARTL.NS",
    "SBIN.NS",
    "TATAMOTORS.NS"
]

# Combined monitored core watchlist
WATCHLIST = CORE_US_WATCHLIST + CORE_INDIA_WATCHLIST

# Complete lookup dictionary for all 105+ universe equities
COMPANY_NAMES = STOCK_UNIVERSE



def get_regular_session_data(ticker):
    stock = yf.Ticker(ticker)

    try:
        daily = stock.history(
            period="5d",
            interval="1d",
            prepost=False
        )

        if daily.empty:
            print(f"[WARN] No daily data available for {ticker}")
            return None

        daily = daily.dropna(subset=["Close"])

        if daily.empty:
            print(f"[WARN] No valid closing price for {ticker}")
            return None

        previous_close = None

        if len(daily) >= 2:
            previous_close = float(daily["Close"].iloc[-2])

        intraday = stock.history(
            period="1d",
            interval="1m",
            prepost=False
        )

        if not intraday.empty and not intraday.dropna(subset=["Close"]).empty:
            intraday_clean = intraday.dropna(subset=["Close"])
            latest_price = float(intraday_clean["Close"].iloc[-1])
            latest_timestamp = intraday_clean.index[-1]
            price_session = "regular-session"
        else:
            latest_price = float(daily["Close"].iloc[-1])
            latest_timestamp = daily.index[-1]
            price_session = "previous_close"

        # Check fast_info quote if available during active sessions
        quote = stock.fast_info.get("last_price")
        if quote is not None and float(quote) > 0:
            latest_price = float(quote)

        if previous_close is None:
            previous_close = float(daily["Close"].iloc[-1])

        change_percent = (
            (latest_price - previous_close) / previous_close
        ) * 100 if previous_close else 0.0

        benchmark_info = get_market_benchmark(ticker)
        is_in = ticker.upper().endswith(".NS") or ticker.upper().endswith(".BO")
        if is_in:
            market_name = "NSE" if ticker.upper().endswith(".NS") else ("BSE" if ticker.upper().endswith(".BO") else "NSE")
        else:
            market_name = "NASDAQ" if ticker.upper() in ("AAPL", "MSFT", "GOOGL", "NVDA", "CRWD") else "NYSE"

        timestamp = format_market_timestamp(latest_timestamp, market=benchmark_info["market"])
        retrieved_at = datetime.now(timezone.utc).isoformat()

        return {
            "ticker": ticker,
            "price": latest_price,
            "previous_close": previous_close,
            "change_percent": change_percent,
            "timestamp": timestamp,
            "exchange_timestamp": timestamp,
            "company": COMPANY_NAMES.get(ticker, ticker),
            "market": market_name,
            "market_region": benchmark_info["market"],
            "currency": benchmark_info["currency"],
            "currency_symbol": benchmark_info["currency_symbol"],
            "price_session": price_session,
            "retrieved_at": retrieved_at
        }

    except Exception as e:
        print(f"[ERROR] {ticker}: {e}")
        return None


DB_NAME = "threat_stock_data.db"


def save_stock_data(data):
    connection = sqlite3.connect(DB_NAME)
    cursor = connection.cursor()

    cursor.execute(
        """
        INSERT OR REPLACE INTO current_stock_data
        (
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
        """,
        (
            data["ticker"],
            data["price"],
            data["previous_close"],
            data["change_percent"],
            data["timestamp"],
            data["company"],
            data.get("exchange_timestamp", data["timestamp"]),
            data.get("market", "NYSE"),
            data.get("currency", "USD"),
            data.get("price_session", "previous_close"),
            data.get("retrieved_at", datetime.now(timezone.utc).isoformat())
        )
    )

    connection.commit()

    connection.close()


def fetch_stock_data(ticker):
    print(f"\n[STOCK] Fetching {ticker}...")

    data = get_regular_session_data(ticker)

    if not data:
        print(f"[WARN] No regular-session data available for {ticker}")
        return None

    save_stock_data(data)

    print(
        f"  {data['company']} ({ticker})"
    )

    formatted_price = format_currency(data['price'], ticker)
    formatted_prev = format_currency(data['previous_close'], ticker)

    print(
        f"  Price: {formatted_price}"
    )

    print(
        f"  Previous Close: {formatted_prev}"
    )

    print(
        f"  Change: {data['change_percent']:+.2f}%"
    )

    print(
        f"  Timestamp: {data['timestamp']}"
    )

    return data



def update_watchlist():
    print("=" * 70)
    print("LIVE STOCK DATA UPDATE")
    print("Regular-session data only")
    print("=" * 70)

    results = []

    for ticker in WATCHLIST:
        data = fetch_stock_data(ticker)

        if data:
            results.append(data)

    print()
    print("=" * 70)
    print(
        f"UPDATED {len(results)}/{len(WATCHLIST)} STOCKS"
    )
    print("=" * 70)

    return results


if __name__ == "__main__":
    update_watchlist()