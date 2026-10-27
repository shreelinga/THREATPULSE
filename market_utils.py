"""
market_utils.py

Unified market, benchmark, currency, and timezone utility module for
the Cyber Threat Stock Intelligence Platform.

Ensures strict market correctness for Indian (NSE/BSE) and US/International
equities, handles benchmark mapping, currency formatting, timezone-aware
quote freshness, and ambiguous ticker resolution safeguards.
"""

from datetime import datetime, date, timedelta, time as dtime, timezone
from typing import Dict, Any, Optional, Tuple
import pytz
import re


# Verified short US/Global tickers (1-2 letters) that are legitimate traded entities
VALID_SHORT_TICKERS = {
    "T": "AT&T Inc.",
    "V": "Visa Inc.",
    "C": "Citigroup Inc.",
    "F": "Ford Motor Co.",
    "K": "Kellanova",
    "M": "Macy's Inc.",
    "BA": "Boeing Co.",
    "GE": "GE Aerospace",
    "GS": "Goldman Sachs Group Inc.",
    "MS": "Morgan Stanley",
    "HD": "Home Depot Inc.",
    "KO": "Coca-Cola Co.",
    "PG": "Procter & Gamble Co.",
    "DE": "Deere & Co.",
    "MA": "Mastercard Inc.",
    "PM": "Philip Morris International",
    "BP": "BP p.l.c.",
    "EA": "Electronic Arts Inc.",
    "HP": "Helmerich & Payne",
}

# Stopwords and common 1-2 letter acronyms that should NEVER resolve dynamically
AMBIGUOUS_SHORT_TOKENS = {
    "SA", "IT", "AI", "ON", "IN", "IS", "OR", "AT", "BY", "AN", "AS", "BE",
    "DO", "GO", "HE", "IF", "ME", "MY", "NO", "OF", "OK", "SO", "TO", "UP",
    "US", "UK", "EU", "UN", "ID", "IP", "OS", "PC", "DB", "UI", "UX", "PR"
}

# Indian Stock Sectors and Benchmarks
INDIAN_SECTOR_MAP = {
    # Information Technology -> ^CNXIT
    "TCS.NS": {"sector": "Information Technology", "benchmark": "^CNXIT"},
    "INFY.NS": {"sector": "Information Technology", "benchmark": "^CNXIT"},
    "HCLTECH.NS": {"sector": "Information Technology", "benchmark": "^CNXIT"},
    "WIPRO.NS": {"sector": "Information Technology", "benchmark": "^CNXIT"},
    "TECHM.NS": {"sector": "Information Technology", "benchmark": "^CNXIT"},

    # Banking & Financial Services -> ^NSEBANK
    "HDFCBANK.NS": {"sector": "Financial Services", "benchmark": "^NSEBANK"},
    "ICICIBANK.NS": {"sector": "Financial Services", "benchmark": "^NSEBANK"},
    "SBIN.NS": {"sector": "Financial Services", "benchmark": "^NSEBANK"},
    "KOTAKBANK.NS": {"sector": "Financial Services", "benchmark": "^NSEBANK"},
    "AXISBANK.NS": {"sector": "Financial Services", "benchmark": "^NSEBANK"},
    "INDUSINDBK.NS": {"sector": "Financial Services", "benchmark": "^NSEBANK"},
    "BAJFINANCE.NS": {"sector": "Financial Services", "benchmark": "^NSEBANK"},
    "BAJAJFINSV.NS": {"sector": "Financial Services", "benchmark": "^NSEBANK"},
    "SHRIRAMFIN.NS": {"sector": "Financial Services", "benchmark": "^NSEBANK"},
    "SBILIFE.NS": {"sector": "Financial Services", "benchmark": "^NSEBANK"},
    "HDFCLIFE.NS": {"sector": "Financial Services", "benchmark": "^NSEBANK"},
    "LICI.NS": {"sector": "Financial Services", "benchmark": "^NSEBANK"},

    # Automotive -> ^CNXAUTO
    "TATAMOTORS.NS": {"sector": "Automotive", "benchmark": "^CNXAUTO"},
    "MARUTI.NS": {"sector": "Automotive", "benchmark": "^CNXAUTO"},
    "EICHERMOT.NS": {"sector": "Automotive", "benchmark": "^CNXAUTO"},
    "HEROMOTOCO.NS": {"sector": "Automotive", "benchmark": "^CNXAUTO"},

    # Pharmaceuticals & Healthcare
    "SUNPHARMA.NS": {"sector": "Healthcare", "benchmark": "^NSEI"},
    "DRREDDY.NS": {"sector": "Healthcare", "benchmark": "^NSEI"},
    "CIPLA.NS": {"sector": "Healthcare", "benchmark": "^NSEI"},
    "DIVISLAB.NS": {"sector": "Healthcare", "benchmark": "^NSEI"},
    "APOLLOHOSP.NS": {"sector": "Healthcare", "benchmark": "^NSEI"},

    # Energy, Oil & Gas, Power
    "RELIANCE.NS": {"sector": "Energy & Conglomerate", "benchmark": "^NSEI"},
    "ONGC.NS": {"sector": "Energy", "benchmark": "^NSEI"},
    "BPCL.NS": {"sector": "Energy", "benchmark": "^NSEI"},
    "IOC.NS": {"sector": "Energy", "benchmark": "^NSEI"},
    "POWERGRID.NS": {"sector": "Utilities", "benchmark": "^NSEI"},
    "NTPC.NS": {"sector": "Utilities", "benchmark": "^NSEI"},
    "COALINDIA.NS": {"sector": "Energy & Materials", "benchmark": "^NSEI"},

    # Metals & Materials
    "TATASTEEL.NS": {"sector": "Materials", "benchmark": "^NSEI"},
    "JSWSTEEL.NS": {"sector": "Materials", "benchmark": "^NSEI"},
    "ULTRACEMCO.NS": {"sector": "Materials", "benchmark": "^NSEI"},
    "GRASIM.NS": {"sector": "Materials", "benchmark": "^NSEI"},
    "PIDILITIND.NS": {"sector": "Materials", "benchmark": "^NSEI"},

    # Consumer & Retail
    "HINDUNILVR.NS": {"sector": "Consumer Goods", "benchmark": "^NSEI"},
    "ITC.NS": {"sector": "Consumer Goods", "benchmark": "^NSEI"},
    "NESTLEIND.NS": {"sector": "Consumer Goods", "benchmark": "^NSEI"},
    "BRITANNIA.NS": {"sector": "Consumer Goods", "benchmark": "^NSEI"},
    "TATACONSUM.NS": {"sector": "Consumer Goods", "benchmark": "^NSEI"},
    "TITAN.NS": {"sector": "Consumer Goods", "benchmark": "^NSEI"},
    "ASIANPAINT.NS": {"sector": "Consumer Goods", "benchmark": "^NSEI"},
    "TRENT.NS": {"sector": "Consumer Retail", "benchmark": "^NSEI"},

    # Conglomerate / Telecom / Real Estate
    "ADANIENT.NS": {"sector": "Conglomerate", "benchmark": "^NSEI"},
    "ADANIPORTS.NS": {"sector": "Infrastructure", "benchmark": "^NSEI"},
    "BHARTIARTL.NS": {"sector": "Telecommunications", "benchmark": "^NSEI"},
    "LT.NS": {"sector": "Engineering & Construction", "benchmark": "^NSEI"},
    "DLF.NS": {"sector": "Real Estate", "benchmark": "^NSEI"},
}

# US Stock Sectors and Benchmarks
US_SECTOR_MAP = {
    # Technology -> XLK
    "MSFT": {"sector": "Information Technology", "benchmark": "XLK"},
    "AAPL": {"sector": "Information Technology", "benchmark": "XLK"},
    "NVDA": {"sector": "Information Technology", "benchmark": "XLK"},
    "AVGO": {"sector": "Information Technology", "benchmark": "XLK"},
    "ORCL": {"sector": "Information Technology", "benchmark": "XLK"},
    "CRM": {"sector": "Information Technology", "benchmark": "XLK"},
    "AMD": {"sector": "Information Technology", "benchmark": "XLK"},
    "CSCO": {"sector": "Information Technology", "benchmark": "XLK"},
    "IBM": {"sector": "Information Technology", "benchmark": "XLK"},
    "INTU": {"sector": "Information Technology", "benchmark": "XLK"},
    "TXN": {"sector": "Information Technology", "benchmark": "XLK"},
    "AMAT": {"sector": "Information Technology", "benchmark": "XLK"},
    "NOW": {"sector": "Information Technology", "benchmark": "XLK"},
    "PLTR": {"sector": "Information Technology", "benchmark": "XLK"},
    "SWI": {"sector": "Information Technology", "benchmark": "XLK"},
    "CRWD": {"sector": "Information Technology", "benchmark": "XLK"},
    "OKTA": {"sector": "Information Technology", "benchmark": "XLK"},

    # Communication Services -> XLC
    "GOOGL": {"sector": "Communication Services", "benchmark": "XLC"},
    "META": {"sector": "Communication Services", "benchmark": "XLC"},
    "NFLX": {"sector": "Communication Services", "benchmark": "XLC"},
    "DIS": {"sector": "Communication Services", "benchmark": "XLC"},
    "TMUS": {"sector": "Communication Services", "benchmark": "XLC"},
    "T": {"sector": "Communication Services", "benchmark": "XLC"},

    # Consumer Discretionary -> XLY
    "AMZN": {"sector": "Consumer Discretionary", "benchmark": "XLY"},
    "TSLA": {"sector": "Consumer Discretionary", "benchmark": "XLY"},
    "HD": {"sector": "Consumer Discretionary", "benchmark": "XLY"},
    "BKNG": {"sector": "Consumer Discretionary", "benchmark": "XLY"},
    "MAR": {"sector": "Consumer Discretionary", "benchmark": "XLY"},
    "UBER": {"sector": "Consumer Discretionary", "benchmark": "XLY"},

    # Financials -> XLF
    "JPM": {"sector": "Financials", "benchmark": "XLF"},
    "BRK-B": {"sector": "Financials", "benchmark": "XLF"},
    "V": {"sector": "Financials", "benchmark": "XLF"},
    "MA": {"sector": "Financials", "benchmark": "XLF"},
    "BAC": {"sector": "Financials", "benchmark": "XLF"},
    "GS": {"sector": "Financials", "benchmark": "XLF"},
    "MS": {"sector": "Financials", "benchmark": "XLF"},
    "AXP": {"sector": "Financials", "benchmark": "XLF"},
    "BLK": {"sector": "Financials", "benchmark": "XLF"},
    "SPGI": {"sector": "Financials", "benchmark": "XLF"},
    "COF": {"sector": "Financials", "benchmark": "XLF"},

    # Healthcare -> XLV
    "LLY": {"sector": "Healthcare", "benchmark": "XLV"},
    "JNJ": {"sector": "Healthcare", "benchmark": "XLV"},
    "ABBV": {"sector": "Healthcare", "benchmark": "XLV"},
    "ISRG": {"sector": "Healthcare", "benchmark": "XLV"},

    # Consumer Staples -> XLP
    "WMT": {"sector": "Consumer Staples", "benchmark": "XLP"},
    "COST": {"sector": "Consumer Staples", "benchmark": "XLP"},
    "PG": {"sector": "Consumer Staples", "benchmark": "XLP"},
    "KO": {"sector": "Consumer Staples", "benchmark": "XLP"},
    "PM": {"sector": "Consumer Staples", "benchmark": "XLP"},

    # Energy -> XLE
    "XOM": {"sector": "Energy", "benchmark": "XLE"},
    "CVX": {"sector": "Energy", "benchmark": "XLE"},

    # Industrials -> XLI
    "GE": {"sector": "Industrials", "benchmark": "XLI"},
    "CAT": {"sector": "Industrials", "benchmark": "XLI"},
    "RTX": {"sector": "Industrials", "benchmark": "XLI"},
    "DE": {"sector": "Industrials", "benchmark": "XLI"},
    "EFX": {"sector": "Industrials", "benchmark": "XLI"},
}

TZ_INDIA = pytz.timezone("Asia/Kolkata")
TZ_US = pytz.timezone("America/New_York")


def is_indian_ticker(ticker: str) -> bool:
    """Checks whether the ticker belongs to Indian markets (.NS or .BO)."""
    if not ticker:
        return False
    t = str(ticker).strip().upper()
    return t.endswith(".NS") or t.endswith(".BO") or t.startswith("^NSE") or t.startswith("^BS") or t.startswith("^CNX")


def get_market_benchmark(ticker: str) -> Dict[str, Any]:
    """
    Unified market and benchmark resolution function.
    Returns market identity, broad-market benchmark, sector, sector benchmark,
    currency details, and market timezone.
    """
    clean_ticker = str(ticker).strip().upper() if ticker else "UNKNOWN"

    if is_indian_ticker(clean_ticker):
        market = "INDIA"
        broad_benchmark = "^NSEI"
        currency = "INR"
        currency_symbol = "₹"
        tz_name = "Asia/Kolkata"

        sector_info = INDIAN_SECTOR_MAP.get(clean_ticker)
        if sector_info:
            sector = sector_info["sector"]
            sector_benchmark = sector_info["benchmark"]
        else:
            sector = "Indian Market Equities"
            sector_benchmark = "^NSEI"

        return {
            "ticker": clean_ticker,
            "market": market,
            "broad_benchmark": broad_benchmark,
            "sector": sector,
            "sector_benchmark": sector_benchmark,
            "currency": currency,
            "currency_symbol": currency_symbol,
            "timezone": tz_name,
        }
    else:
        market = "US"
        broad_benchmark = "SPY"
        currency = "USD"
        currency_symbol = "$"
        tz_name = "America/New_York"

        sector_info = US_SECTOR_MAP.get(clean_ticker)
        if sector_info:
            sector = sector_info["sector"]
            sector_benchmark = sector_info["benchmark"]
        else:
            sector = "US Market Equities"
            sector_benchmark = "SPY"

        return {
            "ticker": clean_ticker,
            "market": market,
            "broad_benchmark": broad_benchmark,
            "sector": sector,
            "sector_benchmark": sector_benchmark,
            "currency": currency,
            "currency_symbol": currency_symbol,
            "timezone": tz_name,
        }


def format_currency(value: Optional[float], ticker: Optional[str] = None, currency_code: Optional[str] = None) -> str:
    """
    Formats price or monetary value with the correct currency symbol.
    Never formats missing/None value as 0.0.
    """
    if value is None:
        return "--"

    try:
        val = float(value)
    except (ValueError, TypeError):
        return "--"

    if currency_code:
        symbol = "₹" if currency_code.upper() in ("INR", "RS") else "$"
    elif ticker and is_indian_ticker(ticker):
        symbol = "₹"
    else:
        symbol = "$"

    return f"{symbol}{val:,.2f}"


def parse_market_timestamp(source_timestamp_str: Optional[str]) -> Tuple[Optional[datetime], Optional[str]]:
    """
    Parses any market timestamp string into a timezone-aware UTC datetime
    and returns (dt_utc, detected_tz_label).
    Handles ISO 8601 strings, 'YYYY-MM-DD HH:MM:SS [ET|IST|UTC]', and 'YYYY-MM-DD'.
    Returns (None, None) for missing or invalid strings.
    """
    if not source_timestamp_str or str(source_timestamp_str).strip() in ("--", "None", "UNKNOWN", "N/A", ""):
        return None, None

    raw = str(source_timestamp_str).strip()

    tz_label = "UTC"
    if "IST" in raw:
        tz_label = "IST"
    elif any(k in raw for k in ("ET", "EDT", "EST")):
        tz_label = "ET"
    elif "UTC" in raw or "Z" in raw:
        tz_label = "UTC"

    # Strip trailing timezone label
    clean = re.sub(r"\s*(IST|ET|EDT|EST|UTC)\s*$", "", raw).strip()

    try:
        # 1. ISO string with explicit offset (e.g. 2026-10-23T15:59:00-04:00 or Z)
        if "+" in clean or ("-" in clean[10:] and "T" in clean):
            dt = datetime.fromisoformat(clean.replace("Z", "+00:00"))
            return dt.astimezone(timezone.utc), tz_label

        # 2. Date only format (YYYY-MM-DD)
        if len(clean) == 10 and clean.count("-") == 2:
            naive = datetime.strptime(clean, "%Y-%m-%d")
            if tz_label == "IST":
                dt = TZ_INDIA.localize(naive)
            elif tz_label == "ET":
                dt = TZ_US.localize(naive)
            else:
                dt = pytz.utc.localize(naive)
            return dt.astimezone(timezone.utc), tz_label

        # 3. Standard datetime string (YYYY-MM-DD HH:MM:SS or YYYY-MM-DDTHH:MM:SS)
        if "T" in clean:
            naive = datetime.fromisoformat(clean)
        else:
            naive = datetime.strptime(clean, "%Y-%m-%d %H:%M:%S")

        if tz_label == "IST":
            dt = TZ_INDIA.localize(naive)
        elif tz_label == "ET":
            dt = TZ_US.localize(naive)
        else:
            dt = pytz.utc.localize(naive)

        return dt.astimezone(timezone.utc), tz_label
    except Exception:
        return None, None


def get_last_market_session_date(market: str = "US", reference_time: Optional[datetime] = None) -> date:
    """
    Returns the calendar date of the last completed regular trading session.
    NSE/BSE (India): Mon-Fri, close at 15:30 IST.
    US Exchanges: Mon-Fri, close at 16:00 ET.
    Accounts for weekends and current intraday session progression.
    """
    ref_utc = reference_time if reference_time is not None else datetime.now(timezone.utc)
    if ref_utc.tzinfo is None:
        ref_utc = pytz.utc.localize(ref_utc)

    market_str = str(market).strip().upper() if market else "US"
    is_india = market_str in ("INDIA", "NSE", "BSE") or is_indian_ticker(market_str)

    if is_india:
        ref_local = ref_utc.astimezone(TZ_INDIA)
        close_time = dtime(15, 30)
    else:
        ref_local = ref_utc.astimezone(TZ_US)
        close_time = dtime(16, 0)

    weekday = ref_local.weekday()  # 0=Monday, 4=Friday, 5=Saturday, 6=Sunday
    local_date = ref_local.date()
    local_time = ref_local.time()

    if weekday == 5:  # Saturday -> Friday
        return local_date - timedelta(days=1)
    elif weekday == 6:  # Sunday -> Friday
        return local_date - timedelta(days=2)
    elif weekday == 0 and local_time < close_time:  # Monday before close -> Friday
        return local_date - timedelta(days=3)
    elif local_time < close_time:  # Tue-Fri before close -> Yesterday
        return local_date - timedelta(days=1)
    else:  # Mon-Fri after close -> Today
        return local_date


def format_market_timestamp(dt: Any, market: str = "US") -> str:
    """
    Formats a datetime, date, or Timestamp to exchange timezone-aware string representation.
    """
    if dt is None:
        return "--"

    # If already a string, check if format needs adjustment
    if isinstance(dt, str):
        parsed_dt, _ = parse_market_timestamp(dt)
        if parsed_dt is None:
            return dt
        dt = parsed_dt

    # If a pure date object
    if isinstance(dt, date) and not isinstance(dt, datetime):
        return dt.strftime("%Y-%m-%d (Close)")

    market_str = str(market).strip().upper() if market else "US"
    is_india = market_str in ("INDIA", "NSE", "BSE") or is_indian_ticker(market_str)
    tz = TZ_INDIA if is_india else TZ_US
    tz_label = "IST" if is_india else "ET"

    if getattr(dt, "tzinfo", None) is None:
        dt = pytz.utc.localize(dt)

    local_dt = dt.astimezone(tz)
    return local_dt.strftime(f"%Y-%m-%d %H:%M:%S {tz_label}")


def calculate_quote_age_minutes(source_timestamp_str: Optional[str], market: str = "US") -> Optional[float]:
    """
    Calculates quote age in minutes relative to the actual source exchange timestamp.
    Returns None if timestamp cannot be parsed.
    """
    dt_utc, _ = parse_market_timestamp(source_timestamp_str)
    if dt_utc is None:
        return None

    now_utc = datetime.now(timezone.utc)
    age_seconds = (now_utc - dt_utc).total_seconds()
    return max(0.0, round(age_seconds / 60.0, 1))


def evaluate_quote_freshness(
    timestamp_str: Optional[str],
    ticker: Optional[str] = None,
    market: str = "US",
    current_price: Optional[float] = None,
    reference_time: Optional[datetime] = None
) -> Dict[str, Any]:
    """
    Evaluates whether a quote's observation timestamp is current or stale.
    A quote is stale if:
      - It lacks a valid timestamp.
      - Its trading session date is strictly before the last completed trading session.
      - The market session is currently OPEN and the quote has aged > 30 minutes without update.
    Never marks delayed data or closed markets as 'live'.
    """
    ref_utc = reference_time if reference_time is not None else datetime.now(timezone.utc)
    if ref_utc.tzinfo is None:
        ref_utc = pytz.utc.localize(ref_utc)

    t_str = str(ticker).strip().upper() if ticker else ""
    is_india = is_indian_ticker(t_str) or str(market).strip().upper() in ("INDIA", "NSE", "BSE")
    target_tz = TZ_INDIA if is_india else TZ_US
    tz_code = "IST" if is_india else "ET"

    # Specific market/exchange name
    if is_india:
        market_name = "NSE" if t_str.endswith(".NS") else ("BSE" if t_str.endswith(".BO") else "NSE")
    else:
        market_name = "NASDAQ" if t_str in ("AAPL", "MSFT", "GOOGL", "NVDA", "CRWD") else "NYSE"

    currency = "INR" if is_india else "USD"
    currency_symbol = "₹" if is_india else "$"

    missing_price = (current_price is None or float(current_price) <= 0)

    dt_utc, _ = parse_market_timestamp(timestamp_str)
    last_session_date = get_last_market_session_date(market="INDIA" if is_india else "US", reference_time=ref_utc)

    if dt_utc is None:
        return {
            "is_stale": True,
            "stale_warning": "Missing observation timestamp; quote freshness cannot be confirmed",
            "age_minutes": None,
            "observation_time_local": None,
            "observation_time_utc": None,
            "observation_date": None,
            "market_session_date": last_session_date.isoformat(),
            "market": market_name,
            "market_region": "India" if is_india else "US",
            "currency": currency,
            "currency_symbol": currency_symbol,
            "price_session": "previous_close",
            "freshness_label": "as of N/A",
            "freshness_status": "MISSING_TIMESTAMP",
            "missing_price": missing_price,
            "is_live": False
        }

    local_dt = dt_utc.astimezone(target_tz)
    obs_date = local_dt.date()
    age_seconds = (ref_utc - dt_utc).total_seconds()
    age_minutes = max(0.0, round(age_seconds / 60.0, 1))

    is_stale = False
    stale_warning = None
    freshness_status = "CURRENT_SESSION"

    # Compare observation session date against last completed session
    if obs_date < last_session_date:
        is_stale = True
        stale_warning = (
            f"Stale quote: observation is from prior session ({obs_date.strftime('%b %d')}), "
            f"older than last session ({last_session_date.strftime('%b %d')})"
        )
        freshness_status = "STALE"
        price_session = "previous_close"
    else:
        # Quote is from the last session date or today
        ref_local = ref_utc.astimezone(target_tz)
        if is_india:
            is_open = (ref_local.weekday() < 5 and dtime(9, 15) <= ref_local.time() <= dtime(15, 30))
        else:
            is_open = (ref_local.weekday() < 5 and dtime(9, 30) <= ref_local.time() <= dtime(16, 0))

        if is_open and age_minutes > 30.0:
            is_stale = True
            stale_warning = f"Delayed quote: {int(age_minutes)}m old during active trading session"
            freshness_status = "STALE_DELAYED"
            price_session = "regular-session"
        elif is_open:
            freshness_status = "ACTIVE_SESSION"
            price_session = "regular-session"
        else:
            freshness_status = "SESSION_CLOSE"
            price_session = "previous_close"

    freshness_label = f"as of {local_dt.strftime('%b %d, %H:%M')} {tz_code}"

    return {
        "is_stale": is_stale,
        "stale_warning": stale_warning,
        "age_minutes": age_minutes,
        "observation_time_local": local_dt.strftime(f"%Y-%m-%d %H:%M:%S {tz_code}"),
        "observation_time_utc": dt_utc.isoformat(),
        "observation_date": obs_date.isoformat(),
        "market_session_date": last_session_date.isoformat(),
        "market": market_name,
        "market_region": "India" if is_india else "US",
        "currency": currency,
        "currency_symbol": currency_symbol,
        "price_session": price_session,
        "freshness_label": freshness_label,
        "freshness_status": freshness_status,
        "missing_price": missing_price,
        "is_live": False  # Public delayed feed / closed session; never falsely reported as live tick
    }



def get_market_session_status(market: str = "US") -> Dict[str, Any]:
    """
    Determines genuine market session state based on exchange time:
    NSE/BSE (India): Mon-Fri 09:15 - 15:30 IST
    US Exchanges: Mon-Fri 09:30 - 16:00 ET
    Accepts market name ("India", "US") or ticker ("TCS.NS", "AAPL").
    """
    now_utc = datetime.now(timezone.utc)
    market_str = str(market).strip().upper() if market else "US"
    is_india = market_str in ("INDIA", "NSE", "BSE") or is_indian_ticker(market_str)

    if is_india:
        now_local = now_utc.astimezone(TZ_INDIA)
        weekday = now_local.weekday()  # 0=Monday, 4=Friday, 5=Saturday, 6=Sunday
        local_time = now_local.time()

        if weekday >= 5:
            return {"status": "CLOSED", "session": "WEEKEND", "local_time": now_local.strftime("%H:%M:%S IST")}

        open_time = dtime(9, 15)
        close_time = dtime(15, 30)

        if open_time <= local_time <= close_time:
            return {"status": "OPEN", "session": "REGULAR", "local_time": now_local.strftime("%H:%M:%S IST")}
        elif dtime(9, 0) <= local_time < open_time:
            return {"status": "PRE_MARKET", "session": "PRE_MARKET", "local_time": now_local.strftime("%H:%M:%S IST")}
        else:
            return {"status": "CLOSED", "session": "AFTER_HOURS", "local_time": now_local.strftime("%H:%M:%S IST")}
    else:
        now_local = now_utc.astimezone(TZ_US)
        weekday = now_local.weekday()
        local_time = now_local.time()

        if weekday >= 5:
            return {"status": "CLOSED", "session": "WEEKEND", "local_time": now_local.strftime("%H:%M:%S ET")}

        open_time = dtime(9, 30)
        close_time = dtime(16, 0)

        if open_time <= local_time <= close_time:
            return {"status": "OPEN", "session": "REGULAR", "local_time": now_local.strftime("%H:%M:%S ET")}
        elif dtime(4, 0) <= local_time < open_time:
            return {"status": "PRE_MARKET", "session": "PRE_MARKET", "local_time": now_local.strftime("%H:%M:%S ET")}
        elif close_time < local_time <= dtime(20, 0):
            return {"status": "AFTER_HOURS", "session": "POST_MARKET", "local_time": now_local.strftime("%H:%M:%S ET")}
        else:
            return {"status": "CLOSED", "session": "CLOSED", "local_time": now_local.strftime("%H:%M:%S ET")}


def validate_ticker_query(query: str) -> Tuple[bool, str]:
    """
    Safeguards ticker resolution from ambiguous short tokens and acronyms.
    Returns (is_valid, reason).
    """
    clean_q = re.sub(r"[^a-zA-Z0-9\s.-]", "", str(query)).strip()
    if not clean_q:
        return False, "Empty query"

    upper_q = clean_q.upper()

    # If it is a known stopword or ambiguous short token, require manual review
    if upper_q in AMBIGUOUS_SHORT_TOKENS:
        return False, f"Ambiguous token '{upper_q}' resembles a country code or common word; automated mapping rejected"

    # If 1-2 letters, ensure it is in the whitelist of verified short equities
    if len(clean_q) <= 2:
        if upper_q in VALID_SHORT_TICKERS:
            return True, f"Verified valid short ticker {upper_q}"
        return False, f"Short ticker query '{clean_q}' is unverified and flagged for review"

    return True, "Valid query format"
