"""
risk_scoring_engine.py
Company/Ticker resolution, transparent composite cyber risk scoring,
and market/sector benchmark comparison.
"""

from typing import Dict, Any, Optional
import re
try:
    import yfinance as yf
except ImportError:
    yf = None
from market_utils import get_market_benchmark, validate_ticker_query, is_indian_ticker


COMPANY_TICKER_MAP = {
    # US Technology
    "MICROSOFT": {"ticker": "MSFT", "name": "Microsoft Corporation", "sector": "Information Technology", "benchmark": "XLK"},
    "MSFT": {"ticker": "MSFT", "name": "Microsoft Corporation", "sector": "Information Technology", "benchmark": "XLK"},
    "APPLE": {"ticker": "AAPL", "name": "Apple Inc.", "sector": "Information Technology", "benchmark": "XLK"},
    "AAPL": {"ticker": "AAPL", "name": "Apple Inc.", "sector": "Information Technology", "benchmark": "XLK"},
    "NVIDIA": {"ticker": "NVDA", "name": "NVIDIA Corporation", "sector": "Information Technology", "benchmark": "XLK"},
    "NVDA": {"ticker": "NVDA", "name": "NVIDIA Corporation", "sector": "Information Technology", "benchmark": "XLK"},
    "ORACLE": {"ticker": "ORCL", "name": "Oracle Corporation", "sector": "Information Technology", "benchmark": "XLK"},
    "ORCL": {"ticker": "ORCL", "name": "Oracle Corporation", "sector": "Information Technology", "benchmark": "XLK"},
    "SOLARWINDS": {"ticker": "SWI", "name": "SolarWinds Corporation", "sector": "Information Technology", "benchmark": "XLK"},
    "SWI": {"ticker": "SWI", "name": "SolarWinds Corporation", "sector": "Information Technology", "benchmark": "XLK"},
    "CROWDSTRIKE": {"ticker": "CRWD", "name": "CrowdStrike Holdings Inc.", "sector": "Information Technology", "benchmark": "XLK"},
    "CRWD": {"ticker": "CRWD", "name": "CrowdStrike Holdings Inc.", "sector": "Information Technology", "benchmark": "XLK"},
    "OKTA": {"ticker": "OKTA", "name": "Okta Inc.", "sector": "Information Technology", "benchmark": "XLK"},
    "PALANTIR": {"ticker": "PLTR", "name": "Palantir Technologies Inc.", "sector": "Information Technology", "benchmark": "XLK"},
    "PLTR": {"ticker": "PLTR", "name": "Palantir Technologies Inc.", "sector": "Information Technology", "benchmark": "XLK"},

    # US Communication & Media
    "GOOGLE": {"ticker": "GOOGL", "name": "Alphabet Inc.", "sector": "Communication Services", "benchmark": "XLC"},
    "ALPHABET": {"ticker": "GOOGL", "name": "Alphabet Inc.", "sector": "Communication Services", "benchmark": "XLC"},
    "GOOGL": {"ticker": "GOOGL", "name": "Alphabet Inc.", "sector": "Communication Services", "benchmark": "XLC"},
    "META": {"ticker": "META", "name": "Meta Platforms Inc.", "sector": "Communication Services", "benchmark": "XLC"},
    "FACEBOOK": {"ticker": "META", "name": "Meta Platforms Inc.", "sector": "Communication Services", "benchmark": "XLC"},
    "T-MOBILE": {"ticker": "TMUS", "name": "T-Mobile US Inc.", "sector": "Communication Services", "benchmark": "XLC"},
    "TMUS": {"ticker": "TMUS", "name": "T-Mobile US Inc.", "sector": "Communication Services", "benchmark": "XLC"},
    "NETFLIX": {"ticker": "NFLX", "name": "Netflix Inc.", "sector": "Communication Services", "benchmark": "XLC"},
    "DISNEY": {"ticker": "DIS", "name": "Walt Disney Co.", "sector": "Communication Services", "benchmark": "XLC"},

    # US Consumer & Retail
    "AMAZON": {"ticker": "AMZN", "name": "Amazon.com Inc.", "sector": "Consumer Discretionary", "benchmark": "XLY"},
    "AMZN": {"ticker": "AMZN", "name": "Amazon.com Inc.", "sector": "Consumer Discretionary", "benchmark": "XLY"},
    "TESLA": {"ticker": "TSLA", "name": "Tesla Inc.", "sector": "Consumer Discretionary", "benchmark": "XLY"},
    "TSLA": {"ticker": "TSLA", "name": "Tesla Inc.", "sector": "Consumer Discretionary", "benchmark": "XLY"},
    "MARRIOTT": {"ticker": "MAR", "name": "Marriott International Inc.", "sector": "Consumer Discretionary", "benchmark": "XLY"},
    "MAR": {"ticker": "MAR", "name": "Marriott International Inc.", "sector": "Consumer Discretionary", "benchmark": "XLY"},
    "UBER": {"ticker": "UBER", "name": "Uber Technologies Inc.", "sector": "Consumer Discretionary", "benchmark": "XLY"},

    # US Financials
    "JPMORGAN": {"ticker": "JPM", "name": "JPMorgan Chase & Co.", "sector": "Financials", "benchmark": "XLF"},
    "JPM": {"ticker": "JPM", "name": "JPMorgan Chase & Co.", "sector": "Financials", "benchmark": "XLF"},
    "CAPITAL ONE": {"ticker": "COF", "name": "Capital One Financial Corp.", "sector": "Financials", "benchmark": "XLF"},
    "COF": {"ticker": "COF", "name": "Capital One Financial Corp.", "sector": "Financials", "benchmark": "XLF"},
    "VISA": {"ticker": "V", "name": "Visa Inc.", "sector": "Financials", "benchmark": "XLF"},
    "MASTERCARD": {"ticker": "MA", "name": "Mastercard Inc.", "sector": "Financials", "benchmark": "XLF"},
    "BANK OF AMERICA": {"ticker": "BAC", "name": "Bank of America Corp.", "sector": "Financials", "benchmark": "XLF"},
    "BAC": {"ticker": "BAC", "name": "Bank of America Corp.", "sector": "Financials", "benchmark": "XLF"},

    # US Industrials
    "EQUIFAX": {"ticker": "EFX", "name": "Equifax Inc.", "sector": "Industrials", "benchmark": "XLI"},
    "EFX": {"ticker": "EFX", "name": "Equifax Inc.", "sector": "Industrials", "benchmark": "XLI"},
    "BOEING": {"ticker": "BA", "name": "Boeing Co.", "sector": "Industrials", "benchmark": "XLI"},
    "GE": {"ticker": "GE", "name": "GE Aerospace", "sector": "Industrials", "benchmark": "XLI"},

    # Indian Equities - Information Technology (^CNXIT)
    "TCS": {"ticker": "TCS.NS", "name": "Tata Consultancy Services", "sector": "Information Technology", "benchmark": "^CNXIT"},
    "TATA CONSULTANCY SERVICES": {"ticker": "TCS.NS", "name": "Tata Consultancy Services", "sector": "Information Technology", "benchmark": "^CNXIT"},
    "INFOSYS": {"ticker": "INFY.NS", "name": "Infosys Ltd.", "sector": "Information Technology", "benchmark": "^CNXIT"},
    "INFY": {"ticker": "INFY.NS", "name": "Infosys Ltd.", "sector": "Information Technology", "benchmark": "^CNXIT"},
    "WIPRO": {"ticker": "WIPRO.NS", "name": "Wipro Ltd.", "sector": "Information Technology", "benchmark": "^CNXIT"},
    "HCL": {"ticker": "HCLTECH.NS", "name": "HCL Technologies", "sector": "Information Technology", "benchmark": "^CNXIT"},
    "HCL TECH": {"ticker": "HCLTECH.NS", "name": "HCL Technologies", "sector": "Information Technology", "benchmark": "^CNXIT"},
    "TECH MAHINDRA": {"ticker": "TECHM.NS", "name": "Tech Mahindra", "sector": "Information Technology", "benchmark": "^CNXIT"},

    # Indian Equities - Banking & Financials (^NSEBANK)
    "HDFC BANK": {"ticker": "HDFCBANK.NS", "name": "HDFC Bank", "sector": "Financial Services", "benchmark": "^NSEBANK"},
    "HDFCBANK": {"ticker": "HDFCBANK.NS", "name": "HDFC Bank", "sector": "Financial Services", "benchmark": "^NSEBANK"},
    "ICICI BANK": {"ticker": "ICICIBANK.NS", "name": "ICICI Bank", "sector": "Financial Services", "benchmark": "^NSEBANK"},
    "ICICIBANK": {"ticker": "ICICIBANK.NS", "name": "ICICI Bank", "sector": "Financial Services", "benchmark": "^NSEBANK"},
    "STATE BANK OF INDIA": {"ticker": "SBIN.NS", "name": "State Bank of India", "sector": "Financial Services", "benchmark": "^NSEBANK"},
    "SBI": {"ticker": "SBIN.NS", "name": "State Bank of India", "sector": "Financial Services", "benchmark": "^NSEBANK"},
    "KOTAK": {"ticker": "KOTAKBANK.NS", "name": "Kotak Mahindra Bank", "sector": "Financial Services", "benchmark": "^NSEBANK"},
    "KOTAK BANK": {"ticker": "KOTAKBANK.NS", "name": "Kotak Mahindra Bank", "sector": "Financial Services", "benchmark": "^NSEBANK"},
    "AXIS BANK": {"ticker": "AXISBANK.NS", "name": "Axis Bank", "sector": "Financial Services", "benchmark": "^NSEBANK"},
    "BAJAJ FINANCE": {"ticker": "BAJFINANCE.NS", "name": "Bajaj Finance", "sector": "Financial Services", "benchmark": "^NSEBANK"},

    # Indian Equities - Automotive (^CNXAUTO)
    "TATA MOTORS": {"ticker": "TATAMOTORS.NS", "name": "Tata Motors", "sector": "Automotive", "benchmark": "^CNXAUTO"},
    "TATAMOTORS": {"ticker": "TATAMOTORS.NS", "name": "Tata Motors", "sector": "Automotive", "benchmark": "^CNXAUTO"},
    "MARUTI": {"ticker": "MARUTI.NS", "name": "Maruti Suzuki India", "sector": "Automotive", "benchmark": "^CNXAUTO"},
    "MARUTI SUZUKI": {"ticker": "MARUTI.NS", "name": "Maruti Suzuki India", "sector": "Automotive", "benchmark": "^CNXAUTO"},

    # Indian Equities - Broad Market / Energy (^NSEI)
    "RELIANCE": {"ticker": "RELIANCE.NS", "name": "Reliance Industries", "sector": "Energy & Conglomerate", "benchmark": "^NSEI"},
    "RELIANCE INDUSTRIES": {"ticker": "RELIANCE.NS", "name": "Reliance Industries", "sector": "Energy & Conglomerate", "benchmark": "^NSEI"},
    "BHARTI AIRTEL": {"ticker": "BHARTIARTL.NS", "name": "Bharti Airtel", "sector": "Telecommunications", "benchmark": "^NSEI"},
    "AIRTEL": {"ticker": "BHARTIARTL.NS", "name": "Bharti Airtel", "sector": "Telecommunications", "benchmark": "^NSEI"},
    "LARSEN & TOUBRO": {"ticker": "LT.NS", "name": "Larsen & Toubro", "sector": "Engineering & Construction", "benchmark": "^NSEI"},
    "LT": {"ticker": "LT.NS", "name": "Larsen & Toubro", "sector": "Engineering & Construction", "benchmark": "^NSEI"},
}


def resolve_ticker(query: str) -> Dict[str, Any]:
    """
    Resolves a company name or ticker symbol to standardized details using
    market-aware validation and strict safeguards:
      1. Ambiguity & short-token validation check
      2. Static dictionary lookup (COMPANY_TICKER_MAP + KNOWN_COMPANIES)
      3. Word-boundary lookup for known entities
      4. Safe dynamic lookup if permissible
    Returns resolved=False and ticker="UNKNOWN" if it does not resolve or is ambiguous.
    """
    clean_q = re.sub(r"[^a-zA-Z0-9\s.-]", "", str(query)).strip()
    if not clean_q:
        return {
            "ticker": "UNKNOWN",
            "name": "Unknown Company",
            "sector": "Unknown",
            "benchmark": "SPY",
            "resolved": False
        }

    # 1. Ambiguity & short-token validation check
    is_valid, validation_msg = validate_ticker_query(clean_q)
    if not is_valid:
        return {
            "ticker": "UNKNOWN",
            "name": clean_q.title(),
            "sector": "Unknown",
            "benchmark": "SPY",
            "resolved": False,
            "review_needed": True,
            "review_reason": validation_msg
        }

    clean_upper = clean_q.upper()
    clean_lower = clean_q.lower()

    # 2. Direct match in COMPANY_TICKER_MAP
    if clean_upper in COMPANY_TICKER_MAP:
        res = dict(COMPANY_TICKER_MAP[clean_upper])
        bm = get_market_benchmark(res["ticker"])
        res["sector"] = bm["sector"]
        res["benchmark"] = bm["sector_benchmark"]
        res["resolved"] = True
        return res

    # 3. Match against KNOWN_COMPANIES
    try:
        from threat_collector import KNOWN_COMPANIES, get_company_name, resolve_dynamic_ticker
    except ImportError:
        KNOWN_COMPANIES = {}
        get_company_name = lambda t: t
        resolve_dynamic_ticker = lambda c: None

    for company_name, ticker in KNOWN_COMPANIES.items():
        if clean_lower == company_name.lower() or clean_upper == ticker.upper():
            c_name = get_company_name(ticker) or company_name
            bm = get_market_benchmark(ticker)
            return {
                "ticker": ticker,
                "name": c_name,
                "sector": bm["sector"],
                "benchmark": bm["sector_benchmark"],
                "resolved": True
            }

    # 4. Exact word boundary match in KNOWN_COMPANIES
    ambiguous_names = {"apple", "amazon", "meta", "alphabet", "uber", "target", "visa", "sa"}
    matches = []
    for company, ticker in KNOWN_COMPANIES.items():
        comp_lower = company.lower()
        pattern = r"\b" + re.escape(comp_lower) + r"\b"
        if re.search(pattern, clean_lower):
            if comp_lower in ambiguous_names:
                original_pattern = r"\b" + re.escape(company) + r"\b"
                capitalized_pattern = r"\b" + re.escape(company.capitalize()) + r"\b"
                if not (re.search(original_pattern, query) or re.search(capitalized_pattern, query)):
                    continue
            matches.append((company, ticker))

    unique_tickers = set(t for _, t in matches)
    if len(unique_tickers) == 1:
        ticker = next(iter(unique_tickers))
        c_name = get_company_name(ticker)
        bm = get_market_benchmark(ticker)
        return {
            "ticker": ticker,
            "name": c_name,
            "sector": bm["sector"],
            "benchmark": bm["sector_benchmark"],
            "resolved": True
        }

    # 5. Strict dynamic lookup (only for queries > 2 characters)
    if len(clean_q) > 2 and callable(resolve_dynamic_ticker):
        dynamic_ticker = resolve_dynamic_ticker(clean_q)
        if dynamic_ticker:
            c_name = get_company_name(dynamic_ticker) or clean_q.title()
            bm = get_market_benchmark(dynamic_ticker)
            return {
                "ticker": dynamic_ticker,
                "name": c_name,
                "sector": bm["sector"],
                "benchmark": bm["sector_benchmark"],
                "resolved": True
            }

    return {
        "ticker": "UNKNOWN",
        "name": clean_q.title(),
        "sector": "Unknown",
        "benchmark": "SPY",
        "resolved": False
    }



def calculate_risk_score(
    severity_score: float,
    data_sensitivity_score: float,
    predicted_price_drop: float = 0.0,
    current_stock_change: float = 0.0,
    users_affected_millions: float = 0.0,
    sector_excess_performance: float = 0.0,
    business_impact: float = 0.0,
    evidence_quality: float = 0.0,
    exploitation_status: str = "NOT_REPORTED"
) -> Dict[str, Any]:
    """
    Transparent Multi-Factor Cyber Risk Scoring Engine.

    Risk score measures cyber-threat severity.
    Market movement is supporting evidence, not proof of causation.

    Components:
      - Threat Severity: 0-35 points
      - Data Sensitivity & Blast Radius: 0-25 points
      - Business Impact: 0-15 points
      - ML Predicted Impact: 0-10 points
      - Market Evidence: 0-5 points
      - Evidence Quality: 0-5 points
      - Exploitation Status: 0-5 points

    Total Score: 0-100
    """

    sev_normalized = min(
        10.0,
        max(0.0, float(severity_score))
    )

    severity_points = (
        sev_normalized / 10.0
    ) * 35.0

    sens_normalized = min(
        10.0,
        max(0.0, float(data_sensitivity_score))
    )

    user_multiplier = 1.0 + (
        min(max(users_affected_millions, 0.0), 100.0) / 500.0
    )

    sensitivity_points = min(
        25.0,
        (sens_normalized / 10.0) * 25.0 * user_multiplier
    )

    drop_magnitude = (
        abs(float(predicted_price_drop))
        if float(predicted_price_drop) < 0
        else 0.0
    )

    ml_points = min(
        10.0,
        drop_magnitude * 1.5
    )

    excess_drop = (
        abs(float(sector_excess_performance))
        if float(sector_excess_performance) < 0
        else 0.0
    )

    market_points = min(
        5.0,
        excess_drop * 1.0
    )

    business_normalized = min(
        10.0,
        max(0.0, float(business_impact))
    )

    business_points = (
        business_normalized / 10.0
    ) * 15.0

    evidence_normalized = min(
        10.0,
        max(0.0, float(evidence_quality))
    )

    evidence_points = (
        evidence_normalized / 10.0
    ) * 5.0

    exploitation_points = 0.0

    if exploitation_status == "CONFIRMED":
        exploitation_points = 5.0
    elif exploitation_status == "REPORTED":
        exploitation_points = 3.0

    composite_score = round(
        severity_points +
        sensitivity_points +
        business_points +
        ml_points +
        market_points +
        evidence_points +
        exploitation_points,
        1
    )

    composite_score = min(
        100.0,
        max(0.0, composite_score)
    )

    if composite_score >= 75.0 and excess_drop >= 2.0:
        tier = "CRITICAL"
        recommendation = "REVIEW"
        rationale = (
            "High cyber-risk score with significant relative "
            "market underperformance. Cyber-related market impact "
            "requires further confirmation."
        )

    elif composite_score >= 50.0 and excess_drop >= 1.0:
        tier = "HIGH"
        recommendation = "MONITOR"
        rationale = (
            "Material cyber risk with observable relative market "
            "underperformance. Causal attribution is not established."
        )

    elif composite_score >= 25.0:
        tier = "MEDIUM"
        recommendation = "MONITOR"
        rationale = (
            "Moderate cyber risk detected. Available market evidence "
            "is insufficient to establish a strong cyber-related signal."
        )

    else:
        tier = "LOW"
        recommendation = "NO SIGNAL"
        rationale = (
            "Insufficient evidence of material cyber-related market impact."
        )

    # Inject historical context from Event Studies (No causal claims)
    try:
        import sqlite3
        conn = sqlite3.connect("threat_stock_data.db", timeout=5)
        c = conn.cursor()
        c.execute(
            "SELECT AVG(abnormal_return_1d) FROM event_studies "
            "WHERE status = 'EVALUATED' AND severity_tier = ?",
            (tier,)
        )
        avg_abnormal = c.fetchone()[0]
        conn.close()
        
        if avg_abnormal is not None:
            rationale += (
                f" Historical context: similar {tier} incidents averaged "
                f"{avg_abnormal:+.2f}% abnormal return."
            )
    except Exception:
        pass

    return {
        "risk_score": int(round(composite_score)),
        "tier": tier,
        "recommendation": recommendation,
        "rationale": rationale,
        "market_evidence": {
            "current_stock_change_percent": round(
                float(current_stock_change),
                2
            ),
            "raw_return_percent": round(
                float(current_stock_change),
                2
            ),
            "sector_excess_performance_percent": round(
                float(sector_excess_performance),
                2
            ),
            "abnormal_return_percent": round(
                float(sector_excess_performance),
                2
            ),
            "causal_attribution_established": False
        },
        "breakdown": {
            "severity_points": round(severity_points, 1),
            "sensitivity_points": round(sensitivity_points, 1),
            "business_impact_points": round(business_points, 1),
            "ml_predicted_points": round(ml_points, 1),
            "market_evidence_points": round(market_points, 1),
            "evidence_quality_points": round(evidence_points, 1),
            "exploitation_points": round(exploitation_points, 1)
        }
    }


def get_sector_comparison(
    ticker: str,
    benchmark_ticker: Optional[str] = None
) -> Dict[str, Any]:
    """
    Pulls current day percentage change for target ticker
    versus its sector or broad market benchmark.

    Calculates relative underperformance without treating missing returns as zero.
    """
    if not benchmark_ticker:
        bm_info = get_market_benchmark(ticker)
        benchmark_ticker = bm_info["sector_benchmark"]

    if yf is None:
        return {
            "ticker": ticker,
            "ticker_change_percent": None,
            "benchmark_ticker": benchmark_ticker,
            "benchmark_change_percent": None,
            "excess_performance": None,
            "drag_detected": False,
            "status": "UNAVAILABLE",
            "error": "yfinance package not available in environment"
        }

    try:
        t_data = yf.Ticker(ticker).history(period="5d")
        b_data = yf.Ticker(benchmark_ticker).history(period="5d")

        t_clean = t_data.dropna(subset=["Close"]) if not t_data.empty else None
        b_clean = b_data.dropna(subset=["Close"]) if not b_data.empty else None

        if t_clean is None or len(t_clean) < 2:
            return {
                "ticker": ticker,
                "ticker_change_percent": None,
                "raw_return": None,
                "benchmark_ticker": benchmark_ticker,
                "benchmark_change_percent": None,
                "benchmark_return": None,
                "excess_performance": None,
                "abnormal_return": None,
                "drag_detected": False,
                "status": "UNAVAILABLE",
                "error": f"Insufficient regular session price history for {ticker}"
            }

        if b_clean is None or len(b_clean) < 2:
            return {
                "ticker": ticker,
                "ticker_change_percent": None,
                "raw_return": None,
                "benchmark_ticker": benchmark_ticker,
                "benchmark_change_percent": None,
                "benchmark_return": None,
                "excess_performance": None,
                "abnormal_return": None,
                "drag_detected": False,
                "status": "UNAVAILABLE",
                "error": f"Insufficient regular session benchmark history for {benchmark_ticker}"
            }

        t_prev = float(t_clean["Close"].iloc[-2])
        t_curr = float(t_clean["Close"].iloc[-1])
        t_pct = ((t_curr - t_prev) / t_prev) * 100.0 if t_prev else None

        b_prev = float(b_clean["Close"].iloc[-2])
        b_curr = float(b_clean["Close"].iloc[-1])
        b_pct = ((b_curr - b_prev) / b_prev) * 100.0 if b_prev else None

        if t_pct is None or b_pct is None:
            return {
                "ticker": ticker,
                "ticker_change_percent": None,
                "raw_return": None,
                "benchmark_ticker": benchmark_ticker,
                "benchmark_change_percent": None,
                "benchmark_return": None,
                "excess_performance": None,
                "abnormal_return": None,
                "drag_detected": False,
                "status": "UNAVAILABLE",
                "error": "Non-zero previous close required for return calculation"
            }

        excess_change = t_pct - b_pct

        return {
            "ticker": ticker,
            "ticker_change_percent": round(t_pct, 2),
            "raw_return": round(t_pct, 2),
            "benchmark_ticker": benchmark_ticker,
            "benchmark_change_percent": round(b_pct, 2),
            "benchmark_return": round(b_pct, 2),
            "excess_performance": round(excess_change, 2),
            "abnormal_return": round(excess_change, 2),
            "drag_detected": excess_change < -0.5,
            "status": "SUCCESS"
        }

    except Exception as e:
        return {
            "ticker": ticker,
            "ticker_change_percent": None,
            "raw_return": None,
            "benchmark_ticker": benchmark_ticker,
            "benchmark_change_percent": None,
            "benchmark_return": None,
            "excess_performance": None,
            "abnormal_return": None,
            "drag_detected": False,
            "status": "ERROR",
            "error": str(e)
        }



if __name__ == "__main__":
    print("Testing Resolver:")

    res = resolve_ticker("Microsoft")
    print("  Resolved:", res)

    print("\nTesting Risk Engine:")

    score = calculate_risk_score(
        severity_score=9.0,
        data_sensitivity_score=8.5,
        predicted_price_drop=-2.4,
        current_stock_change=-1.8,
        users_affected_millions=50.0,
        sector_excess_performance=-1.2,
        business_impact=7.0,
        evidence_quality=8.0,
        exploitation_status="CONFIRMED"
    )

    print("  Risk Assessment:", score)