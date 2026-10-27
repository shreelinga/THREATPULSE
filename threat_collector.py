import feedparser
import re
import html
import time
import argparse
import json
import sqlite3
import os
from typing import Dict, Optional
from datetime import datetime, timedelta, timezone
from email.utils import parsedate_to_datetime


BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

DB_PATH = os.path.join(
    BASE_DIR,
    "threat_stock_data.db"
)


FEEDS = {
    "BleepingComputer - Security":
        "https://www.bleepingcomputer.com/feed/",
    "The Hacker News":
        "https://feeds.feedburner.com/TheHackersNews",
    "SecurityWeek":
        "https://feeds.feedburner.com/Securityweek",
    "Krebs on Security":
        "https://krebsonsecurity.com/feed/",
    "CISA Alerts":
        "https://www.cisa.gov/cybersecurity-advisories/all.xml"
}


MAX_ARTICLE_AGE_DAYS = 14


KNOWN_COMPANIES = {
    "Microsoft Corporation": "MSFT",
    "Microsoft 365": "MSFT",
    "Microsoft": "MSFT",
    "Office 365": "MSFT",
    "Azure": "MSFT",
    "Exchange Server": "MSFT",
    "GitHub": "MSFT",
    "LinkedIn": "MSFT",

    "Apple Inc.": "AAPL",
    "Apple Inc": "AAPL",
    "Apple": "AAPL",

    "Alphabet Inc.": "GOOGL",
    "Alphabet Inc": "GOOGL",
    "Google": "GOOGL",
    "Alphabet": "GOOGL",

    "Meta Platforms": "META",
    "Meta Platforms Inc.": "META",
    "Facebook": "META",
    "Instagram": "META",
    "WhatsApp": "META",

    "Amazon Web Services": "AMZN",
    "Amazon.com": "AMZN",
    "Amazon": "AMZN",
    "AWS": "AMZN",

    "T-Mobile US": "TMUS",
    "T-Mobile": "TMUS",

    "Uber Technologies": "UBER",
    "Uber": "UBER",

    "Marriott International": "MAR",
    "Marriott": "MAR",

    "Capital One": "COF",
    "Capital One Financial": "COF",

    "SolarWinds Corporation": "SWI",
    "SolarWinds": "SWI",

    "JPMorgan Chase": "JPM",
    "JPMorgan": "JPM",
    "Chase Bank": "JPM",

    "Bank of America": "BAC",

    "Wells Fargo": "WFC",

    "Citigroup": "C",
    "Citibank": "C",

    "Goldman Sachs": "GS",

    "Morgan Stanley": "MS",

    "Visa Inc.": "V",
    "Visa": "V",

    "Mastercard": "MA",

    "PayPal Holdings": "PYPL",
    "PayPal": "PYPL",

    "Cisco Systems": "CSCO",
"Cisco": "CSCO",
    "Oracle Corporation": "ORCL",
    "Oracle": "ORCL",

    "IBM Corporation": "IBM",
    "IBM": "IBM",

    "Intel Corporation": "INTC",
    "Intel": "INTC",

    "NVIDIA Corporation": "NVDA",
    "Nvidia": "NVDA",

    "Salesforce": "CRM",

    "Adobe Inc.": "ADBE",
    "Adobe": "ADBE",

    "SAP SE": "SAP",
    "SAP": "SAP",

    "UnitedHealth Group": "UNH",
    "UnitedHealth": "UNH",

    "Anthem Inc.": "ELV",
    "Anthem": "ELV",

    "Equifax Inc.": "EFX",
    "Equifax": "EFX",

    "Target Corporation": "TGT",
    "Target": "TGT",

    "Home Depot": "HD",

    "Walmart Inc.": "WMT",
    "Walmart": "WMT",

    "FedEx Corporation": "FDX",
    "FedEx": "FDX",

    "Boeing Company": "BA",
    "Boeing": "BA",

    "Delta Air Lines": "DAL",
    "Delta Airlines": "DAL",

    "American Airlines": "AAL",

    "Twilio Inc.": "TWLO",
    "Twilio": "TWLO",

    "Okta Inc.": "OKTA",
    "Okta": "OKTA",

    "CrowdStrike Holdings": "CRWD",
    "CrowdStrike": "CRWD",

    "Palo Alto Networks": "PANW",

    "Fortinet Inc.": "FTNT",
    "Fortinet": "FTNT",

    "Zoom Video Communications": "ZM",
    "Zoom": "ZM"
}

# Dynamic resolutions are session-local and only populated after a strict
# yfinance search match. The static dictionary remains the fast path.
DYNAMIC_COMPANIES: Dict[str, str] = {}
_DYNAMIC_TICKER_CACHE: Dict[str, Dict[str, str]] = {}

_COMPANY_STOPWORDS = {
    "a", "an", "and", "at", "by", "for", "from", "in", "inc", "incorporated",
    "is", "ltd", "of", "on", "the", "to", "with", "corp", "corporation", "company",
    "group", "holdings", "plc", "sa", "se"
}


def _company_words(value: str):
    """Normalize a company phrase for transparent search-result matching."""
    return {
        word for word in re.findall(r"[a-z0-9]+", (value or "").lower())
        if word not in _COMPANY_STOPWORDS and len(word) > 1
    }


def extract_dynamic_company_candidate(title: str, description: str) -> Optional[str]:
    """Extract an explicitly named organization phrase; return None if unclear."""
    title = clean_text(title)
    description = clean_text(description)
    patterns = [
        r"^([A-Z][A-Za-z0-9&.\-]*(?:\s+[A-Z][A-Za-z0-9&.\-]*){0,4})\s+(?:confirms|discloses|reports|reveals|suffers|hit by|breached|investigates)",
        r"(?:breach|ransomware|attack|vulnerability|incident)\s+(?:at|against|affecting|hits?|in)\s+([A-Z][A-Za-z0-9&.\-]*(?:\s+[A-Z][A-Za-z0-9&.\-]*){0,4})",
        r"(?:at|against|affecting|hits?)\s+([A-Z][A-Za-z0-9&.\-]*(?:\s+[A-Z][A-Za-z0-9&.\-]*){0,4})",
    ]
    for text in (title, description):
        for pattern in patterns:
            match = re.search(pattern, text)
            if not match:
                continue
            candidate = match.group(1).strip(" -:.,")
            words = _company_words(candidate)
            if words and not words.issubset({"security", "cyber", "threat", "attack", "data"}):
                return candidate
    return None


def resolve_dynamic_ticker(company_name: str) -> Optional[str]:
    """Resolve a named company only when yfinance returns a high-confidence equity match."""
    candidate = clean_text(company_name)
    if not candidate:
        return None
    cache_key = candidate.casefold()
    cached = _DYNAMIC_TICKER_CACHE.get(cache_key)
    if cached:
        return cached["ticker"]

    quotes = []
    try:
        import yfinance as yf
        search = yf.Search(candidate, max_results=5, news_count=0, raise_errors=False)
        quotes = getattr(search, "quotes", []) or []
    except Exception as error:
        safe_print(f"[TICKER] yfinance lookup unavailable for {candidate}: {error}")

    # yfinance can fail before searching when its optional local cache cannot
    # be opened. In that case use Yahoo Finance's matching public search API;
    # it requires no account or key and returns the same quote metadata.
    if not quotes:
        try:
            import requests
            response = requests.get(
                "https://query2.finance.yahoo.com/v1/finance/search",
                params={"q": candidate, "quotesCount": 5, "newsCount": 0},
                headers={"User-Agent": "Mozilla/5.0"},
                timeout=10,
            )
            response.raise_for_status()
            quotes = response.json().get("quotes", []) or []
        except Exception as error:
            safe_print(f"[TICKER] Public lookup unavailable for {candidate}: {error}")
            return None

    candidate_words = _company_words(candidate)
    for quote in quotes:
        if quote.get("quoteType") not in {"EQUITY", "STOCK"}:
            continue
        ticker = clean_text(quote.get("symbol", "")).upper()
        exchange = clean_text(quote.get("exchange", "")).upper()
        result_name = clean_text(quote.get("longname") or quote.get("shortname") or "")
        result_words = _company_words(result_name)
        shared_words = candidate_words & result_words
        # A single exact core name is sufficient; multiword candidates require
        # two shared words. This rejects loosely related search suggestions.
        exact_name = candidate_words == result_words or candidate_words.issubset(result_words)
        strong_overlap = len(candidate_words) > 1 and len(shared_words) >= 2
        eligible_exchange = exchange in {"NMS", "NYQ", "NAS", "NGM", "NCM", "PCX", "ASE"}
        if ticker and eligible_exchange and (exact_name or strong_overlap):
            resolved_name = result_name or candidate
            _DYNAMIC_TICKER_CACHE[cache_key] = {"ticker": ticker, "name": resolved_name}
            DYNAMIC_COMPANIES[ticker] = resolved_name
            safe_print(f"[TICKER] Verified dynamic resolution: {resolved_name} (${ticker})")
            return ticker

    safe_print(f"[TICKER] No high-confidence public-equity match for {candidate}")
    return None


THREAT_KEYWORDS = {
    "zero-day": "Zero-Day Exploit",
    "zero day": "Zero-Day Exploit",
    "zero-days": "Zero-Day Exploit",
    "zero days": "Zero-Day Exploit",
    "ransomware": "Ransomware",
    "data breach": "Data Breach",
    "breach": "Data Breach",
    "credential theft": "Credential Theft",
    "account compromise": "Account Compromise",
    "supply chain": "Supply Chain Attack",
    "sql injection": "Vulnerability",
    "security flaw": "Vulnerability",
    "actively exploited": "Vulnerability",
    "vulnerability": "Vulnerability",
    "phishing": "Phishing",
    "spyware": "Spyware",
    "malware": "Malware",
    "cyberattack": "Cyberattack",
    "cyber attack": "Cyberattack",
    "compromised": "Security Incident",
    "compromise": "Security Incident",
    "hacked": "Security Incident",
    "hack": "Security Incident"
}


EDITORIAL_EXCLUSION_TERMS = [
    "doesn't need to start over",
    "does not need to start over",
    "how to",
    "guide to",
    "best practices",
    "why security teams",
    "security teams should",
    "security teams need",
    "what security teams",
    "lessons learned",
    "industry trends",
    "security trends",
    "analysis:",
    "opinion:",
    "commentary:"
]


def safe_print(text):
    try:
        print(text)
    except UnicodeEncodeError:
        print(
            str(text).encode(
                "ascii",
                "ignore"
            ).decode("ascii")
        )


def clean_text(value):
    if not value:
        return ""

    value = html.unescape(
        str(value)
    )

    value = re.sub(
        r"<[^>]+>",
        " ",
        value
    )

    value = re.sub(
        r"\s+",
        " ",
        value
    )

    return value.strip()


def contains_keyword(text, keyword):
    pattern = (
        r"\b"
        + re.escape(
            keyword.lower()
        )
        + r"\b"
    )

    return re.search(
        pattern,
        text.lower()
    ) is not None


def is_editorial_article(
    title,
    description
):
    text = (
        f"{title} {description}"
        .lower()
    )

    for term in EDITORIAL_EXCLUSION_TERMS:
        if term in text:
            return True

    return False


def classify_threat(
    title,
    description
):
    text = (
        f"{title} {description}"
        .lower()
    )

    vulnerability_terms = [
        "zero-day",
        "zero day",
        "zero-days",
        "zero days",
        "security flaw",
        "vulnerability",
        "sql injection",
        "remote code execution",
        "rce",
        "path traversal",
        "cross-site request forgery",
        "csrf",
        "code execution",
        "privilege escalation",
        "authentication bypass",
        "security bug",
        "data exfiltration",
        "zero-click"
    ]

    for keyword in vulnerability_terms:

        if contains_keyword(
            text,
            keyword
        ):

            if keyword in [
                "zero-day",
                "zero day",
                "zero-days",
                "zero days"
            ]:
                return "Zero-Day Exploit"

            return "Vulnerability"

    priority = [
        "ransomware",
        "data breach",
        "breach",
        "credential theft",
        "account compromise",
        "supply chain",
        "phishing",
        "spyware",
        "malware",
        "cyberattack",
        "cyber attack",
        "compromised",
        "compromise",
        "hacked",
        "hack"
    ]

    for keyword in priority:

        if contains_keyword(
            text,
            keyword
        ):
            return THREAT_KEYWORDS[
                keyword
            ]

    return None


def estimate_severity(
    threat_type,
    title,
    description
):
    text = (
        f"{title} {description}"
        .lower()
    )

    if threat_type == "Zero-Day Exploit":
        return 9.0

    if threat_type == "Ransomware":
        return 8.5

    if threat_type == "Data Breach":
        return 7.5

    if threat_type == "Credential Theft":
        return 7.0

    if threat_type == "Account Compromise":
        return 6.5

    if threat_type == "Supply Chain Attack":
        return 8.0

    if threat_type == "Vulnerability":

        if "actively exploited" in text:
            return 9.0

        if any(
            term in text
            for term in [
                "remote code execution",
                "rce",
                "zero-click",
                "data exfiltration"
            ]
        ):
            return 8.0

        return 6.0

    if threat_type == "Phishing":
        return 5.5

    if threat_type == "Spyware":
        return 7.0

    if threat_type == "Malware":
        return 7.0

    if threat_type == "Cyberattack":
        return 7.0

    if threat_type == "Security Incident":
        return 5.0

    return 0.0


def extract_users_affected(text):

    text_lower = text.lower()

    patterns = [
        r"(\d+(?:\.\d+)?)\s*(million|m)\s+(users|customers|people|accounts|records)",
        r"(\d[\d,]+)\s+(users|customers|people|accounts|records)",
        r"(\d+(?:\.\d+)?)\s*(million|m)\s+accounts",
        r"(\d[\d,]+)\s+accounts"
    ]

    for pattern in patterns:

        match = re.search(
            pattern,
            text_lower
        )

        if not match:
            continue

        number = (
            match.group(1)
            .replace(",", "")
        )

        value = float(number)

        if len(match.groups()) >= 2:

            unit = match.group(2)

            if unit in [
                "million",
                "m"
            ]:
                return value

        return value / 1_000_000

    return 0.0


def extract_company_from_text(
    title,
    description
):
    title_lower = title.lower()

    matches = []

    ambiguous_names = {
        "apple",
        "amazon",
        "meta",
        "alphabet",
        "uber",
        "target",
        "visa"
    }

    for company, ticker in KNOWN_COMPANIES.items():

        comp_lower = company.lower()

        pattern = (
            r"\b"
            + re.escape(comp_lower)
            + r"\b"
        )

        if re.search(
            pattern,
            title_lower
        ):

            if comp_lower in ambiguous_names:

                original_pattern = (
                    r"\b"
                    + re.escape(company)
                    + r"\b"
                )

                capitalized_pattern = (
                    r"\b"
                    + re.escape(
                        company.capitalize()
                    )
                    + r"\b"
                )

                if not (
                    re.search(
                        original_pattern,
                        title
                    )
                    or
                    re.search(
                        capitalized_pattern,
                        title
                    )
                ):
                    continue

            matches.append(
                (
                    company,
                    ticker
                )
            )

    unique_tickers = set(
        ticker
        for _, ticker in matches
    )

    if len(unique_tickers) == 1:
        return next(
            iter(unique_tickers)
        )

    # Static names are intentionally preferred. Only attempt the external
    # lookup after no single static ticker can be established.
    dynamic_candidate = extract_dynamic_company_candidate(title, description)
    return resolve_dynamic_ticker(dynamic_candidate) if dynamic_candidate else None


def get_company_name(ticker):

    if ticker in DYNAMIC_COMPANIES:
        return DYNAMIC_COMPANIES[ticker]

    company_names = {

        "MSFT": "Microsoft Corporation",
        "AAPL": "Apple Inc.",
        "GOOGL": "Alphabet Inc.",
        "META": "Meta Platforms",
        "AMZN": "Amazon.com Inc.",
        "TMUS": "T-Mobile US",
        "UBER": "Uber Technologies",
        "MAR": "Marriott International",
        "COF": "Capital One",
        "SWI": "SolarWinds Corporation",
        "JPM": "JPMorgan Chase",
        "BAC": "Bank of America",
        "WFC": "Wells Fargo",
        "C": "Citigroup",
        "GS": "Goldman Sachs",
        "MS": "Morgan Stanley",
        "V": "Visa Inc.",
        "MA": "Mastercard",
        "PYPL": "PayPal Holdings",
        "CISCO": "Cisco Systems",
        "ORCL": "Oracle Corporation",
        "IBM": "IBM Corporation",
        "INTC": "Intel Corporation",
        "NVDA": "NVIDIA Corporation",
        "CRM": "Salesforce",
        "ADBE": "Adobe Inc.",
        "SAP": "SAP SE",
        "UNH": "UnitedHealth Group",
        "ELV": "Anthem Inc.",
        "EFX": "Equifax Inc.",
        "TGT": "Target Corporation",
        "HD": "Home Depot",
        "WMT": "Walmart Inc.",
        "FDX": "FedEx Corporation",
        "BA": "Boeing Company",
        "DAL": "Delta Air Lines",
        "AAL": "American Airlines",
        "TWLO": "Twilio Inc.",
        "OKTA": "Okta Inc.",
        "CRWD": "CrowdStrike Holdings",
        "PANW": "Palo Alto Networks",
        "FTNT": "Fortinet Inc.",
        "ZM": "Zoom Video Communications"
    }

    return company_names.get(
        ticker,
        "Unknown Target"
    )


def parse_date(entry):

    parsed_fields = [
        "published_parsed",
        "updated_parsed",
        "created_parsed"
    ]

    for field in parsed_fields:

        parsed_value = entry.get(
            field
        )

        if parsed_value:

            try:

                return datetime(
                    parsed_value.tm_year,
                    parsed_value.tm_mon,
                    parsed_value.tm_mday,
                    parsed_value.tm_hour,
                    parsed_value.tm_min,
                    parsed_value.tm_sec,
                    tzinfo=timezone.utc
                )

            except Exception:
                pass

    string_fields = [
        "published",
        "updated",
        "created",
        "date"
    ]

    for field in string_fields:

        value = entry.get(
            field
        )

        if not value:
            continue

        value = str(
            value
        ).strip()

        try:

            dt = parsedate_to_datetime(
                value
            )

            if dt.tzinfo is None:

                dt = dt.replace(
                    tzinfo=timezone.utc
                )

            return dt.astimezone(
                timezone.utc
            )

        except Exception:
            pass

        try:

            value_clean = value.replace(
                "Z",
                "+00:00"
            )

            dt = datetime.fromisoformat(
                value_clean
            )

            if dt.tzinfo is None:

                dt = dt.replace(
                    tzinfo=timezone.utc
                )

            return dt.astimezone(
                timezone.utc
            )

        except Exception:
            pass

    return None


def is_recent_article(
    published_date
):

    if not published_date:
        return False

    now = datetime.now(
        timezone.utc
    )

    cutoff = (
        now
        - timedelta(
            days=MAX_ARTICLE_AGE_DAYS
        )
    )

    return (
        cutoff
        <= published_date
        <= now + timedelta(
            hours=2
        )
    )


def calculate_relevance(
    title,
    description,
    threat_type,
    ticker
):

    score = 0

    if ticker:
        score += 60

    if threat_type:
        score += 20

    text = (
        f"{title} {description}"
        .lower()
    )

    important_terms = [
        "breach",
        "ransomware",
        "zero-day",
        "zero day",
        "zero-days",
        "zero days",
        "actively exploited",
        "cyberattack",
        "hacked",
        "compromised",
        "data leak",
        "data exfiltration",
        "security flaw",
        "vulnerability",
        "credential theft",
        "account compromise",
        "remote code execution",
        "rce"
    ]

    for term in important_terms:

        if term in text:

            score += 5
            break

    return min(
        score,
        100
    )


def fetch_feed(
    feed_name,
    feed_url
):

    safe_print(
        f"\n[FEED] {feed_name} -> "
        f"{feed_url}..."
    )

    try:

        feed = feedparser.parse(
            feed_url,
            request_headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
        )

        if getattr(
            feed,
            "bozo",
            False
        ):

            if not feed.entries:

                safe_print(
                    "  [WARN] Feed could not be parsed"
                )

                return []

        entries = feed.entries[:20]

        safe_print(
            f"  Fetched "
            f"{len(entries)} articles"
        )

        return entries

    except Exception as error:

        safe_print(
            f"  [WARN] Could not reach feed: "
            f"{error}"
        )

        return []


def process_article(
    entry,
    source
):

    title = clean_text(
        entry.get(
            "title",
            ""
        )
    )

    description = clean_text(
        entry.get(
            "summary",
            entry.get(
                "description",
                ""
            )
        )
    )

    if not title:
        return None, "invalid"

    if is_editorial_article(
        title,
        description
    ):
        return None, "editorial"

    published_date = parse_date(
        entry
    )

    if not is_recent_article(
        published_date
    ):
        return None, "old"

    threat_type = classify_threat(
        title,
        description
    )

    if not threat_type:
        return None, "low"

    ticker = extract_company_from_text(
        title,
        description
    )

    relevance = calculate_relevance(
        title,
        description,
        threat_type,
        ticker
    )

    if relevance < 20:
        return None, "low"

    users_affected = (
        extract_users_affected(
            f"{title} {description}"
        )
    )

    severity = estimate_severity(
        threat_type,
        title,
        description
    )

    link = clean_text(
        entry.get(
            "link",
            ""
        )
    )

    result = {

        "headline":
            title,

        "description":
            description,

        "source":
            source,

        "published":
            (
                published_date.strftime(
                    "%a, %d %b %Y "
                    "%H:%M:%S UTC"
                )
                if published_date
                else "Unknown"
            ),

        "ticker":
            (
                ticker
                if ticker
                else "UNKNOWN"
            ),

        "company":
            (
                get_company_name(
                    ticker
                )
                if ticker
                else "Unknown Target"
            ),

        "threat_type":
            threat_type,

        "severity":
            severity,

        "users_affected_millions":
            users_affected,

        "relevance":
            relevance,

        "link":
            link
    }

    return result, "accepted"


def scan_feeds():

    safe_print("=" * 70)

    safe_print(
        "CYBER THREAT COLLECTOR - "
        "LIVE RSS FEED SCANNER"
    )

    safe_print(
        "Scanning 5 threat intelligence feeds..."
    )

    safe_print(
        f"Recency filter: last "
        f"{MAX_ARTICLE_AGE_DAYS} days"
    )

    safe_print("=" * 70)

    all_articles = []

    skipped_old = 0
    skipped_low = 0
    skipped_invalid = 0
    skipped_editorial = 0

    total_articles = 0

    for source, feed_url in FEEDS.items():

        entries = fetch_feed(
            source,
            feed_url
        )

        total_articles += len(
            entries
        )

        for entry in entries:

            result, status = process_article(
                entry,
                source
            )

            if result:

                all_articles.append(
                    result
                )

            elif status == "old":

                skipped_old += 1

            elif status == "low":

                skipped_low += 1

            elif status == "invalid":

                skipped_invalid += 1

            elif status == "editorial":

                skipped_editorial += 1

    safe_print("")

    safe_print(
        f"[SCAN] Total articles fetched: "
        f"{total_articles}"
    )

    safe_print(
        f"\n[FILTER] "
        f"{len(all_articles)} recent "
        f"threat candidates identified"
    )

    safe_print(
        f"[FILTER] Skipped old articles: "
        f"{skipped_old}"
    )

    safe_print(
        f"[FILTER] Skipped low-relevance "
        f"articles: {skipped_low}"
    )

    safe_print(
        f"[FILTER] Skipped editorial "
        f"articles: {skipped_editorial}"
    )

    if skipped_invalid:

        safe_print(
            f"[FILTER] Skipped invalid "
            f"articles: {skipped_invalid}"
        )

    safe_print("")

    safe_print("=" * 70)

    safe_print(
        "THREAT INTELLIGENCE SCAN RESULTS"
    )

    safe_print("=" * 70)

    for index, threat in enumerate(
        all_articles,
        1
    ):

        safe_print(
            f"\n[{index:02d}] "
            + "-" * 60
        )

        safe_headline = (
            threat["headline"][:100]
            .encode(
                "ascii",
                "ignore"
            )
            .decode(
                "ascii"
            )
        )

        safe_print(
            f"  Headline : "
            f"{safe_headline}"
        )

        safe_print(
            f"  Source   : "
            f"{threat['source']} | "
            f"Published: "
            f"{threat['published']}"
        )

        safe_print(
            f"  Target   : "
            f"{threat['company']} "
            f"(${threat['ticker']})"
        )

        safe_print(
            f"  Type     : "
            f"{threat['threat_type']}"
        )

        safe_print(
            f"  Severity : "
            f"{threat['severity']:.1f}/10"
        )

        if (
            threat[
                "users_affected_millions"
            ] > 0
        ):

            safe_print(
                f"  Affected : "
                f"{threat['users_affected_millions']:.4f}M"
            )

        else:

            safe_print(
                "  Affected : Not reported"
            )

        safe_print(
            f"  Relevance: "
            f"{threat['relevance']}%"
        )

        safe_print(
            f"  Link     : "
            f"{threat['link']}"
        )

    return all_articles


def initialize_processed_articles():

    conn = sqlite3.connect(
        DB_PATH
    )

    cursor = conn.cursor()

    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS processed_articles (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            article_url TEXT UNIQUE NOT NULL,
            headline TEXT,
            ticker TEXT,
            processed_at TEXT NOT NULL
        )
        """
    )

    conn.commit()
    conn.close()


def article_already_processed(
    link
):

    if not link:
        return False

    initialize_processed_articles()

    conn = sqlite3.connect(
        DB_PATH
    )

    cursor = conn.cursor()

    cursor.execute(
        """
        SELECT 1
        FROM processed_articles
        WHERE article_url = ?
        LIMIT 1
        """,
        (link,)
    )

    result = cursor.fetchone()

    conn.close()

    return result is not None


def mark_article_processed(
    threat
):

    link = threat.get(
        "link",
        ""
    )

    if not link:
        return

    initialize_processed_articles()

    conn = sqlite3.connect(
        DB_PATH
    )

    cursor = conn.cursor()

    cursor.execute(
        """
        INSERT OR IGNORE INTO processed_articles
        (
            article_url,
            headline,
            ticker,
            processed_at
        )
        VALUES (?, ?, ?, ?)
        """,
        (
            link,
            threat.get(
                "headline",
                ""
            ),
            threat.get(
                "ticker",
                "UNKNOWN"
            ),
            datetime.now(
                timezone.utc
            ).strftime(
                "%Y-%m-%d %H:%M:%S UTC"
            )
        )
    )

    conn.commit()
    conn.close()


def ingest_threats(
    threats,
    limit=1
):

    from realtime_collector import (
        ingest_and_evaluate_threat
    )

    safe_print("")
    safe_print("=" * 70)

    safe_print(
        f"AUTOMATIC THREAT INGESTION "
        f"MODE | LIMIT: {limit}"
    )

    safe_print("=" * 70)

    initialize_processed_articles()

    eligible = [
        threat for threat in threats
        if str(threat.get("ticker") or "").strip().upper() not in {"", "UNKNOWN", "NONE", "NULL"}
        and str(threat.get("company") or "").strip()
        and str(threat.get("link") or "").strip()
    ]

    if not eligible:

        safe_print(
            "\n[INGEST] No threats with a "
            "resolved public-company ticker."
        )

        return

    ingested = 0
    skipped = 0

    for threat in eligible:

        if ingested >= limit:
            break

        if article_already_processed(
            threat["link"]
        ):

            safe_headline = (
                threat["headline"][:70]
                .encode(
                    "ascii",
                    "ignore"
                )
                .decode(
                    "ascii"
                )
            )

            safe_print(
                f"\n[SKIP] Already processed: "
                f"{safe_headline}"
            )

            skipped += 1
            continue

        safe_print("")
        safe_print("-" * 70)

        safe_headline_full = (
            threat["headline"]
            .encode(
                "ascii",
                "ignore"
            )
            .decode(
                "ascii"
            )
        )

        safe_print(
            f"[INGEST] "
            f"{safe_headline_full}"
        )

        safe_print(
            f"         "
            f"{threat['company']} "
            f"(${threat['ticker']})"
        )

        try:

            description = (
                f"{threat['headline']}. "
                f"{threat['description']}"
            )

            result = ingest_and_evaluate_threat(
                company=threat["company"],
                threat_type=threat["threat_type"],
                description=description
            )

            safe_print(
                f"\n[DONE] Risk Score: "
                f"{result.get('risk_score', 'N/A')}/100"
            )

            safe_print(
                f"[DONE] System Signal: "
                f"{result.get('recommendation', 'NO SIGNAL')}"
            )

            if not result or result.get("threat_id") is None or result.get("ticker") == "UNKNOWN":
                safe_print("[SKIP] Ingestion was rejected; article remains eligible for retry.")
                continue

            mark_article_processed(threat)
            ingested += 1

            time.sleep(2)

        except Exception as error:

            safe_print(
                f"\n[ERROR] Ingestion failed: "
                f"{error}"
            )

    safe_print("")
    safe_print("=" * 70)

    safe_print(
        f"[COMPLETE] Successfully ingested: "
        f"{ingested}"
    )

    safe_print(
        f"[COMPLETE] Already processed/skipped: "
        f"{skipped}"
    )

    safe_print("=" * 70)
    return ingested


def export_json(
    threats,
    output_path="threat_feed_snapshot.json"
):

    snapshot = {

        "scanned_at":
            datetime.now(
                timezone.utc
            ).strftime(
                "%Y-%m-%d %H:%M:%S UTC"
            ),

        "total_feeds":
            len(FEEDS),

        "threats_found":
            len(threats),

        "threats":
            threats
    }

    with open(
        output_path,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            snapshot,
            file,
            indent=2,
            ensure_ascii=False
        )

    safe_print(
        f"\n[EXPORT] Snapshot saved -> "
        f"{output_path}"
    )


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description=(
            "Cyber Threat RSS Collector - "
            "Scan, classify, and optionally "
            "ingest threat feeds."
        )
    )

    parser.add_argument(
        "--ingest",
        action="store_true",
        help=(
            "Automatically send resolved "
            "threats through AI + ML pipeline."
        )
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=1,
        help=(
            "Maximum number of threats "
            "to ingest. Default: 1."
        )
    )

    parser.add_argument(
        "--export-json",
        action="store_true",
        help=(
            "Export the scan to "
            "threat_feed_snapshot.json."
        )
    )

    args = parser.parse_args()

    threats = scan_feeds()

    if args.export_json:

        export_json(
            threats
        )

    if args.ingest:

        ingest_threats(
            threats,
            limit=args.limit
        )

    else:

        safe_print(
            "\n[TIP] Run with --ingest "
            "to process resolved threats "
            "through Gemini AI + RandomForest ML."
        )
