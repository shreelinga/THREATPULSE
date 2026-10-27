"""
research/schema.py

THREATPULSE Phase 4: Point-in-Time (PIT) Incident Dataset Schema.
Defines formal, strongly-typed data structures for empirical quantitative cybersecurity
research, strictly separating point-in-time features available at disclosure (T=0)
from ex-post revisions, retrospective outcomes, and market-model labels.
"""

from dataclasses import dataclass, field, asdict
from datetime import datetime, date
from enum import Enum
from typing import List, Optional, Dict, Any
import json


class MarketSession(str, Enum):
    PRE_MARKET = "PRE_MARKET"          # Before regular trading hours (< 09:30 ET)
    REGULAR_HOURS = "REGULAR_HOURS"    # During regular trading hours (09:30 - 16:00 ET)
    POST_MARKET = "POST_MARKET"        # After regular trading close (> 16:00 ET)
    WEEKEND_HOLIDAY = "WEEKEND_HOLIDAY"# Exchange closed; effective T0 is next trading day


class DisclosureSourceType(str, Enum):
    SEC_8K_ITEM_105 = "SEC_8K_ITEM_105"     # Mandatory SEC Cybersecurity Incident Disclosure
    SEC_8K_ITEM_801 = "SEC_8K_ITEM_801"     # Other Events / Voluntary Disclosures
    SEC_10K_10Q = "SEC_10K_10Q"             # Periodic Regulatory Report
    BSE_NSE_REG_30 = "BSE_NSE_REG_30"       # Indian SEBI LODR Regulation 30 Material Event
    OFFICIAL_PRESS_RELEASE = "OFFICIAL_PR"  # Company Newsroom / PR Newswire / BusinessWire
    VENDOR_SECURITY_ADVISORY = "ADVISORY"   # MSRC, Cisco Talos, vendor patch advisory
    GOVERNMENT_DIRECTIVE = "GOV_DIRECTIVE"  # CISA Emergency Directive, FBI Private Industry Notice
    UNOFFICIAL_MEDIA_LEAK = "MEDIA_LEAK"    # Investigative journalist report (e.g. Krebs, NYT)


class ConfoundingStatus(str, Enum):
    CLEAN = "CLEAN"                         # No confounding corporate actions or macro shocks within [-7, +7] days
    CONFOUNDED = "CONFOUNDED"               # Confirmed earnings, M&A, FDA/regulatory probe, or systemic crash
    SUSPECT_DRIFT = "SUSPECT_DRIFT"         # Post-Earnings Announcement Drift (PEAD) within 5 trading days
    NEEDS_MANUAL_REVIEW = "NEEDS_REVIEW"    # Requires qualitative review before inclusion in training sets


class VerificationStatus(str, Enum):
    VERIFIED_PRIMARY = "VERIFIED_PRIMARY"   # Verified against official SEC EDGAR / Exchange regulatory document
    VERIFIED_SECONDARY = "VERIFIED_SECONDARY" # Verified against authoritative contemporaneous news / company PR
    UNVERIFIED = "UNVERIFIED"               # Secondary citation only, pending primary source retrieval
    DISPUTED = "DISPUTED"                   # Discrepancy between filing time and price reaction


@dataclass
class IncidentIdentification:
    """Core corporate and entity identifiers."""
    incident_id: str                        # Stable unique identifier, e.g., "PIT-2024-UNH-01"
    company_name: str                       # Legal corporate entity name
    ticker: str                             # Exchange ticker symbol, e.g. "UNH", "TATAPOWER.NS"
    exchange: str                           # Primary listing exchange: "NYSE", "NASDAQ", "NSE", "BSE"
    cik: Optional[str] = None               # SEC Central Index Key for US public firms (10 digits)
    country: str = "US"                     # Primary operating jurisdiction
    sector: str = "Healthcare"              # GICS Sector classification
    industry: str = "Managed Healthcare"    # GICS Industry classification


@dataclass
class PointInTimeTiming:
    """Exact point-in-time timestamping and market calendar alignment."""
    original_disclosure_timestamp_utc: str  # ISO 8601 UTC timestamp, e.g. "2024-02-22T11:14:00Z"
    local_timestamp_str: str                # Timestamp in exchange local time
    market_timezone: str                    # "America/New_York", "Asia/Kolkata"
    session_classification: MarketSession   # Pre-market, Regular, Post-market, Weekend
    effective_t0_trading_date: str          # Date of first market session able to react (T0, YYYY-MM-DD)
    t_minus_1_date: str                     # Pre-event base closing price date (T-1, strictly < T0)
    t_plus_1_date: str                      # First post-event session closing price date (T+1, strictly > T0)
    t_plus_2_date: str                      # Second post-event session closing price date (T+2, strictly > T+1)
    source_url: str                         # Permalink to official filing or contemporaneous release
    source_type: DisclosureSourceType       # Item 1.05, Regulation 30, PR, etc.
    verification_status: VerificationStatus # Verified primary, secondary, etc.
    verification_notes: str = ""
    is_intraday_disclosure: bool = False    # True if disclosure occurred during regular exchange trading hours

    def __post_init__(self):
        self.validate_trading_offsets()

    def validate_trading_offsets(self) -> bool:
        """Fails if trading-session offsets are not strictly monotonic."""
        d_tm1 = date.fromisoformat(self.t_minus_1_date)
        d_t0 = date.fromisoformat(self.effective_t0_trading_date)
        d_tp1 = date.fromisoformat(self.t_plus_1_date)
        d_tp2 = date.fromisoformat(self.t_plus_2_date)

        if not (d_tm1 < d_t0 < d_tp1 < d_tp2):
            raise ValueError(
                f"Invalid trading-session offset order: T-1 ({d_tm1}) < T0 ({d_t0}) < "
                f"T+1 ({d_tp1}) < T+2 ({d_tp2}) is violated."
            )
        return True


@dataclass
class PointInTimeFeatures:
    """
    Features STRICTLY available at T=0.
    Forbids ex-post revisions, hindsight scores, or ex-post loss figures.
    """
    disclosed_threat_type: str              # Category stated at T=0 (e.g. "Ransomware", "Cloud Exfiltration")
    is_operational_disruption_reported: bool# Stated business disruption, plant shutdown, or IT freeze at T=0
    is_user_count_quantified_at_t0: bool    # True if an explicit user/account count was stated in initial release
    users_affected_disclosed_at_t0: Optional[float] = None  # Quantified users at T=0 (millions), or None if unknown
    cve_identifiers: List[str] = field(default_factory=list) # Exploited CVEs disclosed at T=0
    is_cisa_kev_listed_at_t0: bool = False  # Was the exploited vulnerability on CISA KEV at T=0?
    is_zero_day_at_t0: bool = False         # Was this reported as an unpatched zero-day at T=0?
    threat_actor_attribution_at_t0: Optional[str] = None # Attribution stated at T=0 (e.g. "ALPHV", "Unknown")
    regulatory_mandate_level: str = "MANDATORY_ITEM_105" # Item 1.05 (material) vs voluntary Item 8.01
    has_ransom_demand_reported: bool = False# Stated extortion/ransom demand at T=0
    qualitative_severity_at_t0: str = "HIGH"# Objective tier: "CRITICAL", "HIGH", "MODERATE", "LOW"


@dataclass
class RetrospectiveAuditing:
    """
    Ex-post facts and revisions.
    STRICTLY QUARANTINED: NEVER fed as inputs to predictive models. Used only for ex-post analysis.
    """
    final_revised_users_millions: Optional[float] = None # Eventual confirmed victim count
    revision_count: int = 0                 # Number of subsequent revisions
    first_revision_date: Optional[str] = None # Date when initial user count was first revised
    final_revision_source_url: Optional[str] = None
    direct_recovery_costs_usd_m: Optional[float] = None # 10-K reported breach remediation cost
    ransom_paid_usd_m: Optional[float] = None # Confirmed ransom payment
    regulatory_fines_usd_m: Optional[float] = None # Total fines from SEC, FTC, OCR, GDPR
    class_action_lawsuits_count: Optional[int] = None
    retrospective_notes: str = ""


@dataclass
class MarketReturnsAndLabels:
    """
    Clean separation of raw returns from benchmark-adjusted abnormal returns
    with explicit holding period definitions.
    """
    benchmark_ticker: str                   # "SPY" (US), "^NSEI" (India), sector ETF
    stock_price_t_minus_1: float            # P[T-1] Close (split/dividend adjusted)
    stock_price_t0: float                   # P[T0] Close
    stock_price_t_plus_1: float             # P[T+1] Close
    stock_price_t_plus_2: float             # P[T+2] Close
    benchmark_price_t_minus_1: float        # Benchmark P[T-1] Close
    benchmark_price_t0: float               # Benchmark P[T0] Close
    benchmark_price_t_plus_1: float         # Benchmark P[T+1] Close
    benchmark_price_t_plus_2: float         # Benchmark P[T+2] Close

    # 1. Explicit Raw Stock Returns
    raw_return_tm1_to_t0: float             # ((P[T0] - P[T-1]) / P[T-1]) * 100 (Immediate T0 reaction)
    raw_return_t0_to_t1: float              # ((P[T+1] - P[T0]) / P[T0]) * 100 (Day 1 follow-through)
    raw_return_tm1_to_t1: float             # ((P[T+1] - P[T-1]) / P[T-1]) * 100 (Cumulative 2-day holding return)
    raw_return_tm1_to_t2: float             # ((P[T+2] - P[T-1]) / P[T-1]) * 100 (Cumulative 3-day holding return)

    # 2. Explicit Benchmark Returns
    benchmark_return_tm1_to_t0: float       # ((BM[T0] - BM[T-1]) / BM[T-1]) * 100
    benchmark_return_t0_to_t1: float        # ((BM[T+1] - BM[T0]) / BM[T0]) * 100
    benchmark_return_tm1_to_t1: float       # ((BM[T+1] - BM[T-1]) / BM[T-1]) * 100
    benchmark_return_tm1_to_t2: float       # ((BM[T+2] - BM[T-1]) / BM[T-1]) * 100

    # 3. Explicit Benchmark-Adjusted Abnormal Returns (Constant Beta = 1.0)
    abnormal_return_tm1_to_t0: float        # raw_return_tm1_to_t0 - benchmark_return_tm1_to_t0
    abnormal_return_t0_to_t1: float         # raw_return_t0_to_t1 - benchmark_return_t0_to_t1
    abnormal_return_tm1_to_t1: float        # raw_return_tm1_to_t1 - benchmark_return_tm1_to_t1
    abnormal_return_tm1_to_t2: float        # raw_return_tm1_to_t2 - benchmark_return_tm1_to_t2

    # 4. Market-Model Abnormal Returns (Estimated via OLS over trading days [-120, -21])
    market_model_alpha: Optional[float] = None      # Intercept from estimation window
    market_model_beta: Optional[float] = None       # Systematic risk sensitivity to benchmark
    market_model_r2: Optional[float] = None         # OLS R-squared in estimation window
    market_model_residual_std: Optional[float] = None # Standard error of regression
    market_model_ar_t0: Optional[float] = None      # R_stock[T0] - (alpha + beta * R_bm[T0])
    market_model_ar_t1: Optional[float] = None      # R_stock[T+1] - (alpha + beta * R_bm[T+1])
    cumulative_abnormal_return_car01: Optional[float] = None # market_model_ar_t0 + market_model_ar_t1
    standardized_abnormal_return_z: Optional[float] = None   # AR[T0] / residual_std

    # 5. Backwards-Compatible Aliases (Mapped strictly to standard event windows)
    raw_return_1d: float = 0.0              # Alias for raw_return_tm1_to_t0
    raw_return_2d: float = 0.0              # Alias for raw_return_tm1_to_t1
    raw_return_3d: float = 0.0              # Alias for raw_return_tm1_to_t2
    benchmark_return_1d: float = 0.0        # Alias for benchmark_return_tm1_to_t0
    benchmark_return_2d: float = 0.0        # Alias for benchmark_return_tm1_to_t1
    abnormal_return_1d: float = 0.0         # Alias for abnormal_return_tm1_to_t0
    abnormal_return_2d: float = 0.0         # Alias for abnormal_return_tm1_to_t1
    market_model_abnormal_return_1d: Optional[float] = None # Alias for market_model_ar_t0
    cumulative_abnormal_return_2d: Optional[float] = None   # Alias for cumulative_abnormal_return_car01

    def __post_init__(self):
        # Auto-populate aliases if default
        if self.raw_return_1d == 0.0 and self.raw_return_tm1_to_t0 != 0.0:
            self.raw_return_1d = self.raw_return_tm1_to_t0
        if self.raw_return_2d == 0.0 and self.raw_return_tm1_to_t1 != 0.0:
            self.raw_return_2d = self.raw_return_tm1_to_t1
        if self.raw_return_3d == 0.0 and self.raw_return_tm1_to_t2 != 0.0:
            self.raw_return_3d = self.raw_return_tm1_to_t2
        if self.benchmark_return_1d == 0.0 and self.benchmark_return_tm1_to_t0 != 0.0:
            self.benchmark_return_1d = self.benchmark_return_tm1_to_t0
        if self.benchmark_return_2d == 0.0 and self.benchmark_return_tm1_to_t1 != 0.0:
            self.benchmark_return_2d = self.benchmark_return_tm1_to_t1
        if self.abnormal_return_1d == 0.0 and self.abnormal_return_tm1_to_t0 != 0.0:
            self.abnormal_return_1d = self.abnormal_return_tm1_to_t0
        if self.abnormal_return_2d == 0.0 and self.abnormal_return_tm1_to_t1 != 0.0:
            self.abnormal_return_2d = self.abnormal_return_tm1_to_t1
        if self.market_model_abnormal_return_1d is None:
            self.market_model_abnormal_return_1d = self.market_model_ar_t0
        if self.cumulative_abnormal_return_2d is None:
            self.cumulative_abnormal_return_2d = self.cumulative_abnormal_return_car01


@dataclass
class ConfoundingAndQualityFlags:
    """Quality and confounder audit verification."""
    confounding_status: ConfoundingStatus
    earnings_announcement_proximity_days: Optional[int] = None # Trading days between earnings and disclosure
    macro_economic_shock_flag: bool = False # Severe market dislocation (e.g. COVID liquidity crash March 2020)
    confounding_details: str = ""
    is_clean_research_candidate: bool = True # Flag indicating whether record is admissible for modeling
    audited_by: str = "THREATPULSE Research Team"
    audit_timestamp: str = ""


@dataclass
class PointInTimeIncidentRecord:
    """Complete, self-contained record uniting all audit and quantitative fields."""
    identification: IncidentIdentification
    timing: PointInTimeTiming
    pit_features: PointInTimeFeatures
    retrospective: RetrospectiveAuditing
    market_labels: MarketReturnsAndLabels
    quality_flags: ConfoundingAndQualityFlags

    def to_flat_dict(self) -> Dict[str, Any]:
        """Flattens nested dataclasses into a tabular dictionary for CSV/DataFrame export."""
        flat = {}
        for sub_obj in [self.identification, self.timing, self.pit_features,
                        self.retrospective, self.market_labels, self.quality_flags]:
            for k, v in asdict(sub_obj).items():
                if isinstance(v, (Enum, MarketSession, DisclosureSourceType, ConfoundingStatus, VerificationStatus)):
                    flat[k] = v.value
                elif isinstance(v, list):
                    flat[k] = ";".join(str(item) for item in v)
                else:
                    flat[k] = v
        return flat

    def to_json(self, indent: int = 2) -> str:
        """Serializes record to strict JSON."""
        return json.dumps(self.to_flat_dict(), indent=indent)
