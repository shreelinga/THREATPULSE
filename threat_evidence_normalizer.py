"""
threat_evidence_normalizer.py

Evidence-based Cyber Threat Analysis, CISA KEV Correlation, and Company Mapping.
Performs:
  1. Incident normalization (source URL, publisher, timestamp, threat category, CVE extraction).
  2. CISA KEV catalog correlation (evidence of in-the-wild exploitation, not automatic breach proof).
  3. Evidence-based company exposure classification (DIRECT, INDIRECT, UNKNOWN).
  4. Transparent multi-factor severity scoring separating intrinsic severity from mapping confidence.
"""

import re
import sqlite3
import os
from typing import Dict, Any, List, Optional, Tuple

CVE_PATTERN = re.compile(r"\bCVE-\d{4}-\d{4,7}\b", re.IGNORECASE)

KNOWN_SUPPLY_CHAIN_KEYWORDS = [
    "vendor", "contractor", "third-party", "supply chain", "upstream",
    "dependency", "library", "cloud provider", "saas", "subcontractor"
]

DIRECT_BREACH_KEYWORDS = [
    "breached", "hacked", "compromised", "ransomware hit", "stole data from",
    "exfiltrated from", "infiltrated", "extorted", "unauthorized access to"
]


def extract_cve_ids(text: str) -> List[str]:
    """Extracts standardized uppercase CVE identifiers from free text."""
    if not text:
        return []
    matches = CVE_PATTERN.findall(str(text))
    return sorted(list(set(m.upper() for m in matches)))


def correlate_cisa_kev(
    cve_ids: List[str],
    db_path: str = "threat_stock_data.db"
) -> Dict[str, Any]:
    """
    Correlates extracted CVE IDs against the local CISA Known Exploited Vulnerabilities (KEV) table.
    Enforces the strict rule: KEV status proves the flaw is exploited in the wild,
    NOT that a monitored company was compromised.
    """
    if not cve_ids or not os.path.exists(db_path):
        return {
            "has_kev_entries": False,
            "matched_cves": [],
            "ransomware_use_confirmed": False,
            "kev_summary": "No CVEs identified or matched against CISA KEV catalog."
        }

    matched = []
    ransomware_flag = False

    try:
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        placeholders = ",".join("?" for _ in cve_ids)
        cursor.execute(f"""
            SELECT cve_id, vendor_project, product, vulnerability_name,
                   date_added, due_date, ransomware_use, short_description
            FROM cisa_kev_entries
            WHERE cve_id IN ({placeholders})
        """, cve_ids)

        rows = cursor.fetchall()
        for r in rows:
            entry = dict(r)
            if entry.get("ransomware_use", "").upper() == "KNOWN":
                ransomware_flag = True
            matched.append(entry)

        conn.close()
    except Exception as e:
        return {
            "has_kev_entries": False,
            "matched_cves": [],
            "ransomware_use_confirmed": False,
            "error": str(e)
        }

    has_entries = len(matched) > 0
    return {
        "has_kev_entries": has_entries,
        "matched_cve_count": len(matched),
        "matched_cves": matched,
        "ransomware_use_confirmed": ransomware_flag,
        "exploitation_status": "CONFIRMED_IN_WILD" if has_entries else "NOT_REPORTED",
        "kev_note": (
            "CISA KEV status establishes active in-the-wild exploitation of the vulnerability; "
            "it does not prove compromise of an organization's internal network unless direct evidence exists."
        ) if has_entries else "No matched vulnerabilities in CISA KEV catalog."
    }


def classify_company_exposure(
    text: str,
    company_name: str,
    ticker: str
) -> Dict[str, Any]:
    """
    Classifies company exposure state into:
      - DIRECT: Evidence directly links breach/compromise to the organization.
      - INDIRECT: Organization affected through a third-party, vendor, or supply-chain dependency.
      - UNKNOWN: General vulnerability advisory, research disclosure, or unverified rumor.
    """
    lower_text = str(text or "").lower()
    comp_lower = str(company_name or "").lower()

    # Direct breach indicators
    has_company = comp_lower in lower_text or (ticker and ticker.lower() in lower_text)
    has_direct_keyword = any(kw in lower_text for kw in DIRECT_BREACH_KEYWORDS)
    has_supply_chain = any(kw in lower_text for kw in KNOWN_SUPPLY_CHAIN_KEYWORDS)

    if not has_company:
        return {
            "exposure_level": "UNKNOWN",
            "confidence": 0.3,
            "rationale": f"No explicit mention of {company_name} in threat documentation.",
            "is_direct": False,
            "is_indirect": False
        }

    if has_direct_keyword and not has_supply_chain:
        return {
            "exposure_level": "DIRECT",
            "confidence": 0.85,
            "rationale": f"Explicit reporting links security incident directly to {company_name} systems.",
            "is_direct": True,
            "is_indirect": False
        }
    elif has_supply_chain:
        return {
            "exposure_level": "INDIRECT",
            "confidence": 0.70,
            "rationale": f"Incident appears centered on third-party supply-chain/vendor affecting {company_name}.",
            "is_direct": False,
            "is_indirect": True
        }
    else:
        return {
            "exposure_level": "UNKNOWN",
            "confidence": 0.40,
            "rationale": f"Advisory or disclosure mentions {company_name}, but compromise is unconfirmed.",
            "is_direct": False,
            "is_indirect": False
        }


def compute_evidence_quality(
    publisher: Optional[str] = None,
    source_url: Optional[str] = None,
    has_cves: bool = False,
    is_kev: bool = False
) -> float:
    """
    Calculates evidence quality score (1.0 - 10.0) based on source credibility,
    vulnerability identifiers, and official advisory citations.
    """
    score = 4.0  # Base score for general threat reporting

    pub = str(publisher or "").lower()
    url = str(source_url or "").lower()

    # Trusted primary threat intelligence feeds
    if any(k in pub or k in url for k in ["cisa", "cert", "krebsonsecurity", "bleepingcomputer", "thehackernews", "securityweek"]):
        score += 2.5

    # CVE specificity
    if has_cves:
        score += 1.5

    # CISA KEV verified
    if is_kev:
        score += 1.5

    return min(10.0, max(1.0, score))


def calculate_transparent_severity(
    base_severity: float,
    data_sensitivity: float,
    exposure_level: str = "UNKNOWN",
    is_kev_exploited: bool = False,
    evidence_quality: float = 5.0
) -> Dict[str, Any]:
    """
    Transparent severity scoring that separates intrinsic technical vulnerability severity
    from confidence in company exposure, preventing unconfirmed reports from firing critical alerts.
    """
    base_sev = min(10.0, max(1.0, float(base_severity)))
    data_sens = min(10.0, max(1.0, float(data_sensitivity)))
    ev_qual = min(10.0, max(1.0, float(evidence_quality)))

    # Weights
    # 1. Technical severity (0-40)
    tech_points = (base_sev / 10.0) * 40.0

    # 2. Data sensitivity (0-30)
    data_points = (data_sens / 10.0) * 30.0

    # 3. KEV Exploitation confirmed (0-15)
    kev_points = 15.0 if is_kev_exploited else 5.0

    # 4. Exposure confidence multiplier (Direct: 1.0, Indirect: 0.70, Unknown: 0.40)
    exposure_multiplier = {
        "DIRECT": 1.0,
        "INDIRECT": 0.70,
        "UNKNOWN": 0.40
    }.get(exposure_level.upper(), 0.40)

    # 5. Evidence quality multiplier (0.7 to 1.0)
    evidence_multiplier = 0.7 + (ev_qual / 10.0) * 0.3

    raw_score = (tech_points + data_points + kev_points) * exposure_multiplier * evidence_multiplier
    normalized_score = int(round(min(100.0, max(0.0, raw_score))))


    if normalized_score >= 70 and exposure_level == "DIRECT":
        tier = "CRITICAL"
        action = "REVIEW"
    elif normalized_score >= 50:
        tier = "HIGH"
        action = "REVIEW"
    elif normalized_score >= 30:
        tier = "MEDIUM"
        action = "MONITOR"
    else:
        tier = "LOW"
        action = "NO SIGNAL"

    return {
        "severity_score": round(base_sev, 1),
        "composite_risk_score": normalized_score,
        "severity_tier": tier,
        "recommended_signal": action,
        "exposure_level": exposure_level,
        "exposure_multiplier": exposure_multiplier,
        "evidence_quality_score": round(ev_qual, 1),
        "is_kev_exploited": is_kev_exploited,
        "explanation": (
            f"Intrinsic severity: {base_sev}/10, Data sensitivity: {data_sens}/10. "
            f"Exposure classified as {exposure_level} (factor {exposure_multiplier}). "
            f"Evidence quality assessed at {ev_qual:.1f}/10."
        )
    }
