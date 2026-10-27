"""
tests/test_batch4_threat_analysis.py

Automated test suite for Batch 4: Cyber Threat Log Analysis & Correlation.
Tests:
  1. CVE Extraction from unformatted text.
  2. CISA KEV Catalog correlation with real database records.
  3. Strict evidence rule: KEV status does not infer direct company breach.
  4. Company exposure classification: DIRECT vs INDIRECT vs UNKNOWN.
  5. Transparent severity scoring (separating intrinsic severity from exposure confidence).
"""

import unittest
import os
import sqlite3

from threat_evidence_normalizer import (
    extract_cve_ids,
    correlate_cisa_kev,
    classify_company_exposure,
    compute_evidence_quality,
    calculate_transparent_severity
)


class TestThreatAnalysis(unittest.TestCase):

    def test_cve_extraction(self):
        """Test extraction of standardized uppercase CVE IDs."""
        text = "Vulnerabilities include cve-2021-44228 and CVE-2020-1472, along with CVE-2023-1234567."
        cves = extract_cve_ids(text)
        self.assertEqual(len(cves), 3)
        self.assertIn("CVE-2021-44228", cves)
        self.assertIn("CVE-2020-1472", cves)
        self.assertIn("CVE-2023-1234567", cves)

    def test_cisa_kev_correlation(self):
        """Verify correlation against real 1,741 entries in cisa_kev_entries."""
        cves = ["CVE-2021-44228", "CVE-9999-0000"]  # One real KEV, one nonexistent
        res = correlate_cisa_kev(cves, db_path="threat_stock_data.db")

        self.assertTrue(res["has_kev_entries"])
        self.assertEqual(res["matched_cve_count"], 1)
        matched = res["matched_cves"][0]
        self.assertEqual(matched["cve_id"], "CVE-2021-44228")
        self.assertEqual(matched["vendor_project"], "Apache")
        self.assertEqual(res["exploitation_status"], "CONFIRMED_IN_WILD")
        self.assertIn("does not prove compromise of an organization's internal network", res["kev_note"])

    def test_exposure_classification(self):
        """Test evidence-based exposure classification: DIRECT vs INDIRECT vs UNKNOWN."""
        # Direct
        direct_text = "Hackers breached Microsoft internal corporate email accounts and exfiltrated executive emails."
        res_dir = classify_company_exposure(direct_text, "Microsoft", "MSFT")
        self.assertEqual(res_dir["exposure_level"], "DIRECT")
        self.assertTrue(res_dir["is_direct"])

        # Indirect (supply chain / third party)
        indirect_text = "Okta reports third-party contractor Sitel was compromised by Lapsus$, impacting customer tenants."
        res_ind = classify_company_exposure(indirect_text, "Okta", "OKTA")
        self.assertEqual(res_ind["exposure_level"], "INDIRECT")
        self.assertTrue(res_ind["is_indirect"])

        # Unknown / unconfirmed
        unknown_text = "Researchers discovered a high-severity flaw in open-source Apache libraries used across the industry."
        res_unk = classify_company_exposure(unknown_text, "Apple", "AAPL")
        self.assertEqual(res_unk["exposure_level"], "UNKNOWN")

    def test_transparent_severity_scoring(self):
        """Verify that UNKNOWN exposure dampens composite score, preventing false criticals."""
        # High intrinsic severity with DIRECT exposure
        direct_score = calculate_transparent_severity(
            base_severity=9.0,
            data_sensitivity=9.0,
            exposure_level="DIRECT",
            is_kev_exploited=True,
            evidence_quality=8.0
        )
        self.assertGreaterEqual(direct_score["composite_risk_score"], 70)
        self.assertEqual(direct_score["severity_tier"], "CRITICAL")
        self.assertEqual(direct_score["recommended_signal"], "REVIEW")

        # Same high intrinsic severity with UNKNOWN exposure
        unknown_score = calculate_transparent_severity(
            base_severity=9.0,
            data_sensitivity=9.0,
            exposure_level="UNKNOWN",
            is_kev_exploited=True,
            evidence_quality=8.0
        )
        # Exposure multiplier (0.40) must keep it below CRITICAL
        self.assertLess(unknown_score["composite_risk_score"], 50)
        self.assertNotEqual(unknown_score["severity_tier"], "CRITICAL")
        self.assertIn(unknown_score["recommended_signal"], ["MONITOR", "NO SIGNAL"])


if __name__ == "__main__":
    unittest.main()
