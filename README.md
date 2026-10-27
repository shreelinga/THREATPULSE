# THREATPULSE: Cyber Threat & Stock Market Intelligence Platform

![Python Version](https://img.shields.io/badge/python-3.10%2B-blue.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)
![Platform](https://img.shields.io/badge/platform-Flask%20%7C%20scikit--learn-orange.svg)
![Architecture](https://img.shields.io/badge/market--coverage-US%20(NYSE%2FNASDAQ)%20%7C%20India%20(NSE)-teal.svg)

**THREATPULSE** is an enterprise-grade cybersecurity and equity intelligence platform that bridges the gap between threat telemetry (CISA Known Exploited Vulnerabilities catalog, CVE disclosures, vendor incident advisories, regulatory disclosures) and financial market microstructure.

The platform provides quantitative event-study analytics, abnormal return modeling, evidence-grounded risk scoring, and machine-learning impact estimates across US and Indian equity markets.

---

## Key Capabilities

### 1. Multi-Exchange Market Correctness
- **Exchange-Aware Benchmarking:** Automatically identifies market geography. US equities (`AAPL`, `MSFT`, `CRWD`, etc.) benchmark against the S&P 500 (`SPY`) and tech indices (`XLK`). Indian equities (`TCS.NS`, `INFY.NS`, etc.) benchmark against the Nifty 50 (`^NSEI`) and Nifty IT (`^CNXIT`).
- **Currency & Session Freshness:** Dynamic currency formatting (`$` for US, `₹` for India) with localized quote freshness tracking across market sessions (`America/New_York` and `Asia/Kolkata`).

### 2. Evidence-Based Cyber Risk Scoring
- **Telemetry Grounding:** Direct integration with CISA Known Exploited Vulnerabilities (KEV) catalog and CVE scoring.
- **Exposure Classification:** Categorizes incident exposure into `DIRECT`, `INDIRECT`, or `UNKNOWN` based on verified evidence quality (1–10 scale) with multi-factor impact dampening.

### 3. Quantitative Event Study & Abnormal Return Engine
- **Financial Econometrics:** Implements market-model event study methodology calculating Abnormal Returns ($AR_t = R_{i,t} - (\alpha + \beta R_{m,t})$) and Cumulative Abnormal Returns ($CAR$).
- **Session Alignment:** Enforces event-time monotonicity ($T_{-1} < T_0 < T_{+1} < T_{+2}$), correctly aligning after-hours, weekend, and holiday disclosures to the first reacting trading session.

### 4. Leakage-Safe Machine Learning Impact Pipeline
- **Strict In-Fold Preprocessing:** Scikit-learn preprocessing pipelines (`StandardScaler`, log transforms) are fitted strictly within cross-validation folds to eliminate data leakage.
- **Honest Baseline Comparison:** Regularized Random Forest and linear candidate models are systematically evaluated against in-fold historical mean and median baselines under Leave-One-Out (LOOCV) and Leave-One-Company-Out (LOGO-CV) protocols.
- **Uncertainty Disclosures:** Tree output spread variance is surfaced with explicit disclosures regarding small sample sizes and directional predictability.

### 5. Sandboxed AI Threat Intelligence
- **Prompt Injection Defense:** External incident advisories are sandboxed in isolated tags to resist adversarial manipulation.
- **Resilient Fallback:** Powered by Google Gemini AI with deterministic local rule-based heuristic fallbacks when credentials are not configured.

### 6. Interactive Glassmorphic Visual System
- High-contrast responsive dashboard featuring 8 modular tabs:
  1. **Overview:** Executive KPI metrics, active threats, and high-risk signals.
  2. **Threat Feed:** Real-time threat feed with CISA KEV tags, CVE indicators, and severity badges.
  3. **Live Markets:** Dual-market monitor with live stock quotes, price changes, and benchmark comparisons.
  4. **Event Studies:** Econometric event studies with interactive CAR trajectory visualization.
  5. **ML Impact Engine:** Interactive impact simulator with tree dispersion and baseline comparisons.
  6. **Risk Matrix:** Quadrant scatter plot mapping threat severity against market capitalization.
  7. **Portfolio Impact:** Holding impact calculator estimating dollar/rupee portfolio exposure.
  8. **System Status:** Architecture health monitor, database telemetry, and collector controls.

---

## Repository Structure

```
cyber-threat-stock-intel/
├── app.py                      # Flask REST API backend & static file server
├── dashboard.html              # Single-page glassmorphic dashboard interface
├── requirements.txt            # Python dependencies
├── .env.example                # Template for environment configuration
├── database_setup.py           # SQLite database schema initialization
├── fetch_stocks.py             # Stock universe market quote ingestion (yfinance)
├── realtime_collector.py       # Live threat feed collector & scheduler
├── threat_collector.py         # Historical & live threat scraper/ingestion
├── cisa_kev_collector.py       # CISA Known Exploited Vulnerabilities integration
├── threat_analyzer_agent.py    # AI & heuristic threat intelligence analyzer
├── threat_evidence_normalizer.py# Point-in-time incident evidence normalizer
├── risk_scoring_engine.py      # Multi-factor risk calculation engine
├── correlation_analyzer.py     # Statistical correlation analysis
├── event_study_engine.py       # Market model OLS abnormal return calculation
├── prediction_tracking.py      # Model prediction audit & outcome tracker
├── threat_ml_engine.py         # Model inference & heuristic fallback runtime
├── train_model.py              # Leakage-safe staged ML model training & evaluation
├── stock_universe.py           # Universe definition for US & Indian equities
├── alerts.py                   # Discord webhook notification dispatcher
├── static/brand/               # SVG logo marks, lockups, and favicon
├── tests/                      # Automated unit & regression test suites
│   ├── test_batch1_market_correctness.py
│   ├── test_batch2_data_integrity.py
│   ├── test_batch3_ml_improvement.py
│   ├── test_batch4_threat_analysis.py
│   ├── test_batch5_ai_analysis.py
│   ├── test_batch7_integration.py
│   ├── test_batch8_redesign_and_bugfixes.py
│   ├── test_batch9_ml_reliability.py
│   ├── test_batch10_honest_prediction_reporting.py
│   ├── test_phase1_target_terminology.py
│   └── test_phase2_data_freshness.py
└── research/                   # Point-in-Time econometrics research framework
    ├── README.md               # Research methodology documentation
    ├── schema.py               # Point-in-Time data contracts & schemas
    ├── market_event_study.py   # Research event-study math engine
    ├── validate_dataset.py     # 11-point econometric quality gate auditor
    └── generate_pit_dataset.py # Reproducible research dataset generator
```

---

## Getting Started

### Prerequisites
- Python 3.10 or higher
- Modern web browser (Chrome, Firefox, Edge, Safari)

### 1. Clone the Repository
```bash
git clone https://github.com/<your-username>/threatpulse.git
cd threatpulse
```

### 2. Set Up Virtual Environment
```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Configure Environment
Copy the example environment template:
```bash
cp .env.example .env
```
Edit `.env` to configure optional credentials:
- `GEMINI_API_KEY`: Google Gemini API key for AI threat analysis (optional; falls back to heuristic engine if unset).
- `DISCORD_WEBHOOK_URL`: Discord webhook URL for incident notifications (optional).

### 5. Initialize the Database
```bash
python database_setup.py
```

### 6. Ingest Initial Market Quotes
```bash
python fetch_stocks.py
```

### 7. Run the Application
```bash
python app.py
```
Open your browser and navigate to:
```
http://127.0.0.1:5000
```

---

## Running Tests

The test suite covers market correctness, data integrity, model reliability, and API endpoints:

```bash
# Run ML improvement regression tests (safe from side effects)
python -m unittest tests/test_batch3_ml_improvement.py

# Run ML reliability and leakage prevention tests
python -m unittest tests/test_batch9_ml_reliability.py

# Run market correctness tests
python -m unittest tests/test_batch1_market_correctness.py
```

---

## Research Framework

The `research/` directory provides an isolated Point-in-Time (PIT) event-study framework designed for academic reproducibility and quantitative audit:

```bash
# Validate dataset against the 11 econometric acceptance criteria
python research/validate_dataset.py research/pit_dataset_v1.csv
```

---

## Disclaimer

This software is developed strictly for research, educational, and cybersecurity threat intelligence purposes. Stock price predictions, risk scores, and abnormal return estimates are statistical outputs and do **not** constitute financial, investment, or trading advice.
