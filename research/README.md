# THREATPULSE Research & Point-in-Time (PIT) Framework

## Overview
This directory contains the isolated research framework, data schemas, validation tools, and pilot datasets developed during **Phase 4 Research Design**.

All research files are strictly segregated from the production engine (`threat_ml_engine.py`) and production artifacts (`threat_impact_model.joblib`, `threat_stock_data.db`, `training_dataset.csv`).

## Directory Contents
- `schema.py`: Formal Python dataclasses defining the Point-in-Time research schema, separating T=0 disclosure features from ex-post revisions and market labels.
- `market_event_study.py`: Event-study math engine that computes market session alignment, OLS market models, abnormal returns, and multi-day CAR.
- `validate_dataset.py`: Automated quality gate auditing candidate datasets against the 11 Phase 4 Acceptance Criteria.
- `generate_pit_dataset.py`: Reproducible generator script creating the pilot research dataset.
- `pit_dataset_v1.csv`: 21 audited corporate cyber incidents (14 legacy + 7 Form 8-K Item 1.05 / Indian Reg 30 filings) formatted under the PIT schema.

## Workflow Execution

### 1. Generate Pilot Dataset
```powershell
python research/generate_pit_dataset.py
```

### 2. Validate Dataset Integrity
```powershell
python research/validate_dataset.py research/pit_dataset_v1.csv
```

### 3. Compute Event Study Labels for a Candidate Ticker
```python
from research.market_event_study import compute_market_model_labels
from datetime import date

labels, err = compute_market_model_labels(
    ticker="UNH",
    benchmark_ticker="SPY",
    effective_t0_date=date(2024, 2, 22)
)
print("Abnormal Return 1D:", labels["abnormal_return_1d"])
print("Market Model Beta:", labels["market_model"]["beta"])
```
