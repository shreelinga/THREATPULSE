
import sqlite3


def create_database():
    conn = sqlite3.connect("threat_stock_data.db")
    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS threats (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            company TEXT NOT NULL,
            ticker TEXT NOT NULL,
            threat_type TEXT,
            severity TEXT,
            severity_score REAL DEFAULT 7.0,
            users_affected_millions REAL DEFAULT 0.0,
            data_sensitivity_score REAL DEFAULT 7.0,
            threat_date TEXT,
            description TEXT,
            risk_score INTEGER DEFAULT 0,
            recommendation TEXT DEFAULT 'MONITOR',
            analysis_source TEXT DEFAULT 'Rule-Based Heuristic',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

    cursor.execute("PRAGMA table_info(threats)")
    existing_cols = [row[1] for row in cursor.fetchall()]
    new_cols = [
        ("severity_score", "REAL DEFAULT 7.0"),
        ("users_affected_millions", "REAL DEFAULT 0.0"),
        ("data_sensitivity_score", "REAL DEFAULT 7.0"),
        ("risk_score", "INTEGER DEFAULT 0"),
        ("recommendation", "TEXT DEFAULT 'MONITOR'"),
        ("analysis_source", "TEXT DEFAULT 'Rule-Based Heuristic'")
    ]

    for col_name, col_def in new_cols:
        if col_name not in existing_cols:
            cursor.execute(
                f"ALTER TABLE threats ADD COLUMN {col_name} {col_def}"
            )

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS stock_prices (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ticker TEXT NOT NULL,
            price REAL,
            date TEXT,
            volume INTEGER,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS current_stock_data (
            ticker TEXT PRIMARY KEY,
            company_name TEXT,
            current_price REAL,
            previous_close REAL,
            change_percent REAL,
            timestamp TEXT,
            exchange_timestamp TEXT,
            market TEXT,
            currency TEXT,
            price_session TEXT,
            retrieved_at TEXT
        )
    """)

    cursor.execute("PRAGMA table_info(current_stock_data)")
    curr_cols = [row[1] for row in cursor.fetchall()]

    stock_additions = [
        ("company_name", "TEXT"),
        ("exchange_timestamp", "TEXT"),
        ("market", "TEXT"),
        ("currency", "TEXT"),
        ("price_session", "TEXT"),
        ("retrieved_at", "TEXT")
    ]
    for col_name, col_type in stock_additions:
        if col_name not in curr_cols:
            cursor.execute(
                f"ALTER TABLE current_stock_data ADD COLUMN {col_name} {col_type}"
            )

    # Phase 2: Backfill existing current_stock_data rows with standardized exchange metadata
    cursor.execute("""
        UPDATE current_stock_data
        SET market = CASE
                WHEN ticker LIKE '%.NS' THEN 'NSE'
                WHEN ticker LIKE '%.BO' THEN 'BSE'
                ELSE 'NYSE'
            END,
            currency = CASE
                WHEN ticker LIKE '%.NS' OR ticker LIKE '%.BO' THEN 'INR'
                ELSE 'USD'
            END,
            price_session = COALESCE(price_session, 'previous_close'),
            exchange_timestamp = COALESCE(exchange_timestamp, timestamp),
            retrieved_at = COALESCE(retrieved_at, timestamp)
        WHERE market IS NULL OR currency IS NULL OR price_session IS NULL
    """)


    cursor.execute("""
        CREATE TABLE IF NOT EXISTS correlations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            threat_id INTEGER,
            ticker TEXT,
            threat_date TEXT,
            stock_price_before REAL,
            stock_price_after REAL,
            price_change_percent REAL,
            risk_score INTEGER DEFAULT 0,
            recommendation TEXT DEFAULT 'HOLD',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY(threat_id) REFERENCES threats(id)
        )
    """)

    cursor.execute("PRAGMA table_info(correlations)")
    corr_cols = [row[1] for row in cursor.fetchall()]

    if "risk_score" not in corr_cols:
        cursor.execute(
            "ALTER TABLE correlations ADD COLUMN risk_score INTEGER DEFAULT 0"
        )

    if "recommendation" not in corr_cols:
        cursor.execute(
            "ALTER TABLE correlations ADD COLUMN recommendation TEXT DEFAULT 'HOLD'"
        )

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            threat_id INTEGER NOT NULL,
            ticker TEXT NOT NULL,
            prediction_timestamp TEXT NOT NULL,
            predicted_direction TEXT NOT NULL,
            predicted_impact_percent REAL NOT NULL,
            confidence REAL,
            actual_1d_return REAL,
            actual_3d_return REAL,
            actual_5d_return REAL,
            raw_return_1d REAL,
            raw_return_3d REAL,
            raw_return_5d REAL,
            status TEXT NOT NULL DEFAULT 'PENDING'
                CHECK (status IN ('PENDING', 'EVALUATED')),
            evaluated_at TEXT,
            FOREIGN KEY(threat_id) REFERENCES threats(id)
        )
    """)

    cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_predictions_status_timestamp
        ON predictions(status, prediction_timestamp)
    """)

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS event_studies (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            source_key TEXT NOT NULL UNIQUE,
            company TEXT NOT NULL,
            ticker TEXT NOT NULL,
            threat_type TEXT,
            event_date TEXT NOT NULL,
            severity_score REAL,
            severity_tier TEXT,
            sector_ticker TEXT NOT NULL,
            company_return_1d REAL,
            company_return_3d REAL,
            company_return_5d REAL,
            raw_return_1d REAL,
            raw_return_3d REAL,
            raw_return_5d REAL,
            sector_return_1d REAL,
            sector_return_3d REAL,
            sector_return_5d REAL,
            market_return_1d REAL,
            market_return_3d REAL,
            market_return_5d REAL,
            abnormal_return_1d REAL,
            cumulative_abnormal_return_1d REAL,
            cumulative_abnormal_return_3d REAL,
            cumulative_abnormal_return_5d REAL,
            z_score REAL,
            z_score_label TEXT,
            status TEXT NOT NULL DEFAULT 'PENDING'
                CHECK (status IN ('PENDING', 'EVALUATED')),
            last_error TEXT,
            last_attempt_at TEXT,
            evaluated_at TEXT
        )
    """)

    cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_event_studies_status_date
        ON event_studies(status, event_date)
    """)

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS monitored_stocks (
            ticker TEXT PRIMARY KEY,
            company_name TEXT NOT NULL,
            market TEXT NOT NULL,
            added_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

    cursor.execute("PRAGMA table_info(threats)")
    th_cols = [row[1] for row in cursor.fetchall()]
    th_additions = [
        ("exposure_level", "TEXT DEFAULT 'UNKNOWN'"),
        ("cve_ids", "TEXT"),
        ("affected_products", "TEXT"),
        ("review_needed", "INTEGER DEFAULT 0")
    ]
    for c_name, c_def in th_additions:
        if c_name not in th_cols:
            cursor.execute(f"ALTER TABLE threats ADD COLUMN {c_name} {c_def}")

    cursor.execute("PRAGMA table_info(correlations)")
    co_cols = [row[1] for row in cursor.fetchall()]
    if "review_needed" not in co_cols:
        cursor.execute("ALTER TABLE correlations ADD COLUMN review_needed INTEGER DEFAULT 0")
    if "raw_return" not in co_cols:
        cursor.execute("ALTER TABLE correlations ADD COLUMN raw_return REAL")
    cursor.execute("""
        UPDATE correlations
        SET raw_return = price_change_percent
        WHERE raw_return IS NULL AND price_change_percent IS NOT NULL
    """)

    cursor.execute("PRAGMA table_info(predictions)")
    pred_cols = [row[1] for row in cursor.fetchall()]
    pred_additions = [
        ("model_version", "TEXT DEFAULT 'v1.0.0'"),
        ("target_definition", "TEXT DEFAULT 'raw_return'"),
        ("failure_reason", "TEXT"),
        ("review_needed", "INTEGER DEFAULT 0"),
        ("raw_return_1d", "REAL"),
        ("raw_return_3d", "REAL"),
        ("raw_return_5d", "REAL")
    ]
    for c_name, c_def in pred_additions:
        if c_name not in pred_cols:
            cursor.execute(f"ALTER TABLE predictions ADD COLUMN {c_name} {c_def}")

    # Standardize predictions target definition and sync raw_return_Xd with actual_Xd_return
    cursor.execute("""
        UPDATE predictions
        SET raw_return_1d = actual_1d_return,
            raw_return_3d = actual_3d_return,
            raw_return_5d = actual_5d_return
        WHERE raw_return_1d IS NULL AND actual_1d_return IS NOT NULL
    """)
    cursor.execute("""
        UPDATE predictions
        SET target_definition = 'raw_return'
        WHERE target_definition = 'abnormal_return'
    """)

    # Standardize event_studies: add raw_return_1d/3d/5d columns and backfill from company_return_1d/3d/5d
    cursor.execute("PRAGMA table_info(event_studies)")
    es_cols = [row[1] for row in cursor.fetchall()]
    es_additions = [
        ("raw_return_1d", "REAL"),
        ("raw_return_3d", "REAL"),
        ("raw_return_5d", "REAL"),
    ]
    for c_name, c_def in es_additions:
        if c_name not in es_cols:
            cursor.execute(f"ALTER TABLE event_studies ADD COLUMN {c_name} {c_def}")

    cursor.execute("""
        UPDATE event_studies
        SET raw_return_1d = company_return_1d,
            raw_return_3d = company_return_3d,
            raw_return_5d = company_return_5d
        WHERE raw_return_1d IS NULL AND company_return_1d IS NOT NULL
    """)

    # Flag historical ambiguous ticker records for review without deleting them
    cursor.execute("UPDATE threats SET review_needed = 1 WHERE ticker = 'SA'")
    cursor.execute("UPDATE correlations SET review_needed = 1 WHERE ticker = 'SA'")
    cursor.execute("UPDATE predictions SET review_needed = 1 WHERE ticker = 'SA'")

    # Prediction deduplication index
    cursor.execute("""
        CREATE UNIQUE INDEX IF NOT EXISTS idx_predictions_dedup
        ON predictions(threat_id, ticker, model_version, target_definition)
    """)

    conn.commit()
    conn.close()
    print("Database threat_stock_data.db verified and updated successfully!")


if __name__ == "__main__":
    create_database()
