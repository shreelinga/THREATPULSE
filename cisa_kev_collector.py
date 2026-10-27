
import sqlite3
import requests
from datetime import datetime, timezone

DB_NAME = "threat_stock_data.db"

CISA_KEV_URL = "https://www.cisa.gov/sites/default/files/feeds/known_exploited_vulnerabilities.json"


def create_table():
    connection = sqlite3.connect(DB_NAME)

    connection.execute("""
        CREATE TABLE IF NOT EXISTS cisa_kev_entries (
            cve_id TEXT PRIMARY KEY,
            vendor_project TEXT,
            product TEXT,
            vulnerability_name TEXT,
            date_added TEXT,
            short_description TEXT,
            required_action TEXT,
            due_date TEXT,
            ransomware_use TEXT,
            source_url TEXT,
            fetched_at TEXT
        )
    """)

    connection.commit()
    connection.close()


def fetch_cisa_kev():
    response = requests.get(CISA_KEV_URL, timeout=30)
    response.raise_for_status()

    data = response.json()
    vulnerabilities = data.get("vulnerabilities", [])

    connection = sqlite3.connect(DB_NAME)
    saved = 0

    for item in vulnerabilities:
        connection.execute("""
            INSERT INTO cisa_kev_entries (
                cve_id, vendor_project, product, vulnerability_name,
                date_added, short_description, required_action,
                due_date, ransomware_use, source_url, fetched_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(cve_id) DO UPDATE SET
                vendor_project = excluded.vendor_project,
                product = excluded.product,
                vulnerability_name = excluded.vulnerability_name,
                date_added = excluded.date_added,
                short_description = excluded.short_description,
                required_action = excluded.required_action,
                due_date = excluded.due_date,
                ransomware_use = excluded.ransomware_use,
                source_url = excluded.source_url,
                fetched_at = excluded.fetched_at
        """, (
            item.get("cveID"),
            item.get("vendorProject"),
            item.get("product"),
            item.get("vulnerabilityName"),
            item.get("dateAdded"),
            item.get("shortDescription"),
            item.get("requiredAction"),
            item.get("dueDate"),
            item.get("knownRansomwareCampaignUse"),
            f"https://nvd.nist.gov/vuln/detail/{item.get('cveID', '')}",
            datetime.now(timezone.utc).isoformat()
        ))

        saved += 1

    connection.commit()
    connection.close()

    return saved


if __name__ == "__main__":
    create_table()

    try:
        count = fetch_cisa_kev()
        print(f"CISA KEV records processed: {count}")
        print("CISA KEV collection completed successfully.")
    except requests.RequestException as error:
        print(f"Could not fetch CISA data: {error}")
    except (ValueError, sqlite3.Error) as error:
        print(f"Data processing error: {error}")
