import os, sqlite3
from dotenv import load_dotenv

# 1. File status
files = [
    'app.py', 'fetch_stocks.py', 'threat_collector.py', 'threat_analyzer_agent.py',
    'threat_ml_engine.py', 'risk_scoring_engine.py', 'realtime_collector.py',
    'alerts.py', 'build_training_dataset.py', 'train_model.py',
    'database_setup.py', 'correlation_analyzer.py', 'dashboard.html', 'requirements.txt'
]
print('=== FILE STATUS ===')
all_ok = True
for f in files:
    size = os.path.getsize(f) if os.path.exists(f) else 0
    status = 'OK' if size > 100 else 'EMPTY/MISSING'
    if status != 'OK':
        all_ok = False
    print(f'  [{status}] {f} ({size} bytes)')

# 2. Model
print('\n=== MODEL STATUS ===')
model = 'threat_impact_model.joblib'
model_ok = os.path.exists(model)
print(f'  threat_impact_model.joblib: {"EXISTS" if model_ok else "MISSING"} ({os.path.getsize(model) if model_ok else 0} bytes)')

# 3. DB tables
print('\n=== DATABASE STATUS ===')
conn = sqlite3.connect('threat_stock_data.db')
c = conn.cursor()
c.execute("SELECT name FROM sqlite_master WHERE type='table'")
tables = [r[0] for r in c.fetchall()]
for t in tables:
    c.execute(f'SELECT COUNT(*) FROM {t}')
    count = c.fetchone()[0]
    c.execute(f'PRAGMA table_info({t})')
    cols = [r[1] for r in c.fetchall()]
    print(f'  [{t}] rows={count} | cols: {cols}')
conn.close()

# 4. .env
print('\n=== ENV STATUS ===')
load_dotenv()
gemini = os.getenv('GEMINI_API_KEY', '').strip()
if gemini:
    print(f'  GEMINI_API_KEY: SET ({len(gemini)} chars) — Gemini AI ACTIVE')
else:
    print('  GEMINI_API_KEY: NOT SET — Heuristic fallback active (add key to .env to enable Gemini AI)')

# 5. Verify model name fix
print('\n=== MODEL NAME FIX ===')
with open('threat_analyzer_agent.py', 'r') as fh:
    content = fh.read()
    if 'gemini-2.0-flash' in content:
        print('  threat_analyzer_agent.py: gemini-2.0-flash [CORRECT]')
    elif 'gemini-3.8-flash' in content:
        print('  threat_analyzer_agent.py: gemini-3.8-flash [STILL WRONG]')
    else:
        print('  threat_analyzer_agent.py: model name not found')

print('\n=== SUMMARY ===')
print(f'  Files: {"ALL OK" if all_ok else "SOME ISSUES"}')
print(f'  ML Model: {"TRAINED & READY" if model_ok else "NOT TRAINED - run train_model.py"}')
print(f'  Database: {"INITIALIZED" if tables else "EMPTY - run database_setup.py"}')
