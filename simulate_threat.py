from threat_analyzer_agent import analyze_threat
from risk_scoring_engine import calculate_risk_score


simulated_threat = {
    "headline": "SIMULATED: Microsoft Corporate Data Breach Exposes 2 Million Customer Records",
    "description": (
        "SIMULATED TEST EVENT. This is artificial data created "
        "to test the cyber threat intelligence pipeline. "
        "The simulated incident involves unauthorized access "
        "to a Microsoft corporate database containing customer information."
    ),
    "company": "Microsoft Corporation",
    "ticker": "MSFT",
    "threat_type": "Data Breach",
    "severity": 8.5,
    "users_affected_millions": 2.0
}


print("=" * 70)
print("SIMULATED CYBER THREAT TEST")
print("=" * 70)

print("\nThreat:")
print(simulated_threat["headline"])

print("\nTarget:")
print(
    f'{simulated_threat["company"]} '
    f'(${simulated_threat["ticker"]})'
)

print("\nType:")
print(simulated_threat["threat_type"])

print("\nSeverity:")
print(simulated_threat["severity"])

print("\nAffected users:")
print(
    f'{simulated_threat["users_affected_millions"]} million'
)

print("\nRunning AI analysis...")

analysis = analyze_threat(
    simulated_threat["headline"],
    simulated_threat["description"]
)

print("\nAI ANALYSIS")
print("-" * 50)

for key, value in analysis.items():
    print(f"{key}: {value}")


print("\nRunning risk engine...")

risk = calculate_risk_score(
    severity_score=simulated_threat["severity"],
    data_sensitivity_score=8.0,
    predicted_price_drop=analysis.get(
        "predicted_stock_impact_percent", 0.0
    ),
    current_stock_change=0.0,
    users_affected_millions=simulated_threat[
        "users_affected_millions"
    ]
)

print("\nRISK RESULT")
print("-" * 50)

if isinstance(risk, dict):
    for key, value in risk.items():
        print(f"{key}: {value}")
else:
    print(risk)

print("\n" + "=" * 70)
print("SIMULATION COMPLETE")
print("This event is artificial test data.")
print("It is NOT a real cyber incident.")
print("=" * 70)