import os
import requests
from dotenv import dotenv_values

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ENV_FILE = os.path.join(BASE_DIR, ".env")

config = dotenv_values(ENV_FILE)

DISCORD_WEBHOOK_URL = config.get("DISCORD_WEBHOOK_URL")

def send_discord_alert(
    ticker: str,
    risk_score: int,
    recommendation: str,
    threat_summary: str
):
    """
    Sends a formatted alert to the configured Discord webhook.
    """

    if not DISCORD_WEBHOOK_URL:
        print("[ERROR] DISCORD_WEBHOOK_URL not found in .env")
        return False

    color_map = {
        "REVIEW": 15548997,    # Red   — high risk, warrants review
        "MONITOR": 16776960,   # Yellow — elevated risk, monitor closely
        "NO SIGNAL": 5793266   # Green  — insufficient evidence
    }

    embed_color = color_map.get(
        recommendation.upper(),
        5793266
    )

    payload = {
        "username": "Threat Risk Engine",
        "embeds": [
            {
                "title": f"MATERIAL BREACH ALERT: {ticker}",
                "description": threat_summary,
                "color": embed_color,
                "fields": [
                    {
                        "name": "Target Ticker",
                        "value": f"`{ticker}`",
                        "inline": True
                    },
                    {
                        "name": "Risk Score",
                        "value": f"`{risk_score} / 100`",
                        "inline": True
                    },
                    {
                        "name": "Actionable Signal",
                        "value": f"**{recommendation}**",
                        "inline": True
                    }
                ],
                "footer": {
                    "text": "Cyber Threat Intelligence System - Automated Alert"
                }
            }
        ]
    }

    try:
        response = requests.post(
            DISCORD_WEBHOOK_URL,
            json=payload,
            timeout=10
        )

        if response.status_code in [200, 204]:
            print("[SUCCESS] Discord alert sent successfully!")
            return True

        print(
            f"[ERROR] Failed to send alert. "
            f"HTTP Status Code: {response.status_code}"
        )

        if response.text:
            print(f"[ERROR] Discord response: {response.text}")

        return False

    except requests.RequestException as e:
        print(f"[EXCEPT] Error sending alert: {e}")
        return False


if __name__ == "__main__":
    send_discord_alert(
        ticker="MSFT",
        risk_score=85,
        recommendation="REVIEW",
        threat_summary=(
            "Test alert from the Cyber Threat Intelligence System."
        )
    )