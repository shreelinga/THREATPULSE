import sqlite3
from datetime import datetime
from alerts import send_discord_alert


def calculate_correlation(ticker, threat_date):
    conn = sqlite3.connect('threat_stock_data.db')
    cursor = conn.cursor()

    print(f"\nAnalyzing {ticker}...")

    cursor.execute('''
        SELECT price, date FROM stock_prices
        WHERE ticker = ?
        ORDER BY date DESC
        LIMIT 2
    ''', (ticker,))

    prices = cursor.fetchall()

    if len(prices) >= 2:
        price_after = prices[0][0]
        price_before = prices[1][0]

        if price_before > 0:
            change_percent = (
                (price_after - price_before) / price_before
            ) * 100

            print(f"Price before: ${price_before:.2f}")
            print(f"Price after: ${price_after:.2f}")
            print(f"Change: {change_percent:.2f}%")

            if change_percent < 0:
                recommendation = "SELL"
                risk_score = 70

                threat_summary = (
                    f"{ticker} stock price decreased by "
                    f"{abs(change_percent):.2f}%."
                )

                print(
                    f"Stock DROPPED - Recommendation: "
                    f"{recommendation}"
                )

                send_discord_alert(
                    ticker=ticker,
                    risk_score=risk_score,
                    recommendation=recommendation,
                    threat_summary=threat_summary
                )

            else:
                recommendation = "HOLD"
                risk_score = 0

                print(
                    f"Stock UP - Recommendation: "
                    f"{recommendation}"
                )

        else:
            print("Invalid previous stock price.")

    else:
        print(
            f"Not enough stock price data available "
            f"for {ticker}."
        )

    conn.close()


if __name__ == "__main__":
    companies = ['MSFT', 'AAPL', 'GOOGL']
    threat_date = datetime.now().strftime('%Y-%m-%d')

    for ticker in companies:
        calculate_correlation(ticker, threat_date)