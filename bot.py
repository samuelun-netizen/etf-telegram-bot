import os
import time
import yfinance as yf
import numpy as np
import pandas as pd
from scipy.optimize import minimize
import requests
from datetime import datetime, timezone

TOKEN = os.getenv("TELEGRAM_TOKEN")
CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")
CASH = float(os.getenv("CASH_USD", 2000))

TICKERS = ["ARKB", "ARKK", "BND", "BNDX", "EMB", "EUFN", "EWG", "EWP", "EWQ", "EZU", "FEZ"]

def get_data():
    data = yf.download(TICKERS, period="6mo", interval="1d")["Adj Close"]
    returns = data.pct_change().dropna()
    return returns

def portfolio_performance(weights, mean_returns, cov_matrix, risk_free_rate=0.02):
    returns = np.sum(mean_returns * weights) * 252
    std_dev = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights))) * np.sqrt(252)
    sharpe_ratio = (returns - risk_free_rate) / std_dev
    return std_dev, returns, sharpe_ratio

def neg_sharpe_ratio(weights, mean_returns, cov_matrix, risk_free_rate=0.02):
    return -portfolio_performance(weights, mean_returns, cov_matrix, risk_free_rate)[2]

def max_sharpe_ratio(mean_returns, cov_matrix):
    num_assets = len(mean_returns)
    args = (mean_returns, cov_matrix)
    constraints = {"type": "eq", "fun": lambda x: np.sum(x) - 1}
    bound = (0.0, 1.0)
    bounds = tuple(bound for asset in range(num_assets))
    result = minimize(neg_sharpe_ratio, num_assets * [1. / num_assets], args=args,
                      method="SLSQP", bounds=bounds, constraints=constraints)
    return result

def send_message(text):
    url = f"https://api.telegram.org/bot{TOKEN}/sendMessage"
    payload = {"chat_id": CHAT_ID, "text": text, "parse_mode": "Markdown"}
    requests.post(url, data=payload)

def main():
    returns = get_data()
    mean_returns = returns.mean()
    cov_matrix = returns.cov()
    result = max_sharpe_ratio(mean_returns, cov_matrix)
    weights = result.x
    latest_prices = yf.download(TICKERS, period="1d", interval="1d")["Adj Close"].iloc[-1]

    allocation = []
    for ticker, weight in zip(TICKERS, weights):
        usd_alloc = weight * CASH
        shares = usd_alloc / latest_prices[ticker]
        allocation.append((ticker, weight, usd_alloc, latest_prices[ticker], shares))

    df = pd.DataFrame(allocation, columns=["Ticker", "Weight", "USD Allocation", "Last Price", "Approx Shares"])
    df["Weight"] = df["Weight"].round(4)
    df["USD Allocation"] = df["USD Allocation"].round(2)
    df["Approx Shares"] = df["Approx Shares"].round(2)

    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    message = f"📊 *Distribución óptima* ({now})\n"
    for _, row in df.iterrows():
        message += f"{row['Ticker']}: {row['Weight']*100:.2f}% - ${row['USD Allocation']} ({row['Approx Shares']} sh)\n"

    send_message(message)

if __name__ == "__main__":
    while True:
        try:
            main()
        except Exception as e:
            send_message(f"⚠️ Error en bot: {str(e)}")
        time.sleep(3600)
