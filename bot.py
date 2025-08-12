import os
import yfinance as yf
import pandas as pd
from telegram import Bot

# Configuración desde variables de entorno
TOKEN = os.getenv("TELEGRAM_TOKEN")
CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")
CASH_USD = float(os.getenv("CASH_USD", 2000))

TICKERS = [
    "ARKB", "ARKK", "BND", "BNDX", "EMB", "EUFN", "EWG", "EWP", "EWQ", "EZU",
    "FEZ", "GLD", "HYG", "ICLN", "IEUR", "IGF", "QQQ", "SOXX", "TIP", "VB",
    "VEU", "VGK", "VIG", "VNQ", "VOO", "XLC", "XLE", "XLF", "XLI", "XLP",
    "XLU", "XLV", "XLY"
]

MIN_PCT = 0.035  # Mínimo 3.5% por ETF

bot = Bot(token=TOKEN)

def get_data():
    prices = {}
    for ticker in TICKERS:
        try:
            df = yf.download(ticker, period="6mo", interval="1d", progress=False, threads=False)
            if not df.empty:
                prices[ticker] = df["Adj Close"].iloc[-1]
        except Exception as e:
            print(f"Error al descargar {ticker}: {e}")
    return prices

def optimize_portfolio(prices):
    n = len(prices)
    min_allocation = CASH_USD * MIN_PCT
    allocation = {t: min_allocation for t in prices}
    remaining = CASH_USD - min_allocation * n

    if remaining > 0:
        inv_prices = {t: 1/p for t, p in prices.items()}
        total_inv = sum(inv_prices.values())
        for t in prices:
            allocation[t] += remaining * (inv_prices[t] / total_inv)

    return allocation

def send_message(text):
    bot.send_message(chat_id=CHAT_ID, text=text)

if __name__ == "__main__":
    prices = get_data()
    if prices:
        allocation = optimize_portfolio(prices)
        df = pd.DataFrame([allocation]).T
        df.columns = ["USD"]
        df["%"] = (df["USD"] / CASH_USD) * 100
        msg = "📊 Distribución recomendada:\n" + df.to_string()
        send_message(msg)
    else:
        send_message("No se pudieron obtener precios de los ETFs.")
