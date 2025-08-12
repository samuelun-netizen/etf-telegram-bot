import os
import yfinance as yf
import pandas as pd
from telegram import Bot
import time

# Configuración
TOKEN = os.getenv("TELEGRAM_TOKEN")
CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")
CASH_USD = float(os.getenv("CASH_USD", 2000))

TICKERS = [
    "ARKB", "ARKK", "BND", "BNDX", "EMB", "EUFN", "EWG", "EWP", "EWQ", "EZU", "FEZ",
    "GLD", "HYG", "ICLN", "IEUR", "IGF", "QQQ", "SOXX", "TIP", "VB", "VEU", "VGK", "VIG",
    "VNQ", "VOO", "XLC", "XLE", "XLF", "XLI", "XLP", "XLU", "XLV", "XLY"
]

MIN_PCT = 0.035  # 3.5% mínimo por ETF

bot = Bot(token=TOKEN)

def get_allocation():
    data = yf.download(TICKERS, period="6mo", interval="1d")["Adj Close"]
    returns = data.pct_change().mean()
    weights = returns / returns.sum()

    # Aplicar restricción de mínimo 3.5%
    weights = weights.clip(lower=MIN_PCT)
    weights /= weights.sum()

    allocation = (weights * CASH_USD).round(2)
    return allocation

def send_update():
    allocation = get_allocation()
    msg = "📊 Nueva distribución del portafolio:\n"
    for ticker, amount in allocation.items():
        msg += f"{ticker}: ${amount}\n"
    bot.send_message(chat_id=CHAT_ID, text=msg)

if __name__ == "__main__":
    while True:
        send_update()
        time.sleep(3600)
