import os
import time
import yfinance as yf
import pandas as pd
import telegram
import warnings

# Silenciar warnings de yfinance
warnings.simplefilter(action='ignore', category=FutureWarning)

# Variables de entorno
TOKEN = os.getenv("TELEGRAM_TOKEN")
CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")
CASH_USD = float(os.getenv("CASH_USD", 2000))

# Lista ampliada de ETFs
TICKERS = [
    "ARKB", "ARKK", "BND", "BNDX", "EMB", "EUFN", "EWG", "EWP", "EWQ", "EZU",
    "FEZ", "GLD", "HYG", "ICLN", "IEUR", "IGF", "QQQ", "SOXX", "TIP", "VB",
    "VEU", "VGK", "VIG", "VNQ", "VOO", "XLC", "XLE", "XLF", "XLI", "XLP", "XLU", "XLV", "XLY"
]

MIN_PERCENT = 0.035  # 3.5% mínimo por ETF

bot = telegram.Bot(token=TOKEN)

def get_data():
    data = yf.download(TICKERS, period="6mo", interval="1d")["Adj Close"]
    returns = data.pct_change().mean() * 252  # rendimiento anualizado
    return returns.sort_values(ascending=False)

def allocate_portfolio(returns):
    base_weights = returns / returns.sum()
    
    # Asegurar mínimo del 3.5%
    weights = base_weights.copy()
    weights[weights < MIN_PERCENT] = MIN_PERCENT
    
    # Rebalancear para que sumen 1
    weights = weights / weights.sum()
    return weights

def send_message(message):
    bot.send_message(chat_id=CHAT_ID, text=message)

def main():
    while True:
        returns = get_data()
        weights = allocate_portfolio(returns)
        allocation = (weights * CASH_USD).round(2)
        
        msg = "📊 Nueva distribución recomendada:\n"
        for etf, amount in allocation.items():
            msg += f"{etf}: ${amount} USD ({weights[etf]*100:.2f}%)\n"
        
        send_message(msg)
        time.sleep(3600)  # Esperar 1 hora

if __name__ == "__main__":
    main()
