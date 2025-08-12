import os
import asyncio
import pandas as pd
import yfinance as yf
from telegram import Bot

TOKEN = os.getenv("TELEGRAM_TOKEN")
CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")
CASH_USD = float(os.getenv("CASH_USD", 2000))

TICKERS = [
    "ARKB", "ARKK", "BND", "BNDX", "EMB", "EUFN", "EWG", "EWP", "EWQ", "EZU", "FEZ", "GLD", "HYG", "ICLN",
    "IEUR", "IGF", "QQQ", "SOXX", "TIP", "VB", "VEU", "VGK", "VIG", "VNQ", "VOO", "XLC", "XLE", "XLF",
    "XLI", "XLP", "XLU", "XLV", "XLY"
]

MIN_PCT = 0.035  # 3.5% mínimo por ETF

async def fetch_prices():
    prices = {}
    for ticker in TICKERS:
        try:
            df = yf.download(ticker, period="6mo", interval="1d", progress=False)
            if not df.empty:
                prices[ticker] = df["Adj Close"].iloc[-1]
            else:
                print(f"No data for {ticker}")
        except Exception as e:
            print(f"Failed to get ticker {ticker} reason: {e}")
    return prices

def allocate_portfolio(prices):
    total_inv = CASH_USD
    n = len(prices)
    equal_weight = total_inv / n

    allocations = {}
    min_allocation = total_inv * MIN_PCT

    for ticker, price in prices.items():
        alloc = max(equal_weight, min_allocation)
        allocations[ticker] = alloc

    total_alloc = sum(allocations.values())
    factor = total_inv / total_alloc
    for ticker in allocations:
        allocations[ticker] *= factor

    return allocations

async def main():
    prices = await fetch_prices()
    if not prices:
        print("No prices fetched")
        return

    allocations = allocate_portfolio(prices)
    msg_lines = ["ETF Portfolio Allocation:"]
    for ticker, amount in allocations.items():
        msg_lines.append(f"{ticker}: ${amount:.2f}")
    msg = "\n".join(msg_lines)

    bot = Bot(token=TOKEN)
    await bot.send_message(chat_id=CHAT_ID, text=msg)

if __name__ == "__main__":
    asyncio.run(main())
