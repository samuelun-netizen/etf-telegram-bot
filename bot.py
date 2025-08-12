"""ETF Telegram bot (yfinance + Investing.com fallback)

- Tries to download historical prices from yfinance first.
- If yfinance fails for a ticker, tries to fetch using investpy (Investing.com).
- Estimates expected returns and covariance from available data.
- Computes a mean-variance (max Sharpe) allocation subject to:
    * No short positions
    * Minimum weight per ETF (3.5%)
    * Sum of weights = 1
- Converts weights to USD allocation for a total investment (2000 USD by default).
- Sends a Telegram message with the allocation (synchronous python-telegram-bot v13).
"""

import os
import warnings
from datetime import datetime
import numpy as np
import pandas as pd
import yfinance as yf
try:
    import investpy
    HAS_INVESTPY = True
except Exception:
    HAS_INVESTPY = False
from scipy.optimize import minimize
from telegram import Bot

warnings.filterwarnings('ignore')

TOKEN = os.getenv('TELEGRAM_TOKEN')
CHAT_ID = os.getenv('TELEGRAM_CHAT_ID')
TOTAL_USD = float(os.getenv('CASH_USD', 2000))
LOOKBACK_DAYS = int(os.getenv('LOOKBACK_DAYS', 252))
RISK_FREE = float(os.getenv('RISK_FREE', 0.02))
MIN_WEIGHT = float(os.getenv('MIN_WEIGHT', 0.035))

TICKERS = [
    'ARKB','ARKK','BND','BNDX','EMB','EUFN','EWG','EWP','EWQ','EZU',
    'FEZ','GLD','HYG','ICLN','IEUR','IGF','QQQ','SOXX','TIP','VB',
    'VEU','VGK','VIG','VNQ','VOO','XLC','XLE','XLF','XLI','XLP',
    'XLU','XLV','XLY'
]

bot = Bot(token=TOKEN) if TOKEN else None

def fetch_yf(ticker, days=LOOKBACK_DAYS):
    end = datetime.utcnow().date()
    start = end - pd.Timedelta(days=days)
    try:
        df = yf.download(ticker, start=start.isoformat(), end=(end + pd.Timedelta(days=1)).isoformat(), progress=False, threads=False, auto_adjust=True)
        if df is None or df.empty:
            return None
        # try 'Adj Close' then 'Close' then first numeric column
        if 'Adj Close' in df.columns:
            s = df['Adj Close'].dropna()
        elif 'Close' in df.columns:
            s = df['Close'].dropna()
        else:
            # pick first numeric column
            s = df.select_dtypes(include='number').iloc[:,0].dropna()
        return s
    except Exception:
        return None

def fetch_investing(ticker, days=LOOKBACK_DAYS):
    if not HAS_INVESTPY:
        return None
    try:
        # try to search ETF on investing.com
        results = investpy.search_quotes(text=ticker, products=['etfs'], n_results=5)
        if not results:
            return None
        item = results[0]
        to_date = datetime.utcnow().date()
        from_date = to_date - pd.Timedelta(days=days)
        df = investpy.get_etf_historical_data(etf=item.symbol, country=item.country, from_date=from_date.strftime('%d/%m/%Y'), to_date=to_date.strftime('%d/%m/%Y'))
        if df is None or df.empty:
            return None
        if 'Close' in df.columns:
            s = df['Close'].dropna()
            s.index = pd.to_datetime(s.index)
            return s
        return None
    except Exception:
        return None

def fetch_all(tickers):
    data = {}
    missing = []
    for t in tickers:
        s = fetch_yf(t)
        if s is not None and not s.empty:
            data[t] = s
            continue
        s2 = fetch_investing(t)
        if s2 is not None and not s2.empty:
            data[t] = s2
            continue
        missing.append(t)
    if not data:
        return pd.DataFrame(), missing
    df = pd.DataFrame(data).sort_index().dropna(how='all')
    df = df.fillna(method='ffill').fillna(method='bfill')
    return df, missing

def estimate(mu_series, cov_matrix):
    # mean-variance optimize (max Sharpe) with minimum weight constraint
    mu = mu_series
    cov = cov_matrix
    n = len(mu)
    x0 = np.repeat(1.0/n, n)
    bounds = tuple((MIN_WEIGHT, 1.0) for _ in range(n))
    cons = ({'type':'eq','fun': lambda x: np.sum(x)-1.0},)
    def neg_sharpe(w, mu, cov, rf):
        port_ret = np.dot(w, mu)
        port_vol = np.sqrt(np.dot(w, cov).dot(w))
        if port_vol == 0:
            return 1e6
        return -(port_ret - rf)/port_vol
    res = minimize(neg_sharpe, x0, args=(mu.values, cov.values, RISK_FREE), method='SLSQP', bounds=bounds, constraints=cons, options={'maxiter':1000})
    if not res.success:
        # fallback: enforce min weights and renormalize
        w = np.maximum(x0, MIN_WEIGHT)
        w = w / w.sum()
        return pd.Series(w, index=mu.index)
    w = res.x
    w[w < 1e-8] = 0.0
    w = w / w.sum()
    return pd.Series(w, index=mu.index)

def build_message(weights, prices, missing):
    alloc = (weights * TOTAL_USD).round(2)
    shares = (alloc / prices).replace([np.inf, -np.inf], np.nan).round(6)
    lines = [f"📊 Rebalance recomendada (Total ${TOTAL_USD:.2f})", ""]
    for t in weights.index:
        lines.append(f"{t}: {weights[t]*100:.2f}% -> ${alloc[t]:.2f} (~{shares[t]} sh) | price ${prices[t]:.2f}")
    if missing:
        lines.append("")
        lines.append("⚠️ No se pudieron obtener datos para: " + ", ".join(missing))
    return "\n".join(lines)

def run_once():
    prices_df, missing = fetch_all(TICKERS)
    if prices_df.empty:
        text = "No se obtuvieron precios de los ETFs. Revisa fuentes."
        if bot:
            bot.send_message(chat_id=CHAT_ID, text=text)
        else:
            print(text)
        return
    daily = prices_df.pct_change().dropna()
    mu = daily.mean() * 252
    cov = daily.cov() * 252
    # ensure order matches
    mu = mu.reindex(prices_df.columns)
    cov = cov.reindex(index=prices_df.columns, columns=prices_df.columns)
    weights = estimate(mu, cov)
    last_prices = prices_df.iloc[-1].reindex(weights.index)
    msg = build_message(weights, last_prices, missing)
    if bot:
        bot.send_message(chat_id=CHAT_ID, text=msg)
    else:
        print(msg)

if __name__ == '__main__':
    run_once()
