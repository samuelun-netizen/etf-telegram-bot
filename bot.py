"""Robust ETF Telegram bot

- Tries yfinance for full historical series.
- If unavailable, tries investpy (Investing.com) if installed.
- If still missing, fills expected returns and covariance with sensible estimates so ALL tickers are included.
- Optimizes portfolio by maximizing Sharpe ratio with:
    * no short positions
    * minimum weight (3.5% by default)
    * sum(weights) = 1
- Sends one Telegram message with results.
"""

import os
import time
import math
import warnings
from datetime import datetime
import numpy as np
import pandas as pd
import yfinance as yf
import requests
from io import StringIO

try:
    import investpy
    HAS_INVESTPY = True
except Exception:
    HAS_INVESTPY = False

from scipy.optimize import minimize
from telegram import Bot

warnings.filterwarnings('ignore')

# --- Configuration ---
TOKEN = os.getenv('TELEGRAM_TOKEN')
CHAT_ID = os.getenv('TELEGRAM_CHAT_ID')
TOTAL_USD = float(os.getenv('CASH_USD', 2000))
LOOKBACK_DAYS = int(os.getenv('LOOKBACK_DAYS', 252))  # ~1 trading year
RISK_FREE = float(os.getenv('RISK_FREE', 0.02))
MIN_WEIGHT = float(os.getenv('MIN_WEIGHT', 0.035))

TICKERS = [
    'ARKB','ARKK','BND','BNDX','EMB','EUFN','EWG','EWP','EWQ','EZU',
    'FEZ','GLD','HYG','ICLN','IEUR','IGF','QQQ','SOXX','TIP','VB',
    'VEU','VGK','VIG','VNQ','VOO','XLC','XLE','XLF','XLI','XLP',
    'XLU','XLV','XLY'
]

bot = Bot(token=TOKEN) if TOKEN else None

def yf_series(ticker, days=LOOKBACK_DAYS):
    end = datetime.utcnow().date()
    start = end - pd.Timedelta(days=days)
    try:
        df = yf.download(ticker, start=start.isoformat(), end=(end + pd.Timedelta(days=1)).isoformat(), progress=False, threads=False, auto_adjust=True)
        if df is None or df.empty:
            return None
        # prefer 'Adj Close' else 'Close' else first numeric column
        if 'Adj Close' in df.columns:
            s = df['Adj Close'].dropna()
        elif 'Close' in df.columns:
            s = df['Close'].dropna()
        else:
            s = df.select_dtypes(include='number').iloc[:, 0].dropna()
        s.index = pd.to_datetime(s.index)
        return s
    except Exception as e:
        # Try alternative yf.Ticker.history
        try:
            tk = yf.Ticker(ticker)
            hist = tk.history(period=f"{days}d", auto_adjust=True)
            if hist is None or hist.empty:
                return None
            if 'Close' in hist.columns:
                s = hist['Close'].dropna()
            else:
                s = hist.select_dtypes(include='number').iloc[:, 0].dropna()
            s.index = pd.to_datetime(s.index)
            return s
        except Exception:
            return None

def investpy_series(ticker, days=LOOKBACK_DAYS):
    if not HAS_INVESTPY:
        return None
    try:
        res = investpy.search_quotes(text=ticker, products=['etfs'], n_results=5)
        if not res:
            return None
        item = res[0]
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

def yahoo_csv_series(ticker, days=LOOKBACK_DAYS):
    # Try the Yahoo CSV download endpoint as fallback
    end = int(time.time())
    start = int((datetime.utcnow().date() - pd.Timedelta(days=days)).strftime('%s')) if hasattr(time, 'mktime') else end - days*86400
    url = f"https://query1.finance.yahoo.com/v7/finance/download/{ticker}?period1={start}&period2={end}&interval=1d&events=history&includeAdjustedClose=true"
    try:
        r = requests.get(url, timeout=15)
        if r.status_code != 200:
            return None
        s = pd.read_csv(StringIO(r.text), parse_dates=['Date'], index_col='Date')
        if s.empty:
            return None
        if 'Adj Close' in s.columns:
            return s['Adj Close'].dropna()
        elif 'Close' in s.columns:
            return s['Close'].dropna()
        else:
            return s.select_dtypes(include='number').iloc[:, 0].dropna()
    except Exception:
        return None

def fetch_all_series(tickers):
    series_map = {}
    missing = []
    for t in tickers:
        s = yf_series(t)
        if s is None or s.empty:
            s = investpy_series(t) if HAS_INVESTPY else None
        if s is None or s.empty:
            s = yahoo_csv_series(t)
        if s is None or s.empty:
            missing.append(t)
        else:
            # ensure enough points
            if len(s) < 30:
                # try to extend or consider missing
                missing.append(t)
            else:
                series_map[t] = s
        time.sleep(0.5)  # polite delay
    return series_map, missing

def build_full_mu_cov(series_map, missing, all_tickers):
    # compute mu and cov for available tickers
    if series_map:
        prices_df = pd.DataFrame(series_map).sort_index().dropna(how='all')
        daily = prices_df.pct_change().dropna()
        mu_avail = daily.mean() * 252
        cov_avail = daily.cov() * 252
    else:
        # no real data
        mu_avail = pd.Series(dtype=float)
        cov_avail = pd.DataFrame()

    # Prepare full mu and cov with estimates for missing tickers
    mu_full = pd.Series(index=all_tickers, dtype=float)
    for t in all_tickers:
        if t in mu_avail.index:
            mu_full[t] = mu_avail[t]
    # fill missing mu with mean of available or small positive default
    if mu_avail.size > 0:
        mu_mean = float(mu_avail.mean())
        var_mean = float(np.diag(cov_avail).mean()) if not cov_avail.empty else 0.05
        cov_mean = float(cov_avail.values.mean()) if not cov_avail.empty else 0.01
    else:
        mu_mean = 0.05
        var_mean = 0.05
        cov_mean = 0.01

    for t in all_tickers:
        if pd.isna(mu_full[t]):
            mu_full[t] = mu_mean

    # Build covariance full matrix
    cov_full = pd.DataFrame(cov_mean, index=all_tickers, columns=all_tickers)
    for t in all_tickers:
        cov_full.loc[t, t] = var_mean

    # insert available cov values where present
    for i in cov_avail.index:
        for j in cov_avail.columns:
            if i in cov_full.index and j in cov_full.columns:
                cov_full.loc[i, j] = cov_avail.loc[i, j]

    return mu_full, cov_full, (prices_df if 'prices_df' in locals() else pd.DataFrame())

def max_sharpe(mu, cov, min_w=MIN_WEIGHT, rf=RISK_FREE):
    n = len(mu)
    x0 = np.repeat(1.0/n, n)
    bounds = tuple((min_w, 1.0) for _ in range(n))
    cons = ({'type':'eq','fun': lambda x: np.sum(x)-1.0},)
    def neg_sharpe(w, mu_vals, cov_vals, rf_val):
        port_ret = np.dot(w, mu_vals)
        port_vol = math.sqrt(np.dot(w, cov_vals).dot(w))
        if port_vol == 0:
            return 1e6
        return -(port_ret - rf_val) / port_vol
    res = minimize(neg_sharpe, x0, args=(mu.values, cov.values, rf), method='SLSQP', bounds=bounds, constraints=cons, options={'maxiter':1000})
    if not res.success:
        w = np.maximum(x0, min_w)
        w = w / w.sum()
        return pd.Series(w, index=mu.index)
    w = res.x
    w[w < 1e-8] = 0.0
    w = w / w.sum()
    return pd.Series(w, index=mu.index)

def get_last_prices(prices_df, series_map, all_tickers):
    last = pd.Series(index=all_tickers, dtype=float)
    # fill from prices_df if available
    if not prices_df.empty:
        for t in prices_df.columns:
            last.loc[t] = prices_df[t].iloc[-1]
    # for missing tickers try quick fetch of last price (yfinance Ticker.info or history last)
    for t in all_tickers:
        if pd.isna(last.loc[t]):
            try:
                tk = yf.Ticker(t)
                info = tk.history(period='5d', auto_adjust=True)
                if info is not None and not info.empty:
                    last.loc[t] = info['Close'].iloc[-1]
            except Exception:
                last.loc[t] = float(np.nan)
    # if still NaN, fill with mean price
    mean_price = last.dropna().mean() if last.dropna().size>0 else 1.0
    last = last.fillna(mean_price)
    return last

def build_message(weights, last_prices, missing):
    alloc_usd = (weights * TOTAL_USD).round(2)
    shares = (alloc_usd / last_prices).replace([np.inf, -np.inf], np.nan).round(6)
    lines = []
    lines.append(f"📊 Rebalance recomendada (Total: ${TOTAL_USD:.2f})")
    lines.append("")
    for t in weights.index:
        lines.append(f"{t}: {weights[t]*100:.2f}% -> ${alloc_usd[t]:.2f} (~{shares[t]} sh) | price ${last_prices[t]:.2f}")
    if missing:
        lines.append("")
        lines.append("⚠️ No se pudieron obtener series históricas completas para: " + ", ".join(missing))
        lines.append("   Se usaron estimaciones para esos tickers (mu/cov promedio). Revisa mapeos si necesitas datos exactos.")
    return "\n".join(lines)

def run_once():
    series_map, missing = fetch_all_series(TICKERS)
    mu, cov, prices_df = build_full_mu_cov(series_map, missing, TICKERS)
    weights = max_sharpe(mu, cov, min_w=MIN_WEIGHT)
    last_prices = get_last_prices(prices_df, series_map, TICKERS)
    msg = build_message(weights, last_prices, missing)
    if bot:
        bot.send_message(chat_id=CHAT_ID, text=msg)
    else:
        print(msg)

if __name__ == '__main__':
    run_once()
