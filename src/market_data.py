import numpy as np
import pandas as pd

def download_yfinance_prices(tickers, start, end, benchmark="SPY"):
    """Download only the requested liquid universe plus benchmark; no bulk history."""
    import yfinance as yf
    symbols = sorted(set(tickers) | {benchmark})
    start_ts = pd.Timestamp(start) - pd.Timedelta(days=100)
    end_ts = pd.Timestamp(end) + pd.Timedelta(days=35)
    records = []
    for symbol in symbols:
        raw = yf.download(symbol, start=start_ts.date(), end=end_ts.date(), auto_adjust=False, progress=False, threads=False)
        if raw is None or raw.empty:
            continue
        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = raw.columns.get_level_values(0)
        close_col = "Adj Close" if "Adj Close" in raw.columns else "Close"
        if close_col not in raw.columns: continue
        close = raw[close_col]
        volume = raw.get("Volume", np.nan)
        if isinstance(close, pd.DataFrame): close = close.iloc[:, 0]
        if isinstance(volume, pd.DataFrame): volume = volume.iloc[:, 0]
        part = pd.DataFrame({"date": raw.index, "close": close, "volume": volume})
        part["ticker"] = symbol
        records.append(part.reset_index(drop=True))
    if not records:
        raise RuntimeError("yfinance returned no price data for the configured universe.")
    return pd.concat(records, ignore_index=True)

def enrich_prices(prices, benchmark="SPY"):
    p = prices.copy().sort_values(["ticker", "date"])
    p["log_return"] = p.groupby("ticker")["close"].transform(lambda s: np.log(s / s.shift(1)))
    benchmark_returns = p.loc[p["ticker"].eq(benchmark)].set_index("date")["log_return"]
    p["market_return"] = p["date"].map(benchmark_returns)
    p["abnormal_return"] = p["log_return"] - p["market_return"]
    for h in [1, 5, 10, 20]:
        p[f"return_{h}d"] = p.groupby("ticker")["log_return"].transform(lambda s: s.shift(-1).rolling(h, min_periods=h).sum())
    for w in [20, 60]:
        p[f"volatility_{w}d"] = p.groupby("ticker")["log_return"].transform(lambda s: s.rolling(w, min_periods=w).std() * np.sqrt(252))
    p["abnormal_volatility"] = p["volatility_20d"] - p.groupby("ticker")["volatility_20d"].transform(lambda s: s.rolling(60, min_periods=20).mean())
    return p

def merge_panel(news_daily, prices):
    cols = ["ticker", "date", "return_1d", "return_5d", "return_10d", "return_20d", "log_return", "abnormal_return", "volatility_20d", "volatility_60d", "abnormal_volatility", "volume"]
    return news_daily.merge(prices[[c for c in cols if c in prices.columns]], on=["ticker", "date"], how="inner")
