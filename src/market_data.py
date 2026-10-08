import numpy as np
import pandas as pd

def enrich_prices(prices, benchmark="SPY"):
    p = prices.copy().sort_values(["ticker", "date"])
    p["log_return"] = p.groupby("ticker")["close"].transform(lambda s: np.log(s / s.shift(1)))
    p["market_return"] = p.loc[p["ticker"].eq(benchmark)].set_index("date")["log_return"].reindex(p["date"]).to_numpy()
    p["abnormal_return"] = p["log_return"] - p["market_return"]
    for h in [1,5,10,20]: p[f"return_{h}d"] = p.groupby("ticker")["log_return"].transform(lambda s: s.shift(-1).rolling(h, min_periods=h).sum())
    for w in [20,60]: p[f"volatility_{w}d"] = p.groupby("ticker")["log_return"].transform(lambda s: s.rolling(w, min_periods=w).std() * np.sqrt(252))
    p["abnormal_volatility"] = p["volatility_20d"] - p.groupby("ticker")["volatility_20d"].transform(lambda s: s.rolling(60, min_periods=20).mean())
    return p

def merge_panel(news_daily, prices):
    cols = ["ticker","date","return_1d","return_5d","return_10d","return_20d","log_return","abnormal_return","volatility_20d","volatility_60d","abnormal_volatility","volume"]
    return news_daily.merge(prices[[c for c in cols if c in prices.columns]], on=["ticker","date"], how="inner")
