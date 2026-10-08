import pandas as pd

def clean_news(df, timezone="UTC"):
    out = df.copy()
    out["headline"] = out["headline"].astype("string").str.strip()
    out["ticker"] = out["ticker"].astype("string").str.upper().str.strip().str.replace(r"[^A-Z.\-]", "", regex=True)
    out["timestamp"] = pd.to_datetime(out["timestamp"], errors="coerce", utc=True)
    out = out.dropna(subset=["headline", "ticker", "timestamp"])
    out = out[out["headline"].str.len() >= 10].drop_duplicates(["ticker", "timestamp", "headline"])
    out["date"] = out["timestamp"].dt.tz_convert(timezone).dt.normalize().dt.tz_localize(None)
    return out.sort_values(["ticker", "timestamp"]).reset_index(drop=True)

def clean_prices(df):
    out = df.copy()
    out["ticker"] = out["ticker"].astype("string").str.upper().str.strip()
    out["date"] = pd.to_datetime(out["date"], errors="coerce", utc=True).dt.normalize().dt.tz_localize(None)
    for col in ["close", "volume"]: out[col] = pd.to_numeric(out[col], errors="coerce")
    return out.dropna(subset=["ticker", "date", "close"]).drop_duplicates(["ticker", "date"]).sort_values(["ticker", "date"])
