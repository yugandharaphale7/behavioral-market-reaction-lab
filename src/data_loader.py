from pathlib import Path
import pandas as pd

NEWS_ALIASES = {
    "headline": ["headline", "article title", "title"],
    "date": ["date", "publication timestamp", "timestamp"],
    "ticker": ["stock", "stock ticker symbol", "ticker"],
}

def _resolve(df, names):
    norm = {str(c).strip().lower(): c for c in df.columns}
    for name in names:
        if name.lower() in norm:
            return norm[name.lower()]
    return None

def load_news(path):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"News file not found: {path}. Download the real Kaggle file first.")
    df = pd.read_csv(path)
    mapping = {k: _resolve(df, v) for k, v in NEWS_ALIASES.items()}
    missing = [k for k, v in mapping.items() if v is None]
    if missing:
        raise ValueError(f"News schema missing {missing}; observed columns: {list(df.columns)}")
    out = df[[mapping["headline"], mapping["date"], mapping["ticker"]]].copy()
    out.columns = ["headline", "timestamp", "ticker"]
    return out

def load_prices(path):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Price file not found: {path}. Supply OHLCV data or enable yfinance.")
    df = pd.read_csv(path)
    cols = {str(c).lower().replace(" ", "_"): c for c in df.columns}
    def pick(*names):
        for n in names:
            if n in cols: return cols[n]
        return None
    required = {"ticker": pick("ticker", "symbol"), "date": pick("date", "datetime"), "close": pick("adj_close", "close", "adjclose"), "volume": pick("volume")}
    missing = [k for k, v in required.items() if v is None]
    if missing: raise ValueError(f"Price schema missing {missing}; need ticker,date,close,volume")
    out = df[[v for v in required.values() if v is not None]].copy()
    out.columns = [k for k, v in required.items() if v is not None]
    return out
