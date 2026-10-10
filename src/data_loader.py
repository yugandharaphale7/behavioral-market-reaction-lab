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

def load_news(path, tickers=None, start=None, end=None, chunksize=100_000):
    """Read only required columns in chunks, filtering tickers before concatenation."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"News file not found: {path}. Download the processed Kaggle CSV first.")
    header = pd.read_csv(path, nrows=0)
    mapping = {k: _resolve(header, v) for k, v in NEWS_ALIASES.items()}
    missing = [k for k, v in mapping.items() if v is None]
    if missing:
        raise ValueError(f"News schema missing {missing}; observed columns: {list(header.columns)}")
    wanted = {str(t).upper().strip() for t in tickers} if tickers else None
    chunks = []
    usecols = [mapping["headline"], mapping["date"], mapping["ticker"]]
    for chunk in pd.read_csv(path, usecols=usecols, chunksize=chunksize, low_memory=False):
        chunk.columns = ["headline", "timestamp", "ticker"]
        chunk["ticker"] = chunk["ticker"].astype("string").str.upper().str.strip()
        if wanted is not None:
            chunk = chunk[chunk["ticker"].isin(wanted)]
        if start is not None or end is not None:
            dates = pd.to_datetime(chunk["timestamp"], errors="coerce", utc=True)
            if start is not None:
                chunk = chunk[dates >= pd.Timestamp(start, tz="UTC")]
            if end is not None:
                chunk = chunk[dates <= pd.Timestamp(end, tz="UTC")]
        if not chunk.empty:
            chunks.append(chunk)
    if not chunks:
        raise ValueError("No news records remained after ticker/date filtering.")
    return pd.concat(chunks, ignore_index=True)

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
