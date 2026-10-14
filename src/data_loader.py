from pathlib import Path
import pandas as pd

ALIASES = {'headline':['headline','article title','title'], 'date':['date','publication timestamp','timestamp'], 'ticker':['stock','stock ticker symbol','ticker']}

def _resolve(columns, names):
    norm={str(c).strip().lower():c for c in columns}
    return next((norm[n] for n in names if n.lower() in norm), None)

def load_news(path, tickers=None, start=None, end=None, chunksize=100_000):
    path=Path(path)
    if not path.exists(): raise FileNotFoundError(f'News file not found: {path}')
    header=pd.read_csv(path,nrows=0); mapping={k:_resolve(header.columns,v) for k,v in ALIASES.items()}
    missing=[k for k,v in mapping.items() if v is None]
    if missing: raise ValueError(f'Missing news fields {missing}; observed columns: {list(header.columns)}')
    wanted={str(t).upper().strip() for t in tickers} if tickers else None; parts=[]
    for part in pd.read_csv(path,usecols=list(mapping.values()),chunksize=chunksize,low_memory=False):
        part.columns=['headline','timestamp','ticker']; part['ticker']=part['ticker'].astype('string').str.upper().str.strip()
        if wanted is not None: part=part[part['ticker'].isin(wanted)]
        if start is not None or end is not None:
            dates=pd.to_datetime(part['timestamp'],errors='coerce',utc=True)
            if start is not None: part=part[dates>=pd.Timestamp(start,tz='UTC')]
            if end is not None: part=part[dates<=pd.Timestamp(end,tz='UTC')]
        if not part.empty: parts.append(part)
    if not parts: raise ValueError('No news records remained after filtering.')
    return pd.concat(parts,ignore_index=True)

def load_prices(path):
    path=Path(path)
    if not path.exists(): raise FileNotFoundError(f'Price file not found: {path}')
    df=pd.read_csv(path); cols={str(c).lower().replace(' ','_'):c for c in df.columns}
    def pick(*names): return next((cols[n] for n in names if n in cols),None)
    mapping={'ticker':pick('ticker','symbol'),'date':pick('date','datetime'),'close':pick('adj_close','close','adjclose'),'volume':pick('volume')}
    missing=[k for k,v in mapping.items() if v is None]
    if missing: raise ValueError(f'Missing price fields {missing}')
    out=df[list(mapping.values())].copy(); out.columns=list(mapping); return out
