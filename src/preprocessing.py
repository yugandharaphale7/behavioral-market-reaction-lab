import pandas as pd

def clean_news(df):
    out=df.copy(); out['headline']=out['headline'].astype('string').str.strip(); out['ticker']=out['ticker'].astype('string').str.upper().str.strip(); out['timestamp']=pd.to_datetime(out['timestamp'],errors='coerce',utc=True)
    out=out.dropna(subset=['headline','ticker','timestamp']); out=out[out['headline'].str.len()>=10].drop_duplicates(['ticker','timestamp','headline']); out['date']=out['timestamp'].dt.tz_convert('UTC').dt.normalize().dt.tz_localize(None); return out.sort_values(['ticker','timestamp']).reset_index(drop=True)

def clean_prices(df):
    out=df.copy(); out['ticker']=out['ticker'].astype('string').str.upper().str.strip(); out['date']=pd.to_datetime(out['date'],errors='coerce',utc=True).dt.normalize().dt.tz_localize(None); out['close']=pd.to_numeric(out['close'],errors='coerce'); out['volume']=pd.to_numeric(out['volume'],errors='coerce'); return out.dropna(subset=['ticker','date','close']).drop_duplicates(['ticker','date']).sort_values(['ticker','date'])
