import numpy as np
import pandas as pd
from .event_study import define_events, summarize_events

def winsorize(df, cols, lower=.01, upper=.99):
    out=df.copy()
    for c in cols:
        if c in out: out[c]=out[c].clip(out[c].quantile(lower), out[c].quantile(upper))
    return out

def run_robustness(panel):
    rows=[]
    for q in [.8,.9,.95]:
        e=define_events(panel,q); s=summarize_events(e); s["threshold"]=q; rows.append(s)
    return pd.concat(rows, ignore_index=True)
