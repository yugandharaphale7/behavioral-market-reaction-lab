import pandas as pd
from .event_study import define_events,summarize_events

def winsorize(df,cols,lower=.01,upper=.99):
    out=df.copy()
    for c in cols:
        if c in out: out[c]=out[c].clip(out[c].quantile(lower),out[c].quantile(upper))
    return out

def run_robustness(panel):
    rows=[]
    for q in [.8,.9,.95]:
        s=summarize_events(define_events(panel,q)); s['threshold']=q; rows.append(s)
    return pd.concat(rows,ignore_index=True)
