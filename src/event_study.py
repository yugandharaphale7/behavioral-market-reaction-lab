import numpy as np
import pandas as pd

def define_events(panel,quantile=.9):
    out=panel.copy(); out['sentiment_event']=out.sentiment_score.abs()>=out.sentiment_score.abs().quantile(quantile); out['novelty_event']=out.novelty_score>=out.novelty_score.quantile(quantile); out['repetition_event']=out.repeat_ratio>=out.repeat_ratio.quantile(quantile); out['information_event']=out.information_intensity>=out.information_intensity.quantile(quantile); return out

def event_summary(panel,event_col,horizon=5):
    x=panel.loc[panel[event_col].fillna(False)].copy()
    if x.empty:return {'event':event_col,'n':0,'mean_car':np.nan,'median_car':np.nan,'recovery_days':np.nan}
    car=x.groupby('ticker')['abnormal_return'].rolling(horizon,min_periods=1).sum().reset_index(level=0,drop=True); return {'event':event_col,'n':len(x),'mean_car':float(car.mean()),'median_car':float(car.median()),'recovery_days':np.nan}

def summarize_events(panel): return pd.DataFrame([event_summary(panel,c) for c in ['sentiment_event','novelty_event','repetition_event','information_event']])
