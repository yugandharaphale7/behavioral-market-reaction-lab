import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests

def bootstrap_mean(x, iterations=1000, seed=42):
    x = pd.Series(x).dropna().to_numpy()
    if len(x) == 0: return (np.nan, np.nan, np.nan)
    rng = np.random.default_rng(seed); means = np.array([rng.choice(x, len(x), replace=True).mean() for _ in range(iterations)])
    return float(x.mean()), float(np.quantile(means,.025)), float(np.quantile(means,.975))

def correlations(panel, features=None, outcomes=None):
    features = features or ["sentiment_score","novelty_score","repeat_ratio","information_intensity"]
    outcomes = outcomes or ["return_1d","return_5d","return_20d","volatility_20d"]
    rows=[]
    for a in features:
        for b in outcomes:
            z=panel[[a,b]].dropna()
            if len(z)<5: continue
            r,p=stats.spearmanr(z[a],z[b]); rows.append({"feature":a,"outcome":b,"rho":r,"p_value":p,"n":len(z)})
    out=pd.DataFrame(rows)
    if not out.empty: out["p_value_adj"]=multipletests(out.p_value, method="fdr_bh")[1]
    return out

def bucket_summary(panel, feature, outcome):
    x=panel[[feature,outcome]].dropna().copy(); x["bucket"]=pd.qcut(x[feature], 5, labels=False, duplicates="drop")+1
    return x.groupby("bucket")[outcome].agg(["count","mean","median","std"]).reset_index()

def panel_regression(panel, outcome="return_5d"):
    # Fixed effects via one-hot stock/date dummies; cluster by ticker when statsmodels is available.
    import statsmodels.formula.api as smf
    cols=[outcome,"sentiment_score","novelty_score","repeat_ratio","information_intensity","ticker","date"]
    x=panel[cols].dropna().copy(); x["date"]=x["date"].astype(str)
    if len(x)<30: return pd.DataFrame([{"status":"insufficient_observations","n":len(x)}])
    model=smf.ols(f"{outcome} ~ sentiment_score + novelty_score + repeat_ratio + information_intensity + C(ticker) + C(date)", data=x).fit(cov_type="cluster", cov_kwds={"groups":x["ticker"]})
    return pd.DataFrame({"term":model.params.index,"coef":model.params.values,"p_value":model.pvalues.values})
