import math
import numpy as np
import pandas as pd

def _rank(x): return pd.Series(x).rank(method='average').to_numpy(float)
def _p(z): return math.erfc(abs(float(z))/math.sqrt(2))
def bootstrap_mean(x,iterations=1000,seed=42):
    x=pd.Series(x).dropna().to_numpy(float)
    if len(x)==0:return np.nan,np.nan,np.nan
    rng=np.random.default_rng(seed); vals=np.array([rng.choice(x,len(x),replace=True).mean() for _ in range(iterations)]); return float(x.mean()),float(np.quantile(vals,.025)),float(np.quantile(vals,.975))
def correlations(panel,features=None,outcomes=None):
    features=features or ['sentiment_score','novelty_score','repeat_ratio','information_intensity']; outcomes=outcomes or ['return_1d','return_5d','return_20d','volatility_20d']; rows=[]
    for a in features:
        for b in outcomes:
            z=panel[[a,b]].dropna()
            if len(z)<5: continue
            rows.append({'feature':a,'outcome':b,'rho':float(np.corrcoef(_rank(z[a]),_rank(z[b]))[0,1]),'p_value':np.nan,'p_value_adj':np.nan,'n':len(z)})
    return pd.DataFrame(rows)
def bucket_summary(panel,feature,outcome):
    x=panel[[feature,outcome]].dropna().copy(); x['bucket']=pd.qcut(x[feature],5,labels=False,duplicates='drop')+1; return x.groupby('bucket',observed=True)[outcome].agg(['count','mean','median','std']).reset_index()
def panel_regression(panel,outcome='return_5d'):
    cols=[outcome,'sentiment_score','novelty_score','repeat_ratio','information_intensity','ticker','date']; x=panel[cols].dropna().copy()
    if len(x)<30:return pd.DataFrame([{'status':'insufficient_observations','n':len(x)}])
    f=['sentiment_score','novelty_score','repeat_ratio','information_intensity']; d=pd.get_dummies(x[['ticker','date']].astype(str),drop_first=True,dtype=float).to_numpy(); X=np.column_stack([np.ones(len(x)),x[f].to_numpy(float),d]); y=x[outcome].to_numpy(float); beta=np.linalg.lstsq(X,y,rcond=None)[0]; r=y-X@beta; dof=max(len(y)-X.shape[1],1); cov=((r@r)/dof)*np.linalg.pinv(X.T@X); se=np.sqrt(np.maximum(np.diag(cov),0)); z=np.divide(beta,se,out=np.full_like(beta,np.nan),where=se>0); terms=['const']+f+[f'fixed_effect_{i}' for i in range(len(beta)-len(f)-1)]; return pd.DataFrame({'term':terms[:len(beta)],'coef':beta,'std_err':se,'z_value':z,'p_value':[_p(v) if np.isfinite(v) else np.nan for v in z]})
