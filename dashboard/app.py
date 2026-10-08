import streamlit as st
import pandas as pd
from pathlib import Path

st.set_page_config(page_title="Behavioral Market Reaction Lab", page_icon="BR", layout="wide")
st.markdown("""<style>.stApp{background:#f5f7fb}.block-container{max-width:1400px;padding-top:2rem}.metric-card{background:white;border:1px solid #dfe5ef;border-radius:10px;padding:14px}.small{color:#64748b;font-size:.85rem}</style>""", unsafe_allow_html=True)

@st.cache_data
def load():
    p=Path("data/processed/panel.csv")
    if not p.exists(): return pd.DataFrame()
    x=pd.read_csv(p, parse_dates=["date"]); return x

df=load()
st.title("Behavioral Market Reaction Lab")
st.caption("Observational event-study terminal · statistical association is not causal inference")
if df.empty:
    st.warning("No processed panel found. Add the real Kaggle news file and aligned prices, then run `python run_pipeline.py`.")
    st.stop()

c1,c2,c3,c4=st.columns(4)
c1.metric("Panel observations",f"{len(df):,}"); c2.metric("Stocks",df.ticker.nunique()); c3.metric("News dates",df.date.nunique()); c4.metric("Date range",f"{df.date.min():%Y-%m-%d} → {df.date.max():%Y-%m-%d}")
with st.sidebar:
    st.header("Company explorer")
    ticker=st.selectbox("Ticker",sorted(df.ticker.unique()))
    feature=st.selectbox("Feature",["sentiment_score","novelty_score","repeat_ratio","information_intensity"])
    horizon=st.selectbox("Outcome",["return_1d","return_5d","return_20d","volatility_20d"])

st.subheader("Market reaction")
left,right=st.columns(2)
with left:
    x=df.groupby(pd.qcut(df[feature],5,labels=["Q1","Q2","Q3","Q4","Q5"],duplicates="drop"),observed=True)[horizon].agg(["count","mean","median"]).reset_index(); x.columns=["bucket","count","mean","median"]
    st.bar_chart(x.set_index("bucket")["mean"],height=300)
with right:
    st.line_chart(df[df.ticker.eq(ticker)].set_index("date")[[feature,horizon]].dropna(),height=300)

st.subheader("Company timeline")
st.dataframe(df[df.ticker.eq(ticker)].sort_values("date",ascending=False),use_container_width=True,hide_index=True)
st.caption("Event thresholds and inference tables are written to reports/ by the pipeline. Weak or uncorrected evidence must not be presented as a behavioral conclusion.")
