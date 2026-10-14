from pathlib import Path
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title="Behavioral Market Reaction Lab", page_icon="◈", layout="wide", initial_sidebar_state="expanded")
ROOT = Path(__file__).resolve().parents[1]

# ---------- Theme ----------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Space+Grotesk:wght@500;600;700&display=swap');
:root { --navy:#0b1220; --panel:#111c2e; --panel2:#15243a; --ink:#e8eef7; --muted:#8ea1ba; --cyan:#37d6c0; --blue:#5b8cff; --gold:#f5c86b; --red:#ff7081; }
.stApp { background: radial-gradient(circle at 88% 0%, #172d52 0%, #0b1220 38%, #080e18 100%); color:var(--ink); font-family:'DM Sans',sans-serif; }
[data-testid="stSidebar"] { background:#0a1322; border-right:1px solid #22334d; }
[data-testid="stSidebar"] * { color:var(--ink); }
.block-container { max-width:1500px; padding:2rem 3rem 4rem; }
h1,h2,h3 { font-family:'Space Grotesk',sans-serif !important; letter-spacing:-.03em; }
h1 { font-size:2.55rem !important; margin-bottom:.15rem !important; }
h2 { font-size:1.35rem !important; margin-top:1.2rem !important; }
.hero { border:1px solid #284364; background:linear-gradient(135deg,rgba(20,42,70,.94),rgba(13,25,44,.78)); padding:1.8rem 2rem; border-radius:22px; box-shadow:0 20px 60px rgba(0,0,0,.22); margin-bottom:1.1rem; }
.eyebrow { color:var(--cyan); text-transform:uppercase; font-weight:700; letter-spacing:.14em; font-size:.72rem; }
.subtitle { color:var(--muted); font-size:1rem; margin-top:.35rem; }
.badge { display:inline-block; color:#b7c9df; background:#172a43; border:1px solid #2a4667; border-radius:999px; padding:.32rem .7rem; font-size:.74rem; margin:.55rem .3rem 0 0; }
.kpi { background:linear-gradient(145deg,#14243a,#0f1b2d); border:1px solid #263d5d; border-radius:16px; padding:1rem 1.1rem; min-height:116px; box-shadow:0 10px 30px rgba(0,0,0,.14); }
.kpi-label { color:var(--muted); font-size:.78rem; text-transform:uppercase; letter-spacing:.08em; }
.kpi-value { color:var(--ink); font-family:'Space Grotesk'; font-size:1.65rem; font-weight:700; margin-top:.35rem; }
.kpi-note { color:#6fd9c9; font-size:.75rem; margin-top:.3rem; }
.section-card { background:rgba(17,28,46,.72); border:1px solid #203753; border-radius:18px; padding:1rem 1.15rem; }
.insight { background:#10283a; border-left:4px solid var(--cyan); border-radius:8px; padding:.85rem 1rem; color:#c9d8e9; line-height:1.5; }
.warning { background:#352b18; border-left:4px solid var(--gold); border-radius:8px; padding:.85rem 1rem; color:#f1dfb1; }
.small { color:var(--muted); font-size:.82rem; }
div[data-testid="stMetric"] { background:#111f34; border:1px solid #294361; padding:1rem; border-radius:14px; }
div[data-testid="stMetricLabel"] { color:var(--muted); }
div[data-testid="stMetricValue"] { color:var(--ink); }
button[data-baseweb="tab"] { color:var(--muted); }
button[data-baseweb="tab"][aria-selected="true"] { color:var(--cyan); }
.stDataFrame { border:1px solid #263d5d; border-radius:12px; }
</style>
""", unsafe_allow_html=True)

# ---------- Data ----------
@st.cache_data(show_spinner=False)
def load_csv(path, dates=None):
    p = Path(path)
    if not p.is_absolute():
        p = ROOT / p
    if not p.exists():
        return pd.DataFrame()
    return pd.read_csv(p, parse_dates=dates or [])

panel = load_csv("data/processed/panel.csv", ["date"])
events = load_csv("data/processed/events.csv", ["date"])
event_summary = load_csv("reports/event_summary.csv")
correlations = load_csv("reports/correlations.csv")
regression = load_csv("reports/panel_regression.csv")

# ---------- Header ----------
st.markdown('<div class="hero"><div class="eyebrow">Quantitative research terminal · BRL-01</div><h1>Behavioral Market Reaction Lab</h1><div class="subtitle">How sentiment, novelty, repetition and information intensity relate to subsequent market behavior.</div><span class="badge">OBSERVATIONAL DESIGN</span><span class="badge">EVENT STUDY</span><span class="badge">NO CAUSAL CLAIMS</span></div>', unsafe_allow_html=True)

if panel.empty:
    st.markdown('<div class="warning"><b>No processed panel found.</b><br>Place the real Kaggle CSV in <code>data/raw/analyst_ratings_processed.csv</code>, then run <code>python run_pipeline.py</code>.</div>', unsafe_allow_html=True)
    st.stop()

# ---------- Sidebar controls ----------
st.sidebar.markdown("## Research controls")
st.sidebar.caption("Filter the processed research panel. These controls do not change the underlying source data.")
tickers = sorted(panel["ticker"].dropna().unique())
selected_tickers = st.sidebar.multiselect("Companies", tickers, default=tickers[:min(5, len(tickers))])
if not selected_tickers:
    selected_tickers = tickers
min_date, max_date = panel["date"].min(), panel["date"].max()
date_range = st.sidebar.date_input("Date range", value=(min_date.date(), max_date.date()), min_value=min_date.date(), max_value=max_date.date())
if isinstance(date_range, tuple) and len(date_range) == 2:
    start_date, end_date = pd.Timestamp(date_range[0]), pd.Timestamp(date_range[1])
else:
    start_date, end_date = min_date, max_date
feature_options = {"Sentiment score":"sentiment_score", "Novelty score":"novelty_score", "Repetition ratio":"repeat_ratio", "Information intensity":"information_intensity"}
outcome_options = {"1-day return":"return_1d", "5-day return":"return_5d", "20-day return":"return_20d", "20-day volatility":"volatility_20d"}
feature_label = st.sidebar.selectbox("Research feature", list(feature_options))
outcome_label = st.sidebar.selectbox("Market outcome", list(outcome_options), index=1)
feature, outcome = feature_options[feature_label], outcome_options[outcome_label]

filtered = panel[panel["ticker"].isin(selected_tickers) & panel["date"].between(start_date, end_date)].copy()
if filtered.empty:
    st.warning("No rows match the current filters.")
    st.stop()

# ---------- KPIs ----------
def kpi(label, value, note):
    return f'<div class="kpi"><div class="kpi-label">{label}</div><div class="kpi-value">{value}</div><div class="kpi-note">{note}</div></div>'

k1, k2, k3, k4, k5 = st.columns(5)
with k1: st.markdown(kpi("Observations", f"{len(filtered):,}", "filtered panel rows"), unsafe_allow_html=True)
with k2: st.markdown(kpi("Companies", f"{filtered.ticker.nunique():,}", "in selected universe"), unsafe_allow_html=True)
with k3: st.markdown(kpi("News records", f"{int(filtered.news_count.sum()):,}", "daily aggregated"), unsafe_allow_html=True)
with k4: st.markdown(kpi("Mean outcome", f"{filtered[outcome].mean():.3%}" if filtered[outcome].notna().any() else "—", outcome_label), unsafe_allow_html=True)
with k5: st.markdown(kpi("Median outcome", f"{filtered[outcome].median():.3%}" if filtered[outcome].notna().any() else "—", "robust central tendency"), unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ---------- Tabs ----------
tab_overview, tab_events, tab_company, tab_evidence, tab_data = st.tabs(["Overview", "Event studies", "Company lens", "Evidence", "Data explorer"])
plot_template = "plotly_dark"

with tab_overview:
    left, right = st.columns([1.35, 1])
    with left:
        st.subheader("Market response across feature intensity")
        bucketed = filtered[[feature, outcome]].dropna().copy()
        if len(bucketed) >= 10:
            # Rank first so repeated/constant feature values cannot collapse
            # qcut bin edges and trigger a labels/bin-count mismatch.
            bucketed["bucket"] = pd.qcut(
                bucketed[feature].rank(method="first"),
                5,
                labels=["Q1 · low", "Q2", "Q3", "Q4", "Q5 · high"],
            )
            summary = bucketed.groupby("bucket", observed=True)[outcome].agg(mean="mean", median="median", count="size").reset_index()
            fig = go.Figure()
            fig.add_trace(go.Bar(x=summary["bucket"], y=summary["mean"], name="Mean", marker_color="#37d6c0", hovertemplate="%{x}<br>Mean: %{y:.3%}<extra></extra>"))
            fig.add_trace(go.Scatter(x=summary["bucket"], y=summary["median"], name="Median", mode="lines+markers", line=dict(color="#f5c86b", width=3), hovertemplate="%{x}<br>Median: %{y:.3%}<extra></extra>"))
            fig.update_layout(template=plot_template, height=400, margin=dict(l=10,r=10,t=20,b=10), yaxis_tickformat=".2%", legend=dict(orientation="h"))
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("At least 10 observations are needed for quantile comparison.")
    with right:
        st.subheader("Daily information flow")
        daily = filtered.groupby("date", as_index=False).agg(news_count=("news_count", "sum"), outcome=(outcome, "mean"), intensity=("information_intensity", "mean"))
        fig = px.area(daily, x="date", y="news_count", color_discrete_sequence=["#5b8cff"], template=plot_template)
        fig.update_layout(height=400, margin=dict(l=10,r=10,t=20,b=10), yaxis_title="News count", xaxis_title=None)
        st.plotly_chart(fig, use_container_width=True)
    st.markdown('<div class="insight"><b>Reading the chart:</b> Quintile differences describe conditional association, not a trading signal. Review sample size, dispersion, overlapping events and the corrected statistical outputs before interpreting a pattern.</div>', unsafe_allow_html=True)

with tab_events:
    st.subheader("Extreme-information event monitor")
    if not event_summary.empty:
        show = event_summary.copy()
        rename = {"event":"Event family", "n":"Events", "mean_car":"Mean CAR", "median_car":"Median CAR", "threshold":"Threshold"}
        show = show.rename(columns={k:v for k,v in rename.items() if k in show.columns})
        st.dataframe(show, use_container_width=True, hide_index=True)
        if "Mean CAR" in show:
            fig = px.bar(show, x="Event family", y="Mean CAR", color="Mean CAR", color_continuous_scale=["#ff7081","#f5c86b","#37d6c0"], template=plot_template)
            fig.update_layout(height=360, margin=dict(l=10,r=10,t=20,b=80), yaxis_tickformat=".2%", coloraxis_showscale=False)
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Run the pipeline to generate event-summary results.")

with tab_company:
    st.subheader("Company-level reaction lens")
    company = st.selectbox("Company", selected_tickers, key="company_lens")
    company_df = filtered[filtered.ticker.eq(company)].sort_values("date")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=company_df.date, y=company_df[feature], name=feature_label, line=dict(color="#37d6c0", width=2)))
    fig.add_trace(go.Scatter(x=company_df.date, y=company_df[outcome], name=outcome_label, yaxis="y2", line=dict(color="#f5c86b", width=2)))
    fig.update_layout(template=plot_template, height=430, hovermode="x unified", margin=dict(l=10,r=10,t=20,b=10), yaxis=dict(title=feature_label), yaxis2=dict(title=outcome_label, overlaying="y", side="right"), legend=dict(orientation="h"))
    st.plotly_chart(fig, use_container_width=True)
    st.dataframe(company_df[[c for c in ["date","news_count",feature,outcome,"abnormal_return","volatility_20d"] if c in company_df]].tail(30), use_container_width=True, hide_index=True)

with tab_evidence:
    st.subheader("Statistical evidence")
    if not correlations.empty:
        st.markdown("#### Rank correlations")
        st.dataframe(correlations, use_container_width=True, hide_index=True)
    if not regression.empty:
        st.markdown("#### Fixed-effect OLS output")
        st.dataframe(regression.head(30), use_container_width=True, hide_index=True)
    st.markdown('<div class="warning"><b>Interpretation guardrail.</b> Corrected p-values, confidence intervals, effect size, and sample size must be considered together. These observational outputs do not establish that news features caused returns.</div>', unsafe_allow_html=True)

with tab_data:
    st.subheader("Filtered research panel")
    st.download_button("Download filtered panel CSV", filtered.to_csv(index=False).encode("utf-8"), "filtered_panel.csv", "text/csv")
    st.dataframe(filtered.sort_values("date", ascending=False).head(500), use_container_width=True, height=560, hide_index=True)

st.markdown("<hr><div class='small'>Behavioral Market Reaction Lab · Research prototype · Data source: Kaggle financial-news dataset and yfinance market history · No investment advice.</div>", unsafe_allow_html=True)
