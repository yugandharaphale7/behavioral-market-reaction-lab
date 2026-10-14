# Behavioral Market Reaction Lab

A reproducible observational event-study project investigating whether financial-news sentiment, novelty, repetition, and information intensity are associated with subsequent returns, abnormal returns, volatility, and recovery behavior. It is not a stock-price prediction system and does not claim causality.

## Windows-friendly build

This package intentionally does **not** depend on SciPy, scikit-learn, or statsmodels. Those compiled dependencies caused Windows Application Control DLL errors on the target machine. Novelty uses a pure-Python bounded rolling-history text similarity implementation; correlations and fixed-effect OLS use NumPy/pandas. The pipeline prints seven progress stages and does not silently appear frozen.

## Setup

```powershell
py -3.14 -m venv .venv
.\\.venv\\Scripts\\Activate.ps1
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements.txt
```

If Python is already available as `python`, use `python -m venv .venv` instead.

## Data

Download only the processed Kaggle file `analyst_ratings_processed.csv` from:
https://www.kaggle.com/datasets/miguelaenlle/massive-stock-news-analysis-db-for-nlpbacktests

Place it at `data/raw/analyst_ratings_processed.csv`. The loader reads it in chunks and filters to the ten configured tickers before concatenating. The raw analyst and partner-headline files are not needed.

No manual price CSV is required. With `market.auto_download_yfinance: true`, the pipeline downloads only the selected ten tickers plus SPY. A manual `data/raw/prices.csv` with `ticker,date,close,volume` may be used instead.

## Run

```powershell
python -m compileall src run_pipeline.py
python run_pipeline.py
streamlit run dashboard/app.py
```

Outputs: `data/processed/panel.csv`, `data/processed/events.csv`, `reports/correlations.csv`, `reports/event_summary.csv`, `reports/panel_regression.csv`, `reports/robustness.csv`, and `reports/research_report.html`.

The project never creates synthetic research observations and fails when real inputs are absent. Statistical association is not causal inference. Do not publish raw datasets or secrets.
