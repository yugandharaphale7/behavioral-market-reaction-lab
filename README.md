# Behavioral Market Reaction Lab

A reproducible observational event-study project investigating whether financial-news **sentiment, novelty, repetition, and information intensity** are associated with subsequent returns, abnormal returns, volatility, and recovery behavior. This is **not** a stock-price prediction system and does not claim causality.

## Dataset selected

The primary dataset is Kaggle's [Daily Financial News for 6000+ Stocks](https://www.kaggle.com/datasets/miguelaenlle/massive-stock-news-analysis-db-for-nlpbacktests), selected because it provides dated headlines and ticker identifiers at useful scale. **You only need the processed `analyst_ratings_processed.csv` file**, not the raw analyst-ratings file or the raw partner-headlines file. The processed file is documented as approximately 843,062 records and 6,193 tickers, with minute-level timestamps in UTC-4. The pipeline reads it in chunks and immediately filters to the ten-ticker universe in `config.yaml`, so the full dataset is never held in memory.

Sentiment is not supplied by this dataset; the project derives an interpretable lexicon score rather than fabricating labels. Prices are downloaded only for the configured tickers plus SPY through yfinance when `market.auto_download_yfinance: true`.

## Low-storage quick start

1. Install Python 3.10+ and create an environment: `python -m venv .venv` then activate it.
2. Install dependencies: `pip install -r requirements.txt`.
3. Download **only** Kaggle's `analyst_ratings_processed.csv` into `data/raw/analyst_ratings_processed.csv`. Do not download the other large files.
4. Review `config.yaml`. The default universe is ten liquid tickers: AAPL, MSFT, AMZN, GOOGL, META, NVDA, JPM, XOM, JNJ, and WMT. Set `input_filter.start_date` and `input_filter.end_date` if you want a smaller date window.
5. Run the pipeline: `python run_pipeline.py`. It will download only the selected ticker histories plus SPY from yfinance; no `prices.csv` is required.
6. Launch the dashboard: `streamlit run dashboard/app.py`.
7. Inspect `reports/research_report.html` and `data/processed/panel.csv`.

If you prefer manual price data, place `data/raw/prices.csv` with `ticker,date,open,high,low,close,adj_close,volume`; it takes precedence over yfinance. The pipeline fails loudly when real inputs are missing and never creates synthetic analysis data.

## Reproducible stages

`src/data_loader.py` validates the source schema, reads news in bounded chunks, and filters tickers/date ranges; `preprocessing.py` removes duplicates and invalid records; `sentiment.py` computes a transparent finance lexicon score; `novelty.py` computes TF-IDF cosine similarity against prior same-ticker headlines using only historical text; `features.py` aggregates news daily; `market_data.py` downloads only the configured price universe and computes log returns, forward returns, volatility and benchmark abnormal returns; `event_study.py` defines events from information available on the event day and estimates CAR/recovery; `statistics.py` runs correlations, bootstrap intervals, quantile comparisons, panel regressions with stock/time fixed effects and clustered errors; `robustness.py` runs threshold/window/placebo/winsorization checks; and `visualization.py` writes figures.

## Hypotheses

H1 overreaction, H2 underreaction/drift, H3 attention effect, H4 novelty effect, and H5 sentiment persistence. Results are reported as supported, not supported, or inconclusive using effect sizes, confidence intervals, corrected p-values, and sample sizes. Statistical association is distinguished from economic significance and causal inference.

## Data limitations

The Kaggle source contains news and tickers but no built-in credible sentiment field, and some raw files lack exact publication times. News-to-close timing may therefore be imperfect. yfinance availability, ticker survivorship, corporate actions, delistings, market holidays, and overlapping events can affect alignment. Novelty is not meaningful for very short histories. The analysis is observational and should not be interpreted as a trading strategy or causal estimate.

## Dataset provenance

Kaggle page: https://www.kaggle.com/datasets/miguelaenlle/massive-stock-news-analysis-db-for-nlpbacktests

The project was designed after inspecting the public Kaggle description on 2026-10-08. The report records source-level fields and will fill exact post-filter counts only after the user supplies the processed download and runs the pipeline.
