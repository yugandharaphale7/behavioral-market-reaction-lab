# Data contract

Place the downloaded Kaggle file `analyst_ratings_processed.csv` in `data/raw/`.
It must contain the documented fields `headline`, `date`, and `stock` (the exact
Kaggle file has `article title`, `date`, and `stock` in its processed variant;
the loader accepts both spellings). No sample or synthetic observations are
included in this repository.

Place a compatible daily OHLCV file in `data/raw/prices.csv`, or set
`market.auto_download_yfinance: true` to download prices for the configured
liquid-stock universe plus SPY. Downloaded data are not committed.
