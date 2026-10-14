# Data contract

Place only the real Kaggle `analyst_ratings_processed.csv` in `data/raw/`. It must provide a headline/article-title field, a date/timestamp field, and a stock/ticker field. The loader accepts the documented Kaggle spellings.

The pipeline reads the file in bounded chunks and keeps only the configured liquid-stock universe. Prices are downloaded selectively from yfinance unless a local `data/raw/prices.csv` exists.
