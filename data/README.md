# Data contract and low-storage workflow

You only need to download Kaggle's processed `analyst_ratings_processed.csv` into `data/raw/`. Do not download the raw analyst-ratings file or raw partner-headlines file for this project.

The loader reads the processed CSV in bounded chunks, selects only headline/date/ticker columns, and filters to the ten configured liquid tickers before concatenating. Set `input_filter.start_date` and `input_filter.end_date` in `config.yaml` to restrict the analysis window.

The default pipeline downloads daily prices only for the configured ten tickers plus SPY through yfinance. A manually prepared `data/raw/prices.csv` can be used instead, with columns `ticker,date,open,high,low,close,adj_close,volume`. No synthetic observations are generated.
