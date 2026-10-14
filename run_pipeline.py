from pathlib import Path
import time
import yaml
from src.data_loader import load_news, load_prices
from src.preprocessing import clean_news, clean_prices
from src.sentiment import add_sentiment
from src.novelty import add_novelty
from src.features import aggregate_news
from src.market_data import download_yfinance_prices, enrich_prices, merge_panel
from src.event_study import define_events, summarize_events
from src.statistics import correlations, panel_regression
from src.robustness import run_robustness


def main():
    started = time.time()
    cfg = yaml.safe_load(Path("config.yaml").read_text())
    Path("data/processed").mkdir(parents=True, exist_ok=True)
    Path("reports").mkdir(exist_ok=True)
    filt = cfg.get("input_filter", {})
    tickers = cfg["universe"]["tickers"]

    print("[1/7] Reading and filtering news in chunks...", flush=True)
    news = load_news(cfg["paths"]["news_file"], tickers, start=filt.get("start_date"), end=filt.get("end_date"), chunksize=filt.get("news_chunksize", 100000))
    print(f"      retained {len(news):,} source rows", flush=True)
    print("[2/7] Cleaning and scoring sentiment...", flush=True)
    news = add_sentiment(clean_news(news))
    print("[3/7] Computing bounded rolling novelty...", flush=True)
    news = add_novelty(news, cfg["features"]["repetition_similarity_threshold"], cfg["features"]["novelty_history_days"])
    print("[4/7] Aggregating daily information features...", flush=True)
    daily = aggregate_news(news)
    print("[5/7] Loading or downloading selective price history...", flush=True)
    price_path = Path(cfg["paths"]["prices_file"])
    if price_path.exists():
        prices = clean_prices(load_prices(price_path))
    else:
        prices = clean_prices(download_yfinance_prices(sorted(news.ticker.dropna().unique()), news.date.min(), news.date.max(), cfg["market"]["benchmark"]))
    panel = merge_panel(daily, enrich_prices(prices, cfg["market"]["benchmark"]))
    print(f"      aligned {len(panel):,} panel rows", flush=True)
    print("[6/7] Running event study and statistics...", flush=True)
    events = define_events(panel, cfg["events"]["quantile"])
    panel.to_csv("data/processed/panel.csv", index=False)
    events.to_csv("data/processed/events.csv", index=False)
    corr, reg, ev, rob = correlations(panel), panel_regression(panel), summarize_events(events), run_robustness(panel)
    corr.to_csv("reports/correlations.csv", index=False)
    reg.to_csv("reports/panel_regression.csv", index=False)
    ev.to_csv("reports/event_summary.csv", index=False)
    rob.to_csv("reports/robustness.csv", index=False)
    stats = {"rows": len(panel), "tickers": int(panel.ticker.nunique()), "news_records": len(news), "date_min": str(news.date.min()), "date_max": str(news.date.max())}
    html = "<html><body><h1>Behavioral Market Reaction Lab</h1><h2>Sample</h2><pre>" + str(stats) + "</pre><h2>Events</h2>" + ev.to_html(index=False) + "<h2>Correlations</h2>" + corr.to_html(index=False) + "</body></html>"
    Path("reports/research_report.html").write_text(html, encoding="utf-8")
    print(f"[7/7] Completed in {time.time() - started:.1f}s: {stats}", flush=True)


if __name__ == "__main__":
    main()
