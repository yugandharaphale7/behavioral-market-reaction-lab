from pathlib import Path
import yaml, pandas as pd
from src.data_loader import load_news, load_prices
from src.preprocessing import clean_news, clean_prices
from src.sentiment import add_sentiment
from src.novelty import add_novelty
from src.features import aggregate_news
from src.market_data import enrich_prices, merge_panel, download_yfinance_prices
from src.event_study import define_events, summarize_events
from src.statistics import correlations, panel_regression
from src.robustness import run_robustness

def main():
    cfg = yaml.safe_load(Path("config.yaml").read_text())
    Path("data/processed").mkdir(parents=True, exist_ok=True); Path("reports").mkdir(exist_ok=True)
    tickers = cfg["universe"]["tickers"]
    filt = cfg.get("input_filter", {})
    news = load_news(cfg["paths"]["news_file"], tickers=tickers, start=filt.get("start_date"), end=filt.get("end_date"), chunksize=filt.get("news_chunksize", 100000))
    news = add_sentiment(clean_news(news))
    news = add_novelty(news, cfg["features"]["repetition_similarity_threshold"], cfg["features"]["novelty_history_days"], cfg["features"]["novelty_recent_days"])
    daily = aggregate_news(news)
    price_path = Path(cfg["paths"]["prices_file"])
    if price_path.exists():
        prices = clean_prices(load_prices(price_path))
    elif cfg["market"].get("auto_download_yfinance", False):
        prices = download_yfinance_prices(sorted(news["ticker"].dropna().unique()), news["date"].min(), news["date"].max(), cfg["market"]["benchmark"])
        prices = clean_prices(prices)
    else:
        raise FileNotFoundError(f"{price_path} is missing and auto_download_yfinance is false.")
    prices = enrich_prices(prices, cfg["market"]["benchmark"])
    panel = merge_panel(daily, prices)
    events = define_events(panel, cfg["events"]["quantile"])
    panel.to_csv("data/processed/panel.csv", index=False); events.to_csv("data/processed/events.csv", index=False)
    corr = correlations(panel); reg = panel_regression(panel); ev = summarize_events(events); rob = run_robustness(panel)
    corr.to_csv("reports/correlations.csv", index=False); reg.to_csv("reports/panel_regression.csv", index=False); ev.to_csv("reports/event_summary.csv", index=False); rob.to_csv("reports/robustness.csv", index=False)
    stats = {"rows": len(panel), "tickers": panel.ticker.nunique(), "news_records": len(news), "date_min": str(news.date.min()), "date_max": str(news.date.max())}
    html = f"""<html><head><title>Behavioral Market Reaction Lab</title><style>body{{font-family:Arial;max-width:1100px;margin:40px auto;color:#172033}}table{{border-collapse:collapse}}td,th{{padding:8px;border:1px solid #ddd}}</style></head><body><h1>Behavioral Market Reaction Lab</h1><h2>Abstract</h2><p>This report evaluates observational relationships between financial-news features and subsequent market behavior. It does not establish causality.</p><h2>Dataset and sample</h2><pre>{stats}</pre><h2>Event-study summary</h2>{ev.to_html(index=False)}<h2>Correlations with FDR correction</h2>{corr.to_html(index=False)}<h2>Interpretation</h2><p>Findings should be considered statistically supported only when corrected p-values, confidence intervals, effect size, and sample size are jointly reviewed. This report intentionally avoids inventing conclusions when inputs are absent or evidence is weak.</p><h2>Limitations</h2><p>News timing, sentiment lexicon coverage, market-data alignment, overlapping events, survivorship and observational confounding limit interpretation. See README for full details.</p></body></html>"""
    Path("reports/research_report.html").write_text(html, encoding="utf-8")
    print(f"Completed: {stats}")

if __name__ == "__main__": main()
