import numpy as np

def aggregate_news(df):
    out = df.copy()
    daily = out.groupby(["ticker", "date"], as_index=False).agg(
        news_count=("headline", "size"), sentiment_score=("sentiment_score", "mean"),
        sentiment_intensity=("sentiment_intensity", "mean"), novelty_score=("novelty_score", "mean"),
        repeat_ratio=("repeat_ratio", "mean"), repeated_news_count=("repeated_news_count", "sum"))
    daily["log_news_count"] = np.log1p(daily["news_count"])
    daily["information_intensity"] = daily["log_news_count"] * (1 + daily["sentiment_intensity"].fillna(0))
    for c in ["novelty_score", "repeat_ratio"]: daily[c] = daily[c].fillna(0.0)
    return daily
