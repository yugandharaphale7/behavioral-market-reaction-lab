import re

POSITIVE = {"beat","beats","strong","growth","gain","gains","profit","profits","upgrade","upgraded","surge","surges","record","bullish","positive","success","improve","improved","outperform"}
NEGATIVE = {"miss","misses","weak","loss","losses","downgrade","downgraded","fall","falls","drop","drops","decline","declines","bearish","negative","fraud","lawsuit","cut","cuts","layoff","layoffs"}

def score_text(text):
    tokens = re.findall(r"[a-z]+", str(text).lower())
    p, n = sum(t in POSITIVE for t in tokens), sum(t in NEGATIVE for t in tokens)
    total = max(len(tokens), 1)
    score = (p - n) / max(p + n, 1)
    return score, abs(score), "positive" if score > 0 else "negative" if score < 0 else "neutral"

def add_sentiment(df):
    vals = df["headline"].map(score_text)
    out = df.copy()
    out[["sentiment_score", "sentiment_intensity", "sentiment_direction"]] = vals.tolist()
    return out
