from collections import Counter, deque
from datetime import timedelta


def _tokens(text):
    return {w for w in str(text).lower().split() if len(w) > 2 and any(c.isalpha() for c in w)}


def _jaccard(a, b):
    union = len(a | b)
    return len(a & b) / union if union else 0.0


def add_novelty(df, threshold=0.85, history_days=30, recent_days=7, max_history=50):
    """Compute novelty against at most max_history recent same-ticker headlines.

    The bounded history prevents the quadratic runtime that made the previous
    pure-Python implementation impractical on the full Kaggle file.
    """
    out = df.copy().sort_values(["ticker", "timestamp"]).reset_index(drop=True)
    out["novelty_score"] = 1.0
    out["repeat_ratio"] = 0.0
    out["repeated_news_count"] = 0
    for _, group in out.groupby("ticker", sort=False):
        history = deque()
        for row in group.index:
            current_time = out.at[row, "timestamp"]
            while history and current_time - history[0][0] > timedelta(days=history_days):
                history.popleft()
            current_tokens = _tokens(out.at[row, "headline"])
            if history:
                similarities = [_jaccard(current_tokens, old_tokens) for _, old_tokens in history]
                repeated = sum(score >= threshold for score in similarities)
                out.at[row, "novelty_score"] = 1.0 - max(similarities)
                out.at[row, "repeated_news_count"] = repeated
                out.at[row, "repeat_ratio"] = repeated / len(similarities)
            history.append((current_time, current_tokens))
            while len(history) > max_history:
                history.popleft()
    return out
