import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def add_novelty(df, threshold=0.85, history_days=30, recent_days=7):
    out = df.copy().sort_values(["ticker", "timestamp"]).reset_index(drop=True)
    out["novelty_score"] = np.nan; out["repeat_ratio"] = np.nan; out["repeated_news_count"] = 0
    for ticker, idx in out.groupby("ticker", sort=False).groups.items():
        rows = list(idx)
        texts = out.loc[rows, "headline"].astype(str).tolist()
        vec = TfidfVectorizer(stop_words="english", ngram_range=(1,2), min_df=1)
        mat = vec.fit_transform(texts)
        for pos, row in enumerate(rows):
            current = out.at[row, "timestamp"]
            hist = [j for j in range(pos) if (current - out.at[rows[j], "timestamp"]).days <= history_days]
            recent = [j for j in hist if (current - out.at[rows[j], "timestamp"]).days <= recent_days]
            if not hist: continue
            sims = cosine_similarity(mat[pos], mat[rows[j] for j in hist]).ravel()
            out.at[row, "novelty_score"] = float(1 - np.max(sims))
            out.at[row, "repeated_news_count"] = int(sum(s >= threshold for s in sims))
            out.at[row, "repeat_ratio"] = float(np.mean(sims >= threshold))
    return out
