import re
POS={'beat','beats','strong','growth','gain','gains','profit','profits','upgrade','upgraded','surge','surges','record','bullish','positive','success','improve','improved','outperform'}
NEG={'miss','misses','weak','loss','losses','downgrade','downgraded','fall','falls','drop','drops','decline','declines','bearish','negative','fraud','lawsuit','cut','cuts','layoff','layoffs'}
def score_text(text):
    tokens=re.findall(r'[a-z]+',str(text).lower()); p=sum(t in POS for t in tokens); n=sum(t in NEG for t in tokens); score=(p-n)/max(p+n,1); return score,abs(score),'positive' if score>0 else 'negative' if score<0 else 'neutral'
def add_sentiment(df):
    out=df.copy(); vals=out['headline'].map(score_text); out[['sentiment_score','sentiment_intensity','sentiment_direction']]=vals.tolist(); return out
