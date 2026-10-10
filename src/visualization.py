import plotly.express as px

def bucket_figure(summary, x="bucket", y="mean", title="Bucket comparison"):
    return px.bar(summary, x=x, y=y, error_y="std" if "std" in summary else None, title=title, template="plotly_white")

def timeline_figure(df, x="date", y="return_1d", color=None, title="Market reaction"):
    return px.line(df, x=x, y=y, color=color, title=title, template="plotly_white")
