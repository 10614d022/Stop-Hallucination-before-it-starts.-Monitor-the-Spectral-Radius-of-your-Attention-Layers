import pandas as pd


def summarize(df: pd.DataFrame):
    summary = df.groupby("group").agg({
        "collapsed": "mean",
        "lead_time": "mean",
    }).rename(columns={"collapsed": "collapse_rate"})
    print("\n=== SUMMARY ===")
    print(summary)
    return summary
