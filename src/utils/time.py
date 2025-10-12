import numpy as np
import pandas as pd

def to_dt(df, cols):
    for c in cols:
        df[c] = pd.to_datetime(df[c])
    return df

def time_decay(ts: pd.Series, now=None, alpha_days=14.0):
    now = pd.Timestamp(now) if now else ts.max()
    dt = (now - ts).dt.total_seconds() / (3600*24)
    return np.exp(-dt / alpha_days)

def week_number(ts: pd.Series) -> pd.Series:
    return ts.dt.isocalendar().week.astype(int)
