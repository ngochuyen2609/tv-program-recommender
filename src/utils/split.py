import pandas as pd
from .time import week_number

def temporal_split(df: pd.DataFrame, ts_col: str, train_weeks=(11,24), val_weeks=(25,30)):
    wk = week_number(df[ts_col])
    train = df[(wk>=train_weeks[0]) & (wk<=train_weeks[1])].copy()
    val   = df[(wk>=val_weeks[0])   & (wk<=val_weeks[1])].copy()
    return train, val
