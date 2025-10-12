import pandas as pd
from ..utils.io import to_parquet

def add_sessions(df: pd.DataFrame, user_col="user_id", ts_col="start_time_view", gap_min=45):
    df = df.sort_values([user_col, ts_col]).copy()
    gap = df.groupby(user_col)[ts_col].diff().dt.total_seconds().fillna(0)
    new_sess = (gap > gap_min*60).astype(int)
    df["session_id"] = new_sess.groupby(df[user_col]).cumsum()
    return df

def run(in_path, out_train, out_val, gap_min=45):
    import pandas as pd
    from ..utils.split import temporal_split
    df = pd.read_parquet(in_path)
    df = add_sessions(df, gap_min=gap_min)
    train, val = temporal_split(df, ts_col="start_time_view",
                                train_weeks=(11,24), val_weeks=(25,30))
    to_parquet(train, out_train)
    to_parquet(val, out_val)
    print(f"[SESSION] users={df.user_id.nunique()} sessions={df[['user_id','session_id']].drop_duplicates().shape[0]}")
