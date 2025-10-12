from pathlib import Path
import pandas as pd

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)
    return p

def read_csv(path: str | Path, **kw) -> pd.DataFrame:
    return pd.read_csv(path, **kw)

def to_parquet(df, path: str | Path):
    path = Path(path)
    ensure_dir(path.parent)
    df.to_parquet(path, index=False)
    return str(path)
