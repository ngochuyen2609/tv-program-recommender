from __future__ import annotations
from pathlib import Path
import pandas as pd
import numpy as np
from ..utils.io import read_csv

# Logs
def load_logs(path: str | Path, min_duration_sec: int = 180) -> pd.DataFrame:
    """
    Đọc file logs (dataset11-30.csv), chuẩn hoá cột và lọc lượt xem ngắn.
    - Đổi tên: stop_time->end_time, duraton->duration
    - Parse datetime cho start_time, end_time
    - Lọc duration >= min_duration_sec
    """
    df = read_csv(path)
    df = df.rename(columns={"stop_time": "end_time", "duraton": "duration"}).copy()

    # Bắt buộc có các cột tối thiểu
    required = {"user_id", "vsetv_id", "start_time", "end_time", "duration"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in logs: {missing}")

    # Parse datetime cho logs (thường đã ở dạng ISO)
    df["start_time"] = pd.to_datetime(df["start_time"], errors="raise", utc=False)
    df["end_time"]   = pd.to_datetime(df["end_time"],   errors="raise", utc=False)

    # Duration numeric & filter
    df["duration"] = pd.to_numeric(df["duration"], errors="coerce").fillna(0).astype(int)
    df = df[df["duration"] >= int(min_duration_sec)].reset_index(drop=True)

    # Đổi tên cột nhất quán downstream
    df = df.rename(columns={
        "start_time": "start_time_view",
        "end_time":   "end_time_view",
        "duration":   "duration_view",
    })
    return df[["user_id", "vsetv_id", "start_time_view", "end_time_view", "duration_view"]]


# Metadata (export_arh_*)
def _parse_meta_start(series: pd.Series) -> pd.Series:
    """
    Parse start_time của metadata. Ưu tiên định dạng 'dd.mm.YYYY HH:MM:SS'.
    Fallback sang dayfirst=True nếu không khớp.
    """
    try:
        return pd.to_datetime(series, format="%d.%m.%Y %H:%M:%S", errors="raise")
    except Exception:
        return pd.to_datetime(series, dayfirst=True, errors="raise")

def load_meta(paths: list[str | Path]) -> pd.DataFrame:
    """
    Đọc và gộp các file export_arh_11-20-final.csv, export_arh_21-30-final.csv.
    - Đổi channel_id -> vsetv_id
    - Parse start_time (định dạng dd.mm.YYYY HH:MM:SS)
    - Tính prog_end = start_time + duration(s)
    - Trả về các cột cần cho interval join
    """
    if not paths:
        raise ValueError("`paths` for metadata is empty.")

    dfs = [read_csv(p) for p in paths]
    meta = pd.concat(dfs, ignore_index=True).copy()

    # Đổi tên cột & kiểm tra
    meta = meta.rename(columns={"channel_id": "vsetv_id"})
    required = {"vsetv_id", "start_time", "duration", "tv_show_id"}
    missing = required - set(meta.columns)
    if missing:
        raise ValueError(f"Missing columns in meta: {missing}")

    # Parse time + duration
    meta["start_time"] = _parse_meta_start(meta["start_time"])
    meta["duration"] = pd.to_numeric(meta["duration"], errors="coerce").fillna(0).astype(int)
    meta["prog_end"] = meta["start_time"] + pd.to_timedelta(meta["duration"], unit="s")

    # Cột text có thể vắng -> fill
    for c in ["tv_show_title", "tv_show_category", "tv_show_genre_1", "tv_show_genre_2", "tv_show_genre_3"]:
        if c not in meta.columns:
            meta[c] = ""

    cols = [
        "vsetv_id", "start_time", "prog_end", "tv_show_id",
        "tv_show_title", "tv_show_category",
        "tv_show_genre_1", "tv_show_genre_2", "tv_show_genre_3",
    ]
    return meta[cols]
