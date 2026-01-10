# src/preprocessing/join_program.py
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from .load_raw import load_logs, load_meta
from ..utils.io import to_parquet

def _pick_overlap(logs_ch: pd.DataFrame, meta_ch: pd.DataFrame) -> pd.DataFrame:
    if logs_ch.empty or meta_ch.empty:
        return pd.DataFrame(columns=["user_id","tv_show_id","vsetv_id","start_time_view","end_time_view","duration_view"])

    # sort đúng key
    meta_ch = meta_ch.sort_values("start_time").reset_index(drop=True)

    # BỎ vsetv_id ở meta_ch để tránh _x/_y khi merge
    meta_ch = meta_ch.drop(columns=["vsetv_id"], errors="ignore")

    # gắn id cho từng lượt xem
    logs_ch = logs_ch.copy()
    logs_ch["view_id"] = np.arange(len(logs_ch), dtype=np.int64)

    # 1) ứng viên L theo start_time_view
    logs_L = logs_ch.sort_values("start_time_view").reset_index(drop=True)
    L = pd.merge_asof(
        logs_L, meta_ch,
        left_on="start_time_view", right_on="start_time",
        direction="backward", allow_exact_matches=True
    )
    L["cand"] = "L"

    # 2) ứng viên R theo end_time_view
    logs_R = logs_ch.sort_values("end_time_view").reset_index(drop=True)
    R = pd.merge_asof(
        logs_R, meta_ch,
        left_on="end_time_view", right_on="start_time",
        direction="backward", allow_exact_matches=True
    )
    R["cand"] = "R"

    keep_log  = ["user_id","vsetv_id","start_time_view","end_time_view","duration_view","view_id"]
    keep_meta = ["tv_show_id","start_time","prog_end"]

    cand = pd.concat([
        L[keep_log + keep_meta + ["cand"]],
        R[keep_log + keep_meta + ["cand"]],
    ], ignore_index=True).dropna(subset=["tv_show_id"])

    if cand.empty:
        return pd.DataFrame(columns=["user_id","tv_show_id","vsetv_id","start_time_view","end_time_view","duration_view"])

    # overlap > 0
    s = np.maximum(
        cand["start_time_view"].values.astype("datetime64[s]"),
        cand["start_time"].values.astype("datetime64[s]")
    )
    e = np.minimum(
        cand["end_time_view"].values.astype("datetime64[s]"),
        cand["prog_end"].values.astype("datetime64[s]")
    )
    cand["overlap_s"] = (e - s).astype("timedelta64[s]").astype(int)
    cand = cand[cand["overlap_s"] > 0]
    if cand.empty:
        return pd.DataFrame(columns=["user_id","tv_show_id","vsetv_id","start_time_view","end_time_view","duration_view"])

    # chọn overlap lớn nhất theo view_id
    cand.sort_values(["view_id","overlap_s"], ascending=[True, False], inplace=True)
    cand = cand.drop_duplicates(subset=["view_id"], keep="first")

    return cand[["user_id","tv_show_id","vsetv_id","start_time_view","end_time_view","duration_view"]].reset_index(drop=True)


def interval_join_logs_with_program(logs: pd.DataFrame, meta: pd.DataFrame) -> pd.DataFrame:
    out = []
    # xử lý theo từng kênh để RAM thấp
    for vid, logs_ch in logs.groupby("vsetv_id", sort=False):
        meta_ch = meta[meta["vsetv_id"] == vid]
        matched = _pick_overlap(logs_ch, meta_ch)
        if not matched.empty:
            out.append(matched)
    if not out:
        return pd.DataFrame(columns=["user_id","tv_show_id","vsetv_id","start_time_view","end_time_view","duration_view"])
    return pd.concat(out, ignore_index=True)

def run(in_logs: str, in_meta_list: list[str], out_path: str):
    logs = load_logs(in_logs)      # đã có *_view cột
    meta = load_meta(in_meta_list) # có start_time, prog_end
    joined = interval_join_logs_with_program(logs, meta)
    to_parquet(joined, out_path)
    coverage = len(joined) / len(logs) if len(logs) else 0.0
    print(f"[JOIN] coverage={coverage:.3f} (expect > 0.70) -> {out_path}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-logs", required=True)
    ap.add_argument("--in-meta", required=True, nargs="+")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    run(args.in_logs, args.in_meta, args.out)
