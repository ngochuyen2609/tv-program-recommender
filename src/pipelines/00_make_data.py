# 00_make_data.py — Join logs↔program + Sessionize + Temporal Split
from __future__ import annotations
import argparse
from pathlib import Path

def run_join(in_logs: str, in_meta: list[str], out_parquet: str):
    from src.preprocessing.join_program import run as join_run
    join_run(in_logs, in_meta, out_parquet)

def run_session_split(in_parquet: str, out_train: str, out_val: str, gap_min: int):
    from src.preprocessing.sessionize import run as sess_run
    sess_run(in_parquet, out_train, out_val, gap_min=gap_min)

def build_parser():
    ap = argparse.ArgumentParser("00_make_data")
    ap.add_argument("--logs", required=True, help="Path to dataset11-30.csv")
    ap.add_argument("--meta", required=True, nargs="+",
                    help="Paths to export_arh_11-20-final.csv export_arh_21-30-final.csv")
    ap.add_argument("--interim-dir", default="data/interim", type=str)
    ap.add_argument("--gap-min", default=45, type=int)
    ap.add_argument("--skip-session", action="store_true")
    return ap

def main():
    args = build_parser().parse_args()
    interim = Path(args.interim_dir); interim.mkdir(parents=True, exist_ok=True)

    joined_pq = str(interim / "logs_program.parquet")
    run_join(args.logs, args.meta, joined_pq)

    if not args.skip_session:
        train_pq = str(interim / "train.parquet")
        val_pq   = str(interim / "val.parquet")
        run_session_split(joined_pq, train_pq, val_pq, gap_min=args.gap_min)
        print(f"[OK] train={train_pq}  val={val_pq}")
    print(f"[OK] joined={joined_pq}")

if __name__ == "__main__":
    main()
