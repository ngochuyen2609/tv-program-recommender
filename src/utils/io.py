import pandas as pd, numpy as np
from pathlib import Path

DATA = Path("data")

logs = pd.read_csv(DATA/"dataset11-30.csv")
# Chuẩn hoá tên cột (bộ channel hay có 'stop_time' và 'duraton')
logs = logs.rename(columns={"stop_time":"end_time", "duraton":"duration"})
logs["start_time"] = pd.to_datetime(logs["start_time"])
logs["end_time"]   = pd.to_datetime(logs["end_time"])
# Lọc nhiễu: giữ lượt xem >= 3 phút
logs = logs[logs["duration"] >= 180]
