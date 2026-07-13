# doernenburg.py
import pandas as pd
import numpy as np

def doernenburg(row):
    """Trả về fault hoặc INVALID_LOW_GAS/INCONCLUSIVE."""
    # Kiểm tra ngưỡng khí tối thiểu
    if row["h2"] < 100 or row["ch4"] < 50 or row["c2h2"] < 1 or row["c2h4"] < 50 or row["c2h6"] < 50:
        return "INVALID_LOW_GAS"
    r1 = row["ratio_ch4_h2"]
    r2 = row["ratio_c2h2_ch4"]  # cần tạo nếu chưa có
    r3 = row["ratio_c2h4_c2h6"]
    r4 = row["ratio_c2h2_c2h4"]
    if r1 > 1.0 and r2 < 0.1 and r3 < 1.0 and r4 < 0.5: return "PD"
    if r1 < 0.1 and r2 < 0.1 and 1.0 <= r3 <= 3.0 and 0.5 <= r4 <= 3.0: return "D1"
    if r1 < 1.0 and r2 < 0.1 and r3 > 3.0 and r4 > 3.0: return "D2"
    if r1 < 0.1 and r2 < 0.1 and r3 < 1.0 and r4 < 0.5: return "T1"
    if r1 < 0.1 and r2 < 0.1 and r3 >= 1.0 and r4 < 0.5: return "T2"
    if r1 < 0.1 and r2 < 0.1 and r3 >= 1.0 and r4 >= 0.5: return "T3"
    return "INCONCLUSIVE"

def apply_doernenburg(df: pd.DataFrame) -> pd.DataFrame:
    df["ratio_c2h2_ch4"] = np.where(df["ch4"] > 0, df["c2h2"] / df["ch4"], np.nan)
    df["doernenburg_fault"] = df.apply(doernenburg, axis=1)
    return df