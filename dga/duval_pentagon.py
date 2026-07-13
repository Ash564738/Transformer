# duval_pentagon.py
import pandas as pd
import numpy as np

def duval_pentagon_zone(row):
    """Tạm thời trả về 'UNKNOWN' vì cần dữ liệu %H2, %C2H6,..."""
    # Đây là phần phức tạp, tôi chỉ trả về UNKNOWN để chứng minh pipeline
    return "UNKNOWN"

def apply_duval_pentagon(df: pd.DataFrame) -> pd.DataFrame:
    df["duval_pentagon_fault"] = df.apply(duval_pentagon_zone, axis=1)
    return df