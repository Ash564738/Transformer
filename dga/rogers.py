# rogers.py
import pandas as pd
import numpy as np

def rogers_ratio(row):
    """Trả về mã fault hoặc 'INCONCLUSIVE'."""
    r1 = row["ratio_c2h2_c2h4"]
    r2 = row["ratio_ch4_h2"]
    r3 = row["ratio_c2h4_c2h6"]
    # Rogers 4-ratio đơn giản hóa
    if r2 < 0.1 and r3 < 1.0: return "PD"
    if r1 > 0.1 and 0.1 <= r2 <= 1.0 and r3 > 1.0: return "D1"
    if 0.1 <= r1 <= 0.5 and 0.1 <= r2 <= 1.0 and r3 > 1.0: return "D2"
    if r1 < 0.1 and r2 > 1.0 and 1.0 <= r3 <= 4.0: return "T1"
    if r1 < 0.1 and r2 > 1.0 and r3 > 4.0: return "T2"
    if r1 < 0.2 and r2 > 1.0 and r3 > 10.0: return "T3"
    return "INCONCLUSIVE"

def apply_rogers(df: pd.DataFrame) -> pd.DataFrame:
    df["rogers_fault"] = df.apply(rogers_ratio, axis=1)
    return df