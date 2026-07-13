# iec60599.py
import pandas as pd
import numpy as np

def iec60599_ratios(row):
    """Trả về list các fault codes, hoặc ['INCONCLUSIVE']."""
    faults = []
    r1 = row["ratio_c2h2_c2h4"]
    r2 = row["ratio_ch4_h2"]
    r3 = row["ratio_c2h4_c2h6"]
    
    # PD
    if r2 < 0.1 and r3 < 0.2:
        faults.append("PD")
    # D1
    if r1 > 0.1 and 0.1 < r2 < 0.5 and r3 > 0.2:
        faults.append("D1")
    # D2
    if 0.1 < r1 < 0.5 and 0.1 < r2 < 1.0 and r3 > 0.4:
        faults.append("D2")
    # T1
    if r1 < 0.1 and 0.1 < r2 < 1.0 and 1.0 < r3 < 4.0:
        faults.append("T1")
    # T2
    if r1 < 0.1 and r2 > 1.0 and r3 > 4.0:
        faults.append("T2")
    # T3
    if r1 < 0.2 and r2 > 1.0 and 4.0 < r3 < 10.0:
        faults.append("T3")
    return faults if faults else ["INCONCLUSIVE"]

def apply_iec(df: pd.DataFrame) -> pd.DataFrame:
    df["iec_faults"] = df.apply(iec60599_ratios, axis=1)
    return df