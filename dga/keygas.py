# dga/keygas.py
from __future__ import annotations
import numpy as np
import pandas as pd

MIN_TDCG = 100.0  # ppm

def key_gas_method(row: pd.Series) -> str:
    row_lower = row.copy()
    row_lower.index = row_lower.index.str.lower()

    h2 = float(row_lower.get("h2", 0.0)) if not pd.isna(row_lower.get("h2")) else 0.0
    ch4 = float(row_lower.get("ch4", 0.0)) if not pd.isna(row_lower.get("ch4")) else 0.0
    c2h6 = float(row_lower.get("c2h6", 0.0)) if not pd.isna(row_lower.get("c2h6")) else 0.0
    c2h4 = float(row_lower.get("c2h4", 0.0)) if not pd.isna(row_lower.get("c2h4")) else 0.0
    c2h2 = float(row_lower.get("c2h2", 0.0)) if not pd.isna(row_lower.get("c2h2")) else 0.0
    co = float(row_lower.get("co", 0.0)) if not pd.isna(row_lower.get("co")) else 0.0

    tdcg = row_lower.get("tdcg", np.nan)
    if pd.isna(tdcg):
        tdcg = h2 + ch4 + c2h6 + c2h4 + c2h2 + co

    if tdcg < MIN_TDCG:
        return "NORMAL"

    gases = {
        "h2": h2,
        "ch4": ch4,
        "c2h6": c2h6,
        "c2h4": c2h4,
        "c2h2": c2h2,
        "co": co
    }
    dominant = max(gases, key=gases.get)

    if dominant == "c2h2":
        return "D2"
    elif dominant == "h2":
        return "PD"
    elif dominant == "co":
        return "THERMAL_CELLULOSE"
    elif dominant in ["c2h4", "ch4", "c2h6"]:
        return "THERMAL_OIL"
    else:
        return "UNCERTAIN"

def apply_key_gas(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = df.columns.str.lower()
    df["keygas_fault"] = df.apply(key_gas_method, axis=1)

    # Debug: in 5 dòng đầu
    print("=== DEBUG KEY GAS (first 5 rows) ===")
    sample = df[["h2", "ch4", "c2h6", "c2h4", "c2h2", "co", "tdcg", "keygas_fault"]].head(5)
    print(sample.to_string())
    print("=====================================\n")
    return df