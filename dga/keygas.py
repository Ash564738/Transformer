# dga/keygas.py
from __future__ import annotations
import numpy as np
import pandas as pd
import logging
from config import config as cfg

logger = logging.getLogger(__name__)

def key_gas_method(row: pd.Series) -> str:
    # Dựa vào tên cột đã được chuẩn hóa lowercase từ trước
    h2 = float(row.get("h2", 0.0)) if not pd.isna(row.get("h2")) else 0.0
    ch4 = float(row.get("ch4", 0.0)) if not pd.isna(row.get("ch4")) else 0.0
    c2h6 = float(row.get("c2h6", 0.0)) if not pd.isna(row.get("c2h6")) else 0.0
    c2h4 = float(row.get("c2h4", 0.0)) if not pd.isna(row.get("c2h4")) else 0.0
    c2h2 = float(row.get("c2h2", 0.0)) if not pd.isna(row.get("c2h2")) else 0.0
    co = float(row.get("co", 0.0)) if not pd.isna(row.get("co")) else 0.0

    tdcg = row.get("tdcg", np.nan)
    if pd.isna(tdcg):
        tdcg = h2 + ch4 + c2h6 + c2h4 + c2h2 + co

    if tdcg < cfg.MIN_TDCG:
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
    df["keygas_fault"] = df.apply(key_gas_method, axis=1)

    logger.debug("Key gas fault applied.")
    if logger.isEnabledFor(logging.DEBUG):
        sample_cols = ["h2", "ch4", "c2h6", "c2h4", "c2h2", "co", "tdcg", "keygas_fault"]
        logger.debug("Sample Key Gas results:\n" + df[sample_cols].head(5).to_string())
    return df