# dga/iec60599.py
from __future__ import annotations
import numpy as np
import pandas as pd
import logging
from config import config as cfg

logger = logging.getLogger(__name__)

def _safe_ratio(num: float, den: float) -> float:
    if pd.isna(num) or pd.isna(den):
        return np.nan
    if den <= 0:
        return np.inf if num > 0 else 0.0
    return num / den

def iec_ratio_method(row: pd.Series) -> str:
    l1 = cfg.L1_LIMITS
    h2 = float(row.get("h2", 0))
    ch4 = float(row.get("ch4", 0))
    c2h2 = float(row.get("c2h2", 0))
    c2h4 = float(row.get("c2h4", 0))
    c2h6 = float(row.get("c2h6", 0))

    if not any([
        h2 >= l1["h2"],
        ch4 >= l1["ch4"],
        c2h2 >= l1["c2h2"],
        c2h4 >= l1["c2h4"],
        c2h6 >= l1["c2h6"]
    ]):
        return "NORMAL"

    r1 = _safe_ratio(c2h2, c2h4)
    r2 = _safe_ratio(ch4, h2)
    r3 = _safe_ratio(c2h4, c2h6)

    if (0.6 <= r1 <= 2.5) and (0.1 <= r2 <= 1.0) and (r3 >= 1.0):
        return "D2"
    if (r1 >= 0.1) and (r3 > 3.0):
        return "D1"
    if (r1 < 0.2) and (r2 > 1.0) and (r3 > 4.0):
        return "T3"
    if (r1 < 0.1) and (r2 > 1.0) and (1.0 <= r3 <= 4.0):
        return "T2"
    if (r2 > 1.0) and (r3 < 1.0):
        return "T1"
    if (r1 < 0.1) and (r2 < 0.1) and (r3 < 0.2):
        return "PD"
    return "UNCERTAIN"

def apply_iec(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["iec_r1_c2h2_c2h4"] = df.apply(lambda r: _safe_ratio(r["c2h2"], r["c2h4"]), axis=1)
    df["iec_r2_ch4_h2"] = df.apply(lambda r: _safe_ratio(r["ch4"], r["h2"]), axis=1)
    df["iec_r3_c2h4_c2h6"] = df.apply(lambda r: _safe_ratio(r["c2h4"], r["c2h6"]), axis=1)
    df["iec_fault"] = df.apply(iec_ratio_method, axis=1)

    logger.debug("IEC 60599 fault applied.")
    if logger.isEnabledFor(logging.DEBUG):
        cols = ["h2", "ch4", "c2h2", "c2h4", "c2h6", "iec_r1_c2h2_c2h4", "iec_r2_ch4_h2", "iec_r3_c2h4_c2h6", "iec_fault"]
        logger.debug("Sample IEC results:\n" + df[cols].head(5).to_string())
    return df