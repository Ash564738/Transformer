# dga/doernenburg.py
from __future__ import annotations
import numpy as np
import pandas as pd
import logging
from config import config as cfg

logger = logging.getLogger(__name__)

def _safe_ratio(num: float, den: float) -> float:
    if pd.isna(num) or pd.isna(den) or den <= 0:
        return np.nan
    return num / den

def doernenburg_method(row: pd.Series) -> str:
    l1 = cfg.L1_DOERNENBURG
    h2 = float(row.get("h2", 0))
    ch4 = float(row.get("ch4", 0))
    c2h2 = float(row.get("c2h2", 0))
    c2h4 = float(row.get("c2h4", 0))
    c2h6 = float(row.get("c2h6", 0))

    if not any([
        h2 >= l1["h2"],
        ch4 >= l1["ch4"],
        c2h2 >= l1["c2h2"],
        c2h4 >= l1["c2h4"]
    ]):
        return "NORMAL"

    r1 = _safe_ratio(ch4, h2)
    r2 = _safe_ratio(c2h2, c2h4)
    r3 = _safe_ratio(c2h2, ch4)
    r4 = _safe_ratio(c2h6, c2h2)

    r1_valid = (ch4 > 0 and h2 > 0) and (ch4 >= l1["ch4"] or h2 >= l1["h2"])
    r2_valid = (c2h2 > 0 and c2h4 > 0) and (c2h2 >= l1["c2h2"] or c2h4 >= l1["c2h4"])
    r3_valid = (c2h2 > 0 and ch4 > 0) and (c2h2 >= l1["c2h2"] or ch4 >= l1["ch4"])
    r4_valid = (c2h6 > 0 and c2h2 > 0) and (c2h6 >= l1["c2h6"] or c2h2 >= l1["c2h2"])

    if not any([r1_valid, r2_valid, r3_valid, r4_valid]):
        return "UNCERTAIN"

    if r1_valid and r2_valid and r4_valid:
        if r1 > 1.0 and r2 < 0.75 and r4 > 0.4:
            if not r3_valid or r3 < 0.3:
                return "T3"

    if r1_valid and r4_valid:
        if r1 < 0.1 and r4 > 0.4:
            if not r3_valid or r3 < 0.3:
                return "PD"

    if r1_valid and r2_valid and r3_valid and r4_valid:
        if 0.1 <= r1 <= 1.0 and r2 > 0.75 and r3 > 0.3 and r4 < 0.4:
            return "D2"

    return "UNCERTAIN"

def apply_doernenburg(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["dr_r1_ch4_h2"] = df.apply(lambda r: _safe_ratio(r["ch4"], r["h2"]), axis=1)
    df["dr_r2_c2h2_c2h4"] = df.apply(lambda r: _safe_ratio(r["c2h2"], r["c2h4"]), axis=1)
    df["dr_r3_c2h2_ch4"] = df.apply(lambda r: _safe_ratio(r["c2h2"], r["ch4"]), axis=1)
    df["dr_r4_c2h6_c2h2"] = df.apply(lambda r: _safe_ratio(r["c2h6"], r["c2h2"]), axis=1)
    df["doernenburg_fault"] = df.apply(doernenburg_method, axis=1)

    logger.debug("Doernenburg fault applied.")
    if logger.isEnabledFor(logging.DEBUG):
        cols = ["h2", "ch4", "c2h2", "c2h4", "c2h6",
                "dr_r1_ch4_h2", "dr_r2_c2h2_c2h4", "dr_r3_c2h2_ch4", "dr_r4_c2h6_c2h2",
                "doernenburg_fault"]
        logger.debug("Sample Doernenburg results:\n" + df[cols].head(5).to_string())
    return df