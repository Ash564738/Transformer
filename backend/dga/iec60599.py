# dga/iec60599.py
from __future__ import annotations
import logging
from attrs import field
from typing import List
import numpy as np, pandas as pd
from config import config as cfg
DIAGNOSTIC_GASES: List[str] = ["h2", "ch4", "c2h2", "c2h4", "c2h6"]
L1_LIMITS = {"h2": 100, "ch4": 120, "c2h2": 1, "c2h4": 50, "c2h6": 65}
logger = logging.getLogger(__name__)

def _gas(row: pd.Series, name: str) -> float:
    value = row.get(name, np.nan)
    try: value = float(value)
    except (TypeError, ValueError): return np.nan
    return value if np.isfinite(value) and value >= 0 else np.nan

def _safe_ratio(num: float, den: float) -> float:
    if pd.isna(num) or pd.isna(den): return np.nan
    if den == 0: return np.inf if num > 0 else np.nan
    return float(num) / float(den)

def _applicable(row: pd.Series) -> bool:
    for gas_name in DIAGNOSTIC_GASES:
        value = _gas(row, gas_name)
        if np.isfinite(value) and value >= L1_LIMITS[gas_name]: return True
    return False

def iec_ratio_method(row: pd.Series) -> str:
    if not _applicable(row): return "ABSTAIN"
    h2 = _gas(row, "h2"); ch4 = _gas(row, "ch4"); c2h2 = _gas(row, "c2h2"); c2h4 = _gas(row, "c2h4"); c2h6 = _gas(row, "c2h6")
    values = [h2, ch4, c2h2, c2h4, c2h6]
    if not all(np.isfinite(value) for value in values): return "ABSTAIN"
    r1 = _safe_ratio(c2h2, c2h4); r2 = _safe_ratio(ch4, h2); r3 = _safe_ratio(c2h4, c2h6)
    if np.isfinite(r2) and np.isfinite(r3) and r2 < 0.1 and r3 < 0.2: return "PD"
    if np.isfinite(r1) and np.isfinite(r2) and np.isfinite(r3) and r1 > 1.0 and 0.1 <= r2 <= 0.5 and r3 > 1.0: return "D1"
    if np.isfinite(r1) and np.isfinite(r2) and np.isfinite(r3) and 0.6 <= r1 <= 2.5 and 0.1 <= r2 <= 1.0 and r3 > 2.0: return "D2"
    if np.isfinite(r2) and np.isfinite(r3) and r2 > 1.0 and r3 < 1.0: return "T1"
    if np.isfinite(r1) and np.isfinite(r2) and np.isfinite(r3) and r1 < 0.1 and r2 > 1.0 and 1.0 <= r3 <= 4.0: return "T2"
    if np.isfinite(r1) and np.isfinite(r2) and np.isfinite(r3) and r1 < 0.2 and r2 > 1.0 and r3 > 4.0: return "T3"
    return "ABSTAIN"

def apply_iec(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["iec_r1_c2h2_c2h4"] = df.apply(lambda row: _safe_ratio(_gas(row, "c2h2"), _gas(row, "c2h4")), axis=1)
    df["iec_r2_ch4_h2"] = df.apply(lambda row: _safe_ratio(_gas(row, "ch4"), _gas(row, "h2")), axis=1)
    df["iec_r3_c2h4_c2h6"] = df.apply(lambda row: _safe_ratio(_gas(row, "c2h4"), _gas(row, "c2h6")), axis=1)
    df["iec_fault"] = df.apply(iec_ratio_method, axis=1)
    logger.debug("IEC 60599:2022 ratio diagnostic applied.")
    if logger.isEnabledFor(logging.DEBUG):
        cols = ["h2", "ch4", "c2h2", "c2h4", "c2h6", "iec_r1_c2h2_c2h4", "iec_r2_ch4_h2", "iec_r3_c2h4_c2h6", "iec_fault"]
        available_cols = [col for col in cols if col in df.columns]
        logger.debug("Sample IEC results:\n%s", df[available_cols].head(5).to_string())
    return df