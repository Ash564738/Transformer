# dga/doernenburg.py
from __future__ import annotations
import logging
from typing import Dict
import numpy as np, pandas as pd
from config import config as cfg
logger = logging.getLogger(__name__)
L1_LIMITS = {"h2": 100, "ch4": 120, "c2h2": 1, "c2h4": 50, "c2h6": 65}

def _gas(row: pd.Series, name: str) -> float:
    value = row.get(name, np.nan)
    try: value = float(value)
    except (TypeError, ValueError): return np.nan
    return value if np.isfinite(value) and value >= 0 else np.nan

def _safe_ratio(num: float, den: float) -> float:
    if pd.isna(num) or pd.isna(den): return np.nan
    if den == 0: return np.inf if num > 0 else np.nan
    return float(num) / float(den)

def _applicable_to_doernenburg(gases: Dict[str, float]) -> bool:
    primary = {"h2": gases["h2"], "ch4": gases["ch4"], "c2h2": gases["c2h2"], "c2h4": gases["c2h4"]}
    l1 = L1_LIMITS
    high_gases = [gas for gas, value in primary.items() if np.isfinite(value) and value >= 2.0 * l1[gas]]
    if not high_gases: return False
    for high_gas in high_gases:
        for other_gas, other_value in primary.items():
            if other_gas == high_gas: continue
            if np.isfinite(other_value) and other_value >= l1[other_gas]: return True
    return False

def _ratio_valid(numerator: float, denominator: float, numerator_name: str, denominator_name: str) -> bool:
    l1 = L1_LIMITS
    if pd.isna(numerator) or pd.isna(denominator): return False
    return numerator >= l1[numerator_name] or denominator >= l1[denominator_name]

def _thermal_pattern(r1: float, r2: float, r3: float, r4: float) -> bool:
    return np.isfinite(r1) and np.isfinite(r2) and np.isfinite(r3) and r4 > 0.4 and r1 > 1.0 and r2 < 0.75 and r3 < 0.3

def _pd_pattern(r1: float, r2: float, r3: float, r4: float) -> bool:
    return np.isfinite(r1) and np.isfinite(r3) and r4 > 0.4 and r1 < 0.1 and r3 < 0.3

def _arcing_pattern(r1: float, r2: float, r3: float, r4: float) -> bool:
    return np.isfinite(r1) and np.isfinite(r2) and np.isfinite(r3) and np.isfinite(r4) and 0.1 < r1 < 1.0 and r2 > 0.75 and r3 > 0.3 and r4 < 0.4

def doernenburg_method(row: pd.Series) -> str:
    gases = {name: _gas(row, name) for name in ["h2", "ch4", "c2h2", "c2h4", "c2h6"]}
    if not all(np.isfinite(value) for value in gases.values()): return "ABSTAIN"
    if not _applicable_to_doernenburg(gases): return "ABSTAIN"
    h2 = gases["h2"]; ch4 = gases["ch4"]; c2h2 = gases["c2h2"]; c2h4 = gases["c2h4"]; c2h6 = gases["c2h6"]
    r1 = _safe_ratio(ch4, h2); r2 = _safe_ratio(c2h2, c2h4); r3 = _safe_ratio(c2h2, ch4); r4 = _safe_ratio(c2h6, c2h2)
    ratios_valid = all([
        _ratio_valid(ch4, h2, "ch4", "h2"),
        _ratio_valid(c2h2, c2h4, "c2h2", "c2h4"),
        _ratio_valid(c2h2, ch4, "c2h2", "ch4"),
        _ratio_valid(c2h6, c2h2, "c2h6", "c2h2"),
    ])
    if not ratios_valid: return "ABSTAIN"
    if _thermal_pattern(r1, r2, r3, r4): return "T3"
    if _pd_pattern(r1, r2, r3, r4): return "PD"
    if _arcing_pattern(r1, r2, r3, r4): return "D2"
    return "ABSTAIN"

def apply_doernenburg(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["dr_r1_ch4_h2"] = df.apply(lambda row: _safe_ratio(_gas(row, "ch4"), _gas(row, "h2")), axis=1)
    df["dr_r2_c2h2_c2h4"] = df.apply(lambda row: _safe_ratio(_gas(row, "c2h2"), _gas(row, "c2h4")), axis=1)
    df["dr_r3_c2h2_ch4"] = df.apply(lambda row: _safe_ratio(_gas(row, "c2h2"), _gas(row, "ch4")), axis=1)
    df["dr_r4_c2h6_c2h2"] = df.apply(lambda row: _safe_ratio(_gas(row, "c2h6"), _gas(row, "c2h2")), axis=1)
    df["doernenburg_fault"] = df.apply(doernenburg_method, axis=1)
    logger.debug("Doernenburg diagnostic applied.")
    return df