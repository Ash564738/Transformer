# rogers.py
from __future__ import annotations

import numpy as np
import pandas as pd

# IEEE C57.104 / Rogers Ratio Method


def _safe_ratio(num: float, den: float) -> float:
    if pd.isna(num) or pd.isna(den):
        return np.nan
    if den <= 0:
        return np.inf if num > 0 else 0.0
    return num / den


def rogers_ratio_method(row: pd.Series) -> str:
    """
    Rogers Ratio Method (IEEE C57.104)

    R1 = CH4 / H2
    R2 = C2H2 / C2H4
    R3 = C2H4 / C2H6

    Returns
    -------
    NORMAL
    PD
    D2
    T1
    T2
    T3
    NA
    """

    h2 = float(row.get("H2", 0))
    ch4 = float(row.get("CH4", 0))
    c2h2 = float(row.get("C2H2", 0))
    c2h4 = float(row.get("C2H4", 0))
    c2h6 = float(row.get("C2H6", 0))

    r1 = _safe_ratio(ch4, h2)
    r2 = _safe_ratio(c2h2, c2h4)
    r3 = _safe_ratio(c2h4, c2h6)

    # Case 0 : Normal
    if (
        0.1 <= r1 <= 1.0
        and r2 < 0.1
        and r3 < 1.0
    ):
        return "NORMAL"

    # Case 1 : Partial discharge
    if (
        r1 < 0.1
        and r2 < 0.1
        and r3 < 1.0
    ):
        return "PD"

    # Case 2 : High-energy discharge
    if (
        0.1 <= r1 <= 1.0
        and 0.1 <= r2 <= 3.0
        and r3 > 3.0
    ):
        return "D2"

    # Case 3 : Thermal <300°C
    if (
        0.1 <= r1 <= 1.0
        and r2 < 0.1
        and 1.0 <= r3 <= 3.0
    ):
        return "T1"

    # Case 4 : Thermal 300–700°C
    if (
        r1 > 1.0
        and r2 < 0.1
        and 1.0 <= r3 <= 3.0
    ):
        return "T2"

    # Case 5 : Thermal >700°C
    if (
        r1 > 1.0
        and r2 < 0.1
        and r3 > 3.0
    ):
        return "T3"

    return "NA"


def apply_rogers(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply Rogers Ratio Method.
    """

    df = df.copy()

    df["R1_CH4_H2"] = df.apply(
        lambda r: _safe_ratio(r["ch4"], r["h2"]),
        axis=1,
    )

    df["R2_C2H2_C2H4"] = df.apply(
        lambda r: _safe_ratio(r["c2h2"], r["c2h4"]),
        axis=1,
    )

    df["R3_C2H4_C2H6"] = df.apply(
        lambda r: _safe_ratio(r["c2h4"], r["c2h6"]),
        axis=1,
    )

    df["rogers_fault"] = df.apply(rogers_ratio_method, axis=1)

    return df