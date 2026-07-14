# iec60599.py
from __future__ import annotations

import numpy as np
import pandas as pd

# IEC 60599 minimum gas concentrations (ppm)
MIN_H2 = 100
MIN_CH4 = 120
MIN_C2H2 = 1
MIN_C2H4 = 50
MIN_C2H6 = 65


def _safe_ratio(num: float, den: float) -> float:
    """Safe division."""
    if pd.isna(num) or pd.isna(den):
        return np.nan
    if den <= 0:
        return np.inf if num > 0 else 0.0
    return num / den


def iec_ratio_method(row) -> str:
    """
    IEC 60599 Basic Gas Ratio Method

    Ratios
    ------
    R1 = CH4 / H2
    R2 = C2H2 / C2H4
    R3 = C2H4 / C2H6

    Returns
    -------
    PD
    D1
    D2
    T1
    T2
    T3
    INVALID_LOW_GAS
    UNCERTAIN
    """

    h2 = float(row.get("h2", 0))
    ch4 = float(row.get("ch4", 0))
    c2h2 = float(row.get("c2h2", 0))
    c2h4 = float(row.get("c2h4", 0))
    c2h6 = float(row.get("c2h6", 0))

    # IEC ratio method validity
    valid = (
        h2 >= MIN_H2
        or ch4 >= MIN_CH4
        or c2h2 >= MIN_C2H2
        or c2h4 >= MIN_C2H4
        or c2h6 >= MIN_C2H6
    )

    if not valid:
        return "INVALID_LOW_GAS"

    r1 = _safe_ratio(ch4, h2)
    r2 = _safe_ratio(c2h2, c2h4)
    r3 = _safe_ratio(c2h4, c2h6)

    # PD
    if (
        r1 < 0.1
        and r3 < 0.2
    ):
        return "PD"

    # D1
    if (
        r2 > 1.0
        and 0.1 <= r1 <= 0.5
        and r3 > 1.0
    ):
        return "D1"

    # D2
    if (
        0.6 <= r2 <= 2.5
        and 0.1 <= r1 <= 1.0
        and r3 > 2.0
    ):
        return "D2"

    # T1
    if (
        r1 > 1.0
        and r3 < 1.0
    ):
        return "T1"

    # T2
    if (
        r2 < 0.1
        and r1 > 1.0
        and 1.0 <= r3 <= 4.0
    ):
        return "T2"

    # T3
    if (
        r2 < 0.2
        and r1 > 1.0
        and r3 > 4.0
    ):
        return "T3"

    return "UNCERTAIN"


def apply_iec(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply IEC 60599 ratio diagnosis.
    """

    df = df.copy()

    df["IEC_R1_CH4_H2"] = df.apply(
        lambda r: _safe_ratio(r["ch4"], r["h2"]),
        axis=1,
    )

    df["IEC_R2_C2H2_C2H4"] = df.apply(
        lambda r: _safe_ratio(r["c2h2"], r["c2h4"]),
        axis=1,
    )

    df["IEC_R3_C2H4_C2H6"] = df.apply(
        lambda r: _safe_ratio(r["c2h4"], r["c2h6"]),
        axis=1,
    )

    df["iec_fault"] = df.apply(iec_ratio_method, axis=1)

    return df