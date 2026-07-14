# doernenburg.py
from __future__ import annotations

import numpy as np
import pandas as pd

# IEEE C57.104 Doernenburg Ratio Method

L1 = {
    "h2": 100,
    "ch4": 120,
    "c2h2": 1,
    "c2h4": 50,
    "c2h6": 65,
}


def _safe_ratio(num: float, den: float) -> float:
    if pd.isna(num) or pd.isna(den):
        return np.nan
    if den <= 0:
        return np.inf if num > 0 else 0.0
    return num / den


def doernenburg_method(row: pd.Series) -> str:
    """
    IEEE C57.104 Doernenburg Ratio Method

    Ratios
    ------
    R1 = CH4 / H2
    R2 = C2H2 / C2H4
    R3 = C2H2 / CH4
    R4 = C2H6 / C2H2

    Returns
    -------
    THERMAL
    PD
    D2
    INVALID_LOW_GAS
    UNCERTAIN
    """

    h2 = float(row.get("h2", 0))
    ch4 = float(row.get("ch4", 0))
    c2h2 = float(row.get("c2h2", 0))
    c2h4 = float(row.get("c2h4", 0))
    c2h6 = float(row.get("c2h6", 0))

    # ---------------------------------------------------------
    # Step 2 (IEEE)
    # At least one of H2, CH4, C2H2, C2H4 > 2×L1
    # and one other key gas > L1
    # ---------------------------------------------------------

    trigger = any(
        [
            h2 >= 2 * L1["h2"],
            ch4 >= 2 * L1["ch4"],
            c2h2 >= 2 * L1["c2h2"],
            c2h4 >= 2 * L1["c2h4"],
        ]
    )

    above_l1 = sum(
        [
            h2 >= L1["h2"],
            ch4 >= L1["ch4"],
            c2h2 >= L1["c2h2"],
            c2h4 >= L1["c2h4"],
            c2h6 >= L1["c2h6"],
        ]
    )

    if not (trigger and above_l1 >= 2):
        return "INVALID_LOW_GAS"

    # ---------------------------------------------------------
    # Step 3 (IEEE)
    # At least one gas in each ratio exceeds L1
    # ---------------------------------------------------------

    ratio_valid = (
        (h2 >= L1["h2"] or ch4 >= L1["ch4"])
        and (c2h2 >= L1["c2h2"] or c2h4 >= L1["c2h4"])
        and (c2h2 >= L1["c2h2"] or ch4 >= L1["ch4"])
        and (c2h6 >= L1["c2h6"] or c2h2 >= L1["c2h2"])
    )

    if not ratio_valid:
        return "INVALID_LOW_GAS"

    # ---------------------------------------------------------
    # Step 4
    # ---------------------------------------------------------

    r1 = _safe_ratio(ch4, h2)
    r2 = _safe_ratio(c2h2, c2h4)
    r3 = _safe_ratio(c2h2, ch4)
    r4 = _safe_ratio(c2h6, c2h2)

    # ---------------------------------------------------------
    # Step 5
    # Compare with IEEE Doernenburg table
    # ---------------------------------------------------------

    # Thermal decomposition
    if (
        r1 > 1.0
        and r2 < 0.75
        and r3 < 0.3
        and r4 > 0.4
    ):
        return "THERMAL"

    # Partial discharge (Corona)
    if (
        r1 < 0.1
        and r3 < 0.3
        and r4 > 0.4
    ):
        return "PD"

    # Arcing (High-intensity PD)
    if (
        0.1 <= r1 < 1.0
        and r2 > 0.75
        and r3 > 0.3
        and r4 < 0.4
    ):
        return "D2"

    return "UNCERTAIN"


def apply_doernenburg(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["DR_R1_CH4_H2"] = df.apply(
        lambda r: _safe_ratio(r["ch4"], r["h2"]),
        axis=1,
    )

    df["DR_R2_C2H2_C2H4"] = df.apply(
        lambda r: _safe_ratio(r["c2h2"], r["c2h4"]),
        axis=1,
    )

    df["DR_R3_C2H2_CH4"] = df.apply(
        lambda r: _safe_ratio(r["c2h2"], r["ch4"]),
        axis=1,
    )

    df["DR_R4_C2H6_C2H2"] = df.apply(
        lambda r: _safe_ratio(r["c2h6"], r["c2h2"]),
        axis=1,
    )

    df["doernenburg_fault"] = df.apply(
        doernenburg_method,
        axis=1,
    )

    return df