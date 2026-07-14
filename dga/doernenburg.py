# doernenburg.py
from __future__ import annotations

import numpy as np
import pandas as pd

# IEEE C57.104 Doernenburg Ratio Method

# L1 limits (ppm)
L1 = {
    "H2": 100,
    "CH4": 120,
    "C2H2": 1,
    "C2H4": 50,
    "C2H6": 65,
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
    NORMAL
    PD
    D2
    THERMAL
    NA
    """

    h2 = float(row.get("H2", 0))
    ch4 = float(row.get("CH4", 0))
    c2h2 = float(row.get("C2H2", 0))
    c2h4 = float(row.get("C2H4", 0))
    c2h6 = float(row.get("C2H6", 0))

    # ------------------------------------------------------------------
    # IEEE validity check
    # At least one key gas > 2×L1 and
    # at least one remaining gas > L1
    # ------------------------------------------------------------------

    key = [
        h2 >= 2 * L1["H2"],
        ch4 >= 2 * L1["CH4"],
        c2h2 >= 2 * L1["C2H2"],
        c2h4 >= 2 * L1["C2H4"],
    ]

    other = [
        h2 >= L1["H2"],
        ch4 >= L1["CH4"],
        c2h2 >= L1["C2H2"],
        c2h4 >= L1["C2H4"],
        c2h6 >= L1["C2H6"],
    ]

    if not (any(key) and sum(other) >= 2):
        return "NORMAL"

    r1 = _safe_ratio(ch4, h2)
    r2 = _safe_ratio(c2h2, c2h4)
    r3 = _safe_ratio(c2h2, ch4)
    r4 = _safe_ratio(c2h6, c2h2)

    # Corona (Low-energy PD)
    if (
        r1 < 0.1
        and r3 < 0.3
        and r4 > 0.4
    ):
        return "PD"

    # Thermal decomposition
    if (
        r1 > 1.0
        and r2 < 0.75
        and r3 < 0.3
        and r4 > 0.4
    ):
        return "THERMAL"

    # Arcing (High-energy discharge)
    if (
        0.1 <= r1 < 1.0
        and r2 > 0.75
        and r3 > 0.3
        and r4 < 0.4
    ):
        return "D2"

    return "NA"


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