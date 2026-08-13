# dga/rogers.py

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from config import config as cfg


logger = logging.getLogger(__name__)


def _safe_ratio(num: float, den: float) -> float:
    """
    Safely calculate a DGA gas ratio.

    Returns:
        NaN -> missing / invalid input
        inf -> positive numerator with zero denominator
        0.0 -> zero numerator and zero denominator
        num / den -> otherwise
    """
    if pd.isna(num) or pd.isna(den):
        return np.nan

    if den <= 0:
        return np.inf if num > 0 else 0.0

    return float(num) / float(den)


def _gas(row: pd.Series, name: str) -> float:
    """
    Safely read a gas concentration.

    Missing, invalid, non-finite, or negative values
    are treated as zero.
    """
    value = row.get(name, 0)

    try:
        value = float(value)
    except (TypeError, ValueError):
        return 0.0

    if not np.isfinite(value) or value < 0:
        return 0.0

    return value


def rogers_ratio_method(row: pd.Series) -> str:
    """
    Rogers Ratio Method using the three-ratio scheme.

    Ratios:
        R1 = C2H2 / C2H4
        R2 = CH4 / H2
        R3 = C2H4 / C2H6

    Rogers ratio coding:

        R1:
            < 0.1       -> 0
            0.1 ... 3.0  -> 1
            > 3.0       -> 2

        R2:
            < 0.1       -> 0
            0.1 ... 1.0  -> 1
            > 1.0       -> 2

        R3:
            < 1.0       -> 0
            1.0 ... 3.0  -> 1
            > 3.0       -> 2

    Fault interpretation:

        0-0-0 -> PD
        0-1-0 -> NORMAL
        1-1-2 -> D2
        0-1-1 -> T1
        0-2-1 -> T2
        0-2-2 -> T3

    Important:
        Rogers does not natively use IEC D1/D2 terminology.
        D2 here is the system's canonical mapping for
        high-energy discharge.
    """

    l1 = cfg.L1_LIMITS

    h2 = _gas(row, "h2")
    ch4 = _gas(row, "ch4")
    c2h2 = _gas(row, "c2h2")
    c2h4 = _gas(row, "c2h4")
    c2h6 = _gas(row, "c2h6")

    # ------------------------------------------------------------
    # Applicability gate
    #
    # Do not classify low-concentration data as a fault merely
    # because ratios become numerically extreme.
    # ------------------------------------------------------------

    applicable = any(
        [
            h2 >= l1["h2"],
            ch4 >= l1["ch4"],
            c2h2 >= l1["c2h2"],
            c2h4 >= l1["c2h4"],
            c2h6 >= l1["c2h6"],
        ]
    )

    if not applicable:
        return "ABSTAIN"

    # ------------------------------------------------------------
    # Rogers ratios
    # ------------------------------------------------------------

    r1 = _safe_ratio(c2h2, c2h4)
    r2 = _safe_ratio(ch4, h2)
    r3 = _safe_ratio(c2h4, c2h6)

    if not (
        np.isfinite(r1)
        and np.isfinite(r2)
        and np.isfinite(r3)
    ):
        return "ABSTAIN"

    # ------------------------------------------------------------
    # Convert ratios to Rogers codes
    # ------------------------------------------------------------

    # R1 = C2H2 / C2H4
    if r1 < 0.1:
        c1 = 0
    elif r1 <= 3.0:
        c1 = 1
    else:
        c1 = 2

    # R2 = CH4 / H2
    if r2 < 0.1:
        c2 = 0
    elif r2 <= 1.0:
        c2 = 1
    else:
        c2 = 2

    # R3 = C2H4 / C2H6
    if r3 < 1.0:
        c3 = 0
    elif r3 <= 3.0:
        c3 = 1
    else:
        c3 = 2

    code = (c1, c2, c3)

    # ------------------------------------------------------------
    # Rogers diagnosis
    # ------------------------------------------------------------

    # 0-0-0
    # Low-energy density arcing / PD
    if code == (0, 0, 0):
        return "PD"

    # 0-1-0
    # Normal
    if code == (0, 1, 0):
        return "NORMAL"

    # 1-1-2
    # Arcing / high-energy discharge
    #
    # Canonical system mapping:
    # Rogers high-energy discharge -> D2
    if code == (1, 1, 2):
        return "D2"

    # 0-1-1
    # Low-temperature thermal fault
    if code == (0, 1, 1):
        return "T1"

    # 0-2-1
    # Thermal fault, below approximately 700 °C
    if code == (0, 2, 1):
        return "T2"

    # 0-2-2
    # Thermal fault, above approximately 700 °C
    if code == (0, 2, 2):
        return "T3"

    # ------------------------------------------------------------
    # Unmatched Rogers code
    # ------------------------------------------------------------

    return "ABSTAIN"


def apply_rogers(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply Rogers ratios and fault classification to a DataFrame.
    """

    df = df.copy()

    df["r1_ch4_h2"] = df.apply(
        lambda r: _safe_ratio(
            _gas(r, "ch4"),
            _gas(r, "h2"),
        ),
        axis=1,
    )

    df["r2_c2h2_c2h4"] = df.apply(
        lambda r: _safe_ratio(
            _gas(r, "c2h2"),
            _gas(r, "c2h4"),
        ),
        axis=1,
    )

    df["r3_c2h4_c2h6"] = df.apply(
        lambda r: _safe_ratio(
            _gas(r, "c2h4"),
            _gas(r, "c2h6"),
        ),
        axis=1,
    )

    df["rogers_fault"] = df.apply(
        rogers_ratio_method,
        axis=1,
    )

    logger.debug("Rogers fault applied.")

    if logger.isEnabledFor(logging.DEBUG):
        cols = [
            "h2",
            "ch4",
            "c2h2",
            "c2h4",
            "c2h6",
            "r1_ch4_h2",
            "r2_c2h2_c2h4",
            "r3_c2h4_c2h6",
            "rogers_fault",
        ]

        available_cols = [
            col for col in cols if col in df.columns
        ]

        logger.debug(
            "Sample Rogers results:\n%s",
            df[available_cols].head(5).to_string(),
        )

    return df