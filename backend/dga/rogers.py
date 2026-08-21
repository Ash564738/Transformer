# dga/rogers.py
from __future__ import annotations

from dataclasses import field
import logging
from typing import List

import numpy as np
import pandas as pd

from config import config as cfg

logger = logging.getLogger(__name__)


ROGERS_R1_LOW: float = 0.1
ROGERS_R1_HIGH: float = 3.0
ROGERS_R2_LOW: float = 0.1
ROGERS_R2_HIGH: float = 1.0
ROGERS_R3_LOW: float = 1.0
ROGERS_R3_HIGH: float = 3.0

DIAGNOSTIC_GASES: List[str] = [
    "h2",
    "ch4",
    "c2h2",
    "c2h4",
    "c2h6",
]
L1_LIMITS = {
    "h2": 100,
    "ch4": 120,
    "c2h2": 1,
    "c2h4": 50,
    "c2h6": 65,
}
# ============================================================
# SAFE GAS / RATIO
# ============================================================

def _gas(row: pd.Series, name: str) -> float:
    value = row.get(name, np.nan)

    try:
        value = float(value)
    except (TypeError, ValueError):
        return np.nan

    if not np.isfinite(value) or value < 0:
        return np.nan

    return value


def _safe_ratio(num: float, den: float) -> float:
    """
    Return:

        finite ratio -> normal division
        +inf         -> positive numerator / zero denominator
        NaN          -> zero/zero or invalid data
    """

    if pd.isna(num) or pd.isna(den):
        return np.nan

    if den == 0:
        if num > 0:
            return np.inf
        return np.nan

    return float(num) / float(den)


# ============================================================
# APPLICABILITY
# ============================================================

def _rogers_applicable(row: pd.Series) -> bool:
    """
    Conservative applicability gate.

    Rogers should not be used on very-low-gas samples.
    We require at least one of the five gases used by the
    method to reach its IEEE L1 value.
    """

    for gas in DIAGNOSTIC_GASES:
        value = _gas(row, gas)

        if (
            np.isfinite(value)
            and value >= L1_LIMITS[gas]
        ):
            return True

    return False


# ============================================================
# RATIO CODING
# ============================================================

def _code_r1(ratio: float) -> int | None:
    if not np.isfinite(ratio):
        return None

    if ratio < ROGERS_R1_LOW:
        return 0

    if ratio <= ROGERS_R1_HIGH:
        return 1

    return 2


def _code_r2(ratio: float) -> int | None:
    if not np.isfinite(ratio):
        return None

    if ratio < ROGERS_R2_LOW:
        return 0

    if ratio <= ROGERS_R2_HIGH:
        return 1

    return 2


def _code_r3(ratio: float) -> int | None:
    if not np.isfinite(ratio):
        return None

    if ratio < ROGERS_R3_LOW:
        return 0

    if ratio <= ROGERS_R3_HIGH:
        return 1

    return 2


# ============================================================
# PUBLIC ROGERS METHOD
# ============================================================

def rogers_ratio_method(row: pd.Series) -> str:
    """
    IEEE C57.104-2019 Rogers Ratio Method.

    Ratios:

        R1 = C2H2 / C2H4
        R2 = CH4 / H2
        R3 = C2H4 / C2H6

    IEEE Table 5:

        Code       Diagnosis

        0-0-0      NORMAL
        0-0-1      not defined
        0-1-0      PD
        1-0-2      not used by Table 5
        1-1-2      D2
        0-1-1      T1
        0-2-1      T2
        0-2-2      T3

    More exactly, only the six combinations explicitly
    defined by IEEE are accepted.
    """

    if not _rogers_applicable(row):
        return "ABSTAIN"

    c2h2 = _gas(row, "c2h2")
    c2h4 = _gas(row, "c2h4")
    ch4 = _gas(row, "ch4")
    h2 = _gas(row, "h2")
    c2h6 = _gas(row, "c2h6")

    values = [
        c2h2,
        c2h4,
        ch4,
        h2,
        c2h6,
    ]

    if not all(np.isfinite(value) for value in values):
        return "ABSTAIN"

    # --------------------------------------------------------
    # Rogers ratios
    # --------------------------------------------------------

    r1 = _safe_ratio(c2h2, c2h4)
    r2 = _safe_ratio(ch4, h2)
    r3 = _safe_ratio(c2h4, c2h6)

    if not all(np.isfinite(value) for value in [r1, r2, r3]):
        return "ABSTAIN"

    code = (
        _code_r1(r1),
        _code_r2(r2),
        _code_r3(r3),
    )

    if any(value is None for value in code):
        return "ABSTAIN"

    # --------------------------------------------------------
    # IEEE C57.104-2019 Table 5
    # --------------------------------------------------------

    mapping = {
        (0, 1, 0): "NORMAL",
        (0, 0, 0): "PD",
        (1, 1, 2): "D2",
        (0, 1, 1): "T1",
        (0, 2, 1): "T2",
        (0, 2, 2): "T3",
    }

    return mapping.get(code, "ABSTAIN")


# ============================================================
# DATAFRAME APPLICATION
# ============================================================

def apply_rogers(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["r1_c2h2_c2h4"] = df.apply(
        lambda row: _safe_ratio(
            _gas(row, "c2h2"),
            _gas(row, "c2h4"),
        ),
        axis=1,
    )

    df["r2_ch4_h2"] = df.apply(
        lambda row: _safe_ratio(
            _gas(row, "ch4"),
            _gas(row, "h2"),
        ),
        axis=1,
    )

    df["r3_c2h4_c2h6"] = df.apply(
        lambda row: _safe_ratio(
            _gas(row, "c2h4"),
            _gas(row, "c2h6"),
        ),
        axis=1,
    )

    df["rogers_fault"] = df.apply(
        rogers_ratio_method,
        axis=1,
    )

    logger.debug("Rogers diagnostic applied.")

    return df