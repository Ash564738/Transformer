# dga/doernenburg.py
from __future__ import annotations

import logging
from typing import Dict

import numpy as np
import pandas as pd

from config import config as cfg

logger = logging.getLogger(__name__)


# ============================================================
# SAFE GAS / RATIO HELPERS
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
    Ratio behavior:

        valid / positive -> finite ratio
        positive / zero  -> +inf
        zero / zero      -> NaN
        missing          -> NaN
    """

    if pd.isna(num) or pd.isna(den):
        return np.nan

    if den == 0:
        if num > 0:
            return np.inf
        return np.nan

    return float(num) / float(den)


# ============================================================
# DOERNENBURG APPLICABILITY
# ============================================================

def _applicable_to_doernenburg(
    gases: Dict[str, float],
) -> bool:
    """
    IEEE C57.104-2019 Doernenburg applicability gate.

    At least one of:
        H2
        CH4
        C2H2
        C2H4

    must exceed 2 * L1.

    At least one OTHER gas among the same four gases must
    exceed its L1 value.
    """

    primary = {
        "h2": gases["h2"],
        "ch4": gases["ch4"],
        "c2h2": gases["c2h2"],
        "c2h4": gases["c2h4"],
    }

    l1 = cfg.L1_DOERNENBURG

    high_gases = [
        gas
        for gas, value in primary.items()
        if np.isfinite(value)
        and value >= 2.0 * l1[gas]
    ]

    if not high_gases:
        return False

    for high_gas in high_gases:
        for other_gas, other_value in primary.items():
            if other_gas == high_gas:
                continue

            if (
                np.isfinite(other_value)
                and other_value >= l1[other_gas]
            ):
                return True

    return False


# ============================================================
# RATIO VALIDITY
# ============================================================

def _ratio_valid(
    numerator: float,
    denominator: float,
    numerator_name: str,
    denominator_name: str,
) -> bool:
    """
    A Doernenburg ratio is considered valid when at least one
    of the gases forming that ratio exceeds its L1 limit.
    """

    l1 = cfg.L1_DOERNENBURG

    if pd.isna(numerator) or pd.isna(denominator):
        return False

    return (
        numerator >= l1[numerator_name]
        or denominator >= l1[denominator_name]
    )


# ============================================================
# DIAGNOSTIC PATTERNS
# ============================================================

def _thermal_pattern(
    r1: float,
    r2: float,
    r3: float,
    r4: float,
) -> bool:
    return (
        np.isfinite(r1)
        and np.isfinite(r2)
        and np.isfinite(r3)
        and (
            r4 > 0.4
        )
        and r1 > 1.0
        and r2 < 0.75
        and r3 < 0.3
    )


def _pd_pattern(
    r1: float,
    r2: float,
    r3: float,
    r4: float,
) -> bool:
    """
    IEEE Doernenburg PD:

        R1 < 0.1
        R2 not significant
        R3 < 0.3
        R4 > 0.4
    """

    return (
        np.isfinite(r1)
        and np.isfinite(r3)
        and (
            r4 > 0.4
        )
        and r1 < 0.1
        and r3 < 0.3
    )


def _arcing_pattern(
    r1: float,
    r2: float,
    r3: float,
    r4: float,
) -> bool:
    """
    IEEE Doernenburg arcing:

        0.1 < R1 < 1.0
        R2 > 0.75
        R3 > 0.3
        R4 < 0.4
    """

    return (
        np.isfinite(r1)
        and np.isfinite(r2)
        and np.isfinite(r3)
        and np.isfinite(r4)
        and 0.1 < r1 < 1.0
        and r2 > 0.75
        and r3 > 0.3
        and r4 < 0.4
    )


# ============================================================
# PUBLIC METHOD
# ============================================================

def doernenburg_method(row: pd.Series) -> str:
    """
    IEEE C57.104-2019 Doernenburg ratio method.

    Input:
        Mineral-oil DGA concentrations in ppm.

    Output:
        T3
        PD
        D2
        ABSTAIN

    NORMAL is deliberately not returned by this method.

    Doernenburg is a fault-identification method. Failure to
    satisfy its applicability or ratio criteria means that
    the method is inconclusive, not that the transformer is
    normal.
    """

    gases = {
        name: _gas(row, name)
        for name in [
            "h2",
            "ch4",
            "c2h2",
            "c2h4",
            "c2h6",
        ]
    }

    # --------------------------------------------------------
    # Missing / invalid required gases
    # --------------------------------------------------------

    if not all(np.isfinite(value) for value in gases.values()):
        return "ABSTAIN"

    # --------------------------------------------------------
    # Applicability gate
    # --------------------------------------------------------

    if not _applicable_to_doernenburg(gases):
        return "ABSTAIN"

    h2 = gases["h2"]
    ch4 = gases["ch4"]
    c2h2 = gases["c2h2"]
    c2h4 = gases["c2h4"]
    c2h6 = gases["c2h6"]

    # --------------------------------------------------------
    # Ratios
    #
    # R1 = CH4 / H2
    # R2 = C2H2 / C2H4
    # R3 = C2H2 / CH4
    # R4 = C2H6 / C2H2
    # --------------------------------------------------------

    r1 = _safe_ratio(ch4, h2)
    r2 = _safe_ratio(c2h2, c2h4)
    r3 = _safe_ratio(c2h2, ch4)
    r4 = _safe_ratio(c2h6, c2h2)

    # --------------------------------------------------------
    # Every ratio must have at least one gas above L1.
    # --------------------------------------------------------

    ratios_valid = all([
        _ratio_valid(ch4, h2, "ch4", "h2"),
        _ratio_valid(c2h2, c2h4, "c2h2", "c2h4"),
        _ratio_valid(c2h2, ch4, "c2h2", "ch4"),
        _ratio_valid(c2h6, c2h2, "c2h6", "c2h2"),
    ])

    if not ratios_valid:
        return "ABSTAIN"

    # --------------------------------------------------------
    # Evaluate complete patterns.
    # --------------------------------------------------------

    if _thermal_pattern(r1, r2, r3, r4):
        return "T3"

    if _pd_pattern(r1, r2, r3, r4):
        return "PD"

    if _arcing_pattern(r1, r2, r3, r4):
        return "D2"

    return "ABSTAIN"


# ============================================================
# DATAFRAME APPLICATION
# ============================================================

def apply_doernenburg(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["dr_r1_ch4_h2"] = df.apply(
        lambda row: _safe_ratio(
            _gas(row, "ch4"),
            _gas(row, "h2"),
        ),
        axis=1,
    )

    df["dr_r2_c2h2_c2h4"] = df.apply(
        lambda row: _safe_ratio(
            _gas(row, "c2h2"),
            _gas(row, "c2h4"),
        ),
        axis=1,
    )

    df["dr_r3_c2h2_ch4"] = df.apply(
        lambda row: _safe_ratio(
            _gas(row, "c2h2"),
            _gas(row, "ch4"),
        ),
        axis=1,
    )

    df["dr_r4_c2h6_c2h2"] = df.apply(
        lambda row: _safe_ratio(
            _gas(row, "c2h6"),
            _gas(row, "c2h2"),
        ),
        axis=1,
    )

    df["doernenburg_fault"] = df.apply(
        doernenburg_method,
        axis=1,
    )

    logger.debug("Doernenburg diagnostic applied.")

    return df