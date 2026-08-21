# dga/iec60599.py
from __future__ import annotations

import logging

from attrs import field
from typing import List
import numpy as np
import pandas as pd

from config import config as cfg
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
logger = logging.getLogger(__name__)


# ============================================================
# SAFE INPUT
# ============================================================

def _gas(row: pd.Series, name: str) -> float:
    """
    Read one DGA gas safely.

    Missing / invalid / non-finite / negative values are treated
    as NaN rather than zero.

    This is important because an absent gas measurement must not
    silently become a measured 0 ppm value.
    """
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
    Calculate a DGA ratio safely.

    Returns:
        NaN  -> invalid/missing or 0/0
        +inf -> positive numerator / zero denominator
        ratio -> otherwise
    """
    if pd.isna(num) or pd.isna(den):
        return np.nan

    if den == 0:
        if num > 0:
            return np.inf

        return np.nan

    return float(num) / float(den)


# ============================================================
# IEC 60599:2022 APPLICABILITY
# ============================================================

def _applicable(row: pd.Series) -> bool:
    """
    Conservative applicability gate.

    The IEC ratio method should not be forced onto a sample with
    negligible gas levels.

    We require at least one characteristic gas to reach the
    configured L1 screening level.

    L1 values are from the IEEE C57.104-2019 configuration and
    are used here only as a conservative minimum-gas gate.

    The gate itself is NOT claimed to be the complete IEC
    60599:2022 condition-assessment procedure.
    """

    for gas_name in DIAGNOSTIC_GASES:
        value = _gas(row, gas_name)

        if (
            np.isfinite(value)
            and value >= L1_LIMITS[gas_name]
        ):
            return True

    return False


# ============================================================
# IEC RATIO METHOD
# ============================================================

def iec_ratio_method(row: pd.Series) -> str:
    """
    IEC 60599:2022 three-ratio diagnostic method.

    Ratios:

        R1 = C2H2 / C2H4
        R2 = CH4 / H2
        R3 = C2H4 / C2H6

    IEC 60599:2022 interpretation:

        PD:
            R1 = NS
            R2 < 0.1
            R3 < 0.2

        D1:
            R1 > 1
            R2 = 0.1 ... 0.5
            R3 > 1

        D2:
            R1 = 0.6 ... 2.5
            R2 = 0.1 ... 1
            R3 > 2

        T1:
            R1 = NS
            R2 > 1
            R3 < 1

        T2:
            R1 < 0.1
            R2 > 1
            R3 = 1 ... 4

        T3:
            R1 < 0.2
            R2 > 1
            R3 > 4

    Important:

        NS means "not significant".

        The method therefore intentionally does NOT require R1
        to be finite for PD and T1.

        Failure to match a diagnostic zone returns ABSTAIN,
        not NORMAL.
    """

    if not _applicable(row):
        return "ABSTAIN"

    h2 = _gas(row, "h2")
    ch4 = _gas(row, "ch4")
    c2h2 = _gas(row, "c2h2")
    c2h4 = _gas(row, "c2h4")
    c2h6 = _gas(row, "c2h6")

    values = [
        h2,
        ch4,
        c2h2,
        c2h4,
        c2h6,
    ]

    if not all(np.isfinite(value) for value in values):
        return "ABSTAIN"

    # --------------------------------------------------------
    # IEC ratios
    # --------------------------------------------------------

    r1 = _safe_ratio(c2h2, c2h4)
    r2 = _safe_ratio(ch4, h2)
    r3 = _safe_ratio(c2h4, c2h6)

    # --------------------------------------------------------
    # PD
    #
    # R1 = NS
    # R2 < 0.1
    # R3 < 0.2
    #
    # R1 is deliberately ignored.
    # --------------------------------------------------------

    if (
        np.isfinite(r2)
        and np.isfinite(r3)
        and r2 < 0.1
        and r3 < 0.2
    ):
        return "PD"

    # --------------------------------------------------------
    # D1
    #
    # R1 > 1
    # R2 = 0.1 ... 0.5
    # R3 > 1
    # --------------------------------------------------------

    if (
        np.isfinite(r1)
        and np.isfinite(r2)
        and np.isfinite(r3)
        and r1 > 1.0
        and 0.1 <= r2 <= 0.5
        and r3 > 1.0
    ):
        return "D1"

    # --------------------------------------------------------
    # D2
    #
    # R1 = 0.6 ... 2.5
    # R2 = 0.1 ... 1.0
    # R3 > 2
    # --------------------------------------------------------

    if (
        np.isfinite(r1)
        and np.isfinite(r2)
        and np.isfinite(r3)
        and 0.6 <= r1 <= 2.5
        and 0.1 <= r2 <= 1.0
        and r3 > 2.0
    ):
        return "D2"

    # --------------------------------------------------------
    # T1
    #
    # R1 = NS
    # R2 > 1
    # R3 < 1
    #
    # R1 is intentionally ignored.
    # --------------------------------------------------------

    if (
        np.isfinite(r2)
        and np.isfinite(r3)
        and r2 > 1.0
        and r3 < 1.0
    ):
        return "T1"

    # --------------------------------------------------------
    # T2
    #
    # R1 < 0.1
    # R2 > 1
    # R3 = 1 ... 4
    # --------------------------------------------------------

    if (
        np.isfinite(r1)
        and np.isfinite(r2)
        and np.isfinite(r3)
        and r1 < 0.1
        and r2 > 1.0
        and 1.0 <= r3 <= 4.0
    ):
        return "T2"

    # --------------------------------------------------------
    # T3
    #
    # R1 < 0.2
    # R2 > 1
    # R3 > 4
    # --------------------------------------------------------

    if (
        np.isfinite(r1)
        and np.isfinite(r2)
        and np.isfinite(r3)
        and r1 < 0.2
        and r2 > 1.0
        and r3 > 4.0
    ):
        return "T3"

    # --------------------------------------------------------
    # Unclassified / inconclusive
    # --------------------------------------------------------

    return "ABSTAIN"


# ============================================================
# DATAFRAME APPLICATION
# ============================================================

def apply_iec(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply IEC 60599:2022 ratio diagnostics.

    Output:

        iec_r1_c2h2_c2h4
        iec_r2_ch4_h2
        iec_r3_c2h4_c2h6
        iec_fault
    """

    df = df.copy()

    df["iec_r1_c2h2_c2h4"] = df.apply(
        lambda row: _safe_ratio(
            _gas(row, "c2h2"),
            _gas(row, "c2h4"),
        ),
        axis=1,
    )

    df["iec_r2_ch4_h2"] = df.apply(
        lambda row: _safe_ratio(
            _gas(row, "ch4"),
            _gas(row, "h2"),
        ),
        axis=1,
    )

    df["iec_r3_c2h4_c2h6"] = df.apply(
        lambda row: _safe_ratio(
            _gas(row, "c2h4"),
            _gas(row, "c2h6"),
        ),
        axis=1,
    )

    df["iec_fault"] = df.apply(
        iec_ratio_method,
        axis=1,
    )

    logger.debug(
        "IEC 60599:2022 ratio diagnostic applied."
    )

    if logger.isEnabledFor(logging.DEBUG):

        cols = [
            "h2",
            "ch4",
            "c2h2",
            "c2h4",
            "c2h6",
            "iec_r1_c2h2_c2h4",
            "iec_r2_ch4_h2",
            "iec_r3_c2h4_c2h6",
            "iec_fault",
        ]

        available_cols = [
            col
            for col in cols
            if col in df.columns
        ]

        logger.debug(
            "Sample IEC results:\n%s",
            df[available_cols]
            .head(5)
            .to_string(),
        )

    return df