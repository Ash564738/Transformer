# dga/iec60599.py

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
        NaN  -> missing / invalid input
        inf  -> positive numerator with zero denominator
        0.0  -> zero numerator and zero denominator
        num / den -> otherwise
    """
    if pd.isna(num) or pd.isna(den):
        return np.nan

    if den <= 0:
        return np.inf if num > 0 else 0.0

    return float(num) / float(den)


def _gas(row: pd.Series, name: str) -> float:
    """
    Safely read a gas concentration from a row.

    Missing, invalid, non-finite, or negative values are treated as 0.
    """
    value = row.get(name, 0)

    try:
        value = float(value)
    except (TypeError, ValueError):
        return 0.0

    if not np.isfinite(value) or value < 0:
        return 0.0

    return value


def iec_ratio_method(row: pd.Series) -> str:
    """
    IEC 60599 three-ratio method.

    Ratios:
        R1 = C2H2 / C2H4
        R2 = CH4 / H2
        R3 = C2H4 / C2H6

    IEC 60599:2022 interpretation:

        PD:
            R1 < 0.1
            R2 < 0.2
            R3 = NS

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

    If the gas concentrations do not meet the applicability
    condition, the method abstains rather than declaring NORMAL.
    """

    l1 = cfg.L1_LIMITS

    h2 = _gas(row, "h2")
    ch4 = _gas(row, "ch4")
    c2h2 = _gas(row, "c2h2")
    c2h4 = _gas(row, "c2h4")
    c2h6 = _gas(row, "c2h6")

    # ------------------------------------------------------------
    # IEC applicability gate
    #
    # Apply the ratio interpretation only when at least one
    # characteristic gas reaches its L1 concentration.
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
    # IEC 60599 ratios
    # ------------------------------------------------------------

    r1 = _safe_ratio(c2h2, c2h4)
    r2 = _safe_ratio(ch4, h2)
    r3 = _safe_ratio(c2h4, c2h6)

    # ------------------------------------------------------------
    # PD — Partial Discharges
    #
    # R1 < 0.1
    # R2 < 0.2
    # R3 = NS
    # ------------------------------------------------------------

    if (
        np.isfinite(r1)
        and np.isfinite(r2)
        and r1 < 0.1
        and r2 < 0.2
    ):
        return "PD"

    # ------------------------------------------------------------
    # D1 — Discharges of low energy
    #
    # R1 > 1
    # R2 = 0.1 ... 0.5
    # R3 > 1
    # ------------------------------------------------------------

    if (
        np.isfinite(r1)
        and np.isfinite(r2)
        and np.isfinite(r3)
        and r1 > 1.0
        and 0.1 <= r2 <= 0.5
        and r3 > 1.0
    ):
        return "D1"

    # ------------------------------------------------------------
    # D2 — Discharges of high energy
    #
    # R1 = 0.6 ... 2.5
    # R2 = 0.1 ... 1
    # R3 > 2
    # ------------------------------------------------------------

    if (
        np.isfinite(r1)
        and np.isfinite(r2)
        and np.isfinite(r3)
        and 0.6 <= r1 <= 2.5
        and 0.1 <= r2 <= 1.0
        and r3 > 2.0
    ):
        return "D2"

    # ------------------------------------------------------------
    # T1 — Thermal fault, < 300 °C
    #
    # R1 = NS
    # R2 > 1
    # R3 < 1
    #
    # R1 is intentionally NOT tested because IEC defines it
    # as non-significant for this diagnosis.
    # ------------------------------------------------------------

    if (
        np.isfinite(r2)
        and np.isfinite(r3)
        and r2 > 1.0
        and r3 < 1.0
    ):
        return "T1"

    # ------------------------------------------------------------
    # T2 — Thermal fault, 300 ... 700 °C
    #
    # R1 < 0.1
    # R2 > 1
    # R3 = 1 ... 4
    # ------------------------------------------------------------

    if (
        np.isfinite(r1)
        and np.isfinite(r2)
        and np.isfinite(r3)
        and r1 < 0.1
        and r2 > 1.0
        and 1.0 <= r3 <= 4.0
    ):
        return "T2"

    # ------------------------------------------------------------
    # T3 — Thermal fault, > 700 °C
    #
    # R1 < 0.2
    # R2 > 1
    # R3 > 4
    # ------------------------------------------------------------

    if (
        np.isfinite(r1)
        and np.isfinite(r2)
        and np.isfinite(r3)
        and r1 < 0.2
        and r2 > 1.0
        and r3 > 4.0
    ):
        return "T3"

    # ------------------------------------------------------------
    # No IEC fault zone matched.
    #
    # Do not force NORMAL. The ratio method simply did not
    # establish one of the IEC diagnostic patterns.
    # ------------------------------------------------------------

    return "ABSTAIN"


def apply_iec(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply IEC 60599 ratios and fault classification to a DataFrame.
    """

    df = df.copy()

    df["iec_r1_c2h2_c2h4"] = df.apply(
        lambda r: _safe_ratio(
            _gas(r, "c2h2"),
            _gas(r, "c2h4"),
        ),
        axis=1,
    )

    df["iec_r2_ch4_h2"] = df.apply(
        lambda r: _safe_ratio(
            _gas(r, "ch4"),
            _gas(r, "h2"),
        ),
        axis=1,
    )

    df["iec_r3_c2h4_c2h6"] = df.apply(
        lambda r: _safe_ratio(
            _gas(r, "c2h4"),
            _gas(r, "c2h6"),
        ),
        axis=1,
    )

    df["iec_fault"] = df.apply(
        iec_ratio_method,
        axis=1,
    )

    logger.debug("IEC 60599 fault applied.")

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
            col for col in cols if col in df.columns
        ]

        logger.debug(
            "Sample IEC results:\n%s",
            df[available_cols].head(5).to_string(),
        )

    return df