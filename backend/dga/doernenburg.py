# dga/doernenburg.py

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from config import config as cfg

logger = logging.getLogger(__name__)


def _safe_ratio(num: float, den: float) -> float:
    """
    Safely calculate num / den.

    Returns NaN when either value is missing or denominator
    is zero/non-positive.
    """
    if pd.isna(num) or pd.isna(den) or den <= 0:
        return np.nan

    return float(num) / float(den)


def _gas(row: pd.Series, name: str) -> float:
    """
    Read a gas concentration safely.

    Missing / invalid / negative values are treated as zero.
    """
    value = row.get(name, 0)

    try:
        value = float(value)
    except (TypeError, ValueError):
        return 0.0

    if not np.isfinite(value) or value < 0:
        return 0.0

    return value


def doernenburg_method(row: pd.Series) -> str:
    """
    Doernenburg Ratio Method for dissolved gases in transformer oil.

    Ratios:
        R1 = CH4 / H2
        R2 = C2H2 / C2H4
        R3 = C2H2 / CH4
        R4 = C2H6 / C2H2

    IEEE C57.104 dissolved-gas L1 values:
        H2   = 100 ppm
        CH4  = 120 ppm
        C2H2 = 1 ppm
        C2H4 = 50 ppm
        C2H6 = 65 ppm

    The method returns:
        T3       -> thermal decomposition
        PD       -> corona / low-intensity partial discharge
        D2       -> arcing / high-intensity discharge
        NORMAL   -> no Doernenburg applicability evidence
        ABSTAIN  -> insufficient / inconsistent ratio evidence
    """

    l1 = cfg.L1_DOERNENBURG

    h2 = _gas(row, "h2")
    ch4 = _gas(row, "ch4")
    c2h2 = _gas(row, "c2h2")
    c2h4 = _gas(row, "c2h4")
    c2h6 = _gas(row, "c2h6")

    # ------------------------------------------------------------
    # STEP 2 — Doernenburg applicability gate
    #
    # IEEE C57.104:
    #
    # At least one of H2, CH4, C2H2 or C2H4 must exceed
    # twice its L1 value AND one of the other gases must
    # exceed its L1 value.
    #
    # For C2H2, L1 = 1 ppm, therefore the 2x threshold is
    # 2 ppm, NOT 35 ppm.
    # ------------------------------------------------------------

    primary_gases = {
        "h2": h2,
        "ch4": ch4,
        "c2h2": c2h2,
        "c2h4": c2h4,
    }

    exceeds_2x = any(
        value >= 2.0 * l1[name]
        for name, value in primary_gases.items()
    )

    # The "other" gas must exceed its normal L1 threshold.
    exceeds_l1 = any(
        value >= l1[name]
        for name, value in primary_gases.items()
    )

    if not exceeds_2x or not exceeds_l1:
        return "ABSTAIN"

    # ------------------------------------------------------------
    # STEP 3 — Ratio validity
    #
    # A ratio is considered valid when at least one gas forming
    # that ratio exceeds its L1 concentration.
    # ------------------------------------------------------------

    r1 = _safe_ratio(ch4, h2)
    r2 = _safe_ratio(c2h2, c2h4)
    r3 = _safe_ratio(c2h2, ch4)
    r4 = _safe_ratio(c2h6, c2h2)

    r1_valid = (
        ch4 >= l1["ch4"] or
        h2 >= l1["h2"]
    )

    r2_valid = (
        c2h2 >= l1["c2h2"] or
        c2h4 >= l1["c2h4"]
    )

    r3_valid = (
        c2h2 >= l1["c2h2"] or
        ch4 >= l1["ch4"]
    )

    r4_valid = (
        c2h6 >= l1["c2h6"] or
        c2h2 >= l1["c2h2"]
    )

    # If no ratio can actually be evaluated, abstain.
    if not any([
        r1_valid,
        r2_valid,
        r3_valid,
        r4_valid,
    ]):
        return "ABSTAIN"

    # ------------------------------------------------------------
    # STEP 4 / 5 — Fault classification
    #
    # Oil ratio limits from IEEE C57.104:
    #
    # Thermal:
    #   R1 > 1.0
    #   R2 < 0.75
    #   R3 < 0.3
    #   R4 > 0.4
    #
    # PD / Corona:
    #   R1 < 0.1
    #   R3 < 0.3
    #   R4 > 0.4
    #   R2 is not significant
    #
    # Arcing:
    #   0.1 < R1 < 1.0
    #   R2 > 0.75
    #   R3 > 0.3
    #   R4 < 0.4
    # ------------------------------------------------------------

    # Thermal decomposition
    if (
        r1_valid
        and r2_valid
        and r3_valid
        and r4_valid
        and np.isfinite(r1)
        and np.isfinite(r2)
        and np.isfinite(r3)
        and np.isfinite(r4)
    ):
        if (
            r1 > 1.0
            and r2 < 0.75
            and r3 < 0.3
            and r4 > 0.4
        ):
            return "T3"

    # Partial discharge / corona
    #
    # R2 is explicitly "not significant" for this diagnosis.
    if (
        r1_valid
        and r3_valid
        and r4_valid
        and np.isfinite(r1)
        and np.isfinite(r3)
        and np.isfinite(r4)
    ):
        if (
            r1 < 0.1
            and r3 < 0.3
            and r4 > 0.4
        ):
            return "PD"

    # Arcing / high-intensity discharge
    if (
        r1_valid
        and r2_valid
        and r3_valid
        and r4_valid
        and np.isfinite(r1)
        and np.isfinite(r2)
        and np.isfinite(r3)
        and np.isfinite(r4)
    ):
        if (
            0.1 < r1 < 1.0
            and r2 > 0.75
            and r3 > 0.3
            and r4 < 0.4
        ):
            return "D2"

    # ------------------------------------------------------------
    # No complete Doernenburg pattern.
    #
    # Do NOT call this NORMAL. The method simply could not
    # establish one of its diagnostic patterns.
    # ------------------------------------------------------------

    return "ABSTAIN"


def apply_doernenburg(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply Doernenburg ratios and fault classification to a DataFrame.
    """

    df = df.copy()

    df["dr_r1_ch4_h2"] = df.apply(
        lambda r: _safe_ratio(
            _gas(r, "ch4"),
            _gas(r, "h2"),
        ),
        axis=1,
    )

    df["dr_r2_c2h2_c2h4"] = df.apply(
        lambda r: _safe_ratio(
            _gas(r, "c2h2"),
            _gas(r, "c2h4"),
        ),
        axis=1,
    )

    df["dr_r3_c2h2_ch4"] = df.apply(
        lambda r: _safe_ratio(
            _gas(r, "c2h2"),
            _gas(r, "ch4"),
        ),
        axis=1,
    )

    df["dr_r4_c2h6_c2h2"] = df.apply(
        lambda r: _safe_ratio(
            _gas(r, "c2h6"),
            _gas(r, "c2h2"),
        ),
        axis=1,
    )

    df["doernenburg_fault"] = df.apply(
        doernenburg_method,
        axis=1,
    )

    logger.debug("Doernenburg fault applied.")

    if logger.isEnabledFor(logging.DEBUG):
        cols = [
            "h2",
            "ch4",
            "c2h2",
            "c2h4",
            "c2h6",
            "dr_r1_ch4_h2",
            "dr_r2_c2h2_c2h4",
            "dr_r3_c2h2_ch4",
            "dr_r4_c2h6_c2h2",
            "doernenburg_fault",
        ]

        available_cols = [
            col for col in cols if col in df.columns
        ]

        logger.debug(
            "Sample Doernenburg results:\n%s",
            df[available_cols].head(5).to_string(),
        )

    return df