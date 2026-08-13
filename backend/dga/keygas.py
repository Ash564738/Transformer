# dga/keygas.py

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from config import config as cfg


logger = logging.getLogger(__name__)


def _safe_gas(row: pd.Series, name: str) -> float:
    """
    Safely read a gas concentration.

    Missing, invalid, non-finite, or negative values are treated as 0.
    """
    value = row.get(name, 0.0)

    try:
        value = float(value)
    except (TypeError, ValueError):
        return 0.0

    if not np.isfinite(value) or value < 0:
        return 0.0

    return value


def _safe_tdcg(row: pd.Series, gases: dict[str, float]) -> float:
    """
    Read TDCG if available.

    If TDCG is missing, calculate it from the six combustible /
    diagnostic gases used by this implementation.
    """
    value = row.get("tdcg", np.nan)

    try:
        value = float(value)
    except (TypeError, ValueError):
        value = np.nan

    if np.isfinite(value) and value >= 0:
        return value

    return float(sum(gases.values()))


def key_gas_method(row: pd.Series) -> str:
    """
    Key Gas Method.

    This implementation uses dominant-gas patterns rather than simply
    returning the label of the largest gas.

    Canonical system mapping:

        H2 dominant              -> PD
        C2H2 dominant            -> D2
        C2H4 dominant            -> THERMAL_OIL
        CH4/C2H6 dominant        -> THERMAL_OIL
        CO dominant with
        significant CO level     -> THERMAL_CELLULOSE

    IMPORTANT:
        Key Gas is a screening / supporting diagnostic method.
        It should not override stronger ratio-based methods or
        consensus logic.

        NORMAL is returned only when the sample is below the
        configured TDCG applicability threshold.

        If gas levels are elevated but no characteristic key-gas
        pattern is sufficiently clear, return ABSTAIN.
    """

    # ------------------------------------------------------------
    # Read gases
    # ------------------------------------------------------------

    h2 = _safe_gas(row, "h2")
    ch4 = _safe_gas(row, "ch4")
    c2h6 = _safe_gas(row, "c2h6")
    c2h4 = _safe_gas(row, "c2h4")
    c2h2 = _safe_gas(row, "c2h2")
    co = _safe_gas(row, "co")
    co2 = _safe_gas(row, "co2")

    gases = {
        "h2": h2,
        "ch4": ch4,
        "c2h6": c2h6,
        "c2h4": c2h4,
        "c2h2": c2h2,
        "co": co,
    }

    tdcg = _safe_tdcg(row, gases)

    # ------------------------------------------------------------
    # Applicability gate
    # ------------------------------------------------------------

    if tdcg < cfg.MIN_TDCG:
        return "NORMAL"

    # ------------------------------------------------------------
    # No usable gas information
    # ------------------------------------------------------------

    if max(gases.values(), default=0.0) <= 0:
        return "ABSTAIN"

    # ------------------------------------------------------------
    # Key gas thresholds
    #
    # Use the configured gas thresholds where available.
    # The first threshold represents the onset of elevated gas.
    # ------------------------------------------------------------

    thresholds = cfg.SEVERITY_GAS_THRESHOLDS

    h2_l1 = thresholds["h2"][0]
    ch4_l1 = thresholds["ch4"][0]
    c2h6_l1 = thresholds["c2h6"][0]
    c2h4_l1 = thresholds["c2h4"][0]
    c2h2_l1 = thresholds["c2h2"][0]
    co_l1 = thresholds["co"][0]

    # ------------------------------------------------------------
    # Determine dominant hydrocarbon / fault gases.
    #
    # CO is handled separately because its interpretation is
    # associated with cellulose degradation and should not compete
    # directly with hydrocarbon gases through max().
    # ------------------------------------------------------------

    hydrocarbon_gases = {
        "h2": h2,
        "ch4": ch4,
        "c2h6": c2h6,
        "c2h4": c2h4,
        "c2h2": c2h2,
    }

    dominant = max(
        hydrocarbon_gases,
        key=hydrocarbon_gases.get,
    )

    dominant_value = hydrocarbon_gases[dominant]

    # ------------------------------------------------------------
    # Cellulose / paper degradation
    #
    # CO should be materially elevated before using it as a
    # cellulose indicator. CO2 is used as supporting evidence,
    # not as a standalone fault label.
    # ------------------------------------------------------------

    cellulose_signal = co >= co_l1

    if cellulose_signal:
        # Strong CO relative to hydrocarbon gases.
        hydrocarbon_max = max(hydrocarbon_gases.values(), default=0.0)

        if hydrocarbon_max <= 0:
            return "THERMAL_CELLULOSE"

        # If CO clearly dominates the combustible gas profile,
        # classify as cellulose-related thermal degradation.
        if co >= 1.5 * hydrocarbon_max:
            return "THERMAL_CELLULOSE"

        # CO and CO2 together provide additional supporting
        # evidence for cellulose involvement.
        if co2 > 0 and co2 >= 3.0 * co:
            return "THERMAL_CELLULOSE"

    # ------------------------------------------------------------
    # Acetylene -> electrical discharge / arcing
    #
    # C2H2 is the characteristic high-energy discharge indicator.
    # Do not require it to be the absolute largest gas because
    # H2 may be larger in genuine arcing cases.
    # ------------------------------------------------------------

    if c2h2 >= c2h2_l1:
        other_hydrocarbon_max = max(
            h2,
            ch4,
            c2h6,
            c2h4,
            0.0,
        )

        # Acetylene is significant relative to the hydrocarbon
        # profile. A dominant acetylene signal is strong evidence.
        if c2h2 >= other_hydrocarbon_max:
            return "D2"

        # Acetylene accompanied by substantial hydrogen is also
        # characteristic of electrical discharge.
        if h2 >= h2_l1 and c2h2 >= 0.1 * h2:
            return "D2"

    # ------------------------------------------------------------
    # Hydrogen -> partial discharge
    #
    # H2 is associated with low-energy electrical discharge,
    # but H2 alone is not enough if hydrocarbon gases strongly
    # indicate a thermal fault.
    # ------------------------------------------------------------

    if h2 >= h2_l1:
        hydrocarbon_without_h2 = max(
            ch4,
            c2h6,
            c2h4,
            c2h2,
            0.0,
        )

        if hydrocarbon_without_h2 == 0:
            return "PD"

        if h2 >= 2.0 * hydrocarbon_without_h2:
            return "PD"

        # H2 dominant with relatively low acetylene.
        if dominant == "h2" and c2h2 < c2h2_l1:
            return "PD"

    # ------------------------------------------------------------
    # Ethylene -> thermal oil fault
    #
    # C2H4 is the strongest hydrocarbon indicator for higher
    # temperature oil thermal faults.
    # ------------------------------------------------------------

    if c2h4 >= c2h4_l1:
        if c2h4 >= ch4 and c2h4 >= c2h6:
            return "THERMAL_OIL"

    # ------------------------------------------------------------
    # Methane / ethane -> lower-temperature oil thermal pattern
    # ------------------------------------------------------------

    if ch4 >= ch4_l1 or c2h6 >= c2h6_l1:
        thermal_hydrocarbons = {
            "ch4": ch4,
            "c2h6": c2h6,
            "c2h4": c2h4,
        }

        thermal_dominant = max(
            thermal_hydrocarbons,
            key=thermal_hydrocarbons.get,
        )

        if thermal_dominant in {"ch4", "c2h6"}:
            return "THERMAL_OIL"

    # ------------------------------------------------------------
    # Final fallback
    # ------------------------------------------------------------

    return "ABSTAIN"


def apply_key_gas(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply Key Gas classification to the DataFrame.
    """

    df = df.copy()

    df["keygas_fault"] = df.apply(
        key_gas_method,
        axis=1,
    )

    logger.debug("Key Gas fault applied.")

    if logger.isEnabledFor(logging.DEBUG):
        sample_cols = [
            "h2",
            "ch4",
            "c2h6",
            "c2h4",
            "c2h2",
            "co",
            "co2",
            "tdcg",
            "keygas_fault",
        ]

        available_cols = [
            col for col in sample_cols
            if col in df.columns
        ]

        logger.debug(
            "Sample Key Gas results:\n%s",
            df[available_cols].head(5).to_string(),
        )

    return df