# severity.py
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ============================================================================
# IEEE Std C57.104-2019
#
# DGA Status:
#
#   Status 1:
#       Low gas levels and no indication of gassing.
#
#   Status 2:
#       Intermediate gas levels and/or possible gassing.
#
#   Status 3:
#       High gas levels and/or probable active gassing.
#
# Table 1:
#   90th percentile gas concentration.
#
# Table 2:
#   95th percentile gas concentration.
#
# Table 3:
#   95th percentile absolute change between successive lab samples.
#
# Table 4:
#   95th percentile multi-point rate, based on 3-6 samples over
#   4-24 months.
#
# The values below are the published C57.104-2019 norms.
# ============================================================================


GASES = (
    "h2",
    "ch4",
    "c2h6",
    "c2h4",
    "c2h2",
    "co",
    "co2",
)


# ============================================================================
# Table 1 - 90th percentile concentration
#
# key:
#   (o2_n2_section, age_bucket)
#
# o2_n2_section:
#   "LE_0_2"  -> O2/N2 <= 0.2
#   "GT_0_2"  -> O2/N2 > 0.2
#
# age buckets:
#   unknown
#   1_9
#   10_30
#   gt_30
# ============================================================================

TABLE_1_90TH: Dict[str, Dict[str, Dict[str, float]]] = {
    "LE_0_2": {
        "unknown": {
            "h2": 80.0,
            "ch4": 90.0,
            "c2h6": 90.0,
            "c2h4": 50.0,
            "c2h2": 1.0,
            "co": 900.0,
            "co2": 9000.0,
        },
        "1_9": {
            "h2": 75.0,
            "ch4": 45.0,
            "c2h6": 30.0,
            "c2h4": 20.0,
            "c2h2": 1.0,
            "co": 900.0,
            "co2": 5000.0,
        },
        "10_30": {
            "h2": 90.0,
            "ch4": 90.0,
            "c2h6": 90.0,
            "c2h4": 50.0,
            "c2h2": 1.0,
            "co": 900.0,
            "co2": 10000.0,
        },
        "gt_30": {
            "h2": 100.0,
            "ch4": 110.0,
            "c2h6": 150.0,
            "c2h4": 90.0,
            "c2h2": 1.0,
            "co": 900.0,
            "co2": 10000.0,
        },
    },
    "GT_0_2": {
        "unknown": {
            "h2": 40.0,
            "ch4": 20.0,
            "c2h6": 15.0,
            "c2h4": 50.0,
            "c2h2": 2.0,
            "co": 500.0,
            "co2": 5000.0,
        },
        "1_9": {
            "h2": 40.0,
            "ch4": 20.0,
            "c2h6": 15.0,
            "c2h4": 25.0,
            "c2h2": 2.0,
            "co": 500.0,
            "co2": 3500.0,
        },
        "10_30": {
            "h2": 40.0,
            "ch4": 20.0,
            "c2h6": 15.0,
            "c2h4": 60.0,
            "c2h2": 2.0,
            "co": 500.0,
            "co2": 5500.0,
        },
        "gt_30": {
            "h2": 40.0,
            "ch4": 20.0,
            "c2h6": 15.0,
            "c2h4": 60.0,
            "c2h2": 2.0,
            "co": 500.0,
            "co2": 5500.0,
        },
    },
}


# ============================================================================
# Table 2 - 95th percentile concentration
# ============================================================================

TABLE_2_95TH: Dict[str, Dict[str, Dict[str, float]]] = {
    "LE_0_2": {
        "unknown": {
            "h2": 200.0,
            "ch4": 150.0,
            "c2h6": 175.0,
            "c2h4": 100.0,
            "c2h2": 2.0,
            "co": 1100.0,
            "co2": 12500.0,
        },
        "1_9": {
            "h2": 200.0,
            "ch4": 100.0,
            "c2h6": 70.0,
            "c2h4": 40.0,
            "c2h2": 2.0,
            "co": 1100.0,
            "co2": 7000.0,
        },
        "10_30": {
            "h2": 200.0,
            "ch4": 150.0,
            "c2h6": 175.0,
            "c2h4": 95.0,
            "c2h2": 2.0,
            "co": 1100.0,
            "co2": 14000.0,
        },
        "gt_30": {
            "h2": 200.0,
            "ch4": 250.0,
            "c2h6": 175.0,
            "c2h4": 175.0,
            "c2h2": 4.0,
            "co": 1100.0,
            "co2": 14000.0,
        },
    },
    "GT_0_2": {
        "unknown": {
            "h2": 90.0,
            "ch4": 50.0,
            "c2h6": 40.0,
            "c2h4": 100.0,
            "c2h2": 7.0,
            "co": 600.0,
            "co2": 7000.0,
        },
        "1_9": {
            "h2": 90.0,
            "ch4": 60.0,
            "c2h6": 30.0,
            "c2h4": 80.0,
            "c2h2": 7.0,
            "co": 600.0,
            "co2": 5000.0,
        },
        "10_30": {
            "h2": 90.0,
            "ch4": 60.0,
            "c2h6": 40.0,
            "c2h4": 125.0,
            "c2h2": 7.0,
            "co": 600.0,
            "co2": 8000.0,
        },
        "gt_30": {
            "h2": 90.0,
            "ch4": 30.0,
            "c2h6": 40.0,
            "c2h4": 125.0,
            "c2h2": 7.0,
            "co": 600.0,
            "co2": 8000.0,
        },
    },
}


# ============================================================================
# Table 3 - 95th percentile absolute delta
#
# "Any increase" for acetylene means any positive increase.
# ============================================================================

TABLE_3_DELTA_95TH: Dict[str, Dict[str, float | None]] = {
    "LE_0_2": {
        "h2": 40.0,
        "ch4": 30.0,
        "c2h6": 25.0,
        "c2h4": 20.0,
        "c2h2": None,
        "co": 250.0,
        "co2": 2500.0,
    },
    "GT_0_2": {
        "h2": 25.0,
        "ch4": 10.0,
        "c2h6": 7.0,
        "c2h4": 20.0,
        "c2h2": None,
        "co": 175.0,
        "co2": 1750.0,
    },
}


# ============================================================================
# Table 4 - 95th percentile multi-point rate, ppm/year
#
# Period:
#   4-9 months
#   10-24 months
#
# C2H2: any increasing rate.
# ============================================================================

TABLE_4_RATE_95TH: Dict[str, Dict[str, Dict[str, float | None]]] = {
    "LE_0_2": {
        "4_9": {
            "h2": 25.0,
            "ch4": 4.0,
            "c2h6": 3.0,
            "c2h4": 7.0,
            "c2h2": None,
            "co": 100.0,
            "co2": 1000.0,
        },
        "10_24": {
            "h2": 10.0,
            "ch4": 3.0,
            "c2h6": 2.0,
            "c2h4": 5.0,
            "c2h2": None,
            "co": 80.0,
            "co2": 800.0,
        },
    },
    "GT_0_2": {
        "4_9": {
            "h2": 50.0,
            "ch4": 15.0,
            "c2h6": 15.0,
            "c2h4": 10.0,
            "c2h2": None,
            "co": 200.0,
            "co2": 1750.0,
        },
        "10_24": {
            "h2": 20.0,
            "ch4": 10.0,
            "c2h6": 9.0,
            "c2h4": 7.0,
            "c2h2": None,
            "co": 100.0,
            "co2": 1000.0,
        },
    },
}


@dataclass
class DGASeverityResult:
    status: int
    status_label: str
    status_reason: str
    o2_n2_ratio: float
    age_years: float
    table_section: str
    age_bucket: str
    level_exceeds_table1: Tuple[str, ...]
    level_exceeds_table2: Tuple[str, ...]
    delta_exceeds_table3: Tuple[str, ...]
    rate_exceeds_table4: Tuple[str, ...]
    confirmation_required: bool
    extreme_flag: bool
    severity_score: float


# ============================================================================
# Helpers
# ============================================================================

def _safe_float(value) -> float:
    try:
        value = float(value)
    except (TypeError, ValueError):
        return np.nan

    if not np.isfinite(value):
        return np.nan

    return value


def _safe_gas(row: pd.Series, gas: str) -> float:
    value = _safe_float(row.get(gas, np.nan))

    if np.isnan(value) or value < 0:
        return np.nan

    return value


def _get_age_years(row: pd.Series) -> float:
    candidates = [
        row.get("transformer_age_years", np.nan),
        row.get("age_years", np.nan),
    ]

    for value in candidates:
        value = _safe_float(value)
        if np.isfinite(value) and value >= 0:
            return value

    sample_year = _safe_float(row.get("sample_year", np.nan))
    energized_year = _safe_float(
        row.get("year_energized", np.nan)
    )

    if (
        np.isfinite(sample_year)
        and np.isfinite(energized_year)
        and sample_year >= energized_year
    ):
        return sample_year - energized_year

    tested_year = _safe_float(
        row.get("tested_year", np.nan)
    )

    if (
        np.isfinite(tested_year)
        and np.isfinite(energized_year)
        and tested_year >= energized_year
    ):
        return tested_year - energized_year

    return np.nan


def _get_o2_n2_ratio(row: pd.Series) -> float:
    value = _safe_float(
        row.get("o2_n2_ratio", np.nan)
    )

    if np.isfinite(value) and value >= 0:
        return value

    o2 = _safe_float(row.get("o2", np.nan))
    n2 = _safe_float(row.get("n2", np.nan))

    if (
        np.isfinite(o2)
        and np.isfinite(n2)
        and n2 > 0
        and o2 >= 0
    ):
        return o2 / n2

    return np.nan


def _o2_n2_section(ratio: float) -> str:
    """
    IEEE:
        if O2/N2 unavailable -> use >0.2 section
        if close to 0.2 -> >0.2 section recommended
    """
    if not np.isfinite(ratio):
        return "GT_0_2"

    if ratio <= 0.2:
        return "LE_0_2"

    return "GT_0_2"


def _age_bucket(age_years: float) -> str:
    if not np.isfinite(age_years):
        return "unknown"

    if age_years < 10:
        return "1_9"

    if age_years <= 30:
        return "10_30"

    return "gt_30"


def _rate_bucket(span_months: float) -> Optional[str]:
    if not np.isfinite(span_months):
        return None

    if 4.0 <= span_months <= 9.0:
        return "4_9"

    if 10.0 <= span_months <= 24.0:
        return "10_24"

    return None


def _finite_positive_rate(value: float) -> bool:
    return np.isfinite(value) and value > 0


# ============================================================================
# Current-row deltas / rates
# ============================================================================

def _get_delta(
    row: pd.Series,
    gas: str,
) -> float:
    """
    Prefer already engineered delta columns.
    """
    for name in (
        f"{gas}_delta",
        f"{gas}_delta_ppm",
        f"delta_{gas}",
    ):
        value = _safe_float(row.get(name, np.nan))
        if np.isfinite(value):
            return value

    return np.nan


def _get_rate_per_year(
    row: pd.Series,
    gas: str,
) -> float:
    """
    Prefer a multi-point rate already computed by feature engineering.

    If only ppm/day is present, convert it to ppm/year.
    This fallback is less preferred than a true 3-6 point regression.
    """

    for name in (
        f"{gas}_rate_per_year",
        f"{gas}_rate_ppm_per_year",
        f"rate_{gas}_per_year",
    ):
        value = _safe_float(row.get(name, np.nan))
        if np.isfinite(value):
            return value

    rate_day = _safe_float(
        row.get(
            f"{gas}_rate_per_day",
            np.nan,
        )
    )

    if np.isfinite(rate_day):
        return rate_day * 365.25

    return np.nan


# ============================================================================
# IEEE status calculation
# ============================================================================

def calculate_ieee_status(
    row: pd.Series,
) -> DGASeverityResult:

    ratio = _get_o2_n2_ratio(row)
    section = _o2_n2_section(ratio)

    age_years = _get_age_years(row)
    age_bucket = _age_bucket(age_years)

    table1 = TABLE_1_90TH[section][age_bucket]
    table2 = TABLE_2_95TH[section][age_bucket]
    table3 = TABLE_3_DELTA_95TH[section]

    gas_values = {
        gas: _safe_gas(row, gas)
        for gas in GASES
    }

    level_exceeds_t1 = []
    level_exceeds_t2 = []
    delta_exceeds_t3 = []
    rate_exceeds_t4 = []

    # --------------------------------------------------------
    # Gas level checks
    # --------------------------------------------------------

    for gas, value in gas_values.items():

        if not np.isfinite(value):
            continue

        if value > table1[gas]:
            level_exceeds_t1.append(gas)

        if value > table2[gas]:
            level_exceeds_t2.append(gas)

    # --------------------------------------------------------
    # Delta checks
    # --------------------------------------------------------

    for gas in GASES:

        delta = _get_delta(
            row,
            gas,
        )

        if not np.isfinite(delta):
            continue

        if gas == "c2h2":
            if delta > 0:
                delta_exceeds_t3.append(gas)
            continue

        threshold = table3[gas]

        if threshold is not None and delta > threshold:
            delta_exceeds_t3.append(gas)

    # --------------------------------------------------------
    # Rate checks
    #
    # Only use multi-point rate when its span is available.
    # Otherwise, a daily rate converted to annual rate is used
    # only as a fallback.
    # --------------------------------------------------------

    rate_span_months = _safe_float(
        row.get(
            "rate_span_months",
            np.nan,
        )
    )

    if not np.isfinite(rate_span_months):
        rate_span_days = _safe_float(
            row.get(
                "rate_span_days",
                np.nan,
            )
        )

        if np.isfinite(rate_span_days):
            rate_span_months = (
                rate_span_days / 30.4375
            )

    rate_bucket = _rate_bucket(
        rate_span_months
    )

    if rate_bucket is not None:

        table4 = TABLE_4_RATE_95TH[
            section
        ][rate_bucket]

        for gas in GASES:

            rate = _get_rate_per_year(
                row,
                gas,
            )

            if not np.isfinite(rate):
                continue

            if gas == "c2h2":
                if rate > 0:
                    rate_exceeds_t4.append(gas)
                continue

            threshold = table4[gas]

            if (
                threshold is not None
                and rate > threshold
            ):
                rate_exceeds_t4.append(gas)

    # --------------------------------------------------------
    # Status decision
    #
    # Status 3:
    #   any level > Table 2
    #   OR any rate > Table 4
    #
    # Status 2:
    #   any level between Table 1 and Table 2
    #   OR a delta above Table 3 requiring confirmation.
    #
    # Status 1:
    #   all levels below Table 1
    #   and no significant delta/rate.
    #
    # A delta-only result is conservatively represented as
    # Status 2 with confirmation_required=True, because the
    # IEEE flowchart requires confirmation before finalizing
    # the subsequent status.
    # --------------------------------------------------------

    if level_exceeds_t2 or rate_exceeds_t4:
        status = 3
        reason_parts = []

        if level_exceeds_t2:
            reason_parts.append(
                "gas_above_table2"
            )

        if rate_exceeds_t4:
            reason_parts.append(
                "rate_above_table4"
            )

        confirmation_required = False

    elif level_exceeds_t1 or delta_exceeds_t3:
        status = 2
        reason_parts = []

        if level_exceeds_t1:
            reason_parts.append(
                "gas_above_table1"
            )

        if delta_exceeds_t3:
            reason_parts.append(
                "delta_above_table3"
            )

        confirmation_required = bool(
            delta_exceeds_t3
        )

    else:
        status = 1
        reason_parts = [
            "gas_below_table1",
            "no_significant_gassing",
        ]

        confirmation_required = False

    # --------------------------------------------------------
    # Extreme DGA
    #
    # IEEE gives examples such as C2H4 delta around 200 ppm or
    # C2H6 around 1000 ppm as extreme conditions.
    # --------------------------------------------------------

    extreme_flag = False

    c2h4 = gas_values.get("c2h4", np.nan)
    c2h6 = gas_values.get("c2h6", np.nan)

    if (
        np.isfinite(c2h4)
        and c2h4 >= 200.0
    ):
        extreme_flag = True

    if (
        np.isfinite(c2h6)
        and c2h6 >= 1000.0
    ):
        extreme_flag = True

    if extreme_flag:
        reason_parts.append(
            "extreme_level"
        )
        status = 3

    # --------------------------------------------------------
    # Application score
    #
    # This is NOT an IEEE quantity.
    # It is an internal 0-100 prioritization score.
    #
    # It deliberately uses DGA Status as the primary factor.
    # Anomaly is not mixed into DGA status.
    # --------------------------------------------------------

    base_score = {
        1: 20.0,
        2: 55.0,
        3: 85.0,
    }[status]

    if extreme_flag:
        base_score = 100.0

    # Slight deterministic escalation when multiple independent
    # indicators are present.
    indicator_count = (
        len(level_exceeds_t2)
        + len(rate_exceeds_t4)
        + len(delta_exceeds_t3)
    )

    if status == 2:
        base_score = min(
            80.0,
            base_score + 5.0 * max(
                0,
                indicator_count - 1,
            ),
        )

    if status == 3:
        base_score = min(
            100.0,
            base_score + 3.0 * max(
                0,
                indicator_count - 1,
            ),
        )

    score = float(
        np.clip(
            base_score,
            0.0,
            100.0,
        )
    )

    status_label = {
        1: "NORMAL",
        2: "WATCHLIST",
        3: "CRITICAL",
    }[status]

    return DGASeverityResult(
        status=status,
        status_label=status_label,
        status_reason=";".join(reason_parts),
        o2_n2_ratio=ratio,
        age_years=age_years,
        table_section=section,
        age_bucket=age_bucket,
        level_exceeds_table1=tuple(
            level_exceeds_t1
        ),
        level_exceeds_table2=tuple(
            level_exceeds_t2
        ),
        delta_exceeds_table3=tuple(
            delta_exceeds_t3
        ),
        rate_exceeds_table4=tuple(
            rate_exceeds_t4
        ),
        confirmation_required=confirmation_required,
        extreme_flag=extreme_flag,
        severity_score=score,
    )


# ============================================================================
# Public API
# ============================================================================

def compute_gas_level_score(row: pd.Series) -> int:
    """
    Backward-compatible coarse gas-level score.

    0:
        all gases below Table 1
    1:
        at least one gas above Table 1
    2:
        at least one gas above Table 2
    3:
        extreme condition
    """

    result = calculate_ieee_status(row)

    if result.extreme_flag:
        return 3

    if result.level_exceeds_table2:
        return 2

    if result.level_exceeds_table1:
        return 1

    return 0


def compute_trend_score(row: pd.Series) -> int:
    """
    Backward-compatible trend score.

    This no longer uses arbitrary ppm/day thresholds.

    The score is based on IEEE Table 3/4 exceedances:

        0 -> no significant delta/rate
        1 -> Table 3 delta exceeded
        2 -> Table 4 rate exceeded
        3 -> both significant delta and rate
    """

    result = calculate_ieee_status(row)

    delta_flag = bool(
        result.delta_exceeds_table3
    )

    rate_flag = bool(
        result.rate_exceeds_table4
    )

    if delta_flag and rate_flag:
        return 3

    if rate_flag:
        return 2

    if delta_flag:
        return 1

    return 0


def apply_severity(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Apply IEEE C57.104-2019 DGA status to each row.

    This function deliberately separates:

        ieee_dga_status
        severity_score
        anomaly_percentile

    DGA Status is NOT computed from anomaly score.

    Output status:

        1 -> NORMAL
        2 -> WATCHLIST
        3 -> CRITICAL

    The score is an internal 0-100 prioritization score and should
    not be represented as an IEEE quantity.
    """

    logger.info(
        "Calculating IEEE C57.104-2019 DGA status..."
    )

    df = df.copy()

    results = [
        calculate_ieee_status(row)
        for _, row in df.iterrows()
    ]

    df["ieee_dga_status"] = [
        result.status
        for result in results
    ]

    df["ieee_dga_status_label"] = [
        result.status_label
        for result in results
    ]

    df["ieee_dga_status_reason"] = [
        result.status_reason
        for result in results
    ]

    df["o2_n2_ratio"] = [
        result.o2_n2_ratio
        for result in results
    ]

    df["transformer_age_years"] = [
        result.age_years
        for result in results
    ]

    df["ieee_norm_section"] = [
        result.table_section
        for result in results
    ]

    df["ieee_norm_age_bucket"] = [
        result.age_bucket
        for result in results
    ]

    df["ieee_table1_exceeding_gases"] = [
        list(result.level_exceeds_table1)
        for result in results
    ]

    df["ieee_table2_exceeding_gases"] = [
        list(result.level_exceeds_table2)
        for result in results
    ]

    df["ieee_table3_exceeding_gases"] = [
        list(result.delta_exceeds_table3)
        for result in results
    ]

    df["ieee_table4_exceeding_gases"] = [
        list(result.rate_exceeds_table4)
        for result in results
    ]

    df["ieee_confirmation_required"] = [
        result.confirmation_required
        for result in results
    ]

    df["ieee_extreme_dga"] = [
        result.extreme_flag
        for result in results
    ]

    df["severity_gas_score"] = [
        compute_gas_level_score(row)
        for _, row in df.iterrows()
    ]

    df["severity_trend_score"] = [
        compute_trend_score(row)
        for _, row in df.iterrows()
    ]

    df["severity_score"] = [
        result.severity_score
        for result in results
    ]

    df["severity_label"] = [
        {
            1: 0,
            2: 1,
            3: 2,
        }[result.status]
        for result in results
    ]

    # Human-readable label preserved separately.
    df["severity_label_text"] = [
        result.status_label
        for result in results
    ]

    # --------------------------------------------------------
    # Anomaly remains an independent signal.
    # It is NOT injected into IEEE DGA Status.
    # --------------------------------------------------------

    if "anomaly_percentile" in df.columns:
        df["severity_anomaly_score"] = (
            pd.to_numeric(
                df["anomaly_percentile"],
                errors="coerce",
            )
            .clip(
                lower=0.0,
                upper=1.0,
            )
            .fillna(0.0)
        )
    else:
        df["severity_anomaly_score"] = 0.0

    logger.info(
        "IEEE DGA Status distribution:\n%s",
        df["ieee_dga_status_label"]
        .value_counts(dropna=False)
        .to_string(),
    )

    logger.info(
        "Severity score range: %.1f - %.1f",
        df["severity_score"].min(),
        df["severity_score"].max(),
    )

    return df