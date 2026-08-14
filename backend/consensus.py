# consensus.py
from __future__ import annotations

import logging
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from config import config as cfg

logger = logging.getLogger(__name__)


# ============================================================
# GAS HELPERS
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


def _safe_tdcg(row: pd.Series) -> float:
    """
    TDCG:

        H2 + CH4 + C2H6 + C2H4 + C2H2 + CO

    CO2 is excluded.

    This value is retained for reporting only and is NOT
    used as the IEEE 2019 normal/fault gate.
    """

    value = row.get("tdcg", np.nan)

    try:
        value = float(value)
    except (TypeError, ValueError):
        value = np.nan

    if np.isfinite(value) and value >= 0:
        return value

    required = [
        "h2",
        "ch4",
        "c2h6",
        "c2h4",
        "c2h2",
        "co",
    ]

    values = [
        _gas(row, gas)
        for gas in required
    ]

    if not all(np.isfinite(value) for value in values):
        return np.nan

    return float(sum(values))


# ============================================================
# NORMAL / HEALTHY GATE
# ============================================================

def is_below_l1(row: pd.Series) -> bool:
    """
    Conservative concentration screen.

    This is NOT equivalent to the complete IEEE C57.104-2019
    statistical assessment.

    It only means that all six combustible/key gases are below
    the L1 concentrations used by the Doernenburg procedure.
    """

    for gas, limit in cfg.L1_LIMITS.items():

        value = _gas(row, gas)

        if not np.isfinite(value):
            return False

        if value >= limit:
            return False

    return True


def is_normal(row: pd.Series) -> bool:
    """
    Compatibility alias.

    In this implementation "NORMAL" means:

        all six IEEE L1 gases are below their L1 values.

    It does NOT use a fixed TDCG threshold.
    """

    return is_below_l1(row)


# ============================================================
# FAULT NORMALIZATION
# ============================================================

def normalize_fault(label: str) -> str:

    if label is None:
        return "ABSTAIN"

    label = str(label).strip().upper()

    if label in {
        "",
        "ABSTAIN",
        "NA",
        "N/A",
        "NONE",
        "NAN",
    }:
        return "ABSTAIN"

    legacy_map = {
        "T3-H": "T3_H",
        "ARCING": "D2",
        "HIGH_ENERGY_DISCHARGE": "D2",
        "LOW_ENERGY_ARCING": "PD",
        "PARTIAL_DISCHARGE": "PD",
        "CORONA": "PD",
        "THERMAL": "T3",
        "NORMAL": "NORMAL",
    }

    return legacy_map.get(label, label)


def unify_fault(label: str) -> str:

    normalized = normalize_fault(label)

    if normalized == "ABSTAIN":
        return "ABSTAIN"

    return cfg.FAULT_GROUPS.get(
        normalized,
        "ABSTAIN",
    )


# ============================================================
# VOTE PROCESSING
# ============================================================

def _compute_group_weights(
    votes: Dict[str, str],
) -> Tuple[
    Dict[str, float],
    Dict[str, str],
    Dict[str, float],
    int,
]:
    """
    Return:

        group_weights
        fault_by_method
        ambiguous_weights
        active_method_count

    DT is treated as an ambiguous observation, not as two
    independent votes.
    """

    group_weights: Dict[str, float] = {}
    fault_by_method: Dict[str, str] = {}
    ambiguous_weights: Dict[str, float] = {}

    active_methods = 0

    for method, raw_fault in votes.items():

        fault = normalize_fault(raw_fault)

        if fault == "ABSTAIN":
            continue

        weight = cfg.METHOD_WEIGHTS.get(
            method,
            1.0,
        )

        if weight <= 0:
            continue

        active_methods += 1
        fault_by_method[method] = fault

        # ----------------------------------------------------
        # NORMAL is diagnostic output but not a fault vote.
        # ----------------------------------------------------

        if fault == "NORMAL":
            continue

        # ----------------------------------------------------
        # DT is explicitly mixed electrical/thermal.
        # Do not convert one DT into two ordinary votes.
        # ----------------------------------------------------

        if fault == "DT":
            ambiguous_weights["DT"] = (
                ambiguous_weights.get("DT", 0.0)
                + weight
            )
            continue

        group = unify_fault(fault)

        if group in {
            "ABSTAIN",
            "NORMAL",
            "MIXED",
        }:
            continue

        group_weights[group] = (
            group_weights.get(group, 0.0)
            + weight
        )

    return (
        group_weights,
        fault_by_method,
        ambiguous_weights,
        active_methods,
    )


# ============================================================
# CONSENSUS
# ============================================================

def aggregate_votes_with_row(
    row: pd.Series,
    votes: Dict[str, str],
) -> Tuple[str, List[str]]:

    (
        group_weights,
        fault_by_method,
        ambiguous_weights,
        active_methods,
    ) = _compute_group_weights(votes)

    # --------------------------------------------------------
    # 1. Explicit Rogers / method NORMAL results do not prove
    #    the whole transformer is normal.
    #
    #    Only return NORMAL automatically when the concentration
    #    screen is below all L1 limits and there is no positive
    #    fault evidence.
    # --------------------------------------------------------

    if is_below_l1(row) and not group_weights:
        return "NORMAL", ["NORMAL"]

    # --------------------------------------------------------
    # 2. No fault method produced a usable fault result.
    # --------------------------------------------------------

    if not group_weights:

        if is_below_l1(row):
            return "NORMAL", ["NORMAL"]

        return "ABSTAIN", []

    # --------------------------------------------------------
    # 3. Calculate fault-group distribution.
    # --------------------------------------------------------

    total_weight = sum(
        group_weights.values()
    )

    if total_weight <= 0:
        return "ABSTAIN", []

    sorted_groups = sorted(
        group_weights.items(),
        key=lambda item: item[1],
        reverse=True,
    )

    top_group, top_weight = sorted_groups[0]

    second_weight = (
        sorted_groups[1][1]
        if len(sorted_groups) > 1
        else 0.0
    )

    top_ratio = top_weight / total_weight
    second_ratio = second_weight / total_weight

    # --------------------------------------------------------
    # 4. DT itself is evidence of mixed electrical/thermal
    #    behavior.
    #
    #    If there is a DT vote plus substantial independent
    #    thermal/discharge evidence, return MIXED.
    # --------------------------------------------------------

    if ambiguous_weights.get("DT", 0.0) > 0:

        independent_groups = [
            group
            for group, weight in group_weights.items()
            if weight > 0
        ]

        if len(independent_groups) >= 2:
            return "MIXED", independent_groups

        if len(independent_groups) == 1:
            return "MIXED", [
                independent_groups[0],
                "MIXED",
            ]

        return "MIXED", ["THERMAL", "DISCHARGE"]

    # --------------------------------------------------------
    # 5. Mixed fault.
    # --------------------------------------------------------

    if (
        top_ratio < cfg.MIXED_THRESHOLD
        or second_ratio >= cfg.MIN_SECOND_GROUP_WEIGHT_RATIO
    ):
        return "MIXED", [
            group
            for group, weight in sorted_groups
            if weight > 0
        ]

    # --------------------------------------------------------
    # 6. Determine specific fault within winning group.
    #
    # Prefer the method with the highest configured weight.
    # --------------------------------------------------------

    best_fault = None
    best_weight = -1.0

    for method, fault in fault_by_method.items():

        if fault in {
            "ABSTAIN",
            "NORMAL",
            "DT",
        }:
            continue

        if unify_fault(fault) != top_group:
            continue

        method_weight = cfg.METHOD_WEIGHTS.get(
            method,
            1.0,
        )

        if method_weight > best_weight:
            best_weight = method_weight
            best_fault = fault

    if best_fault is not None:
        return best_fault, [top_group]

    return top_group, [top_group]


# ============================================================
# CONFIDENCE
# ============================================================

def confidence_with_row(
    row: pd.Series,
    votes: Dict[str, str],
) -> float:
    """
    Consensus agreement score.

    Important:
        This is NOT a calibrated statistical probability.

    It represents weighted agreement among active fault
    diagnostic methods.

    Coverage is applied so that three agreeing methods out of
    six available methods do not automatically become 100%.
    """

    (
        group_weights,
        fault_by_method,
        ambiguous_weights,
        active_methods,
    ) = _compute_group_weights(votes)

    if not group_weights:

        if is_below_l1(row):
            return 100.0

        return 0.0

    total_weight = sum(
        group_weights.values()
    )

    if total_weight <= 0:
        return 0.0

    top_group = max(
        group_weights,
        key=group_weights.get,
    )

    top_weight = group_weights[top_group]

    agreement = (
        top_weight / total_weight
    )

    # Number of methods that actually returned a usable
    # diagnostic result, including NORMAL/DT.
    total_methods = len(votes)

    coverage = (
        active_methods / total_methods
        if total_methods > 0
        else 0.0
    )

    # Agreement is the main factor.
    # Coverage prevents sparse votes from becoming falsely
    # perfect confidence.
    confidence = (
        100.0
        * agreement
        * (0.5 + 0.5 * coverage)
    )

    return round(
        float(np.clip(confidence, 0.0, 100.0)),
        1,
    )


# ============================================================
# APPLY CONSENSUS
# ============================================================

def apply_consensus(
    df: pd.DataFrame,
) -> pd.DataFrame:

    logger.info(
        "Applying IEEE C57.104-2019 DGA diagnostic modules..."
    )

    from dga import (
        keygas,
        iec60599,
        rogers,
        doernenburg,
        duval_triangle,
        duval_pentagon,
    )

    df = keygas.apply_key_gas(df)

    df = iec60599.apply_iec(df)

    df = rogers.apply_rogers(df)

    df = doernenburg.apply_doernenburg(df)

    df = duval_triangle.apply_duval_triangle(df)

    # --------------------------------------------------------
    # Only ONE Duval Pentagon is included.
    # P1 and P2 are not independent votes.
    # --------------------------------------------------------

    df = duval_pentagon.apply_duval_pentagon(
        df,
        pentagon="P2",
    )

    # --------------------------------------------------------
    # Vote sources
    # --------------------------------------------------------

    vote_columns = {
        "keygas_fault": "keygas_fault",
        "iec_fault": "iec_fault",
        "rogers_fault": "rogers_fault",
        "doernenburg_fault": "doernenburg_fault",
        "duval_triangle_fault": "duval_triangle_fault",
        "duval_pentagon_p2_fault": "duval_pentagon_p2_fault",
    }

    def make_votes(row: pd.Series) -> Dict[str, str]:
        return {
            method: normalize_fault(
                row.get(column, "ABSTAIN")
            )
            for method, column in vote_columns.items()
        }

    # --------------------------------------------------------
    # Aggregate
    # --------------------------------------------------------

    results = df.apply(
        lambda row: aggregate_votes_with_row(
            row,
            make_votes(row),
        ),
        axis=1,
    )

    df["consensus_fault"] = results.apply(
        lambda result: result[0]
    )

    df["mixed_components"] = results.apply(
        lambda result: result[1]
    )

    df["diagnostic_confidence"] = df.apply(
        lambda row: confidence_with_row(
            row,
            make_votes(row),
        ),
        axis=1,
    )

    df["diagnostic_votes"] = df.apply(
        make_votes,
        axis=1,
    )

    # --------------------------------------------------------
    # Diagnostic coverage
    # --------------------------------------------------------

    def active_vote_count(
        row: pd.Series,
    ) -> int:
        votes = make_votes(row)

        return sum(
            normalize_fault(vote) != "ABSTAIN"
            for vote in votes.values()
        )

    df["diagnostic_active_methods"] = df.apply(
        active_vote_count,
        axis=1,
    )

    df["diagnostic_method_count"] = len(
        vote_columns
    )

    df["diagnostic_coverage"] = (
        df["diagnostic_active_methods"]
        / df["diagnostic_method_count"]
        * 100.0
    ).round(1)

    # --------------------------------------------------------
    # TDCG reporting only
    # --------------------------------------------------------

    if "tdcg" not in df.columns:
        df["tdcg"] = df.apply(
            _safe_tdcg,
            axis=1,
        )

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    n_total = len(df)

    n_normal = (
        df["consensus_fault"] == "NORMAL"
    ).sum()

    n_abstain = (
        df["consensus_fault"] == "ABSTAIN"
    ).sum()

    n_mixed = (
        df["consensus_fault"] == "MIXED"
    ).sum()

    n_faults = (
        n_total
        - n_normal
        - n_abstain
        - n_mixed
    )

    avg_conf = (
        df["diagnostic_confidence"].mean()
        if n_total
        else 0.0
    )

    logger.info(
        "Consensus complete: "
        "NORMAL=%d, FAULTS=%d, MIXED=%d, "
        "ABSTAIN=%d, avg_conf=%.1f%%",
        n_normal,
        n_faults,
        n_mixed,
        n_abstain,
        avg_conf,
    )

    return df


# ============================================================
# CONSENSUS + STUDENT MODEL
# ============================================================

def combine_consensus_and_student(
    df: pd.DataFrame,
    student_fault_col: str = "student_fault",
    student_conf_col: str = "student_confidence",
    consensus_conf_threshold: float = 60.0,
) -> pd.DataFrame:
    """
    Combine consensus and student model WITHOUT overwriting
    the original consensus result.

    Output:

        consensus_fault
        diagnostic_confidence
        student_fault
        student_confidence
        final_fault
        final_fault_source

    Policy:

        Strong consensus:
            consensus != MIXED/ABSTAIN
            and confidence >= threshold
                -> consensus

        Otherwise:
            valid student prediction
                -> student

        Otherwise:
            -> consensus

    This is a fusion policy, not part of IEEE C57.104-2019.
    """

    df = df.copy()

    if student_fault_col not in df.columns:
        logger.warning(
            "Student fault column missing; "
            "final_fault will equal consensus_fault."
        )

        df["final_fault"] = df.get(
            "consensus_fault",
            "ABSTAIN",
        )

        df["final_fault_source"] = "CONSENSUS"

        return df

    final_faults: List[str] = []
    final_sources: List[str] = []

    for _, row in df.iterrows():

        consensus_fault = normalize_fault(
            row.get(
                "consensus_fault",
                "ABSTAIN",
            )
        )

        try:
            consensus_conf = float(
                row.get(
                    "diagnostic_confidence",
                    0.0,
                )
            )
        except (TypeError, ValueError):
            consensus_conf = 0.0

        student_fault = normalize_fault(
            row.get(
                student_fault_col,
                "ABSTAIN",
            )
        )

        use_consensus = (
            consensus_fault not in {
                "ABSTAIN",
                "MIXED",
            }
            and consensus_conf >= consensus_conf_threshold
        )

        if use_consensus:
            final_faults.append(
                consensus_fault
            )
            final_sources.append(
                "CONSENSUS"
            )
            continue

        if student_fault != "ABSTAIN":
            final_faults.append(
                student_fault
            )
            final_sources.append(
                "STUDENT"
            )
            continue

        final_faults.append(
            consensus_fault
        )
        final_sources.append(
            "CONSENSUS_FALLBACK"
        )

    df["final_fault"] = final_faults
    df["final_fault_source"] = final_sources

    logger.info(
        "Consensus/student fusion complete:\n%s",
        df["final_fault"].value_counts(
            dropna=False
        ).to_string(),
    )

    return df