# ranking.py
from __future__ import annotations

import ast
import logging
import numpy as np
import pandas as pd

from config import config as cfg
from consensus import normalize_fault, unify_fault

logger = logging.getLogger(__name__)


def _to_float(value):
    try:
        x = float(value)
    except (TypeError, ValueError):
        return np.nan
    return x if np.isfinite(x) else np.nan


def _status_ordinal(value):
    if isinstance(value, str):
        return int(cfg.SEVERITY_ORDER.get(value.strip().upper(), 0))
    x = _to_float(value)
    return int(np.clip(round(x), 0, 3)) if np.isfinite(x) else 0


def _parse_list(value):
    if isinstance(value, (list, tuple, set)):
        return list(value)
    if value is None:
        return []
    if isinstance(value, float) and np.isnan(value):
        return []
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return []
        try:
            parsed = ast.literal_eval(text)
            if isinstance(parsed, (list, tuple, set)):
                return list(parsed)
        except Exception:
            pass
        return [x.strip() for x in text.split(",") if x.strip()]
    return []


def classify_fault_criticality(label):
    fault = normalize_fault(label)
    return cfg.FAULT_CRITICALITY_CONTEXT.get(fault, "UNKNOWN")


def fault_criticality_source():
    return cfg.FAULT_CRITICALITY_SOURCE


def _fault_sequence(group):
    if "final_fault" in group.columns:
        source = group["final_fault"]
    elif "weak_fine_fault" in group.columns:
        source = group["weak_fine_fault"]
    elif "consensus_fault" in group.columns:
        source = group["consensus_fault"]
    else:
        return []

    out = []
    for value in source:
        fault = normalize_fault(value)
        if fault not in {"ABSTAIN", "NORMAL"}:
            out.append(fault)
    return out


def _coarse_fault_sequence(group):
    return [
        coarse
        for coarse in (unify_fault(x) for x in _fault_sequence(group))
        if coarse not in {"ABSTAIN", "NORMAL"}
    ]


def _dominant_value(values):
    if not values:
        return "ABSTAIN"
    counts = pd.Series(values).value_counts()
    top = counts[counts == counts.max()].index.tolist()
    return "MIXED" if len(top) > 1 else str(top[0])


def _entropy(values):
    if not values:
        return np.nan
    p = pd.Series(values).value_counts().to_numpy(dtype=float)
    p /= p.sum()
    return float(-np.sum(p * np.log(np.clip(p, 1e-12, None))))


def _run_length(values):
    if not values:
        return 0
    current = values[-1]
    length = 0
    for value in reversed(values):
        if value != current:
            break
        length += 1
    return length


def _transition_stats(statuses):
    if len(statuses) < 2:
        return {
            "worsening_count": 0,
            "improving_count": 0,
            "stable_count": 0,
            "observed_transition_count": 0,
            "worsening_ratio": np.nan,
            "improving_ratio": np.nan,
        }

    worsening = improving = stable = 0
    for previous, current in zip(statuses[:-1], statuses[1:]):
        if previous == 0 or current == 0:
            continue
        if current > previous:
            worsening += 1
        elif current < previous:
            improving += 1
        else:
            stable += 1

    total = worsening + improving + stable
    return {
        "worsening_count": worsening,
        "improving_count": improving,
        "stable_count": stable,
        "observed_transition_count": total,
        "worsening_ratio": worsening / total if total else np.nan,
        "improving_ratio": improving / total if total else np.nan,
    }


def _history_fault_stats(group):
    fine = _fault_sequence(group)
    coarse = _coarse_fault_sequence(group)
    counts = pd.Series(coarse).value_counts() if coarse else pd.Series(dtype=int)
    dominant = int(counts.max()) if len(counts) else 0
    dominant_fraction = dominant / len(coarse) if coarse else np.nan

    latest = group.iloc[-1]
    latest_fault = normalize_fault(
        latest.get("final_fault", latest.get("weak_fine_fault", "ABSTAIN"))
    )
    latest_group = unify_fault(latest_fault)

    current_occurrences = (
        sum(x == latest_group for x in coarse)
        if latest_group not in {"ABSTAIN", "NORMAL"}
        else 0
    )

    return {
        "history_fault_occurrence_count": len(coarse),
        "history_dominant_fault": _dominant_value(coarse),
        "history_fault_entropy": _entropy(coarse),
        "history_dominant_fault_count": dominant,
        "history_recurrent_fault_fraction": dominant_fraction,
        "history_current_fault_recurrence_fraction": (
            current_occurrences / len(coarse) if coarse else np.nan
        ),
        "history_fault_persistence": _run_length(fine),
        "current_fault": latest_fault,
        "current_fault_group": latest_group,
        "fault_criticality_class": classify_fault_criticality(latest_fault),
        "fault_criticality_source": fault_criticality_source(),
    }


def _evidence_key(row):
    """Unweighted lexicographic maintenance-order key.

    The ordering is deliberately hierarchical:
    1) current IEEE ordinal status;
    2) current standardized exceedance;
    3) number of independent IEEE trigger tables;
    4) current Table-2 exceedance count;
    5) current Table-4 rate exceedance count;
    6) current Table-3 delta exceedance count;
    7) historical maximum status;
    8) historical maximum standardized exceedance;
    9) current-fault recurrence;
    10) worsening-transition ratio.

    This is not a weighted score. No arbitrary coefficients are introduced.
    A transformer with a higher current IEEE status always precedes one with a
    lower current status. Historical evidence is used only after current status
    has been fixed.
    """
    status = int(_to_float(row.get("transformer_overall_severity_level", 0)) or 0)
    current_exceedance = _to_float(row.get("current_standardized_exceedance", 1.0))
    current_exceedance = current_exceedance if np.isfinite(current_exceedance) else 1.0

    triggers = int(_to_float(row.get("current_standard_trigger_count", 0)) or 0)
    table2 = int(_to_float(row.get("table2_exceed_count", 0)) or 0)
    table4 = int(_to_float(row.get("table4_exceed_count", 0)) or 0)
    table3 = int(_to_float(row.get("table3_exceed_count", 0)) or 0)

    historical_max_status = int(
        _to_float(row.get("history_max_status_before_current", 0)) or 0
    )
    historical_exceedance = _to_float(
        row.get("historical_max_standardized_exceedance", 1.0)
    )
    historical_exceedance = (
        historical_exceedance if np.isfinite(historical_exceedance) else 1.0
    )

    recurrence = _to_float(
        row.get("history_current_fault_recurrence_fraction", 0.0)
    )
    recurrence = recurrence if np.isfinite(recurrence) else 0.0

    worsening = _to_float(
        row.get("history_worsening_transition_ratio", 0.0)
    )
    worsening = worsening if np.isfinite(worsening) else 0.0

    return (
        status,
        current_exceedance,
        triggers,
        table2,
        table4,
        table3,
        historical_max_status,
        historical_exceedance,
        recurrence,
        worsening,
    )


def _priority_label(status):
    # This field names the IEEE condition class, not an invented maintenance
    # severity class.
    return {
        3: "STATUS_3",
        2: "STATUS_2",
        1: "STATUS_1",
        0: "INSUFFICIENT_DATA",
    }.get(int(status), "INSUFFICIENT_DATA")


def _priority_reason(status):
    if status == 3:
        return (
            "IEEE Status 3 is the current condition class. Maintenance order "
            "is determined separately by current standardized exceedance and "
            "independent IEEE trigger evidence, then by historical evidence."
        )
    if status == 2:
        return (
            "IEEE Status 2 is the current condition class. Maintenance order "
            "is determined separately within Status 2 by standardized evidence "
            "and history."
        )
    if status == 1:
        return (
            "IEEE Status 1 is the current condition class. Routine surveillance "
            "is indicated; fleet rank is still reported for comparison."
        )
    return "Required IEEE screening evidence is insufficient."


def _build_transformer_summary(transformer_id, group):
    group = group.copy()
    group["sample_day"] = pd.to_datetime(group["sample_day"], errors="coerce")
    group = group.dropna(subset=["sample_day"]).sort_values("sample_day")

    if group.empty:
        return None

    latest = group.iloc[-1]
    statuses = [_status_ordinal(x) for x in group["ieee_dga_status"].tolist()]
    current_status = statuses[-1]
    previous = statuses[:-1]

    span_days = (
        group["sample_day"].max() - group["sample_day"].min()
    ).total_seconds() / 86400.0
    record_count = len(group)
    transition = _transition_stats(statuses)

    prior_exceedance = (
        pd.to_numeric(
            group.iloc[:-1]["ieee_max_standardized_exceedance"],
            errors="coerce",
        )
        if len(group) > 1 and "ieee_max_standardized_exceedance" in group
        else pd.Series(dtype=float)
    )
    history_max_exceedance = max(
        [float(x) for x in prior_exceedance if np.isfinite(x)],
        default=1.0,
    )

    current_exceedance = _to_float(
        latest.get("ieee_max_standardized_exceedance", 1.0)
    )
    current_exceedance = current_exceedance if np.isfinite(current_exceedance) else 1.0

    current_s3_exceedance = _to_float(
        latest.get("ieee_max_status3_standardized_exceedance", 1.0)
    )
    current_s3_exceedance = (
        current_s3_exceedance if np.isfinite(current_s3_exceedance) else 1.0
    )

    t1 = _parse_list(latest.get("ieee_table1_exceeding_gases", []))
    t2 = _parse_list(latest.get("ieee_table2_exceeding_gases", []))
    t3 = _parse_list(latest.get("ieee_table3_exceeding_gases", []))
    t4 = _parse_list(latest.get("ieee_table4_exceeding_gases", []))

    table2_ratio = _to_float(
        latest.get("ieee_table2_max_exceedance_ratio", np.nan)
    )
    table4_ratio = _to_float(
        latest.get("ieee_table4_max_exceedance_ratio", np.nan)
    )

    evidence_candidates = []
    for gas in t2:
        evidence_candidates.append(
            (table2_ratio, "TABLE_2_95TH_CONCENTRATION", str(gas))
        )
    for gas in t4:
        evidence_candidates.append(
            (table4_ratio, "TABLE_4_95TH_RATE", str(gas))
        )

    if evidence_candidates:
        evidence_ratio, evidence_table, evidence_gas = max(
            evidence_candidates,
            key=lambda x: x[0] if np.isfinite(x[0]) else 1.0,
        )
    else:
        evidence_ratio = current_s3_exceedance
        evidence_table = (
            "CURRENT_STANDARD_EVIDENCE" if current_status >= 2 else "NONE"
        )
        evidence_gas = None

    fault_stats = _history_fault_stats(group)

    abnormal_count = int(sum(s >= 2 for s in statuses))
    critical_count = int(sum(s >= 3 for s in statuses))
    history_status_values = [int(x) for x in previous if int(x) > 0]

    if len(statuses) >= 2:
        history_worsening_slope = float(
            np.polyfit(
                np.arange(len(statuses)),
                np.asarray(statuses, dtype=float),
                1,
            )[0]
        )
    else:
        history_worsening_slope = np.nan

    return {
        "transformer_id": transformer_id,
        "loc": latest.get("loc"),
        "name": latest.get("name"),
        "ser": latest.get("ser"),
        "codetx": latest.get("codetx"),
        "sample_day": latest["sample_day"],

        # One standardized overall severity level: IEEE ordinal status.
        # No fabricated weighted composite is used.
        "transformer_overall_severity_level": current_status,
        "transformer_overall_severity_label": cfg.ORDINAL_TO_SEVERITY.get(
            current_status, "INSUFFICIENT_DATA"
        ),
        "transformer_overall_severity_score": float(current_status),
        "overall_score_type": "IEEE_ORDINAL_CURRENT_STATUS",
        "overall_score_formula": "current IEEE status (0..3)",
        "overall_score_is_weighted": False,
        "overall_score_is_failure_probability": False,
        "overall_score_history_is_bounded": False,

        "history_mean_status_before_current": (
            float(np.mean(previous)) if previous else np.nan
        ),
        "history_status_slope_per_observation": history_worsening_slope,

        "maintenance_priority": _priority_label(current_status),
        "maintenance_priority_ordinal": current_status,
        "maintenance_priority_reason": _priority_reason(current_status),
        "maintenance_priority_is_weighted": False,
        "maintenance_priority_score_type": (
            "UNWEIGHTED_LEXICOGRAPHIC_EVIDENCE_ORDER"
        ),

        "critical_front_flag": False,
        "critical_rule": "NOT_USED",
        "critical_reference": (
            "No additional severity class is created; IEEE Status 1/2/3 "
            "remain the condition classes."
        ),

        "critical_evidence_table": evidence_table,
        "critical_evidence_gas": evidence_gas,
        "critical_evidence_ratio": (
            float(evidence_ratio)
            if np.isfinite(evidence_ratio)
            else np.nan
        ),

        "current_fault": fault_stats["current_fault"],
        "current_fault_group": fault_stats["current_fault_group"],
        "fault_criticality_class": fault_stats["fault_criticality_class"],
        "fault_criticality_source": fault_stats["fault_criticality_source"],
        "current_fault_posterior_max": _to_float(
            latest.get(
                "weak_fine_posterior_max",
                latest.get("weak_coarse_posterior_max", np.nan),
            )
        ),
        "current_fault_entropy": _to_float(
            latest.get(
                "weak_fine_entropy",
                latest.get("weak_coarse_entropy", np.nan),
            )
        ),
        "diagnostic_agreement_ratio": _to_float(
            latest.get("diagnostic_agreement_ratio", np.nan)
        ),

        "history_record_count": record_count,
        "history_span_days": span_days,
        "history_max_status_before_current": max(previous, default=0),
        "history_nonzero_status_record_count": len(history_status_values),
        "history_abnormal_record_count": abnormal_count,
        "history_critical_record_count": critical_count,
        "history_abnormal_record_ratio": abnormal_count / record_count,
        "history_critical_record_ratio": critical_count / record_count,

        **fault_stats,

        "history_worsening_transition_count": transition["worsening_count"],
        "history_improving_transition_count": transition["improving_count"],
        "history_stable_transition_count": transition["stable_count"],
        "history_observed_transition_count": transition["observed_transition_count"],
        "history_worsening_transition_ratio": transition["worsening_ratio"],
        "history_improving_transition_ratio": transition["improving_ratio"],
        "history_has_observed_trend": record_count >= cfg.MIN_RECORDS_FOR_OBSERVED_TREND,
        "single_record_transformer": record_count == 1,
        "history_data_sufficiency_level": (
            0 if record_count == 1 else 1 if record_count == 2 else 2
        ),

        "current_standardized_exceedance": current_exceedance,
        "current_status3_standardized_exceedance": current_s3_exceedance,
        "current_delta_exceedance": int(bool(t3)),
        "current_standard_trigger_count": int(
            _to_float(latest.get("ieee_standard_trigger_count", 0)) or 0
        ),
        "historical_max_standardized_exceedance": history_max_exceedance,

        "table1_exceed_count": len(t1),
        "table2_exceed_count": len(t2),
        "table3_exceed_count": len(t3),
        "table4_exceed_count": len(t4),
        "ieee_table1_exceeding_gases": t1,
        "ieee_table2_exceeding_gases": t2,
        "ieee_table3_exceeding_gases": t3,
        "ieee_table4_exceeding_gases": t4,

        "ieee_confirmation_required": bool(
            latest.get("ieee_confirmation_required", False)
        ),
        "ieee_delta_available": bool(
            latest.get("ieee_delta_available", False)
        ),
        "ieee_rate_available": bool(
            latest.get("ieee_rate_available", False)
        ),
        "ieee_rate_span_months": _to_float(
            latest.get("ieee_rate_span_months", np.nan)
        ),
    }


def build_transformer_ranking(df):
    logger.debug("build_transformer_ranking: input rows=%d", len(df))

    required = {"transformer_id", "sample_day", "ieee_dga_status"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Missing columns for transformer ranking: {missing}")

    work = df.copy()
    work["sample_day"] = pd.to_datetime(work["sample_day"], errors="coerce")
    work = work.dropna(subset=["transformer_id", "sample_day"])

    summaries = []
    for transformer_id, group in work.sort_values(
        ["transformer_id", "sample_day"],
        kind="mergesort",
    ).groupby("transformer_id", sort=False):
        item = _build_transformer_summary(transformer_id, group)
        if item is not None:
            summaries.append(item)

    ranking = pd.DataFrame(summaries)
    if ranking.empty:
        logger.warning("build_transformer_ranking: no valid transformer summaries")
        return ranking

    ranking["_ranking_key"] = ranking.apply(_evidence_key, axis=1)
    ranking = ranking.sort_values(
        "_ranking_key",
        ascending=False,
        kind="mergesort",
    ).reset_index(drop=True)

    # Unique rank follows the complete evidence vector. Equal vectors receive
    # the same rank; this is not a weighted average.
    ranks = []
    previous_key = None
    current_rank = 0

    for position, (_, row) in enumerate(ranking.iterrows(), start=1):
        key = _evidence_key(row)
        if previous_key is None or key != previous_key:
            current_rank = position
        ranks.append(current_rank)
        previous_key = key

    ranking["rank"] = ranks
    ranking["rank_tie"] = ranking["rank"].duplicated(keep=False)
    ranking["rank_group_size"] = ranking.groupby("rank")["transformer_id"].transform("size")

    ranking["severity_rank_within_class"] = (
        ranking.groupby(
            "transformer_overall_severity_level",
            sort=False,
        ).cumcount() + 1
    )
    ranking["severity_class_size"] = ranking.groupby(
        "transformer_overall_severity_level",
        sort=False,
    )["transformer_id"].transform("size")
    ranking["severity_rank_text"] = ranking.apply(
        lambda r: (
            f"{int(r['severity_rank_within_class'])}/"
            f"{int(r['severity_class_size'])}"
        ),
        axis=1,
    )

    n = len(ranking)
    ranking["rank_percentile"] = (
        100.0 * (n - ranking["rank"]) / max(n - 1, 1)
    ).round(1)
    ranking["relative_fleet_rank_percentile"] = ranking["rank_percentile"]
    ranking["fleet_rank_percentile"] = ranking["rank_percentile"]

    # Important: this is an ordinal vector, not a scalar health score.
    ranking["priority_score"] = np.nan
    ranking["priority_score_type"] = "UNWEIGHTED_LEXICOGRAPHIC_EVIDENCE_ORDER"
    ranking["overall_score_is_weighted"] = False
    ranking["overall_score_is_failure_probability"] = False
    ranking["severity_priority_score_type"] = (
        "UNWEIGHTED_LEXICOGRAPHIC_EVIDENCE_ORDER"
    )

    ranking["severity_evidence_vector"] = ranking.apply(
        lambda row: "|".join(str(v) for v in _evidence_key(row)),
        axis=1,
    )
    ranking["ranking_policy"] = (
        "current IEEE status; current standardized exceedance; "
        "independent IEEE trigger-table count; current Table-2 count; "
        "current Table-4 count; current Table-3 count; historical maximum "
        "status; historical maximum standardized exceedance; current-fault "
        "recurrence; worsening-transition ratio"
    )
    ranking["ranking_is_weighted"] = False
    ranking["ranking_is_health_score"] = False
    ranking["ranking_uses_record_count_as_severity"] = False
    ranking["ranking_uses_fault_criticality_as_severity"] = False

    ranking["recommended_action"] = [
        _recommended_action(int(status))
        for status in ranking["transformer_overall_severity_level"]
    ]

    ranking["maintenance_priority_rank"] = ranking["rank"]
    ranking["maintenance_priority_rank_percentile"] = ranking["rank_percentile"]

    # Kept as compatibility fields. They are deliberately unused.
    ranking["pareto_dominance_count"] = 0
    ranking["pareto_front"] = False
    ranking["pareto_interpretation"] = (
        "Compatibility field only; maintenance order uses the documented "
        "unweighted lexicographic evidence vector."
    )

    ranking = ranking.drop(columns=["_ranking_key"])
    logger.debug(
        "build_transformer_ranking: final ranking rows=%d",
        len(ranking),
    )
    return ranking


def _recommended_action(status):
    if int(status) == 3:
        return "STATUS_3_RANKED_INVESTIGATION"
    if int(status) == 2:
        return "STATUS_2_CONFIRM_AND_INCREASE_SURVEILLANCE"
    if int(status) == 1:
        return "STATUS_1_ROUTINE_DGA_SURVEILLANCE"
    return "INSUFFICIENT_DATA_REVIEW"


def log_ranking_diagnostics(ranking: pd.DataFrame, top_n: int = 20):
    if ranking is None or ranking.empty:
        logger.warning("MAINTENANCE QUEUE | no transformer ranking rows")
        return

    logger.debug("=" * 120)
    logger.debug(
        "DGA FLEET CONDITION/RANKING DIAGNOSTICS | "
        "IEEE condition status is separate from maintenance order."
    )
    logger.debug(
        "IEEE STATUS COUNTS | %s",
        ranking["transformer_overall_severity_level"]
        .value_counts()
        .sort_index()
        .to_dict(),
    )

    display = ranking.head(max(int(top_n), 1))
    logger.debug("TOP %d FLEET PRIORITY", len(display))

    for _, row in display.iterrows():
        logger.debug(
            "%03d | %-20s | STATUS_%d | %.3fx | triggers=%d | "
            "T2=%d | T4=%d | T3=%d | fault=%s | action=%s",
            int(row["rank"]),
            str(row["transformer_id"])[:20],
            int(row["transformer_overall_severity_level"]),
            float(row.get("current_standardized_exceedance", 1.0)),
            int(row.get("current_standard_trigger_count", 0)),
            int(row.get("table2_exceed_count", 0)),
            int(row.get("table4_exceed_count", 0)),
            int(row.get("table3_exceed_count", 0)),
            row.get("current_fault", "ABSTAIN"),
            str(row.get("recommended_action", ""))[:60],
        )

    logger.debug("=" * 120)


__all__ = [
    "build_transformer_ranking",
    "log_ranking_diagnostics",
    "classify_fault_criticality",
    "fault_criticality_source",
]