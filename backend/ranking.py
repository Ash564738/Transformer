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
    result = x if np.isfinite(x) else np.nan
    logger.debug("_to_float: input=%r output=%r", value, result)
    return result


def _status_ordinal(value):
    if isinstance(value, str):
        result = int(cfg.SEVERITY_ORDER.get(value.strip().upper(), 0))
        logger.debug("_status_ordinal: str input=%r output=%d", value, result)
        return result
    x = _to_float(value)
    result = int(np.clip(round(x), 0, 3)) if np.isfinite(x) else 0
    logger.debug("_status_ordinal: numeric input=%r output=%d", value, result)
    return result


def _parse_list(value):
    logger.debug("_parse_list: input type=%s value=%r", type(value).__name__, value)
    if isinstance(value, (list, tuple, set)):
        result = list(value)
        logger.debug("_parse_list: direct collection -> %s", result)
        return result
    if value is None:
        logger.debug("_parse_list: None -> []")
        return []
    if isinstance(value, float) and np.isnan(value):
        logger.debug("_parse_list: NaN -> []")
        return []
    if isinstance(value, str):
        text = value.strip()
        if not text:
            logger.debug("_parse_list: empty string -> []")
            return []
        try:
            parsed = ast.literal_eval(text)
            if isinstance(parsed, (list, tuple, set)):
                result = list(parsed)
                logger.debug("_parse_list: ast literal -> %s", result)
                return result
        except Exception:
            pass
        result = [x.strip() for x in text.split(",") if x.strip()]
        logger.debug("_parse_list: comma split -> %s", result)
        return result
    logger.debug("_parse_list: fallback -> []")
    return []


def classify_fault_criticality(label):
    fault = normalize_fault(label)
    result = cfg.FAULT_CRITICALITY_CONTEXT.get(fault, "UNKNOWN")
    logger.debug("classify_fault_criticality: label=%r fault=%s result=%s", label, fault, result)
    return result


def fault_criticality_source():
    result = cfg.FAULT_CRITICALITY_SOURCE
    logger.debug("fault_criticality_source: %s", result)
    return result


def _fault_sequence(group):
    if "final_fault" in group.columns:
        source = group["final_fault"]
    elif "weak_fine_fault" in group.columns:
        source = group["weak_fine_fault"]
    elif "consensus_fault" in group.columns:
        source = group["consensus_fault"]
    else:
        logger.debug("_fault_sequence: no valid fault column found; returning []")
        return []
    out = []
    for value in source:
        fault = normalize_fault(value)
        if fault not in {"ABSTAIN", "NORMAL"}:
            out.append(fault)
    logger.debug("_fault_sequence: column=%s out=%s", source.name if hasattr(source, 'name') else 'unknown', out)
    return out


def _coarse_fault_sequence(group):
    result = [
        coarse for coarse in (unify_fault(x) for x in _fault_sequence(group))
        if coarse not in {"ABSTAIN", "NORMAL"}
    ]
    logger.debug("_coarse_fault_sequence: %s", result)
    return result


def _dominant_value(values):
    if not values:
        logger.debug("_dominant_value: empty -> ABSTAIN")
        return "ABSTAIN"
    counts = pd.Series(values).value_counts()
    top = counts[counts == counts.max()].index.tolist()
    result = "MIXED" if len(top) > 1 else str(top[0])
    logger.debug("_dominant_value: values=%s counts=%s result=%s", values, counts.to_dict(), result)
    return result


def _entropy(values):
    if not values:
        logger.debug("_entropy: empty -> NaN")
        return np.nan
    p = pd.Series(values).value_counts().to_numpy(dtype=float)
    p /= p.sum()
    result = float(-np.sum(p * np.log(np.clip(p, 1e-12, None))))
    logger.debug("_entropy: values=%s entropy=%s", values, result)
    return result


def _run_length(values):
    if not values:
        logger.debug("_run_length: empty -> 0")
        return 0
    current = values[-1]
    length = 0
    for value in reversed(values):
        if value != current:
            break
        length += 1
    logger.debug("_run_length: values=%s length=%d", values, length)
    return length


def _transition_stats(statuses):
    logger.debug("_transition_stats: input statuses=%s", statuses)
    if len(statuses) < 2:
        result = {
            "worsening_count": 0,
            "improving_count": 0,
            "stable_count": 0,
            "observed_transition_count": 0,
            "worsening_ratio": np.nan,
            "improving_ratio": np.nan,
        }
        logger.debug("_transition_stats: not enough data -> %s", result)
        return result
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
    result = {
        "worsening_count": worsening,
        "improving_count": improving,
        "stable_count": stable,
        "observed_transition_count": total,
        "worsening_ratio": worsening / total if total else np.nan,
        "improving_ratio": improving / total if total else np.nan,
    }
    logger.debug("_transition_stats: result=%s", result)
    return result


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
    stats = {
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
    logger.debug("_history_fault_stats: stats=%s", stats)
    return stats


def _evidence_key(row):
    """Explicit lexicographic ordering; no hand-assigned numeric weights."""
    key = (
        _status_ordinal(row.get("transformer_overall_severity_level", 0)),
        _to_float(row.get("current_status3_standardized_exceedance", 1.0))
        if np.isfinite(_to_float(row.get("current_status3_standardized_exceedance", 1.0)))
        else 1.0,
        int(_to_float(row.get("table2_exceed_count", 0)) or 0),
        int(_to_float(row.get("table4_exceed_count", 0)) or 0),
        int(_to_float(row.get("table3_exceed_count", 0)) or 0),
        int(_to_float(row.get("current_standard_trigger_count", 0)) or 0),
        int(_to_float(row.get("history_max_status_before_current", 0)) or 0),
        _to_float(row.get("historical_max_standardized_exceedance", 1.0))
        if np.isfinite(_to_float(row.get("historical_max_standardized_exceedance", 1.0)))
        else 1.0,
        _to_float(row.get("history_current_fault_recurrence_fraction", 0.0))
        if np.isfinite(_to_float(row.get("history_current_fault_recurrence_fraction", 0.0)))
        else 0.0,
        _to_float(row.get("history_worsening_transition_ratio", 0.0))
        if np.isfinite(_to_float(row.get("history_worsening_transition_ratio", 0.0)))
        else 0.0,
    )
    logger.debug("_evidence_key: transformer_id=%s key=%s", row.get("transformer_id", "?"), key)
    return key


def _priority(status):
    result = {3: 3, 2: 2, 1: 1}.get(int(status), 0)
    logger.debug("_priority: status=%s priority=%d", status, result)
    return result


def _priority_label(priority):
    result = cfg.MAINTENANCE_PRIORITY_LABELS.get(int(priority), "DATA_REVIEW")
    logger.debug("_priority_label: priority=%s label=%s", priority, result)
    return result


def _priority_reason(status):
    if status == 3:
        reason = "IEEE Status 3; ranked by current standardized DGA evidence and historical evidence using explicit lexicographic ordering."
    elif status == 2:
        reason = "IEEE Status 2; ranked by current standardized DGA evidence and historical evidence using explicit lexicographic ordering."
    elif status == 1:
        reason = "IEEE Status 1; ranked below Status 2 and Status 3, then ordered by explicit evidence fields."
    else:
        reason = "Required IEEE screening evidence is insufficient."
    logger.debug("_priority_reason: status=%s reason=%s", status, reason)
    return reason


def _build_transformer_summary(transformer_id, group):
    logger.debug("_build_transformer_summary: transformer_id=%s input rows=%d", transformer_id, len(group))
    group = group.copy()
    group["sample_day"] = pd.to_datetime(group["sample_day"], errors="coerce")
    group = group.dropna(subset=["sample_day"]).sort_values("sample_day")
    if group.empty:
        logger.debug("_build_transformer_summary: empty after date filtering; return None")
        return None

    latest = group.iloc[-1]
    statuses = [_status_ordinal(x) for x in group["ieee_dga_status"].tolist()]
    current_status = statuses[-1]
    previous = statuses[:-1]
    transition = _transition_stats(statuses)
    span_days = (group["sample_day"].max() - group["sample_day"].min()).total_seconds() / 86400.0
    record_count = len(group)

    prior_exceedance = pd.to_numeric(
        group.iloc[:-1]["ieee_max_standardized_exceedance"], errors="coerce"
    ) if len(group) > 1 and "ieee_max_standardized_exceedance" in group else pd.Series(dtype=float)
    history_max_exceedance = max(
        [float(x) for x in prior_exceedance if np.isfinite(x)],
        default=1.0,
    )

    current_exceedance = _to_float(latest.get("ieee_max_standardized_exceedance", 1.0))
    current_exceedance = current_exceedance if np.isfinite(current_exceedance) else 1.0
    current_s3_exceedance = _to_float(latest.get("ieee_max_status3_standardized_exceedance", 1.0))
    current_s3_exceedance = current_s3_exceedance if np.isfinite(current_s3_exceedance) else 1.0

    t1 = _parse_list(latest.get("ieee_table1_exceeding_gases", []))
    t2 = _parse_list(latest.get("ieee_table2_exceeding_gases", []))
    t3 = _parse_list(latest.get("ieee_table3_exceeding_gases", []))
    t4 = _parse_list(latest.get("ieee_table4_exceeding_gases", []))

    table2_ratio = _to_float(latest.get("ieee_table2_max_exceedance_ratio", np.nan))
    table4_ratio = _to_float(latest.get("ieee_table4_max_exceedance_ratio", np.nan))
    if t2 and np.isfinite(table2_ratio):
        evidence_ratio, evidence_table = table2_ratio, "TABLE_2_95TH_CONCENTRATION"
    elif t4 and np.isfinite(table4_ratio):
        evidence_ratio, evidence_table = table4_ratio, "TABLE_4_95TH_RATE"
    else:
        evidence_ratio, evidence_table = current_s3_exceedance, "CURRENT_STANDARD_EVIDENCE"

    evidence_candidates = []
    for gas in t2:
        evidence_candidates.append((table2_ratio, "TABLE_2_95TH_CONCENTRATION", str(gas)))
    for gas in t4:
        evidence_candidates.append((table4_ratio, "TABLE_4_95TH_RATE", str(gas)))
    if evidence_candidates:
        evidence_ratio, evidence_table, evidence_gas = max(
            evidence_candidates,
            key=lambda x: x[0] if np.isfinite(x[0]) else 1.0,
        )
    else:
        evidence_gas = None

    fault_stats = _history_fault_stats(group)
    abnormal_count = int(sum(s >= 2 for s in statuses))
    critical_count = int(sum(s >= 3 for s in statuses))

    summary = {
        "transformer_id": transformer_id,
        "loc": latest.get("loc"),
        "name": latest.get("name"),
        "ser": latest.get("ser"),
        "codetx": latest.get("codetx"),
        "sample_day": latest["sample_day"],
        "transformer_overall_severity_level": current_status,
        "transformer_overall_severity_label": cfg.ORDINAL_TO_SEVERITY.get(current_status, "INSUFFICIENT_DATA"),
        "overall_score_type": "LATEST_IEEE_STATUS_WITH_HISTORY_AS_TIE_BREAK_EVIDENCE",
        "overall_score_is_weighted": False,
        "overall_score_is_failure_probability": False,
        "maintenance_priority_ordinal": _priority(current_status),
        "maintenance_priority": _priority_label(_priority(current_status)),
        "maintenance_priority_reason": _priority_reason(current_status),
        "maintenance_priority_is_weighted": False,
        "critical_front_flag": False,
        "critical_rule": "NOT_USED",
        "critical_reference": "No additional severity level is created; IEEE Status 1/2/3 remain the condition classes.",
        "critical_evidence_table": evidence_table,
        "critical_evidence_gas": evidence_gas,
        "critical_evidence_ratio": float(evidence_ratio) if np.isfinite(evidence_ratio) else np.nan,
        "current_fault": fault_stats["current_fault"],
        "current_fault_group": fault_stats["current_fault_group"],
        "fault_criticality_class": fault_stats["fault_criticality_class"],
        "fault_criticality_source": fault_stats["fault_criticality_source"],
        "current_fault_posterior_max": _to_float(latest.get("weak_fine_posterior_max", latest.get("weak_coarse_posterior_max", np.nan))),
        "current_fault_entropy": _to_float(latest.get("weak_fine_entropy", latest.get("weak_coarse_entropy", np.nan))),
        "diagnostic_agreement_ratio": _to_float(latest.get("diagnostic_agreement_ratio", np.nan)),
        "history_record_count": record_count,
        "history_span_days": span_days,
        "history_max_status_before_current": max(previous, default=0),
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
        "history_data_sufficiency_level": 0 if record_count == 1 else 1 if record_count == 2 else 2,
        "current_standardized_exceedance": current_exceedance,
        "current_status3_standardized_exceedance": current_s3_exceedance,
        "current_delta_exceedance": int(bool(t3)),
        "current_standard_trigger_count": int(_to_float(latest.get("ieee_standard_trigger_count", 0)) or 0),
        "historical_max_standardized_exceedance": history_max_exceedance,
        "table1_exceed_count": len(t1),
        "table2_exceed_count": len(t2),
        "table3_exceed_count": len(t3),
        "table4_exceed_count": len(t4),
        "ieee_table1_exceeding_gases": t1,
        "ieee_table2_exceeding_gases": t2,
        "ieee_table3_exceeding_gases": t3,
        "ieee_table4_exceeding_gases": t4,
        "ieee_confirmation_required": bool(latest.get("ieee_confirmation_required", False)),
        "ieee_delta_available": bool(latest.get("ieee_delta_available", False)),
        "ieee_rate_available": bool(latest.get("ieee_rate_available", False)),
        "ieee_rate_span_months": _to_float(latest.get("ieee_rate_span_months", np.nan)),
    }
    logger.debug("_build_transformer_summary: completed for %s with status=%d", transformer_id, current_status)
    return summary


def _ranking_key(row):
    return _evidence_key(row)


def build_transformer_ranking(df):
    logger.info("build_transformer_ranking: input rows=%d columns=%s", len(df), list(df.columns))
    required = {"transformer_id", "sample_day", "ieee_dga_status"}
    missing = sorted(required - set(df.columns))
    if missing:
        logger.error("build_transformer_ranking: missing columns %s", missing)
        raise ValueError(f"Missing columns for transformer ranking: {missing}")

    work = df.copy()
    work["sample_day"] = pd.to_datetime(work["sample_day"], errors="coerce")
    work = work.dropna(subset=["transformer_id", "sample_day"])
    logger.info("build_transformer_ranking: after cleaning rows=%d", len(work))

    summaries = []

    for transformer_id, group in work.sort_values(
        ["transformer_id", "sample_day"], kind="mergesort"
    ).groupby("transformer_id", sort=False):
        logger.debug("build_transformer_ranking: processing transformer %s with %d records", transformer_id, len(group))
        item = _build_transformer_summary(transformer_id, group)
        if item is not None:
            summaries.append(item)

    ranking = pd.DataFrame(summaries)
    logger.info("build_transformer_ranking: built %d transformer summaries", len(ranking))
    if ranking.empty:
        logger.warning("build_transformer_ranking: no valid transformer summaries; returning empty DataFrame")
        return ranking

    ranking["_ranking_key"] = ranking.apply(_ranking_key, axis=1)
    ranking = ranking.sort_values("_ranking_key", ascending=False, kind="mergesort").reset_index(drop=True)

    ranks = []
    previous = None
    current_rank = 0
    for position, (_, row) in enumerate(ranking.iterrows(), start=1):
        key = _ranking_key(row)
        if previous is None or key != previous:
            current_rank = position
        ranks.append(current_rank)
        previous = key
    ranking["rank"] = ranks
    ranking["rank_tie"] = ranking["rank"].duplicated(keep=False)
    ranking["rank_group_size"] = ranking.groupby("rank")["rank"].transform("size")

    ranking["severity_rank_within_class"] = (
        ranking.groupby("transformer_overall_severity_level", sort=False).cumcount() + 1
    )
    ranking["severity_class_size"] = ranking.groupby(
        "transformer_overall_severity_level", sort=False
    )["transformer_id"].transform("size")
    ranking["severity_rank_text"] = ranking.apply(
        lambda r: f"{int(r['severity_rank_within_class'])}/{int(r['severity_class_size'])}", axis=1
    )

    n = len(ranking)
    ranking["rank_percentile"] = (
        100.0 * (n - ranking["rank"]) / max(n - 1, 1)
    ).round(1)
    ranking["relative_fleet_rank_percentile"] = ranking["rank_percentile"]
    ranking["fleet_rank_percentile"] = ranking["rank_percentile"]
    ranking["priority_score"] = ranking["rank_percentile"]
    ranking["priority_score_type"] = "RELATIVE_FLEET_RANK_PERCENTILE_NOT_HEALTH_SCORE"

    ranking["severity_evidence_vector"] = ranking.apply(
        lambda row: "|".join(str(v) for v in _ranking_key(row)), axis=1
    )
    ranking["ranking_policy"] = "; ".join(cfg.RANKING_POLICY)
    ranking["ranking_is_weighted"] = False
    ranking["ranking_is_health_score"] = False
    ranking["ranking_uses_record_count_as_severity"] = False
    ranking["ranking_uses_fault_criticality_as_severity"] = False
    ranking["severity_priority_score_type"] = "RELATIVE_RANK_PERCENTILE_NOT_HEALTH_SCORE"
    ranking["maintenance_priority_score_type"] = "LEXICOGRAPHIC_STANDARD_EVIDENCE_NOT_WEIGHTED"
    ranking["recommended_action"] = [
        _recommended_action(int(s))
        for s in ranking["transformer_overall_severity_level"]
    ]
    ranking["maintenance_priority_rank"] = ranking["rank"]
    ranking["maintenance_priority_rank_percentile"] = ranking["rank_percentile"]
    ranking["pareto_dominance_count"] = 0
    ranking["pareto_front"] = False
    ranking["pareto_interpretation"] = "Deprecated compatibility field; Pareto frontier is not used for maintenance priority."
    ranking = ranking.drop(columns=["_ranking_key"])
    logger.info("build_transformer_ranking: final ranking rows=%d", len(ranking))
    return ranking


def _recommended_action(status):
    if int(status) == 3:
        action = "PRIORITY_1_INVESTIGATE_AND_INCREASE_SURVEILLANCE"
    elif int(status) == 2:
        action = "PRIORITY_2_CONFIRM_DGA_AND_INCREASE_SURVEILLANCE"
    elif int(status) == 1:
        action = "PRIORITY_3_ROUTINE_DGA_SURVEILLANCE"
    else:
        action = "PRIORITY_4_REVIEW_DATA_BEFORE_CONDITION_ASSESSMENT"
    logger.debug("_recommended_action: status=%d -> %s", status, action)
    return action


def log_ranking_diagnostics(ranking: pd.DataFrame, top_n: int = 20):
    if ranking is None or ranking.empty:
        logger.warning("MAINTENANCE QUEUE | no transformer ranking rows")
        return
    logger.info("=" * 120)
    logger.info("DGA MAINTENANCE PRIORITY DIAGNOSTICS")
    logger.info("Severity classes are IEEE Status 1/2/3; ranking uses explicit lexicographic evidence, no weights.")
    logger.info("TRANSFORMER STATUS COUNTS | %s", ranking["transformer_overall_severity_level"].value_counts().sort_index().to_dict())
    logger.info("MAINTENANCE PRIORITY COUNTS | %s", ranking["maintenance_priority"].value_counts().to_dict())
    display = ranking.head(max(int(top_n), 1))
    logger.info("TOP %d FLEET PRIORITY", len(display))
    for _, row in display.iterrows():
        logger.info(
            "%03d | %-20s | %-10s | S%d | %.2fx | T2=%d | T4=%d | T3=%d | %-10s | %s",
            int(row["rank"]), str(row["transformer_id"])[:20], row["maintenance_priority"],
            int(row["transformer_overall_severity_level"]), float(row.get("critical_evidence_ratio", 1.0)),
            int(row.get("table2_exceed_count", 0)), int(row.get("table4_exceed_count", 0)),
            int(row.get("table3_exceed_count", 0)), row.get("current_fault", "ABSTAIN"),
            str(row.get("recommended_action", ""))[:70],
        )
    logger.info("=" * 120)


__all__ = [
    "build_transformer_ranking",
    "log_ranking_diagnostics",
    "classify_fault_criticality",
    "fault_criticality_source",
]