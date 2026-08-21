# consensus.py
from __future__ import annotations
import logging
from typing import Iterable
import numpy as np
import pandas as pd
from config import config as cfg

logger = logging.getLogger(__name__)
ABSTAIN = "ABSTAIN"

def _gas(row, name):
    try: value = float(row.get(name, np.nan))
    except (TypeError, ValueError): return np.nan
    if not np.isfinite(value) or value < 0: return np.nan
    return value

def normalize_fault(label):
    if label is None: return ABSTAIN
    try:
        if pd.isna(label): return ABSTAIN
    except (TypeError, ValueError): return ABSTAIN
    text = str(label).strip().upper()
    if not text: return ABSTAIN
    text = text.replace("–","-").replace("—","-").replace("−","-")
    aliases = cfg.BENCHMARK_FAULT_ALIASES
    if text in aliases: return aliases[text]
    variants = (text.replace("-", "_"), text.replace(" ", "_"), text.replace("-", "_").replace(" ", "_"))
    for variant in variants:
        if variant in aliases: return aliases[variant]
    canonical = text.replace("-", "_").replace(" ", "_")
    if canonical in cfg.FINE_FAULT_CLASSES: return canonical
    if canonical in cfg.COARSE_FAULT_GROUPS: return canonical
    return ABSTAIN

def unify_fault(label):
    fine = normalize_fault(label)
    if fine in cfg.FAULT_GROUPS: return cfg.FAULT_GROUPS[fine]
    if fine in cfg.COARSE_FAULT_GROUPS: return fine
    return ABSTAIN

def _safe_ratio(value):
    try: value = float(value)
    except (TypeError, ValueError): return np.nan
    return value if np.isfinite(value) and value >= 0 else np.nan

def _safe_age(row):
    for column in ("transformer_age_years", "age_years"):
        value = _safe_ratio(row.get(column, np.nan))
        if np.isfinite(value):
            return value
    sample_year = _safe_ratio(row.get("sample_year", np.nan))
    energized_year = _safe_ratio(row.get("year_energized", np.nan))
    if np.isfinite(sample_year) and np.isfinite(energized_year) and sample_year >= energized_year:
        return sample_year - energized_year
    return np.nan

def _age_bucket(age):
    if not np.isfinite(age) or age <= 0:
        return "unknown"
    if age < 10:
        return "1_9"
    if age <= 30:
        return "10_30"
    return "gt_30"

def _o2_n2_ratio(row):
    for column in ("o2_n2_ratio", "o2n2_ratio"):
        value = _safe_ratio(row.get(column, np.nan))
        if np.isfinite(value):
            return value
    o2 = _gas(row, "o2")
    n2 = _gas(row, "n2")
    if np.isfinite(o2) and np.isfinite(n2) and n2 > 0:
        return o2 / n2
    return np.nan

def is_below_l1(row):
    for gas in cfg.SEVERITY_REQUIRED_GASES:
        if not np.isfinite(_gas(row, gas)):
            return False
    ratio = _o2_n2_ratio(row)
    if not np.isfinite(ratio):
        return False
    section = "LE_0_2" if ratio <= 0.2 else "GT_0_2"
    age_bucket = _age_bucket(_safe_age(row))
    limits = cfg.TABLE_1_90TH[section][age_bucket]
    return all(_gas(row, gas) < float(limit) for gas, limit in limits.items())

def diagnostic_agreement_ratio(votes, selected_group, selected_methods=None):
    methods = list(selected_methods or cfg.DIAGNOSTIC_METHODS)
    groups = []
    for method in methods:
        group = unify_fault(votes.get(method, ABSTAIN))
        if group != ABSTAIN: groups.append(group)
    if not groups:
        return 0.0
    return round(100.0 * sum(g == selected_group for g in groups) / len(groups), 1)

def _active_votes(votes, selected_methods=None):
    methods = list(selected_methods or cfg.DIAGNOSTIC_METHODS)
    out = {}
    for method in methods:
        value = normalize_fault(votes.get(method, ABSTAIN))
        if value != ABSTAIN: out[method] = value
    return out

def _group_vote_counts(votes, selected_methods=None):
    counts = {group: 0 for group in cfg.COARSE_FAULT_GROUPS}
    for value in _active_votes(votes, selected_methods).values():
        group = unify_fault(value)
        if group in counts: counts[group] += 1
    return counts

def _select_fine_fault(votes, winning_group, selected_methods=None):
    counts = {}
    for value in _active_votes(votes, selected_methods).values():
        if value in {"NORMAL", "ABSTAIN", "MIXED"}: continue
        if unify_fault(value) != winning_group: continue
        counts[value] = counts.get(value, 0) + 1
    if not counts:
        return winning_group
    ranked = sorted(counts.items(), key=lambda item: (-item[1], item[0]))
    if len(ranked) > 1 and ranked[0][1] == ranked[1][1]:
        return ABSTAIN
    return ranked[0][0]

def aggregate_votes_with_row(row, votes, selected_methods):
    active = _active_votes(votes, selected_methods)
    if not active:
        return (("NORMAL", ["NORMAL"], False) if is_below_l1(row) else (ABSTAIN, [], False))
    if all(value == "NORMAL" for value in active.values()): return "NORMAL", ["NORMAL"], False
    counts = _group_vote_counts(votes, selected_methods)
    concrete = {group: count for group, count in counts.items() if group not in {"NORMAL", "MIXED", "ABSTAIN"} and count > 0}
    mixed_count = sum(unify_fault(value) == "MIXED" for value in active.values())
    if not concrete:
        if mixed_count: return "MIXED", ["MIXED"], False
        return ("NORMAL", ["NORMAL"], False) if is_below_l1(row) else (ABSTAIN, [], False)
    ranked = sorted(concrete.items(), key=lambda item: (-item[1], item[0]))
    top_count = ranked[0][1]
    if len(ranked) > 1 and ranked[1][1] == top_count:
        components = [group for group, _ in ranked]
        if mixed_count: components.append("MIXED")
        return ABSTAIN, components, True
    winning_group = ranked[0][0]
    fine = _select_fine_fault(votes, winning_group, selected_methods)
    if fine == ABSTAIN: return winning_group, [winning_group], True
    return fine, [winning_group], False

def _apply_traditional_methods(df):
    logger.debug("_apply_traditional_methods: input shape=%s", df.shape)
    from dga import (doernenburg, duval_pentagon, duval_triangle, iec60599, keygas, rogers)
    out = df.copy()
    out = keygas.apply_key_gas(out)
    out = iec60599.apply_iec(out)
    out = rogers.apply_rogers(out)
    out = doernenburg.apply_doernenburg(out)
    out = duval_triangle.apply_duval_triangle(out)
    out = duval_pentagon.apply_duval_pentagon_dual(out)
    for column in cfg.DIAGNOSTIC_METHOD_TO_COLUMN.values():
        if column not in out.columns: out[column] = ABSTAIN
    logger.debug("_apply_traditional_methods: output shape=%s", out.shape)
    return out

def make_vote_frame(df, methods=None):
    selected = list(methods or cfg.DIAGNOSTIC_METHODS)
    unknown = set(selected) - set(cfg.DIAGNOSTIC_METHODS)
    if unknown:
        logger.error("make_vote_frame: unknown methods=%s", sorted(unknown))
        raise ValueError(f"Unknown diagnostic methods: {sorted(unknown)}")
    if not selected:
        logger.error("make_vote_frame: no methods selected")
        raise ValueError("At least one diagnostic method is required.")
    logger.debug("make_vote_frame: rows=%d methods=%s", len(df), selected)
    rows = []
    for _, row in df.iterrows():
        rows.append({
            method: normalize_fault(row.get(cfg.DIAGNOSTIC_METHOD_TO_COLUMN[method], ABSTAIN)) if method in selected else ABSTAIN
            for method in cfg.DIAGNOSTIC_METHODS
        })
    vote_frame = pd.DataFrame(rows, index=df.index, columns=cfg.DIAGNOSTIC_METHODS)
    logger.debug("make_vote_frame: vote_frame shape=%s", vote_frame.shape)
    return vote_frame

def consensus_from_vote_frame(df, vote_frame, selected_methods=None):
    selected = list(selected_methods or vote_frame.columns)
    if len(df) != len(vote_frame):
        logger.error("consensus_from_vote_frame: length mismatch df=%s vote=%s", len(df), len(vote_frame))
        raise ValueError("df and vote_frame length mismatch.")
    logger.debug("consensus_from_vote_frame: rows=%d methods=%s", len(df), selected)
    out = df.copy()
    faults, groups, components = [], [], []
    agreements, conflicts, active_counts = [], [], []
    for position, (_, row) in enumerate(df.iterrows()):
        votes = vote_frame.iloc[position].to_dict()
        active = sum(normalize_fault(votes.get(method, ABSTAIN)) != ABSTAIN for method in selected)
        fault, comp, conflict = aggregate_votes_with_row(row, votes, selected)
        group = unify_fault(fault)
        faults.append(fault); groups.append(group); components.append(comp); conflicts.append(bool(conflict)); active_counts.append(int(active))
        agreements.append(diagnostic_agreement_ratio(votes, group, selected) if group != ABSTAIN else 0.0)
    out["consensus_fault_traditional"] = faults
    out["consensus_fault"] = faults
    out["consensus_fault_group"] = groups
    out["mixed_components"] = components
    out["diagnostic_confidence"] = agreements
    out["diagnostic_agreement_ratio"] = agreements
    out["diagnostic_votes"] = [vote_frame.iloc[position].to_dict() for position in range(len(out))]
    out["diagnostic_active_methods"] = active_counts
    out["diagnostic_abstain_methods"] = len(selected) - np.asarray(active_counts)
    out["diagnostic_method_count"] = len(selected)
    out["diagnostic_coverage"] = np.asarray(active_counts) / max(len(selected), 1) * 100.0
    out["diagnostic_unweighted"] = True
    out["gas_below_l1"] = [is_below_l1(row) for _, row in df.iterrows()]
    out["diagnostic_conflict"] = conflicts
    out["diagnostic_selected_methods"] = [list(selected) for _ in range(len(out))]
    logger.debug("consensus_from_vote_frame: done rows=%d", len(out))
    return out

def apply_consensus(df, methods=None):
    logger.debug("apply_consensus: input shape=%s", df.shape)
    calculated = _apply_traditional_methods(df)
    selected = list(methods or cfg.DIAGNOSTIC_METHODS)
    logger.debug("apply_consensus: selected_methods=%s", selected)
    votes = make_vote_frame(calculated, selected)
    result = consensus_from_vote_frame(calculated, votes, selected)
    logger.debug("apply_consensus: output shape=%s", result.shape)
    return result

def apply_consensus_from_existing_diagnostics(df, methods):
    selected = list(methods)
    missing = [method for method in selected if cfg.DIAGNOSTIC_METHOD_TO_COLUMN[method] not in df.columns]
    if missing:
        logger.error("apply_consensus_from_existing_diagnostics: missing columns=%s", missing)
        raise ValueError(f"Missing diagnostic columns: {missing}")
    logger.debug("apply_consensus_from_existing_diagnostics: rows=%d methods=%s", len(df), selected)
    votes = make_vote_frame(df, selected)
    result = consensus_from_vote_frame(df, votes, selected)
    logger.debug("apply_consensus_from_existing_diagnostics: output shape=%s", result.shape)
    return result

def canonicalize_truth_fine(series): return series.map(normalize_fault)
def canonicalize_truth_coarse(series): return series.map(unify_fault)

def diagnostic_method_summary(df):
    logger.debug("diagnostic_method_summary: rows=%d", len(df))
    rows = []
    for method, column in cfg.DIAGNOSTIC_METHOD_TO_COLUMN.items():
        labels = df.get(column, pd.Series(ABSTAIN, index=df.index)).map(normalize_fault)
        active = labels != ABSTAIN
        coarse = labels.map(unify_fault)
        rows.append({
            "method": method, "column": column,
            "coverage_percent": float(active.mean() * 100.0) if len(df) else 0.0,
            "active_count": int(active.sum()),
            "abstain_count": int((~active).sum()),
            "unique_fine_labels": int(labels[active].nunique()),
            "unique_coarse_groups": int(coarse[coarse != ABSTAIN].nunique()),
        })
    summary = pd.DataFrame(rows)
    logger.debug("diagnostic_method_summary: summary=%s", summary.to_dict(orient="records"))
    return summary

def pairwise_label_agreement(df, methods=None):
    selected = list(methods or cfg.DIAGNOSTIC_METHODS)
    logger.debug("pairwise_label_agreement: rows=%d methods=%s", len(df), selected)
    rows = []
    for i, method_a in enumerate(selected):
        a = df.get(cfg.DIAGNOSTIC_METHOD_TO_COLUMN[method_a], pd.Series(ABSTAIN, index=df.index)).map(unify_fault)
        for method_b in selected[i + 1:]:
            b = df.get(cfg.DIAGNOSTIC_METHOD_TO_COLUMN[method_b], pd.Series(ABSTAIN, index=df.index)).map(unify_fault)
            valid = (a != ABSTAIN) & (b != ABSTAIN)
            n = int(valid.sum())
            agreement = float((a[valid].to_numpy() == b[valid].to_numpy()).mean()) if n else np.nan
            rows.append({
                "method_a": method_a, "method_b": method_b, "n_joint_active": n,
                "joint_coverage_percent": 100.0 * n / max(len(df), 1),
                "conditional_agreement_percent": 100.0 * agreement if np.isfinite(agreement) else np.nan,
            })
    result = pd.DataFrame(rows)
    logger.debug("pairwise_label_agreement: result rows=%d", len(result))
    return result

def get_coarse_vote_matrix(df, methods=None): return make_vote_frame(df, methods).map(unify_fault)

def get_fine_vote_matrix(df, methods=None):
    matrix = make_vote_frame(df, methods)
    allowed = set(cfg.BENCHMARK_FINE_CLASSES) | {"T1_T2", "MIXED"}
    return matrix.map(lambda value: value if value in allowed else ABSTAIN)

def evaluate_method_labels(y_true, predicted_labels, allowed_labels):
    from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score, precision_score, recall_score)
    truth = pd.Series(y_true).reset_index(drop=True).map(normalize_fault)
    pred = pd.Series(predicted_labels).reset_index(drop=True).map(normalize_fault)
    allowed = set(normalize_fault(label) for label in allowed_labels)
    logger.debug("evaluate_method_labels: truth=%d pred=%d allowed=%s", len(truth), len(pred), sorted(allowed))
    valid_truth = truth.isin(allowed)
    truth = truth[valid_truth].reset_index(drop=True)
    pred = pred[valid_truth].reset_index(drop=True)
    active = pred.isin(allowed)
    n_truth = len(truth)
    n_eval = int(active.sum())
    coverage = n_eval / max(n_truth, 1)
    if n_eval == 0:
        logger.debug("evaluate_method_labels: n_eval=0")
        return {
            "accuracy": np.nan, "balanced_accuracy": np.nan,
            "macro_precision": np.nan, "macro_recall": np.nan,
            "macro_f1": np.nan, "weighted_f1": np.nan,
            "coverage": coverage, "selective_accuracy": np.nan,
            "overall_accuracy_with_abstain_error": 0.0,
            "n_truth": n_truth, "n_evaluated": 0,
        }
    y_t, y_p = truth[active], pred[active]
    correct = int((y_t == y_p).sum())
    labels = sorted(allowed)
    # Compute balanced accuracy from an explicitly labelled confusion matrix.
    # This avoids sklearn's single-label warning on small/ambiguous evaluation splits
    # while preserving the balanced-accuracy definition used for the benchmark.
    cm = confusion_matrix(y_t, y_p, labels=labels)
    support = cm.sum(axis=1)
    present = support > 0
    recalls = np.divide(
        np.diag(cm),
        support,
        out=np.zeros_like(support, dtype=float),
        where=support > 0,
    )
    balanced_accuracy = float(np.mean(recalls[present])) if np.any(present) else 0.0

    metrics = {
        "accuracy": float(accuracy_score(y_t, y_p)),
        "balanced_accuracy": balanced_accuracy,
        "macro_precision": float(precision_score(y_t, y_p, average="macro", labels=labels, zero_division=0)),
        "macro_recall": float(recall_score(y_t, y_p, average="macro", labels=labels, zero_division=0)),
        "macro_f1": float(f1_score(y_t, y_p, average="macro", labels=labels, zero_division=0)),
        "weighted_f1": float(f1_score(y_t, y_p, average="weighted", labels=labels, zero_division=0)),
        "coverage": coverage,
        "selective_accuracy": correct / n_eval,
        "overall_accuracy_with_abstain_error": correct / max(n_truth, 1),
        "n_truth": n_truth, "n_evaluated": n_eval,
    }
    logger.debug(
        "evaluate_method_labels: n_truth=%d n_evaluated=%d selective_accuracy=%.4f",
        n_truth, n_eval, metrics["selective_accuracy"],
    )
    return metrics