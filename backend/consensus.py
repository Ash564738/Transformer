# consensus.py
from typing import Dict, Tuple, List
import pandas as pd
import logging
from config import config as cfg

logger = logging.getLogger(__name__)


def unify_fault(label: str) -> str:
    """Map detailed fault label to broader fault group."""
    if label is None:
        return "ABSTAIN"
    label = str(label).strip().upper()
    if label == "T3-H":
        label = "T3_H"
    if label == "DT":
        return "DT"
    return cfg.FAULT_GROUPS.get(label, "ABSTAIN")


def normalize_fault(label: str) -> str:
    """Normalize a raw fault label from any diagnostic method."""
    if label is None:
        return "ABSTAIN"
    label = str(label).strip().upper()
    if label in ("", "ABSTAIN", "NA"):
        return "ABSTAIN"
    legacy_map = {
        "T3-H": "T3_H",
        "ARCING": "D2",
        "PARTIAL_DISCHARGE": "PD",
        "THERMAL": "T3",
        "NORMAL": "NORMAL"
    }
    return legacy_map.get(label, label)


def _compute_group_weights(votes: Dict[str, str]) -> Tuple[Dict[str, float], Dict[str, str]]:
    """Compute weights for fault groups based on individual method votes."""
    group_weights = {}
    fault_by_method = {}
    for method, raw_fault in votes.items():
        norm = normalize_fault(raw_fault)
        if norm == "ABSTAIN":
            continue
        if norm == "DT":  # DT is ambiguous – contributes to both THERMAL and DISCHARGE
            for grp in ["THERMAL", "DISCHARGE"]:
                weight = cfg.METHOD_WEIGHTS.get(method, 1.0)
                group_weights[grp] = group_weights.get(grp, 0) + weight
            fault_by_method[method] = norm
            continue
        group = unify_fault(norm)
        weight = cfg.METHOD_WEIGHTS.get(method, 1.0)
        group_weights[group] = group_weights.get(group, 0) + weight
        fault_by_method[method] = norm
    return group_weights, fault_by_method


def aggregate_votes(votes: Dict[str, str]) -> Tuple[str, List[str]]:
    """Aggregate individual method votes into a consensus fault label."""
    group_weights, fault_by_method = _compute_group_weights(votes)

    if not group_weights:
        return "ABSTAIN", []

    non_normal = {g: w for g, w in group_weights.items() if g != "NORMAL"}
    total_non_normal = sum(non_normal.values())

    if total_non_normal == 0:
        return "NORMAL", ["NORMAL"]

    top_group = max(non_normal, key=non_normal.get)
    top_weight = non_normal[top_group]
    sorted_weights = sorted(non_normal.values(), reverse=True)
    second_weight = sorted_weights[1] if len(sorted_weights) > 1 else 0

    # Mixed fault condition: no single group dominates sufficiently
    if (top_weight / total_non_normal < cfg.MIXED_THRESHOLD) or \
       (second_weight / total_non_normal >= cfg.MIN_SECOND_GROUP_WEIGHT_RATIO):
        mixed_groups = [g for g in non_normal if non_normal[g] > 0]
        return "MIXED", mixed_groups

    # Pick the specific fault label from the method with highest weight within the top group
    best_fault = None
    best_weight = -1
    for method, fault in fault_by_method.items():
        if fault == "DT":
            continue
        if unify_fault(fault) == top_group:
            w = cfg.METHOD_WEIGHTS.get(method, 1.0)
            if w > best_weight:
                best_weight = w
                best_fault = fault

    return best_fault if best_fault else top_group, [top_group]


def confidence(votes: Dict[str, str]) -> float:
    """Calculate diagnostic confidence as the proportion of weight of the top group."""
    group_weights, _ = _compute_group_weights(votes)
    if not group_weights:
        return 0.0
    total_weight = sum(group_weights.values())
    top_group = max(group_weights, key=group_weights.get)
    top_weight = group_weights[top_group]
    return round((top_weight / total_weight) * 100, 1)


def apply_consensus(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply all DGA diagnostic modules, collect their votes, and produce a consensus fault label.
    """
    logger.info("Applying DGA diagnostic modules (Key Gas, IEC, Rogers, Doernenburg, Duval Triangle, Duval Pentagon)...")

    from dga import keygas, iec60599, rogers, doernenburg, duval_triangle, duval_pentagon

    # Apply each diagnostic method
    df = keygas.apply_key_gas(df)
    df = iec60599.apply_iec(df)
    df = rogers.apply_rogers(df)
    df = doernenburg.apply_doernenburg(df)
    df = duval_triangle.apply_duval_triangle(df)
    df = duval_pentagon.apply_duval_pentagon(df, pentagon="P2")
    logger.info("All DGA modules applied. Starting vote aggregation...")

    # Collect votes from each method
    vote_columns = {
        "keygas_fault": "keygas_fault",
        "iec_fault": "iec_fault",
        "rogers_fault": "rogers_fault",
        "doernenburg_fault": "doernenburg_fault",
        "duval_triangle_fault": "duval_triangle_fault",
        "duval_pentagon_p1_fault": "fault_p1",
        "duval_pentagon_p2_fault": "duval_pentagon_fault"
    }

    def make_votes(row):
        return {method: row.get(col, "ABSTAIN") for method, col in vote_columns.items()}

    votes_series = df.apply(make_votes, axis=1)

    # Compute consensus
    consensus_results = votes_series.apply(aggregate_votes)
    df["consensus_fault"] = consensus_results.apply(lambda x: x[0])
    df["mixed_components"] = consensus_results.apply(lambda x: x[1])
    df["diagnostic_confidence"] = votes_series.apply(confidence)
    df["diagnostic_votes"] = votes_series

    # Log summary statistics
    n_total = len(df)
    n_normal = (df["consensus_fault"] == "NORMAL").sum()
    n_abstain = (df["consensus_fault"] == "ABSTAIN").sum()
    n_mixed = (df["consensus_fault"] == "MIXED").sum()
    n_faults = n_total - n_normal - n_abstain - n_mixed
    avg_conf = df["diagnostic_confidence"].mean()

    logger.info(
        f"Consensus aggregation complete: {n_total} samples -> "
        f"NORMAL={n_normal}, FAULTS={n_faults}, MIXED={n_mixed}, ABSTAIN={n_abstain}. "
        f"Average confidence = {avg_conf:.1f}%"
    )

    # Show a few examples
    sample_cols = ["transformer_id", "sample_day", "consensus_fault", "mixed_components", "diagnostic_confidence"]
    if all(c in df.columns for c in sample_cols):
        sample = df[sample_cols].head(5)
        logger.info("Sample consensus results:\n" + sample.to_string())
    else:
        logger.warning("Some sample columns missing, cannot show example rows.")

    return df

def combine_consensus_and_student(
    df: pd.DataFrame,
    student_fault_col: str = "student_fault",
    student_conf_col: str = "student_confidence",
    consensus_conf_threshold: float = 60.0,
) -> pd.DataFrame:
    """
    Kết hợp nhãn consensus truyền thống và student model:
    - Nếu consensus tự tin (confidence >= threshold) và không phải MIXED/ABSTAIN -> dùng consensus.
    - Ngược lại dùng student model nếu có.
    """
    if student_fault_col not in df.columns:
        logger.warning("Student fault column missing, keep consensus only.")
        return df

    final_faults = []
    for idx, row in df.iterrows():
        cons_fault = row.get("consensus_fault", "ABSTAIN")
        cons_conf = row.get("diagnostic_confidence", 0.0)
        if cons_fault in ("ABSTAIN", "MIXED") or cons_conf < consensus_conf_threshold:
            # Dùng student model
            final_faults.append(row[student_fault_col])
        else:
            final_faults.append(cons_fault)

    df["final_fault"] = final_faults
    # Cập nhật consensus_fault để các bước sau dùng nhãn cuối
    df["consensus_fault"] = df["final_fault"]
    logger.info("Combined consensus and student faults. Final distribution:\n%s",
                df["consensus_fault"].value_counts().to_string())
    return df