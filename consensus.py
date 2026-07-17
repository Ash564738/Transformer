# consensus.py
from typing import Dict, Tuple
import pandas as pd
import logging
from config import FAULT_GROUPS, METHOD_WEIGHTS
from dga import (
    keygas,
    iec60599,
    rogers,
    doernenburg,
    duval_triangle,
    duval_pentagon
)

logger = logging.getLogger(__name__)

MIXED_THRESHOLD = 0.55   # ngưỡng xác định nhóm chiếm ưu thế

def unify_fault(label: str) -> str:
    if label is None:
        return "UNCERTAIN"
    label = str(label).strip().upper()
    if label == "T3-H":
        label = "T3_H"
    elif label in ("THERMAL", "CELLULOSE", "MIXED"):
        if label == "THERMAL": return "THERMAL"
        if label == "CELLULOSE": return "CELLULOSE"
        if label == "MIXED": return "MIXED"
    return FAULT_GROUPS.get(label, "UNCERTAIN")

def normalize_fault(label: str) -> str:
    if label is None:
        return "UNCERTAIN"
    label = str(label).strip().upper()
    if label in ("", "UNCERTAIN", "NA"):
        return "UNCERTAIN"
    legacy_map = {
        "T3-H": "T3_H",
        "ARCING": "D2",
        "PARTIAL_DISCHARGE": "PD",
        "THERMAL": "T3",
        "NORMAL": "NORMAL"
    }
    return legacy_map.get(label, label)

def aggregate_votes(votes: Dict[str, str]) -> Tuple[str, list]:
    """
    Weighted voting để xác định consensus.
    Trả về (fault_label, list_of_groups)
    """
    # 1. Lọc các vote hợp lệ và tính trọng số theo nhóm
    group_weights = {}
    fault_by_method = {}   # để biết fault cụ thể từng phương pháp
    for method, raw_fault in votes.items():
        norm = normalize_fault(raw_fault)
        if norm == "UNCERTAIN":
            continue
        group = unify_fault(norm)
        weight = METHOD_WEIGHTS.get(method, 1.0)
        group_weights[group] = group_weights.get(group, 0) + weight
        fault_by_method[method] = norm

    if not group_weights:
        return "UNCERTAIN", []

    # 2. Tổng trọng số không tính NORMAL
    non_normal = {g: w for g, w in group_weights.items() if g != "NORMAL"}
    total_non_normal = sum(non_normal.values())

    # Nếu tất cả là NORMAL
    if total_non_normal == 0:
        return "NORMAL", ["NORMAL"]

    # 3. Nhóm có trọng số cao nhất
    top_group = max(non_normal, key=non_normal.get)
    top_weight = non_normal[top_group]

    # 4. Xác định có MIXED không
    if top_weight / total_non_normal < MIXED_THRESHOLD:
        # MIXED: trả về danh sách nhóm có trọng số > 0
        mixed_groups = [g for g in non_normal if non_normal[g] > 0]
        return "MIXED", mixed_groups

    # 5. Chọn fault cụ thể đại diện cho nhóm thắng
    # Ưu tiên phương pháp có trọng số cao nhất trong nhóm đó
    best_fault = None
    best_weight = -1
    for method, fault in fault_by_method.items():
        if unify_fault(fault) == top_group:
            w = METHOD_WEIGHTS.get(method, 1.0)
            if w > best_weight:
                best_weight = w
                best_fault = fault

    return best_fault if best_fault else top_group, [top_group]

def confidence(votes: Dict[str, str]) -> float:
    """
    Tính confidence dựa trên trọng số của nhóm thắng / tổng trọng số không phải NORMAL.
    """
    group_weights = {}
    for method, raw_fault in votes.items():
        norm = normalize_fault(raw_fault)
        if norm == "UNCERTAIN":
            continue
        group = unify_fault(norm)
        weight = METHOD_WEIGHTS.get(method, 1.0)
        group_weights[group] = group_weights.get(group, 0) + weight

    non_normal = {g: w for g, w in group_weights.items() if g != "NORMAL"}
    total_non_normal = sum(non_normal.values())
    if total_non_normal == 0:
        return 100.0   # tất cả NORMAL -> confidence cao

    top_group = max(non_normal, key=non_normal.get)
    top_weight = non_normal[top_group]
    return round((top_weight / total_non_normal) * 100, 1)

def apply_consensus(df: pd.DataFrame) -> pd.DataFrame:
    logger.info("Bắt đầu áp dụng các module chẩn đoán...")
    df = keygas.apply_key_gas(df)
    df = iec60599.apply_iec(df)
    df = rogers.apply_rogers(df)
    df = doernenburg.apply_doernenburg(df)
    df = duval_triangle.apply_duval_triangle(df)
    df = duval_pentagon.apply_duval_pentagon(df, pentagon="P2")
    logger.info("Hoàn tất tính toán tỉ số và lỗi. Đang tổng hợp votes...")

    def make_votes(row):
        return {
            "keygas_fault": row.get("keygas_fault", "UNCERTAIN"),
            "iec_fault": row.get("iec_fault", "UNCERTAIN"),
            "rogers_fault": row.get("rogers_fault", "UNCERTAIN"),
            "doernenburg_fault": row.get("doernenburg_fault", "UNCERTAIN"),
            "duval_triangle_fault": row.get("duval_triangle_fault", "UNCERTAIN"),
            "duval_pentagon_p1_fault": row.get("fault_p1", "UNCERTAIN"),
            "duval_pentagon_p2_fault": row.get("duval_pentagon_fault", "UNCERTAIN")
        }

    df["diagnostic_votes"] = df.apply(lambda r: make_votes(r), axis=1)
    results = df.apply(lambda r: aggregate_votes(make_votes(r)), axis=1)
    df["consensus_fault"] = results.apply(lambda x: x[0])
    df["mixed_components"] = results.apply(lambda x: x[1])
    df["diagnostic_confidence"] = df.apply(lambda r: confidence(make_votes(r)), axis=1)

    # Debug
    sample = df[["transformer_id", "sample_day", "consensus_fault", "mixed_components", "diagnostic_confidence"]].head(5)
    logger.info("Mẫu 5 dòng đầu consensus:\n" + sample.to_string())
    print("=== DEBUG CONSENSUS (5 first rows) ===")
    print(sample.to_string())
    print("======================================")

    return df