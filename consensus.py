# consensus.py
from typing import Dict, Tuple, List
import pandas as pd
import logging
from config import config as cfg

logger = logging.getLogger(__name__)

def unify_fault(label: str) -> str:
    """Chuyển đổi nhãn lỗi về nhóm lớn (THERMAL, DISCHARGE,...)"""
    if label is None:
        return "UNCERTAIN"
    label = str(label).strip().upper()
    # Xử lý một số trường hợp đặc biệt
    if label == "T3-H":
        label = "T3_H"
    return cfg.FAULT_GROUPS.get(label, "UNCERTAIN")

def normalize_fault(label: str) -> str:
    """Chuẩn hóa nhãn lỗi về dạng code chung (PD, D1, T1,...)"""
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

def aggregate_votes(votes: Dict[str, str]) -> Tuple[str, List[str]]:
    """
    Weighted voting để xác định consensus.
    Trả về (fault_label, list_of_groups).
    Sửa điều kiện MIXED dùng tỉ lệ trọng số thay vì giá trị tuyệt đối.
    """
    group_weights = {}
    fault_by_method = {}

    for method, raw_fault in votes.items():
        norm = normalize_fault(raw_fault)
        if norm == "UNCERTAIN":
            continue
        group = unify_fault(norm)
        weight = cfg.METHOD_WEIGHTS.get(method, 1.0)
        group_weights[group] = group_weights.get(group, 0) + weight
        fault_by_method[method] = norm

    if not group_weights:
        return "UNCERTAIN", []

    # Loại bỏ NORMAL để tính non‑normal tỉ lệ
    non_normal = {g: w for g, w in group_weights.items() if g != "NORMAL"}
    total_non_normal = sum(non_normal.values())

    if total_non_normal == 0:
        return "NORMAL", ["NORMAL"]

    # Nhóm có trọng số cao nhất
    top_group = max(non_normal, key=non_normal.get)
    top_weight = non_normal[top_group]

    # Sắp xếp trọng số giảm dần
    sorted_weights = sorted(non_normal.values(), reverse=True)
    second_weight = sorted_weights[1] if len(sorted_weights) > 1 else 0

    # Xác định MIXED: nếu top không chiếm ưu thế rõ rệt hoặc nhóm thứ hai đủ lớn
    if (top_weight / total_non_normal < cfg.MIXED_THRESHOLD) or \
       (second_weight / total_non_normal >= cfg.MIN_SECOND_GROUP_WEIGHT_RATIO):
        mixed_groups = [g for g in non_normal if non_normal[g] > 0]
        return "MIXED", mixed_groups

    # Chọn fault cụ thể đại diện cho nhóm thắng
    best_fault = None
    best_weight = -1
    for method, fault in fault_by_method.items():
        if unify_fault(fault) == top_group:
            w = cfg.METHOD_WEIGHTS.get(method, 1.0)
            if w > best_weight:
                best_weight = w
                best_fault = fault

    return best_fault if best_fault else top_group, [top_group]

def confidence(votes: Dict[str, str]) -> float:
    """
    Tính độ tin cậy dựa trên tỉ lệ trọng số của nhóm thắng so với tổng.
    Nếu không có phiếu hợp lệ nào, trả về 0.0.
    """
    group_weights = {}
    for method, raw_fault in votes.items():
        norm = normalize_fault(raw_fault)
        if norm == "UNCERTAIN":
            continue
        group = unify_fault(norm)
        weight = cfg.METHOD_WEIGHTS.get(method, 1.0)
        group_weights[group] = group_weights.get(group, 0) + weight

    non_normal = {g: w for g, w in group_weights.items() if g != "NORMAL"}
    total_non_normal = sum(non_normal.values())

    # Trường hợp tất cả NORMAL hoặc không có phiếu hợp lệ
    if total_non_normal == 0:
        # Nếu có ít nhất 1 phiếu NORMAL, coi là hoàn toàn tin tưởng (100%)
        if group_weights.get("NORMAL", 0) > 0:
            return 100.0
        # Không phiếu nào
        return 0.0

    top_group = max(non_normal, key=non_normal.get)
    top_weight = non_normal[top_group]
    return round((top_weight / total_non_normal) * 100, 1)

def apply_consensus(df: pd.DataFrame) -> pd.DataFrame:
    """
    Áp dụng tất cả module chẩn đoán, tổng hợp votes và tạo cột consensus.
    Phiên bản tối ưu: chỉ gọi make_votes một lần.
    """
    logger.info("Bắt đầu áp dụng các module chẩn đoán...")
    # Gọi các module DGA (giả sử chúng đã được import và hoạt động)
    from dga import keygas, iec60599, rogers, doernenburg, duval_triangle, duval_pentagon
    df = keygas.apply_key_gas(df)
    df = iec60599.apply_iec(df)
    df = rogers.apply_rogers(df)
    df = doernenburg.apply_doernenburg(df)
    df = duval_triangle.apply_duval_triangle(df)
    df = duval_pentagon.apply_duval_pentagon(df, pentagon="P2")
    logger.info("Hoàn tất tính toán tỉ số và lỗi. Đang tổng hợp votes...")

    # Chuẩn bị votes một lần
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
        return {method: row.get(col, "UNCERTAIN") for method, col in vote_columns.items()}

    votes_series = df.apply(make_votes, axis=1)

    # Tính consensus và confidence
    consensus_results = votes_series.apply(aggregate_votes)
    df["consensus_fault"] = consensus_results.apply(lambda x: x[0])
    df["mixed_components"] = consensus_results.apply(lambda x: x[1])
    df["diagnostic_confidence"] = votes_series.apply(confidence)
    df["diagnostic_votes"] = votes_series   # <-- THÊM DÒNG NÀY

    # Log mẫu
    sample_cols = ["transformer_id", "sample_day", "consensus_fault", "mixed_components", "diagnostic_confidence"]
    if all(c in df.columns for c in sample_cols):
        sample = df[sample_cols].head(5)
        logger.info("Mẫu 5 dòng đầu consensus:\n" + sample.to_string())
    else:
        logger.info("Không đủ cột để hiển thị debug.")

    return df