# severity.py
import numpy as np
import pandas as pd
import logging
from config import config as cfg

logger = logging.getLogger(__name__)

def score_by_threshold(value: float, thresholds: list) -> int:
    """Gán điểm dựa trên ngưỡng (0,1,2,3)."""
    if pd.isna(value) or value < 0:
        return 0
    for i, thr in enumerate(thresholds):
        if value < thr:
            return i
    return 3

def compute_gas_level_score(row: pd.Series) -> int:
    """Điểm dựa trên nồng độ khí vượt ngưỡng."""
    score = 0
    for gas, thresholds in cfg.SEVERITY_GAS_THRESHOLDS.items():
        val = row.get(gas, np.nan)
        score += score_by_threshold(val, thresholds)
    return score

def compute_trend_score(row: pd.Series) -> int:
    """Điểm dựa trên tốc độ tăng của khí."""
    score = 0
    tdcg_rate = row.get("tdcg_rate_per_day", np.nan)
    if pd.notna(tdcg_rate):
        if tdcg_rate >= 10:
            score += 3
        elif tdcg_rate >= 3:
            score += 2
        elif tdcg_rate > 0:
            score += 1

    num_inc = row.get("num_gases_increasing", np.nan)
    if pd.notna(num_inc):
        if num_inc >= 5:
            score += 2
        elif num_inc >= 3:
            score += 1

    for gas, limit in [("h2", 3), ("c2h2", 0.5), ("c2h4", 3), ("co", 3)]:
        rate = row.get(f"{gas}_rate_per_day", np.nan)
        if pd.isna(rate):
            continue
        if rate >= limit:
            score += 2
        elif rate > 0:
            score += 1
    return score

def compute_aging_score(row: pd.Series) -> int:
    """Điểm dựa trên dấu hiệu lão hóa cách điện."""
    score = 0
    co = row.get("co", np.nan)
    co2 = row.get("co2", np.nan)
    water = row.get("water", np.nan)
    temp = row.get("temp", np.nan)

    if pd.notna(co):
        if co >= 1400:
            score += 3
        elif co >= 700:
            score += 2
        elif co >= 350:
            score += 1
    if pd.notna(co2):
        if co2 >= 10000:
            score += 2
        elif co2 >= 5000:
            score += 1

    # Tính tỉ số CO2/CO, tránh chia cho 0
    if pd.notna(co) and pd.notna(co2) and co > 0:
        ratio = co2 / co
        if ratio < 3:
            score += 3
        elif ratio < 5:
            score += 2
        elif ratio < 7:
            score += 1
    if pd.notna(water):
        if water >= 40:
            score += 2
        elif water >= 25:
            score += 1
    if pd.notna(temp):
        if temp >= 90:
            score += 2
        elif temp >= 70:
            score += 1
    return score

def severity_class_from_score(score: float) -> str:
    """Phân loại mức độ nghiêm trọng dựa trên điểm tổng."""
    if pd.isna(score):
        return "UNCERTAIN"
    boundaries = cfg.SEVERITY_CLASS_BOUNDARIES
    if score < boundaries[0]:
        return "NORMAL"
    if score < boundaries[1]:
        return "WATCHLIST"
    if score < boundaries[2]:
        return "WARNING"
    return "CRITICAL"

def get_fault_points(row: pd.Series) -> int:
    """
    Tính điểm severity từ consensus fault.
    Với MIXED: lấy max severity của các nhóm thành phần.
    """
    fault_label = row.get("consensus_fault")
    if pd.isna(fault_label):
        return cfg.FAULT_SEVERITY_POINTS["UNCERTAIN"]

    fl = str(fault_label).strip().upper()

    if fl == "MIXED":
        components = row.get("mixed_components", [])
        if not components:
            return cfg.SEVERITY_BY_GROUP.get("MIXED", 5)
        max_sev = 0
        for comp in components:
            sev = cfg.SEVERITY_BY_GROUP.get(comp, 1)
            if sev > max_sev:
                max_sev = sev
        return max_sev

    if fl == "T3-H":
        fl = "T3_H"
    return cfg.FAULT_SEVERITY_POINTS.get(fl, 1)

def apply_severity(df: pd.DataFrame) -> pd.DataFrame:
    """Tính toán các chỉ số severity và gán nhãn."""
    logger.info("Bắt đầu tính severity scores...")
    df["severity_gas_score"] = df.apply(compute_gas_level_score, axis=1)
    df["severity_trend_score"] = df.apply(compute_trend_score, axis=1)
    df["severity_aging_score"] = df.apply(compute_aging_score, axis=1)
    df["severity_fault_score"] = df.apply(get_fault_points, axis=1)

    w = cfg.SEVERITY_WEIGHTS
    df["severity_score"] = (
        w["gas"] * df["severity_gas_score"] +
        w["trend"] * df["severity_trend_score"] +
        w["fault"] * df["severity_fault_score"] +
        w["aging"] * df["severity_aging_score"]
    )

    df["severity_label"] = df["severity_score"].apply(severity_class_from_score)

    # Log mẫu
    sample_cols = ["transformer_id", "severity_gas_score", "severity_trend_score",
                   "severity_fault_score", "severity_score", "severity_label"]
    if all(c in df.columns for c in sample_cols):
        sample = df[sample_cols].head(5)
        logger.info("Mẫu severity (5 dòng đầu):\n" + sample.to_string())
    else:
        logger.warning("Không đủ cột để hiển thị debug severity.")
    return df