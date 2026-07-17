# severity.py
import numpy as np
import pandas as pd
import logging
from config import FAULT_SEVERITY_POINTS, SEVERITY_BY_GROUP

logger = logging.getLogger(__name__)

THRESHOLDS = {
    "h2": [100, 500, 1000],
    "ch4": [120, 400, 1000],
    "c2h6": [65, 100, 150],
    "c2h4": [50, 100, 200],
    "c2h2": [1, 9, 35],
    "co": [350, 700, 1400],
    "co2": [2500, 5000, 10000],
    "tcg": [720, 1920, 4630],
    "tdcg": [720, 1920, 4630],
}

def score_by_threshold(value: float, thresholds: list) -> int:
    if pd.isna(value):
        return 0
    if value >= thresholds[2]: return 3
    if value >= thresholds[1]: return 2
    if value >= thresholds[0]: return 1
    return 0

def compute_gas_level_score(row: pd.Series) -> int:
    score = 0
    for gas, thr in THRESHOLDS.items():
        val = row.get(gas, np.nan)
        score += score_by_threshold(val, thr)
    return score

def compute_trend_score(row: pd.Series) -> int:
    score = 0
    tdcg_rate = row.get("tdcg_rate_per_day", np.nan)
    if pd.notna(tdcg_rate):
        if tdcg_rate >= 10: score += 3
        elif tdcg_rate >= 3: score += 2
        elif tdcg_rate > 0: score += 1

    num_inc = row.get("num_gases_increasing", np.nan)
    if pd.notna(num_inc):
        if num_inc >= 5: score += 2
        elif num_inc >= 3: score += 1

    for gas in ["h2", "c2h2", "c2h4", "co"]:
        rate = row.get(f"{gas}_rate_per_day", np.nan)
        if pd.isna(rate):
            continue
        if gas == "c2h2":
            if rate >= 0.5: score += 2
            elif rate > 0: score += 1
        else:
            if rate >= 3: score += 2
            elif rate > 0: score += 1
    return score

def compute_aging_score(row: pd.Series) -> int:
    score = 0
    co = row.get("co", np.nan)
    co2 = row.get("co2", np.nan)
    ratio = row.get("ratio_co2_co", np.nan)
    water = row.get("water", np.nan)
    temp = row.get("temp", np.nan)

    if pd.notna(co):
        if co >= 1400: score += 3
        elif co >= 700: score += 2
        elif co >= 350: score += 1
    if pd.notna(co2):
        if co2 >= 10000: score += 2
        elif co2 >= 5000: score += 1
    if pd.notna(ratio):
        if ratio < 3: score += 3
        elif ratio < 5: score += 2
        elif ratio < 7: score += 1
    if pd.notna(water):
        if water >= 40: score += 2
        elif water >= 25: score += 1
    if pd.notna(temp):
        if temp >= 90: score += 2
        elif temp >= 70: score += 1
    return score

def severity_class_from_score(score: float) -> str:
    if pd.isna(score): 
        return "UNCERTAIN"
    if score < 4: 
        return "NORMAL"
    if score < 8: 
        return "WATCHLIST"
    if score < 13: 
        return "WARNING"
    return "CRITICAL"

def get_fault_points(row: pd.Series) -> int:
    """
    Tính điểm severity từ consensus fault.
    Nếu MIXED, dùng max severity của các nhóm thành phần.
    """
    fault_label = row.get("consensus_fault")
    if pd.isna(fault_label):
        return 1
    fl = str(fault_label).strip().upper()

    if fl == "MIXED":
        components = row.get("mixed_components", [])
        if not components:
            return SEVERITY_BY_GROUP.get("MIXED", 5)
        max_sev = 0
        for comp in components:
            sev = SEVERITY_BY_GROUP.get(comp, 1)
            if sev > max_sev:
                max_sev = sev
        return max_sev

    # Xử lý các fault thông thường
    if fl == "T3-H":
        fl = "T3_H"
    return FAULT_SEVERITY_POINTS.get(fl, 1)

def apply_severity(df: pd.DataFrame) -> pd.DataFrame:
    logger.info("Bắt đầu tính severity scores...")
    df["severity_gas_score"] = df.apply(compute_gas_level_score, axis=1)
    df["severity_trend_score"] = df.apply(compute_trend_score, axis=1)
    df["severity_aging_score"] = df.apply(compute_aging_score, axis=1)
    df["severity_fault_score"] = df.apply(get_fault_points, axis=1)

    df["severity_score"] = (
        1.0 * df["severity_gas_score"] +
        1.2 * df["severity_trend_score"] +
        1.0 * df["severity_fault_score"] +
        0.8 * df["severity_aging_score"]
    )

    df["severity_label"] = df["severity_score"].apply(severity_class_from_score)

    # Debug
    sample = df[["transformer_id", "severity_gas_score", "severity_trend_score", "severity_fault_score", "severity_score", "severity_label"]].head(5)
    logger.info("Mẫu severity (5 dòng đầu):\n" + sample.to_string())
    print("=== DEBUG SEVERITY (5 first rows) ===")
    print(sample.to_string())
    print("====================================")

    return df