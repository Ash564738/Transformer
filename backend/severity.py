# severity.py
import numpy as np
import pandas as pd
import logging
from config import config as cfg

logger = logging.getLogger(__name__)

def score_by_threshold(value: float, thresholds: list) -> int:
    """Điểm 0-3 dựa trên ngưỡng IEEE."""
    if pd.isna(value) or value < 0:
        return 0
    for i, thr in enumerate(thresholds):
        if value < thr:
            return i
    return 3

def compute_gas_level_score(row: pd.Series) -> int:
    """Tổng điểm vượt ngưỡng khí (mỗi khí 0-3)."""
    score = 0
    for gas, thresholds in cfg.SEVERITY_GAS_THRESHOLDS.items():
        val = row.get(gas, np.nan)
        score += score_by_threshold(val, thresholds)
    return score

def compute_trend_score(row: pd.Series) -> int:
    """
    Điểm xu hướng dựa trên tốc độ tăng (ppm/ngày).
    Ngưỡng tốc độ tham khảo từ IEEE C57.104 (bảng rate of change).
    """
    score = 0
    tdcg_rate = row.get("tdcg_rate_per_day", np.nan)
    if pd.notna(tdcg_rate):
        if tdcg_rate >= 10:
            score += 3
        elif tdcg_rate >= 3:
            score += 2
        elif tdcg_rate > 0:
            score += 1

    # Số khí đang tăng
    num_inc = row.get("num_gases_increasing", np.nan)
    if pd.notna(num_inc):
        if num_inc >= 5:
            score += 2
        elif num_inc >= 3:
            score += 1

    # Tốc độ tăng riêng lẻ cho các khí quan trọng
    for gas, limit in [("h2", 3), ("c2h2", 0.5), ("c2h4", 3), ("co", 3)]:
        rate = row.get(f"{gas}_rate_per_day", np.nan)
        if pd.isna(rate):
            continue
        if rate >= limit:
            score += 2
        elif rate > 0:
            score += 1
    return score

def apply_severity(df: pd.DataFrame) -> pd.DataFrame:
    """
    Tính severity score hoàn toàn dựa trên phương pháp thống kê:
    1. Gas score: tổng điểm vượt ngưỡng IEEE.
    2. Trend score: điểm tốc độ tăng.
    3. Anomaly score: percentile từ ensemble (0-1).
    Cả ba được chuyển về percentile rank, sau đó cộng lại.
    Phân loại severity bằng percentile của tổng điểm.
    """
    logger.info("Calculating severity scores...")

    # 1. Gas score
    df["severity_gas_score"] = df.apply(compute_gas_level_score, axis=1)
    # 2. Trend score
    df["severity_trend_score"] = df.apply(compute_trend_score, axis=1)

    # 3. Anomaly score (nếu có)
    if "anomaly_percentile" in df.columns:
        df["severity_anomaly_score"] = df["anomaly_percentile"].astype(float)
        logger.info("Anomaly percentile feature found.")
    else:
        df["severity_anomaly_score"] = 0.0

    # Chuyển gas và trend sang percentile rank (0-1)
    df["severity_gas_rank"] = df["severity_gas_score"].rank(pct=True)
    df["severity_trend_rank"] = df["severity_trend_score"].rank(pct=True)

    # Tổng hợp: tổng các percentile rank (mỗi thành phần đóng góp 0-1)
    df["severity_score"] = (
        df["severity_gas_rank"]
        + df["severity_trend_rank"]
        + df["severity_anomaly_score"]  # anomaly đã là percentile
    )

    # Phân loại severity dùng percentile của severity_score
    # (đảm bảo phân bố cân bằng, không lệch)
    q = [50, 75, 90]
    thresholds = np.percentile(df["severity_score"], q)
    logger.info("Severity score percentile thresholds: P50=%.3f, P75=%.3f, P90=%.3f",
                thresholds[0], thresholds[1], thresholds[2])

    def classify(score):
        if score < thresholds[0]:
            return "NORMAL"
        elif score < thresholds[1]:
            return "WATCHLIST"
        elif score < thresholds[2]:
            return "WARNING"
        else:
            return "CRITICAL"

    df["severity_label"] = df["severity_score"].apply(classify)

    # Log phân bố
    label_counts = df["severity_label"].value_counts()
    logger.info("Severity label distribution:\n%s", label_counts.to_string())
    logger.info("Overall severity score statistics: mean=%.3f, median=%.3f, min=%.3f, max=%.3f",
                df["severity_score"].mean(), df["severity_score"].median(),
                df["severity_score"].min(), df["severity_score"].max())

    sample_cols = [
        "transformer_id", "severity_gas_score", "severity_trend_score",
        "severity_anomaly_score", "severity_score", "severity_label",
    ]
    available = [c for c in sample_cols if c in df.columns]
    if available:
        logger.info("Sample severity rows (first 5):\n" + df[available].head(5).to_string())
    return df