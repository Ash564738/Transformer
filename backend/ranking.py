# ranking.py
import numpy as np
import pandas as pd
import logging
from config import config as cfg

logger = logging.getLogger(__name__)

def compute_trend_slope(series: pd.Series, window: int) -> pd.Series:
    """Tính slope trên cửa sổ rolling (dùng np.polyfit)."""
    def slope_func(arr):
        if len(arr) < 2:
            return 0.0
        x = np.arange(len(arr))
        return np.polyfit(x, arr, 1)[0]
    return series.rolling(window, min_periods=2).apply(slope_func, raw=True)

def compute_historical_severity(series: pd.Series, window: int) -> pd.Series:
    """Trung bình có trọng số mũ (gần đây hơn có trọng số cao hơn)."""
    # Sử dụng ewm
    return series.ewm(span=window, min_periods=1, adjust=False).mean()

def fault_persistence_score(series: pd.Series, window: int) -> pd.Series:
    """Tỉ lệ số mẫu có severity >= WARNING trong cửa sổ."""
    # Tạo mặt nạ boolean
    high_sev = series.isin(["WARNING", "CRITICAL"]).astype(float)
    return high_sev.rolling(window, min_periods=1).mean()

def build_transformer_ranking(df: pd.DataFrame) -> pd.DataFrame:
    """
    Xây dựng bảng xếp hạng transformer dựa trên điểm tổng hợp.
    Phiên bản tối ưu: sử dụng groupby transform thay vì vòng lặp.
    """
    logger.info("Bắt đầu xây dựng ranking...")

    # Đảm bảo dữ liệu được sắp xếp theo thời gian
    df = df.sort_values(["transformer_id", "sample_day"])

    # Tính slope trend cho từng transformer
    df["trend_slope"] = df.groupby("transformer_id")["severity_score"].transform(
        lambda s: compute_trend_slope(s, cfg.RECENT_SAMPLES_FOR_TREND)
    )

    # Tính historical severity (weighted)
    df["historical_severity"] = df.groupby("transformer_id")["severity_score"].transform(
        lambda s: compute_historical_severity(s, cfg.RECENT_SAMPLES_FOR_HISTORY)
    )

    # Tính fault persistence
    df["fault_persistence"] = df.groupby("transformer_id")["severity_label"].transform(
        lambda s: fault_persistence_score(s, cfg.RECENT_SAMPLES_FOR_PERSISTENCE)
    )

    # Lấy chỉ số dòng mới nhất của mỗi transformer
    latest_idx = df.groupby("transformer_id")["sample_day"].idxmax()
    latest_df = df.loc[latest_idx].copy()

    # Xác định trend string
    def trend_class(slope):
        if pd.isna(slope):
            return "stable"
        if slope > cfg.TREND_SLOPE_WORSENING:
            return "worsening"
        if slope < cfg.TREND_SLOPE_IMPROVING:
            return "improving"
        return "stable"

    latest_df["trend"] = latest_df["trend_slope"].apply(trend_class)

    # Tính số lần critical trong quá khứ và ngày gần nhất
    # Sử dụng groupby trên toàn bộ df và merge
    crit_mask = df["severity_label"] == "CRITICAL"
    crit_history = df[crit_mask].groupby("transformer_id").agg(
        n_critical_past=("sample_day", "count"),
        last_critical_date=("sample_day", "max")
    ).reset_index()

    # Merge vào latest_df
    latest_df = latest_df.merge(crit_history, on="transformer_id", how="left")
    latest_df["n_critical_past"] = latest_df["n_critical_past"].fillna(0).astype(int)
    latest_df["days_since_last_critical"] = (
        latest_df["sample_day"] - latest_df["last_critical_date"]
    ).dt.days

    # Tính điểm thành phần
    sev_current = latest_df["severity_score"].astype(float)
    trend_bonus = latest_df["trend"].map({"worsening": 2.0, "stable": 0.0, "improving": -1.0}).fillna(0)
    crit_bonus = latest_df["n_critical_past"].clip(upper=3).fillna(0)
    conf_penalty = (latest_df["diagnostic_confidence"] < 50).astype(float) * 1.0

    rw = cfg.RANKING_WEIGHTS
    base_score = (
        rw["current"] * sev_current +
        rw["history"] * latest_df["historical_severity"] +
        rw["trend"] * trend_bonus +
        rw["critical_history"] * crit_bonus +
        rw["confidence"] * (1.0 - conf_penalty)
    )

    # Áp dụng persistence bonus (dạng cộng thêm, không khuếch đại quá mức)
    persistence_bonus = cfg.PERSISTENCE_BONUS_FACTOR * latest_df["fault_persistence"]
    latest_df["final_score"] = base_score * (1 + persistence_bonus)

    # Sắp xếp và gán rank
    latest_df = latest_df.sort_values("final_score", ascending=False).reset_index(drop=True)
    latest_df["rank"] = range(1, len(latest_df) + 1)

    # Khuyến nghị hành động
    def recommend_action(row):
        base = ""
        if row["final_score"] >= cfg.ACTION_THRESHOLD_CRITICAL:
            base = f"CRITICAL: Inspect urgently – likely {row.get('consensus_fault', 'unknown fault')}"
        elif row["final_score"] >= cfg.ACTION_THRESHOLD_WARNING:
            base = "WARNING: Increase monitoring frequency"
        else:
            base = "Routine monitoring"

        if row["trend"] == "worsening":
            base += " (trend worsening)"
        if row["n_critical_past"] > 0:
            base += f" – history: {row['n_critical_past']} critical events"
        if row["diagnostic_confidence"] < 50:
            base += " – low confidence diagnosis"
        return base

    latest_df["recommended_action"] = latest_df.apply(recommend_action, axis=1)

    # Chọn các cột đầu ra
    out_cols = [
        "transformer_id", "loc", "name",
        "severity_score", "severity_label", "consensus_fault",
        "sample_day", "diagnostic_confidence",
        "historical_severity", "n_critical_past",
        "days_since_last_critical", "fault_persistence",
        "trend", "final_score", "rank", "recommended_action"
    ]
    available_cols = [c for c in out_cols if c in latest_df.columns]
    ranking = latest_df[available_cols].copy()

    # Log top 5
    top5 = ranking.head(5)
    log_cols = ["rank", "transformer_id", "final_score", "severity_label", "consensus_fault"]
    if all(c in top5.columns for c in log_cols):
        logger.info("Top 5 ranking:\n" + top5[log_cols].to_string())
    else:
        logger.warning("Thiếu cột cần thiết để log top 5 ranking.")

    return ranking