# ranking.py
import numpy as np
import pandas as pd
import logging

logger = logging.getLogger(__name__)

def compute_trend_slope(series: pd.Series, window: int = 5) -> pd.Series:
    """Hồi quy tuyến tính trên cửa sổ trượt – độ dốc xu hướng."""
    def slope_func(arr):
        if len(arr) < 2:
            return 0.0
        x = np.arange(len(arr))
        return np.polyfit(x, arr, 1)[0]
    return series.rolling(window, min_periods=2).apply(slope_func, raw=True)

def build_transformer_ranking(df: pd.DataFrame) -> pd.DataFrame:
    logger.info("Building transformer ranking based on severity score and trend...")

    df = df.sort_values(["transformer_id", "sample_day"])

    # Tính trend slope của severity_score cho mỗi transformer
    df["trend_slope"] = df.groupby("transformer_id")["severity_score"].transform(
        lambda s: compute_trend_slope(s, window=5)
    )

    # Lấy mẫu mới nhất của mỗi transformer
    latest_idx = df.groupby("transformer_id")["sample_day"].idxmax()
    latest_df = df.loc[latest_idx].copy()
    logger.info("Number of transformers in ranking: %d", len(latest_df))

    # Chuyển trend_slope sang percentile rank (0-1)
    latest_df["trend_slope_rank"] = latest_df["trend_slope"].rank(pct=True)

    # Final score = severity_score (đã tổng hợp gas + trend + anomaly)
    # cộng thêm một phần nhỏ trend_slope_rank để ưu tiên xu hướng xấu.
    # Hệ số 0.2 có thể thay bằng 0 nếu muốn chỉ dùng severity_score.
    latest_df["final_score"] = (
        latest_df["severity_score"] + 0.2 * latest_df["trend_slope_rank"]
    )

    # Xếp hạng giảm dần
    latest_df = latest_df.sort_values("final_score", ascending=False).reset_index(drop=True)
    latest_df["rank"] = range(1, len(latest_df) + 1)

    logger.info("Final score range: min=%.3f, max=%.3f",
                latest_df["final_score"].min(), latest_df["final_score"].max())

    # Đề xuất hành động dựa trên severity_label
    def recommend_action(row):
        if row["severity_label"] == "CRITICAL":
            return "Inspect urgently"
        elif row["severity_label"] == "WARNING":
            return "Increase monitoring frequency"
        elif row["severity_label"] == "WATCHLIST":
            return "Watchlist"
        else:
            return "Routine monitoring"

    latest_df["recommended_action"] = latest_df.apply(recommend_action, axis=1)

    # Chọn cột xuất
    out_cols = [
        "transformer_id", "loc", "name",
        "severity_score", "severity_label", "consensus_fault",
        "sample_day", "diagnostic_confidence",
        "trend_slope", "final_score", "rank", "recommended_action",
    ]
    available_cols = [c for c in out_cols if c in latest_df.columns]
    ranking = latest_df[available_cols].copy()

    top5 = ranking.head(5)
    log_cols = ["rank", "transformer_id", "final_score", "severity_label", "consensus_fault"]
    if all(c in top5.columns for c in log_cols):
        logger.info("Top 5 ranking:")
        logger.info("\n" + top5[log_cols].to_string())
    logger.info("Ranking generation complete.")
    return ranking