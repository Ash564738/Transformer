# ranking.py
from __future__ import annotations
import logging
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def compute_trend_slope(series: pd.Series, window: int = 5) -> pd.Series:
    def slope_func(arr):
        arr = np.asarray(arr, dtype=float)
        finite = np.isfinite(arr)
        arr = arr[finite]
        if len(arr) < 2:
            return 0.0
        x = np.arange(len(arr), dtype=float)
        slope = np.polyfit(x, arr, 1)[0]
        if not np.isfinite(slope):
            return 0.0
        return float(slope)

    return series.rolling(window=window, min_periods=2).apply(slope_func, raw=True)


def build_transformer_ranking(df: pd.DataFrame) -> pd.DataFrame:
    logger.info("Building transformer priority ranking...")
    if "transformer_id" not in df.columns:
        raise ValueError("Missing transformer_id.")
    if "sample_day" not in df.columns:
        raise ValueError("Missing sample_day.")

    df = df.copy()
    df["sample_day"] = pd.to_datetime(df["sample_day"], errors="coerce")
    df = df.dropna(subset=["transformer_id", "sample_day"])
    df = df.sort_values(["transformer_id", "sample_day"])

    if "severity_score" in df.columns:
        df["fleet_severity_trend_slope"] = (
            df.groupby("transformer_id")["severity_score"]
            .transform(lambda s: compute_trend_slope(s, window=5))
        )
    else:
        df["fleet_severity_trend_slope"] = 0.0

    latest_idx = df.groupby("transformer_id")["sample_day"].idxmax()
    latest_df = df.loc[latest_idx].copy()
    logger.info("Transformers in ranking: %d", len(latest_df))

    if "severity_score" in latest_df.columns:
        severity_score = pd.to_numeric(latest_df["severity_score"], errors="coerce").fillna(0.0)
        severity_score = severity_score.clip(0.0, 100.0) / 100.0
    else:
        severity_score = pd.Series(0.0, index=latest_df.index)

    if "anomaly_percentile" in latest_df.columns:
        anomaly_score = pd.to_numeric(latest_df["anomaly_percentile"], errors="coerce").fillna(0.0)
        anomaly_score = anomaly_score.clip(0.0, 1.0)
    else:
        anomaly_score = pd.Series(0.0, index=latest_df.index)

    trend = pd.to_numeric(latest_df["fleet_severity_trend_slope"], errors="coerce")
    positive_trend = trend.clip(lower=0.0)
    if positive_trend.notna().any():
        trend_rank = positive_trend.rank(pct=True, method="average").fillna(0.0)
    else:
        trend_rank = pd.Series(0.0, index=latest_df.index)

    if "ieee_dga_status" in latest_df.columns:
        dga_status = pd.to_numeric(latest_df["ieee_dga_status"], errors="coerce").fillna(1.0)
    else:
        dga_status = pd.Series(1.0, index=latest_df.index)

    status_score = ((dga_status - 1.0) / 2.0).clip(0.0, 1.0)

    latest_df["fleet_priority_score"] = (
        0.50 * severity_score + 0.30 * anomaly_score + 0.20 * trend_rank
    )
    latest_df["fleet_priority_score"] = np.maximum(
        latest_df["fleet_priority_score"].to_numpy(), 0.50 * status_score.to_numpy()
    )
    latest_df["fleet_priority_score"] = latest_df["fleet_priority_score"].clip(0.0, 1.0)
    latest_df["fleet_priority_percent"] = (latest_df["fleet_priority_score"] * 100.0).round(1)

    latest_df = latest_df.sort_values(
        ["fleet_priority_score", "sample_day"], ascending=[False, False]
    ).reset_index(drop=True)
    latest_df["rank"] = np.arange(len(latest_df)) + 1

    def recommend_action(row):
        status = int(row.get("ieee_dga_status", 1))
        fault = str(row.get("consensus_fault", "ABSTAIN")).upper()
        confirmation = bool(row.get("ieee_confirmation_required", False))
        extreme = bool(row.get("ieee_extreme_dga", False))
        if extreme:
            return "Immediate engineering review"
        if status >= 3:
            return "Increase surveillance and investigate"
        if confirmation:
            return "Obtain confirmation DGA"
        if fault in {"D2", "T3", "T3_H", "MIXED"} and status >= 2:
            return "Increase monitoring and fault investigation"
        if status == 2:
            return "Increase monitoring frequency"
        return "Routine monitoring"

    latest_df["recommended_action"] = latest_df.apply(recommend_action, axis=1)

    out_cols = [
        "transformer_id", "loc", "name", "sample_day", "ieee_dga_status",
        "ieee_dga_status_label", "ieee_dga_status_reason", "severity_score",
        "severity_label_text", "consensus_fault", "diagnostic_confidence",
        "anomaly_percentile", "fleet_severity_trend_slope", "fleet_priority_percent",
        "fleet_priority_score", "rank", "recommended_action",
    ]
    available_cols = [col for col in out_cols if col in latest_df.columns]
    ranking = latest_df[available_cols].copy()
    ranking["final_score"] = (
        ranking["fleet_priority_percent"] if "fleet_priority_percent" in ranking.columns else np.nan
    )

    log_cols = [col for col in ["rank", "transformer_id", "fleet_priority_percent",
                                "ieee_dga_status_label", "consensus_fault"] if col in ranking.columns]
    if log_cols:
        logger.info("Top 5 transformer priority ranking:\n%s", ranking.head(5)[log_cols].to_string(index=False))

    logger.info("Ranking generation complete.")
    return ranking