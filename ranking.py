# ranking.py
from __future__ import annotations
import numpy as np
import pandas as pd
import logging
from config import FAULT_GROUPS

logger = logging.getLogger(__name__)

WEIGHT_CURRENT = 0.55
WEIGHT_HISTORY = 0.25
WEIGHT_TREND = 0.10
WEIGHT_CONFIDENCE = 0.05
WEIGHT_CRITICAL_HISTORY = 0.05
PERSISTENCE_BONUS_FACTOR = 0.15   # hệ số nhân cho persistence

TREND_SLOPE_WORSENING = 0.5
TREND_SLOPE_IMPROVING = -0.5

RECENT_SAMPLES_FOR_TREND = 5
RECENT_SAMPLES_FOR_HISTORY = 5
RECENT_SAMPLES_FOR_PERSISTENCE = 5

ACTION_THRESHOLD_CRITICAL = 12
ACTION_THRESHOLD_WARNING = 6

def compute_trend(row: pd.Series, df: pd.DataFrame) -> str:
    mask = (df["transformer_id"] == row["transformer_id"]) & (df["sample_day"] <= row["sample_day"])
    recent = df.loc[mask].sort_values("sample_day").tail(RECENT_SAMPLES_FOR_TREND)
    if len(recent) < 2:
        return "stable"
    scores = recent["severity_score"].values
    x = np.arange(len(scores))
    slope = np.polyfit(x, scores, 1)[0]
    if slope > TREND_SLOPE_WORSENING:
        return "worsening"
    elif slope < TREND_SLOPE_IMPROVING:
        return "improving"
    else:
        return "stable"

def compute_historical_severity(row: pd.Series, df: pd.DataFrame) -> float:
    mask = (df["transformer_id"] == row["transformer_id"]) & (df["sample_day"] <= row["sample_day"])
    hist = df.loc[mask].sort_values("sample_day").tail(RECENT_SAMPLES_FOR_HISTORY)
    if len(hist) < 2:
        return row["severity_score"]
    weights = np.arange(1, len(hist) + 1, dtype=float)
    weights /= weights.sum()
    return float(np.average(hist["severity_score"].values, weights=weights))

def compute_critical_history(transformer_id: str, df: pd.DataFrame, current_date) -> dict:
    hist = df[(df["transformer_id"] == transformer_id) & (df["sample_day"] <= current_date)]
    critical_mask = hist["severity_label"] == "CRITICAL"
    n_critical = critical_mask.sum()
    if n_critical > 0:
        last_critical_date = hist.loc[critical_mask, "sample_day"].max()
        days_since = (current_date - last_critical_date).days
    else:
        days_since = np.nan
    return {"n_critical": n_critical, "days_since_last_critical": days_since}

def fault_persistence_score(row: pd.Series, df: pd.DataFrame) -> float:
    """
    Tỉ lệ số mẫu có severity >= WARNING trong lịch sử gần đây (5 mẫu).
    """
    mask = (df["transformer_id"] == row["transformer_id"]) & (df["sample_day"] <= row["sample_day"])
    recent = df.loc[mask].sort_values("sample_day").tail(RECENT_SAMPLES_FOR_PERSISTENCE)
    if len(recent) == 0:
        return 0.0
    high_sev = recent["severity_label"].isin(["WARNING", "CRITICAL"]).sum()
    return high_sev / len(recent)

def build_transformer_ranking(df: pd.DataFrame) -> pd.DataFrame:
    logger.info("Bắt đầu xây dựng ranking...")
    latest_idx = df.groupby("transformer_id")["sample_day"].idxmax()
    latest_df = df.loc[latest_idx].copy()

    trend_list = []
    hist_sev_list = []
    n_crit_list = []
    days_since_list = []
    persistence_list = []

    for _, row in latest_df.iterrows():
        trend = compute_trend(row, df)
        trend_list.append(trend)
        hist_score = compute_historical_severity(row, df)
        hist_sev_list.append(hist_score)
        crit = compute_critical_history(row["transformer_id"], df, row["sample_day"])
        n_crit_list.append(crit["n_critical"])
        days_since_list.append(crit["days_since_last_critical"])
        persistence = fault_persistence_score(row, df)
        persistence_list.append(persistence)

    latest_df["trend"] = trend_list
    latest_df["historical_severity"] = hist_sev_list
    latest_df["n_critical_past"] = n_crit_list
    latest_df["days_since_last_critical"] = days_since_list
    latest_df["fault_persistence"] = persistence_list

    sev_current = latest_df["severity_score"].astype(float)
    trend_bonus = latest_df["trend"].map({"worsening": 2.0, "stable": 0.0, "improving": -1.0}).fillna(0)
    crit_bonus = latest_df["n_critical_past"].clip(upper=3).fillna(0)
    conf_penalty = (latest_df["diagnostic_confidence"] < 50).astype(float) * 1.0

    base_score = (
        WEIGHT_CURRENT * sev_current +
        WEIGHT_HISTORY * latest_df["historical_severity"] +
        WEIGHT_TREND * trend_bonus +
        WEIGHT_CRITICAL_HISTORY * crit_bonus +
        WEIGHT_CONFIDENCE * (1.0 - conf_penalty)
    )

    latest_df["final_score"] = base_score * (1 + PERSISTENCE_BONUS_FACTOR * latest_df["fault_persistence"])

    latest_df = latest_df.sort_values("final_score", ascending=False).reset_index(drop=True)
    latest_df["rank"] = range(1, len(latest_df) + 1)

    def recommend_action(row):
        base = ""
        if row["final_score"] >= ACTION_THRESHOLD_CRITICAL:
            base = f"CRITICAL: Inspect urgently – likely {row.get('consensus_fault', 'unknown fault')}"
        elif row["final_score"] >= ACTION_THRESHOLD_WARNING:
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

    # Debug
    top5 = ranking.head(5)
    logger.info("Top 5 ranking:\n" + top5[["rank", "transformer_id", "final_score", "severity_label", "consensus_fault"]].to_string())
    print("=== DEBUG RANKING (top 5) ===")
    print(top5[["rank", "transformer_id", "final_score", "severity_label", "consensus_fault"]].to_string())
    print("=============================")

    return ranking