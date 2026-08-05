"""
Tạo LF stats và transformer ranking từ weak labels (dga_weak_labels.parquet).
Chỉ giữ phần quan trọng, không sao lưu hay replace UNCERTAIN.
"""
from pathlib import Path
import pandas as pd
import numpy as np
from logging_config import init_logging
init_logging()
import logging
logger = logging.getLogger(__name__)

from weak_supervision import ABSTAIN, build_label_matrix
from severity import apply_severity
from config import config as cfg, BACKEND_DATA_DIR

DATA_DIR = Path(BACKEND_DATA_DIR)
WEAK_LABEL_PATH = DATA_DIR / "dga_weak_labels.parquet"
REPORT_DIR = Path("reports")
REPORT_DIR.mkdir(exist_ok=True)

def generate_lf_stats(df: pd.DataFrame):
    L, lf_names, groups = build_label_matrix(df)
    stats = []
    for j, lf in enumerate(lf_names):
        abstain_rate = (L[:, j] == ABSTAIN).mean()
        votes = {}
        for idx, g in enumerate(groups):
            votes[g] = int((L[:, j] == idx).sum())
        stats.append({"lf": lf, "abstain_rate": float(abstain_rate), **votes})
    return pd.DataFrame(stats)

def generate_transformer_ranking(df: pd.DataFrame):
    # Đảm bảo có severity score (sử dụng consensus_fault nếu có, nếu không có thì tạo)
    if "consensus_fault" not in df.columns:
        from consensus import apply_consensus
        df = apply_consensus(df)
    df = apply_severity(df)

    w = cfg.RANKING_WEIGHTS
    results = []
    for tid, grp in df.groupby("transformer_id"):
        grp2 = grp.sort_values("sample_day")
        latest = grp2.iloc[-1]
        latest_sev = float(latest.get("severity_score", np.nan))
        latest_conf = float(latest.get("weak_fault_confidence", 0.0))
        sev_series = grp2["severity_score"].fillna(0.0)
        ewma = float(sev_series.ewm(alpha=0.3).mean().iloc[-1]) if len(sev_series) > 0 else 0.0
        recent = sev_series.tail(cfg.RECENT_SAMPLES_FOR_TREND)
        if len(recent) >= 2:
            x = np.arange(len(recent))
            y = recent.values
            slope = float(np.polyfit(x, y, 1)[0])
        else:
            slope = 0.0
        crit_count = int((grp2["severity_label"] == "CRITICAL").sum()) if "severity_label" in grp2.columns else 0
        if "weak_fault_group" in grp2.columns:
            recent_groups = grp2["weak_fault_group"].tail(cfg.RECENT_SAMPLES_FOR_PERSISTENCE)
            persistence_bonus = float((recent_groups == recent_groups.iloc[-1]).sum() / max(1, len(recent_groups)))
        else:
            persistence_bonus = 0.0

        raw_score = (
            w["current"] * latest_sev +
            w["history"] * ewma +
            w["trend"] * slope +
            w["confidence"] * (latest_conf * 10.0) +
            w["critical_history"] * crit_count +
            cfg.PERSISTENCE_BONUS_FACTOR * persistence_bonus
        )
        results.append({
            "transformer_id": tid,
            "latest_severity": latest_sev,
            "ewma_severity": ewma,
            "trend_slope": slope,
            "critical_count": crit_count,
            "latest_confidence": latest_conf,
            "persistence_bonus": persistence_bonus,
            "raw_score": raw_score,
        })

    rank_df = pd.DataFrame(results)
    if not rank_df.empty:
        minv, maxv = rank_df["raw_score"].min(), rank_df["raw_score"].max()
        if maxv - minv > 0:
            rank_df["final_score"] = 100.0 * (rank_df["raw_score"] - minv) / (maxv - minv)
        else:
            rank_df["final_score"] = 0.0
        rank_df = rank_df.sort_values("final_score", ascending=False).reset_index(drop=True)
        rank_df["rank"] = rank_df.index + 1
    return rank_df

if __name__ == "__main__":
    if not WEAK_LABEL_PATH.exists():
        logger.error(f"Không tìm thấy {WEAK_LABEL_PATH}. Hãy chạy train_models.py --weak-supervision --weak-only --use-snorkel trước.")
        exit(1)

    df = pd.read_parquet(WEAK_LABEL_PATH)

    lf_stats = generate_lf_stats(df)
    lf_stats.to_csv(REPORT_DIR / "lf_stats.csv", index=False)
    logger.info(f"LF stats → {REPORT_DIR / 'lf_stats.csv'}")

    ranking = generate_transformer_ranking(df)
    ranking.to_csv(REPORT_DIR / "transformer_ranking.csv", index=False)
    logger.info(f"Transformer ranking → {REPORT_DIR / 'transformer_ranking.csv'}")
    logger.info("Hoàn tất.")