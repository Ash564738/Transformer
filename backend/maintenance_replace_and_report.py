import os
import shutil
import pandas as pd
import numpy as np
from pathlib import Path

from weak_supervision import ABSTAIN, build_label_matrix
from consensus import apply_consensus
from severity import apply_severity
from config import config as cfg

DATA_DIR = Path("dataset/processed")
BACKUP_DIR = DATA_DIR / "backups_uncertain"
REPORT_DIR = Path("reports")
REPORT_DIR.mkdir(exist_ok=True)


def backup_file(src: Path, dst_dir: Path):
    dst_dir.mkdir(parents=True, exist_ok=True)
    dst = dst_dir / src.name
    if not dst.exists():
        shutil.copy2(src, dst)


def replace_uncertain_in_parquet(parquet_path: Path) -> int:
    """Replace literal 'UNCERTAIN' strings in string/object columns with 'ABSTAIN'.
    Returns number of replacements made.
    """
    df = pd.read_parquet(parquet_path)
    replaced = 0
    cols = [c for c in df.columns if df[c].dtype == object or pd.api.types.is_string_dtype(df[c].dtype)]
    if not cols:
        return 0

    # Backup original
    backup_file(parquet_path, BACKUP_DIR)

    for c in cols:
        try:
            mask = df[c].astype(str) == "UNCERTAIN"
        except Exception:
            # skip columns that cannot be cast to str
            continue
        if mask.any():
            df.loc[mask, c] = "ABSTAIN"
            replaced += int(mask.sum())
    if replaced > 0:
        df.to_parquet(parquet_path)
    return replaced


def scan_and_replace_all():
    parquet_files = list(DATA_DIR.glob("*.parquet"))
    total_replaced = 0
    per_file = {}
    for p in parquet_files:
        cnt = replace_uncertain_in_parquet(p)
        per_file[p.name] = cnt
        total_replaced += cnt
    return total_replaced, per_file


def generate_lf_stats(df: pd.DataFrame):
    # Build label matrix using default LF mapping
    L, lf_names, groups = build_label_matrix(df)
    stats = []
    for j, lf in enumerate(lf_names):
        col = lf
        abstain_rate = (L[:, j] == ABSTAIN).mean()
        votes = {}
        for idx, g in enumerate(groups):
            votes[g] = int((L[:, j] == idx).sum())
        stats.append({"lf": lf, "abstain_rate": float(abstain_rate), **votes})
    return pd.DataFrame(stats)


def generate_transformer_ranking(df: pd.DataFrame):
    # Ensure consensus_fault exists
    if "consensus_fault" not in df.columns:
        df = apply_consensus(df)
    df = apply_severity(df)

    # For each transformer, get latest sample and compute history EWMA and trend
    w = cfg.RANKING_WEIGHTS
    results = []
    for tid, grp in df.groupby("transformer_id"):
        grp2 = grp.sort_values("sample_day")
        latest = grp2.iloc[-1]
        latest_sev = float(latest.get("severity_score", np.nan))
        latest_conf = float(latest.get("weak_fault_confidence", 0.0))
        # EWMA over severity_score
        sev_series = grp2["severity_score"].fillna(0.0)
        if len(sev_series) == 0:
            ewma = 0.0
        else:
            ewma = float(sev_series.ewm(alpha=0.3).mean().iloc[-1])
        # Trend: slope over last N points
        recent = sev_series.tail(cfg.RECENT_SAMPLES_FOR_TREND)
        if len(recent) >= 2:
            x = np.arange(len(recent))
            y = recent.values
            slope = float(np.polyfit(x, y, 1)[0])
        else:
            slope = 0.0
        # critical history count
        crit_count = int((grp2["severity_label"] == "CRITICAL").sum()) if "severity_label" in grp2.columns else 0
        # persistence: count of repeated same weak_fault_group in recent samples
        if "weak_fault_group" in grp2.columns:
            recent_groups = grp2["weak_fault_group"].tail(cfg.RECENT_SAMPLES_FOR_PERSISTENCE)
            persistence_bonus = float((recent_groups == recent_groups.iloc[-1]).sum() / max(1, len(recent_groups)))
        else:
            persistence_bonus = 0.0

        # Compose raw score
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
    # Normalize raw_score to 0-100
    if not rank_df.empty:
        minv = rank_df["raw_score"].min()
        maxv = rank_df["raw_score"].max()
        if maxv - minv > 0:
            rank_df["final_score"] = 100.0 * (rank_df["raw_score"] - minv) / (maxv - minv)
        else:
            rank_df["final_score"] = 0.0
        rank_df = rank_df.sort_values("final_score", ascending=False).reset_index(drop=True)
        rank_df["rank"] = rank_df.index + 1
    return rank_df


if __name__ == "__main__":
    print("Backing up and replacing 'UNCERTAIN' -> 'ABSTAIN' in Parquet files...")
    total, per = scan_and_replace_all()
    print(f"Total replacements: {total}")
    for k, v in per.items():
        print(f"  {k}: {v}")

    print("\nGenerating LF stats and transformer ranking from dga_weak_labels.parquet...")
    wp = DATA_DIR / "dga_weak_labels.parquet"
    if not wp.exists():
        raise FileNotFoundError(wp)
    df = pd.read_parquet(wp)

    lf_stats = generate_lf_stats(df)
    lf_stats.to_csv(REPORT_DIR / "lf_stats.csv", index=False)
    print(f"LF stats written to {REPORT_DIR / 'lf_stats.csv'}")

    ranking = generate_transformer_ranking(df)
    ranking.to_csv(REPORT_DIR / "transformer_ranking.csv", index=False)
    print(f"Transformer ranking written to {REPORT_DIR / 'transformer_ranking.csv'}")

    print("Done.")
