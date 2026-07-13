# generate_labels.py
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


# =============================================================================
# CONFIG
# =============================================================================

FEATURES_PATH = Path(r"dataset\processed\dga_features.parquet")
OUTPUT_DIR = Path(r"dataset\processed")

CORE_GASES = ["h2", "ch4", "c2h6", "c2h4", "c2h2", "co", "co2"]
REQUIRED_BASE_COLS = ["transformer_id", "sample_day"]

# IEEE-like threshold baseline for severity scoring
THRESHOLDS = {
    "h2": [100, 200, 500],
    "ch4": [120, 400, 1000],
    "c2h6": [65, 100, 150],
    "c2h4": [50, 100, 200],
    "c2h2": [3, 35, 80],
    "co": [350, 700, 1400],
    "co2": [2500, 5000, 10000],
    "tdcg": [720, 1920, 4630],
}

FAULT_ORDER = [
    "Normal",
    "PD",
    "D1",
    "D2",
    "T1",
    "T2",
    "T3",
    "Cellulose",
    "Mixed",
    "Uncertain",
]

FAULT_SEVERITY_POINTS = {
    "Normal": 0,
    "PD": 1,
    "T1": 1,
    "T2": 2,
    "D1": 2,
    "T3": 3,
    "D2": 4,
    "Cellulose": 2,
    "Mixed": 3,
    "Uncertain": 1,
}


# =============================================================================
# HELPERS
# =============================================================================

def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def safe_numeric(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    out = df.copy()
    for c in cols:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce")
    return out


def validate_input(df: pd.DataFrame) -> None:
    missing = [c for c in REQUIRED_BASE_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Input feature file is missing required columns: {missing}")


def score_by_threshold(value: float, thresholds: List[float]) -> int:
    if pd.isna(value):
        return 0
    if value >= thresholds[2]:
        return 3
    if value >= thresholds[1]:
        return 2
    if value >= thresholds[0]:
        return 1
    return 0


def compute_gas_level_score(row: pd.Series) -> Tuple[int, Dict[str, int]]:
    scores = {}
    total = 0
    for gas, thr in THRESHOLDS.items():
        val = row.get(gas, np.nan)
        s = score_by_threshold(val, thr)
        scores[gas] = s
        total += s
    return total, scores


def compute_trend_score(row: pd.Series) -> Tuple[int, Dict[str, int]]:
    """
    Trend severity:
    - tdcg_rate_per_day
    - số khí tăng
    - delta/rate của H2, C2H2, C2H4, CO
    """
    score = 0
    detail = {}

    tdcg_rate = row.get("tdcg_rate_per_day", np.nan)
    if pd.notna(tdcg_rate):
        if tdcg_rate >= 10:
            score += 3
            detail["tdcg_rate"] = 3
        elif tdcg_rate >= 3:
            score += 2
            detail["tdcg_rate"] = 2
        elif tdcg_rate > 0:
            score += 1
            detail["tdcg_rate"] = 1
        else:
            detail["tdcg_rate"] = 0
    else:
        detail["tdcg_rate"] = 0

    num_inc = row.get("num_gases_increasing", np.nan)
    if pd.notna(num_inc):
        if num_inc >= 5:
            score += 2
            detail["num_gases_increasing"] = 2
        elif num_inc >= 3:
            score += 1
            detail["num_gases_increasing"] = 1
        else:
            detail["num_gases_increasing"] = 0
    else:
        detail["num_gases_increasing"] = 0

    for gas in ["h2", "c2h2", "c2h4", "co"]:
        rate = row.get(f"{gas}_rate_per_day", np.nan)
        if pd.isna(rate):
            detail[f"{gas}_rate"] = 0
            continue

        if gas == "c2h2":
            if rate >= 0.5:
                score += 2
                detail[f"{gas}_rate"] = 2
            elif rate > 0:
                score += 1
                detail[f"{gas}_rate"] = 1
            else:
                detail[f"{gas}_rate"] = 0
        else:
            if rate >= 3:
                score += 2
                detail[f"{gas}_rate"] = 2
            elif rate > 0:
                score += 1
                detail[f"{gas}_rate"] = 1
            else:
                detail[f"{gas}_rate"] = 0

    return score, detail


# =============================================================================
# DGA RULE METHODS
# =============================================================================

def _norm_fault(label: Optional[str]) -> str:
    if label is None:
        return "Uncertain"
    x = str(label).strip()
    if x in FAULT_ORDER:
        return x
    return "Uncertain"


def classify_ieee_key_gas(row: pd.Series) -> Tuple[str, Dict[str, str]]:
    """
    Key-gas style heuristic theo IEEE baseline.
    Đây không phải full annex table parser, nhưng là logic thực dụng để gán weak label.
    """
    h2 = row.get("h2", np.nan)
    ch4 = row.get("ch4", np.nan)
    c2h6 = row.get("c2h6", np.nan)
    c2h4 = row.get("c2h4", np.nan)
    c2h2 = row.get("c2h2", np.nan)
    co = row.get("co", np.nan)
    co2 = row.get("co2", np.nan)
    tdcg = row.get("tdcg", np.nan)
    ratio_co2_co = row.get("ratio_co2_co", np.nan)

    if (
        pd.notna(tdcg) and tdcg < 300
        and (pd.isna(h2) or h2 < 100)
        and (pd.isna(ch4) or ch4 < 120)
        and (pd.isna(c2h2) or c2h2 < 3)
        and (pd.isna(c2h4) or c2h4 < 50)
        and (pd.isna(co) or co < 350)
    ):
        return "Normal", {"method": "ieee_key_gas", "rule": "low_all"}

    cellulose_signal = (
        pd.notna(co) and co >= 700 and pd.notna(ratio_co2_co) and ratio_co2_co < 3
    )
    if cellulose_signal:
        return "Cellulose", {"method": "ieee_key_gas", "rule": "high_co_low_co2co"}

    if pd.notna(c2h2) and c2h2 >= 35:
        return "D2", {"method": "ieee_key_gas", "rule": "high_c2h2"}

    if pd.notna(c2h2) and c2h2 >= 3:
        return "D1", {"method": "ieee_key_gas", "rule": "c2h2_present"}

    if pd.notna(h2) and h2 >= 200 and (pd.isna(c2h2) or c2h2 < 3) and (pd.isna(c2h4) or c2h4 < 50):
        return "PD", {"method": "ieee_key_gas", "rule": "high_h2"}

    if pd.notna(c2h4) and c2h4 >= 200:
        return "T3", {"method": "ieee_key_gas", "rule": "very_high_c2h4"}

    if pd.notna(c2h4) and 100 <= c2h4 < 200:
        return "T2", {"method": "ieee_key_gas", "rule": "high_c2h4"}

    if pd.notna(ch4) and ch4 >= 120:
        if pd.notna(c2h6) and c2h6 >= 65:
            return "T1", {"method": "ieee_key_gas", "rule": "ch4_c2h6"}
        return "T1", {"method": "ieee_key_gas", "rule": "high_ch4"}

    return "Uncertain", {"method": "ieee_key_gas", "rule": "fallback"}


def classify_iec_ratio(row: pd.Series) -> Tuple[str, Dict[str, str]]:
    """
    IEC 60599 ratio baseline using:
    - CH4/H2
    - C2H2/C2H4
    - C2H4/C2H6
    """
    r1 = row.get("ratio_ch4_h2", np.nan)
    r2 = row.get("ratio_c2h2_c2h4", np.nan)
    r3 = row.get("ratio_c2h4_c2h6", np.nan)

    if pd.isna(r1) and pd.isna(r2) and pd.isna(r3):
        return "Uncertain", {"method": "iec_ratio", "rule": "missing_ratios"}

    # baseline practical mapping
    # PD: CH4/H2 < 0.1 and C2H2/C2H4 < 0.2 and C2H4/C2H6 < 1
    if pd.notna(r1) and pd.notna(r2) and pd.notna(r3):
        if r1 < 0.1 and r2 < 0.2 and r3 < 1:
            return "PD", {"method": "iec_ratio", "rule": "pd_pattern"}
        if r1 > 0.1 and r1 < 1 and r2 > 1 and r3 > 1:
            return "D1", {"method": "iec_ratio", "rule": "d1_pattern"}
        if r1 >= 0.6 and r2 > 2.5 and r3 > 2:
            return "D2", {"method": "iec_ratio", "rule": "d2_pattern"}
        if r2 < 0.1 and r3 < 1 and r1 > 1:
            return "T1", {"method": "iec_ratio", "rule": "t1_pattern"}
        if r2 < 0.2 and 1 <= r3 < 4:
            return "T2", {"method": "iec_ratio", "rule": "t2_pattern"}
        if r2 < 0.2 and r3 >= 4:
            return "T3", {"method": "iec_ratio", "rule": "t3_pattern"}

    # soft fallback
    if pd.notna(r2) and r2 >= 1:
        return "D2", {"method": "iec_ratio", "rule": "high_c2h2_c2h4"}
    if pd.notna(r3) and r3 >= 4:
        return "T3", {"method": "iec_ratio", "rule": "high_c2h4_c2h6"}
    if pd.notna(r3) and 1 <= r3 < 4:
        return "T2", {"method": "iec_ratio", "rule": "mid_c2h4_c2h6"}
    if pd.notna(r1) and r1 < 0.1:
        return "PD", {"method": "iec_ratio", "rule": "low_ch4_h2"}

    return "Uncertain", {"method": "iec_ratio", "rule": "fallback"}


def classify_rogers_ratio(row: pd.Series) -> Tuple[str, Dict[str, str]]:
    """
    Rogers-style heuristic using:
    - CH4/H2
    - C2H2/C2H4
    - C2H4/C2H6
    - C2H6/CH4
    """
    r1 = row.get("ratio_ch4_h2", np.nan)
    r2 = row.get("ratio_c2h2_c2h4", np.nan)
    r3 = row.get("ratio_c2h4_c2h6", np.nan)
    r4 = row.get("ratio_c2h6_ch4", np.nan)

    if pd.isna(r1) and pd.isna(r2) and pd.isna(r3):
        return "Uncertain", {"method": "rogers_ratio", "rule": "missing_ratios"}

    if pd.notna(r1) and pd.notna(r2) and pd.notna(r3):
        if r1 < 0.1 and r2 < 0.1 and r3 < 1:
            return "PD", {"method": "rogers_ratio", "rule": "pd"}
        if 0.1 <= r1 < 1 and r2 >= 0.1 and r2 < 3 and r3 > 1:
            return "D1", {"method": "rogers_ratio", "rule": "d1"}
        if r2 >= 3:
            return "D2", {"method": "rogers_ratio", "rule": "d2"}
        if r2 < 0.1 and r3 < 1:
            return "T1", {"method": "rogers_ratio", "rule": "t1"}
        if r2 < 0.1 and 1 <= r3 < 3:
            return "T2", {"method": "rogers_ratio", "rule": "t2"}
        if r2 < 0.1 and r3 >= 3:
            return "T3", {"method": "rogers_ratio", "rule": "t3"}

    if pd.notna(r4) and pd.notna(r3):
        if r4 > 1 and r3 < 1:
            return "T1", {"method": "rogers_ratio", "rule": "t1_fallback"}

    return "Uncertain", {"method": "rogers_ratio", "rule": "fallback"}


def classify_duval_triangle_1(row: pd.Series) -> Tuple[str, Dict[str, str]]:
    """
    Duval Triangle 1 simplified zone approximation using
    %CH4, %C2H4, %C2H2.
    Đây là polygon-free approximation để sinh weak label.
    """
    ch4 = row.get("duval1_pct_ch4", np.nan)
    c2h4 = row.get("duval1_pct_c2h4", np.nan)
    c2h2 = row.get("duval1_pct_c2h2", np.nan)

    if pd.isna(ch4) or pd.isna(c2h4) or pd.isna(c2h2):
        return "Uncertain", {"method": "duval_triangle_1", "rule": "missing_pct"}

    # Approximate zones
    if c2h2 <= 4 and c2h4 <= 20 and ch4 >= 70:
        return "PD", {"method": "duval_triangle_1", "rule": "pd_zone_like"}

    if 4 < c2h2 < 23 and c2h4 <= 40:
        return "D1", {"method": "duval_triangle_1", "rule": "d1_zone_like"}

    if c2h2 >= 23:
        return "D2", {"method": "duval_triangle_1", "rule": "d2_zone_like"}

    if c2h4 < 20 and ch4 >= 50 and c2h2 < 4:
        return "T1", {"method": "duval_triangle_1", "rule": "t1_zone_like"}

    if 20 <= c2h4 < 50 and c2h2 < 15:
        return "T2", {"method": "duval_triangle_1", "rule": "t2_zone_like"}

    if c2h4 >= 50 and c2h2 < 15:
        return "T3", {"method": "duval_triangle_1", "rule": "t3_zone_like"}

    return "Uncertain", {"method": "duval_triangle_1", "rule": "fallback"}

def classify_duval_pentagon(row: pd.Series) -> Tuple[str, Dict[str, str]]:
    """
    Simplified Duval Pentagon 1 zone approximation using
    %H2, %CH4, %C2H6, %C2H4, %C2H2.
    This yields a weak label for voting.
    """
    h2 = row.get("duval_pent_pct_h2", np.nan)
    ch4 = row.get("duval_pent_pct_ch4", np.nan)
    c2h6 = row.get("duval_pent_pct_c2h6", np.nan)
    c2h4 = row.get("duval_pent_pct_c2h4", np.nan)
    c2h2 = row.get("duval_pent_pct_c2h2", np.nan)

    if any(pd.isna(v) for v in [h2, ch4, c2h6, c2h4, c2h2]):
        return "Uncertain", {"method": "duval_pentagon", "rule": "missing_pct"}

    # Very rough zone approximations (based on common zone boundaries)
    # PD: high H2, very low C2H2, moderate CH4
    if h2 > 60 and c2h2 < 5 and ch4 < 40:
        return "PD", {"method": "duval_pentagon", "rule": "pd_zone"}

    # D1: moderate C2H2, low C2H4
    if 5 <= c2h2 < 30 and c2h4 < 30:
        return "D1", {"method": "duval_pentagon", "rule": "d1_zone"}

    # D2: high C2H2
    if c2h2 >= 30:
        return "D2", {"method": "duval_pentagon", "rule": "d2_zone"}

    # T1: low C2H4, high CH4
    if c2h4 < 20 and ch4 > 40 and c2h2 < 5:
        return "T1", {"method": "duval_pentagon", "rule": "t1_zone"}

    # T2: medium C2H4
    if 20 <= c2h4 < 50 and c2h2 < 10:
        return "T2", {"method": "duval_pentagon", "rule": "t2_zone"}

    # T3: high C2H4
    if c2h4 >= 50 and c2h2 < 15:
        return "T3", {"method": "duval_pentagon", "rule": "t3_zone"}

    return "Uncertain", {"method": "duval_pentagon", "rule": "fallback"}

def classify_cellulose_override(row: pd.Series) -> Optional[str]:
    co = row.get("co", np.nan)
    ratio_co2_co = row.get("ratio_co2_co", np.nan)
    if pd.notna(co) and co >= 700 and pd.notna(ratio_co2_co) and ratio_co2_co < 3:
        return "Cellulose"
    return None


def aggregate_fault_votes(votes: Dict[str, str], row: pd.Series) -> Tuple[str, Dict[str, object]]:
    """
    Consensus / voting:
    1) cellulose override nếu rõ
    2) nếu >=2 methods cùng 1 nhãn -> lấy nhãn đó
    3) nếu có conflict mạnh giữa discharge/thermal/pd/cellulose -> Mixed
    4) fallback ưu tiên IEEE -> IEC -> Duval -> Rogers
    """
    cellulose = classify_cellulose_override(row)
    if cellulose is not None:
        return cellulose, {
            "strategy": "cellulose_override",
            "votes": votes,
        }

    labels = [_norm_fault(v) for v in votes.values()]
    valid = [x for x in labels if x != "Uncertain"]

    if not valid:
        return "Uncertain", {"strategy": "all_uncertain", "votes": votes}

    vc = pd.Series(valid).value_counts()
    top_label = vc.index[0]
    top_count = int(vc.iloc[0])

    if top_count >= 2:
        return top_label, {
            "strategy": "majority_vote",
            "votes": votes,
            "vote_count": vc.to_dict(),
        }

    discharge = {"PD", "D1", "D2"}
    thermal = {"T1", "T2", "T3"}

    groups = set()
    for x in valid:
        if x in discharge:
            groups.add("discharge")
        elif x in thermal:
            groups.add("thermal")
        elif x == "Cellulose":
            groups.add("cellulose")
        elif x == "Normal":
            groups.add("normal")

    if len(groups) >= 2 and "normal" not in groups:
        return "Mixed", {
            "strategy": "conflict_groups_mixed",
            "votes": votes,
            "groups": sorted(groups),
        }

    # fallback priority
    for k in ["ieee_key_gas", "iec_ratio", "duval_triangle_1", "rogers_ratio"]:
        v = _norm_fault(votes.get(k))
        if v != "Uncertain":
            return v, {"strategy": f"priority_{k}", "votes": votes}

    return "Uncertain", {"strategy": "fallback_uncertain", "votes": votes}


def classify_fault_consensus(row: pd.Series) -> Tuple[str, Dict[str, object]]:
    ieee_label, ieee_info = classify_ieee_key_gas(row)
    iec_label, iec_info = classify_iec_ratio(row)
    duval_label, duval_info = classify_duval_triangle_1(row)
    duval_pent_label, duval_pent_info = classify_duval_pentagon(row)
    rogers_label, rogers_info = classify_rogers_ratio(row)

    votes = {
        "ieee_key_gas": ieee_label,
        "iec_ratio": iec_label,
        "duval_triangle_1": duval_label,
        "duval_pentagon": duval_pent_label,
        "rogers_ratio": rogers_label,
    }

    final_label, agg = aggregate_fault_votes(votes, row)
    detail = {
        "final_label": final_label,
        "votes": votes,
        "ieee_detail": ieee_info,
        "iec_detail": iec_info,
        "duval_detail": duval_info,
        "duval_pentagon_detail": duval_pent_info,
        "rogers_detail": rogers_info,
        "aggregation": agg,
    }
    return final_label, detail


def compute_fault_score(fault_label: str) -> int:
    return FAULT_SEVERITY_POINTS.get(fault_label, 0)


def compute_aging_score(row: pd.Series) -> Tuple[int, Dict[str, int]]:
    """
    Cellulose / insulation aging score:
    - CO
    - CO2
    - CO2/CO
    - water
    - temp
    """
    score = 0
    detail = {}

    co = row.get("co", np.nan)
    co2 = row.get("co2", np.nan)
    ratio = row.get("ratio_co2_co", np.nan)
    water = row.get("water", np.nan)
    temp = row.get("temp", np.nan)

    if pd.notna(co):
        if co >= 1400:
            score += 3
            detail["co"] = 3
        elif co >= 700:
            score += 2
            detail["co"] = 2
        elif co >= 350:
            score += 1
            detail["co"] = 1
        else:
            detail["co"] = 0
    else:
        detail["co"] = 0

    if pd.notna(co2):
        if co2 >= 10000:
            score += 2
            detail["co2"] = 2
        elif co2 >= 5000:
            score += 1
            detail["co2"] = 1
        else:
            detail["co2"] = 0
    else:
        detail["co2"] = 0

    if pd.notna(ratio):
        if ratio < 3:
            score += 3
            detail["co2_co"] = 3
        elif ratio < 5:
            score += 2
            detail["co2_co"] = 2
        elif ratio < 7:
            score += 1
            detail["co2_co"] = 1
        else:
            detail["co2_co"] = 0
    else:
        detail["co2_co"] = 0

    # không tự fill water/temp; nếu missing thì score 0
    if pd.notna(water):
        if water >= 40:
            score += 2
            detail["water"] = 2
        elif water >= 25:
            score += 1
            detail["water"] = 1
        else:
            detail["water"] = 0
    else:
        detail["water"] = 0

    if pd.notna(temp):
        if temp >= 90:
            score += 2
            detail["temp"] = 2
        elif temp >= 70:
            score += 1
            detail["temp"] = 1
        else:
            detail["temp"] = 0
    else:
        detail["temp"] = 0

    return score, detail


def severity_class_from_score(score: float) -> str:
    if pd.isna(score):
        return "Uncertain"
    if score < 4:
        return "Normal"
    if score < 8:
        return "Watchlist"
    if score < 13:
        return "Warning"
    return "Critical"


def summarize_labels(df: pd.DataFrame) -> Dict:
    summary = {
        "rows": int(len(df)),
        "transformers": int(df["transformer_id"].nunique()),
        "date_min": str(df["sample_day"].min()) if len(df) else None,
        "date_max": str(df["sample_day"].max()) if len(df) else None,
        "fault_type_distribution": df["fault_type_label"].value_counts(dropna=False).to_dict(),
        "severity_distribution": df["severity_label"].value_counts(dropna=False).to_dict(),
    }
    return summary


# =============================================================================
# MAIN
# =============================================================================

def main() -> None:
    ensure_dir(OUTPUT_DIR)

    print("=" * 90)
    print("[INFO] Reading features file")
    print(f"[INFO] {FEATURES_PATH}")

    df = pd.read_parquet(FEATURES_PATH)
    validate_input(df)

    df = df.copy()
    df["transformer_id"] = df["transformer_id"].astype(str).str.strip()
    df["sample_day"] = pd.to_datetime(df["sample_day"], errors="coerce")
    df = df.dropna(subset=["transformer_id", "sample_day"]).copy()

    numeric_cols = [
        c for c in df.columns
        if c not in ["transformer_id", "sample_day", "loc", "name", "ser", "codetx", "mfg", "tdcg_source"]
    ]
    df = safe_numeric(df, numeric_cols)

    print("[INFO] Sorting and deduplicating")
    df = (
        df.sort_values(["transformer_id", "sample_day"])
          .drop_duplicates(subset=["transformer_id", "sample_day"], keep="last")
          .reset_index(drop=True)
    )

    # -------------------------------------------------------------------------
    # Fault labels by consensus
    # -------------------------------------------------------------------------
    print("[INFO] Generating rule-based fault labels with consensus (IEEE/IEC/Duval/Rogers)")

    final_fault_labels = []
    final_fault_rules = []

    ieee_labels = []
    iec_labels = []
    duval_labels = []
    rogers_labels = []

    fault_detail_json = []

    for _, row in df.iterrows():
        ieee_label, _ = classify_ieee_key_gas(row)
        iec_label, _ = classify_iec_ratio(row)
        duval_label, _ = classify_duval_triangle_1(row)
        rogers_label, _ = classify_rogers_ratio(row)

        final_label, detail = classify_fault_consensus(row)

        final_fault_labels.append(final_label)
        final_fault_rules.append(detail.get("aggregation", {}).get("strategy", ""))

        ieee_labels.append(ieee_label)
        iec_labels.append(iec_label)
        duval_labels.append(duval_label)
        rogers_labels.append(rogers_label)

        fault_detail_json.append(json.dumps(detail, ensure_ascii=False))

    df["fault_type_label"] = final_fault_labels
    df["fault_rule"] = final_fault_rules

    df["fault_ieee_key_gas"] = ieee_labels
    df["fault_iec_ratio"] = iec_labels
    df["fault_duval_triangle_1"] = duval_labels
    df["fault_rogers_ratio"] = rogers_labels
    df["fault_detail_json"] = fault_detail_json

    # -------------------------------------------------------------------------
    # Severity scores
    # -------------------------------------------------------------------------
    print("[INFO] Computing severity sub-scores")

    gas_scores = []
    trend_scores = []
    aging_scores = []
    fault_scores = []

    gas_detail_json = []
    trend_detail_json = []
    aging_detail_json = []

    for _, row in df.iterrows():
        gas_s, gas_detail = compute_gas_level_score(row)
        trend_s, trend_detail = compute_trend_score(row)
        aging_s, aging_detail = compute_aging_score(row)
        fault_s = compute_fault_score(row["fault_type_label"])

        gas_scores.append(gas_s)
        trend_scores.append(trend_s)
        aging_scores.append(aging_s)
        fault_scores.append(fault_s)

        gas_detail_json.append(json.dumps(gas_detail, ensure_ascii=False))
        trend_detail_json.append(json.dumps(trend_detail, ensure_ascii=False))
        aging_detail_json.append(json.dumps(aging_detail, ensure_ascii=False))

    df["severity_gas_score"] = gas_scores
    df["severity_trend_score"] = trend_scores
    df["severity_fault_score"] = fault_scores
    df["severity_aging_score"] = aging_scores

    df["severity_gas_detail"] = gas_detail_json
    df["severity_trend_detail"] = trend_detail_json
    df["severity_aging_detail"] = aging_detail_json

    # Composite severity score
    # trend + threshold + fault + aging
    df["severity_score"] = (
        1.0 * df["severity_gas_score"]
        + 1.2 * df["severity_trend_score"]
        + 1.0 * df["severity_fault_score"]
        + 0.8 * df["severity_aging_score"]
    )

    df["severity_label"] = df["severity_score"].apply(severity_class_from_score)

    # -------------------------------------------------------------------------
    # Fleet latest ranking
    # -------------------------------------------------------------------------
    latest_idx = df.groupby("transformer_id")["sample_day"].idxmax()
    latest_df = df.loc[latest_idx].copy()
    latest_df = latest_df.sort_values(["severity_score", "tdcg"], ascending=[False, False]).reset_index(drop=True)
    latest_df["fleet_priority_rank"] = np.arange(1, len(latest_df) + 1)

    df = df.merge(
        latest_df[["transformer_id", "fleet_priority_rank"]],
        on="transformer_id",
        how="left",
    )

    # convenience target columns for ML
    df["target_fault_type"] = df["fault_type_label"]
    df["target_severity"] = df["severity_label"]
    df["target_severity_score"] = df["severity_score"]

    # output
    labels_path = OUTPUT_DIR / "dga_labeled.parquet"
    csv_path = OUTPUT_DIR / "dga_labeled.csv"
    summary_path = OUTPUT_DIR / "dga_label_summary.json"
    latest_rank_path = OUTPUT_DIR / "fleet_latest_ranking.csv"

    print(f"[INFO] Writing {labels_path}")
    df.to_parquet(labels_path, index=False)

    print(f"[INFO] Writing {csv_path}")
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")

    latest_df.to_csv(latest_rank_path, index=False, encoding="utf-8-sig")

    summary = summarize_labels(df)
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("=" * 90)
    print("[DONE] Label generation complete")
    print(f"Rows: {len(df):,}")
    print(f"Transformers: {df['transformer_id'].nunique():,}")
    print(f"Columns: {df.shape[1]:,}")
    print(f"Saved labeled dataset to: {labels_path}")
    print(f"Saved csv copy to:       {csv_path}")
    print(f"Saved summary to:        {summary_path}")
    print(f"Saved latest ranking to: {latest_rank_path}")
    print("-" * 90)

    preview_cols = [
        "transformer_id",
        "sample_day",
        "h2",
        "ch4",
        "c2h2",
        "c2h4",
        "co",
        "co2",
        "tdcg",
        "fault_ieee_key_gas",
        "fault_iec_ratio",
        "fault_duval_triangle_1",
        "fault_rogers_ratio",
        "fault_type_label",
        "severity_gas_score",
        "severity_trend_score",
        "severity_fault_score",
        "severity_aging_score",
        "severity_score",
        "severity_label",
        "fleet_priority_rank",
    ]
    preview_cols = [c for c in preview_cols if c in df.columns]

    print(df[preview_cols].head(20).to_string(index=False))
    print("=" * 90)


if __name__ == "__main__":
    main()