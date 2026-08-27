# feature_engineering.py
from __future__ import annotations
import json, logging, re
from pathlib import Path
from typing import Iterable, List, Optional
import warnings
import numpy as np
import pandas as pd
from config import DATASET_DIR

warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)
logger = logging.getLogger(__name__)

INPUT_PATH = DATASET_DIR / "processed" / "dga_unlabeled.parquet"; OUTPUT_PATH = DATASET_DIR / "processed" / "dga_unlabeled_features.parquet"; FEATURE_META_PATH = DATASET_DIR / "processed" / "dga_unlabeled_feature_columns.json"

CORE_GASES = ["h2", "ch4", "c2h6", "c2h4", "c2h2", "co", "co2"]
COMBUSTIBLE_GASES = ["h2", "ch4", "c2h6", "c2h4", "c2h2", "co"]
OPTIONAL_NUMERIC = ["o2", "n2", "water", "temp"]
ROLL_WINDOWS = [3, 5]; EWMA_SPANS = [3, 5]; LAG_STEPS = [1, 2, 3]
MULTIPOINT_RATE_MIN_POINTS = 3; MULTIPOINT_RATE_MAX_POINTS = 6
MULTIPOINT_RATE_MIN_MONTHS = 4.0; MULTIPOINT_RATE_MAX_MONTHS = 24.0

EVENT_KEYWORDS = [
    "trip", "alarm", "buchholz", "bouchholz", "sudden pressure", "pressure relief",
    "oil flow relay", "oil flow", "differential", "diff relay", "diff", "tx.diff",
    "relay", "87k", "87t", "63", "50", "51", "51g", "oc-g", "lock out", "fault",
    "flash", "arc", "arcing", "ground fault", "single line to ground", "short circuit",
    "overcurrent", "over current", "lightning", "bushing", "oltc", "off load tap changer",
    "neutral bushing", "lead", "ระเบิด", "explosion", "burst", "fire", "smoke", "ไหม้",
    "รั่ว", "leak", "leakage", "oil leak", "น้ำมันไหล", "น้ำมันรั่ว", "low oil", "overheat",
    "over heating", "hot spot", "hotspot", "high temp", "temperature alarm", "oil temperature alarm",
    "ร้อนผิดปกติ", "เสียงดัง", "มีเสียงดัง", "c2h2", "acetylene", "hydran detect", "gas alarm",
    "gassing", "de-energize", "de energize", "cold standby", "no energize", "no-energize",
    "first energized", "mea", "pea", "กฟน", "กฟภ", "breaker ระเบิด", "bkr. ระเบิด", "cvt ระเบิด",
    "surge arrester", "after trip", "test after transformer trip",
]
IGNORE_KEYWORDS = [
    "-", "research", "repeat", "ตามผล", "ครั้ง2", "ครั้ง3", "สี", "before test",
    "after dielectric test", "before high voltage test", "after high voltage test",
    "high voltage test", "hv test", "impulse", "routine test", "commissioning",
    "sampling point", "ย้ายมาจาก", "ก่อนนำเข้าใช้งาน", "ทดสอบก่อนนำเข้าใช้งาน",
    "หลังทดสอบทางไฟฟ้า", "after oil purify", "hot oil purify", "oil purify", "purify",
    "replace bushing", "replace oltc", "overhaul", "oh",
]

def has_event(nb_text) -> bool:
    if pd.isna(nb_text): return False
    text = str(nb_text).lower()
    if any(keyword in text for keyword in IGNORE_KEYWORDS): return False
    return any(keyword in text for keyword in EVENT_KEYWORDS)

def classify_event(nb_text) -> str:
    if pd.isna(nb_text): return "Other"
    text = str(nb_text).lower()
    if any(w in text for w in ["buchholz", "diff", "flash", "arc", "relay", "trip", "f87", "f63", "f50", "f51", "discharge"]): return "Electrical"
    if any(w in text for w in ["overheat", "high temp", "hot spot", "hotspot", "thermal", "ไหม้", "ความร้อนสูง"]): return "Thermal"
    if any(w in text for w in ["bushing", "ระเบิด", "burst", "explosion", "leak", "รั่ว", "pressure", "prd", "spr"]): return "Bushing/Mechanical"
    if "c2h2" in text: return "C2H2_detected"
    if any(w in text for w in ["de-energize", "cold standby", "shutdown", "shut down", "outage"]): return "Outage"
    if any(w in text for w in ["repair", "replace", "maintenance", "inspect", "ซ่อม", "เปลี่ยน", "purify"]): return "Maintenance"
    return "Other"

def ensure_required_columns(df: pd.DataFrame) -> None:
    required = ["transformer_id", "sample_day", *CORE_GASES]
    missing = [col for col in required if col not in df.columns]
    if missing: raise ValueError(f"Missing required columns: {missing}")

def coerce_numeric(s: pd.Series) -> pd.Series:
    if pd.api.types.is_numeric_dtype(s): return pd.to_numeric(s, errors="coerce")
    x = s.astype(str).str.strip()
    x = x.replace({"": np.nan, "-": np.nan, "--": np.nan, "nan": np.nan, "None": np.nan, "NONE": np.nan, "#VALUE!": np.nan})
    x = x.str.replace(",", "", regex=False)
    x = x.str.replace(r"[^0-9.\-]+", "", regex=True)
    x = x.replace({"": np.nan, "-": np.nan, ".": np.nan, "-.": np.nan})
    return pd.to_numeric(x, errors="coerce")

def safe_div(a: pd.Series, b: pd.Series) -> pd.Series:
    a = pd.to_numeric(a, errors="coerce"); b = pd.to_numeric(b, errors="coerce")
    out = pd.Series(np.nan, index=a.index, dtype="float64")
    mask = a.notna() & b.notna() & np.isfinite(a) & np.isfinite(b) & (b != 0)
    out.loc[mask] = a.loc[mask] / b.loc[mask]
    return out

def slope_from_series(values: np.ndarray) -> float:
    arr = np.asarray(values, dtype=float); mask = np.isfinite(arr)
    if mask.sum() < 2: return np.nan
    y = arr[mask]; x = np.arange(len(arr), dtype=float)[mask]
    x_mean = x.mean(); y_mean = y.mean()
    denominator = ((x - x_mean) ** 2).sum()
    if denominator <= 0: return np.nan
    return float(((x - x_mean) * (y - y_mean)).sum() / denominator)

def slope_per_year_from_dates(dates: pd.Series, values: pd.Series) -> float:
    dates = pd.to_datetime(dates, errors="coerce"); values = pd.to_numeric(values, errors="coerce")
    mask = dates.notna() & values.notna() & np.isfinite(values)
    dates = dates[mask]; values = values[mask]
    if len(values) < MULTIPOINT_RATE_MIN_POINTS: return np.nan
    if len(values) > MULTIPOINT_RATE_MAX_POINTS:
        dates = dates.iloc[-MULTIPOINT_RATE_MAX_POINTS:]; values = values.iloc[-MULTIPOINT_RATE_MAX_POINTS:]
    elapsed_days = (dates - dates.iloc[0]).dt.total_seconds() / 86400.0
    span_days = float(elapsed_days.iloc[-1]); span_months = span_days / 30.4375
    if not (MULTIPOINT_RATE_MIN_MONTHS <= span_months <= MULTIPOINT_RATE_MAX_MONTHS): return np.nan
    x = elapsed_days.to_numpy() / 365.25; y = values.to_numpy(dtype=float)
    if not np.isfinite(x).all() or not np.isfinite(y).all(): return np.nan
    x_centered = x - x.mean()
    denominator = (x_centered ** 2).sum()
    if denominator <= 0: return np.nan
    return float((x_centered * (y - y.mean())).sum() / denominator)

def extract_numbers_from_rating(x) -> List[float]:
    if pd.isna(x): return []
    s = str(x).strip()
    if not s or s.lower() in {"nan", "none", "null"}: return []
    nums = re.findall(r"\d+(?:\.\d+)?", s.replace(",", ""))
    return [float(number) for number in nums]

def rating_stats_series(s: pd.Series, prefix: str) -> pd.DataFrame:
    values = s.apply(extract_numbers_from_rating)
    return pd.DataFrame({
        f"{prefix}_count": values.apply(len).astype(float),
        f"{prefix}_min": values.apply(lambda xs: min(xs) if xs else np.nan),
        f"{prefix}_max": values.apply(lambda xs: max(xs) if xs else np.nan),
        f"{prefix}_mean": values.apply(lambda xs: float(np.mean(xs)) if xs else np.nan),
        f"{prefix}_first": values.apply(lambda xs: xs[0] if xs else np.nan),
    }, index=s.index)

def preprocess_types(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "sample_day" not in out.columns: raise ValueError("Missing sample_day.")
    out["sample_day"] = pd.to_datetime(out["sample_day"], errors="coerce")
    out = out[out["sample_day"].notna()].copy()
    numeric_cols = CORE_GASES + OPTIONAL_NUMERIC + ["tdcg_raw", "tcg", "year_energized"]
    for col in numeric_cols:
        if col in out.columns: out[col] = coerce_numeric(out[col])
    return out

def sort_and_deduplicate(df):
    out = df.copy()
    out = out[out["transformer_id"].notna() & out["sample_day"].notna()].copy()
    out = out.sort_values(["transformer_id", "sample_day"], kind="mergesort")
    hashable_cols = []
    for col in out.columns:
        if len(out[col]) == 0:
            hashable_cols.append(col); continue
        try:
            hash(out[col].iloc[0]); hashable_cols.append(col)
        except (TypeError,): pass
    out = out.drop_duplicates(subset=hashable_cols, keep="last")
    return out.reset_index(drop=True)

def filter_rows_for_model(df: pd.DataFrame, max_missing_core: int = 3) -> pd.DataFrame:
    out = df.copy()
    out = out[out["transformer_id"].notna() & out["sample_day"].notna()].copy()
    available_core = [gas for gas in CORE_GASES if gas in out.columns]
    if not available_core: raise ValueError("No core gas columns available.")
    missing_count = out[available_core].isna().sum(axis=1)
    out = out[missing_count <= max_missing_core].copy()
    return out

def impute_optional_context_by_transformer(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    # Only static metadata may be propagated forward. O2/N2 are sample-specific
    # inputs for IEEE screening and must remain missing when not measured.
    if "year_energized" in out.columns:
        out["year_energized"] = out.groupby("transformer_id", sort=False)["year_energized"].transform("ffill")
    return out

def add_missingness_flags(df: pd.DataFrame, cols: Iterable[str]) -> pd.DataFrame:
    out = df.copy()
    for col in cols:
        if col not in out.columns: continue
        flag = f"{col}_missing"
        if flag not in out.columns: out[flag] = out[col].isna().astype("int8")
    return out

def add_nb_event_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "nb" not in out.columns:
        out["has_event"] = 0; out["event_type"] = "No NB"; return out
    out["has_event"] = out["nb"].apply(has_event).astype("int8")
    out["event_type"] = out["nb"].apply(classify_event)
    return out

def add_tdcg(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    available = [gas for gas in COMBUSTIBLE_GASES if gas in out.columns]
    count_measured = out[available].notna().sum(axis=1)
    out["tdcg_gas_count"] = count_measured
    out["tdcg_recalc"] = out[available].sum(axis=1, min_count=1)
    out["tdcg_complete"] = (count_measured == len(available)).astype("int8")
    if "tdcg_raw" not in out.columns: out["tdcg_raw"] = np.nan
    out["tdcg"] = out["tdcg_raw"].where(out["tdcg_raw"].notna(), out["tdcg_recalc"])
    out["tdcg_source"] = np.where(out["tdcg_raw"].notna(), "raw_available", "recalculated")
    return out

def add_rating_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "mva" in out.columns: out = pd.concat([out, rating_stats_series(out["mva"], "mva")], axis=1)
    if "kv" in out.columns: out = pd.concat([out, rating_stats_series(out["kv"], "kv")], axis=1)
    return out

def add_metadata_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "year_energized" in out.columns:
        out["transformer_age_years"] = out["sample_day"].dt.year - pd.to_numeric(out["year_energized"], errors="coerce")
        out.loc[out["transformer_age_years"] < 0, "transformer_age_years"] = np.nan
    else: out["transformer_age_years"] = np.nan
    if "o2" in out.columns and "n2" in out.columns: out["o2_n2_ratio"] = safe_div(out["o2"], out["n2"])
    else: out["o2_n2_ratio"] = np.nan
    return out

def add_ratio_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["ratio_ch4_h2"] = safe_div(out["ch4"], out["h2"]); out["ratio_c2h2_c2h4"] = safe_div(out["c2h2"], out["c2h4"])
    out["ratio_c2h4_c2h6"] = safe_div(out["c2h4"], out["c2h6"]); out["ratio_c2h6_ch4"] = safe_div(out["c2h6"], out["ch4"])
    out["ratio_c2h2_h2"] = safe_div(out["c2h2"], out["h2"]); out["ratio_c2h2_ch4"] = safe_div(out["c2h2"], out["ch4"])
    out["ratio_co2_co"] = safe_div(out["co2"], out["co"]); out["ratio_co_co2"] = safe_div(out["co"], out["co2"])
    for gas in CORE_GASES + ["tdcg"]:
        if gas in out.columns: out[f"log1p_{gas}"] = np.log1p(out[gas].clip(lower=0))
    return out

def add_duval_input_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    triangle_gases = ["ch4", "c2h4", "c2h2"]
    tri_sum = out[triangle_gases].sum(axis=1, min_count=1)
    out["duval1_sum"] = tri_sum
    for gas in triangle_gases: out[f"duval1_pct_{gas}"] = safe_div(out[gas] * 100.0, tri_sum)
    pentagon_gases = ["h2", "ch4", "c2h6", "c2h4", "c2h2"]
    pent_sum = out[pentagon_gases].sum(axis=1, min_count=1)
    out["duval_pent_sum"] = pent_sum
    for gas in pentagon_gases: out[f"duval_pent_pct_{gas}"] = safe_div(out[gas] * 100.0, pent_sum)
    return out

def add_calendar_and_sequence_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["sample_year"] = out["sample_day"].dt.year; out["sample_month"] = out["sample_day"].dt.month
    out["sample_quarter"] = out["sample_day"].dt.quarter; out["sample_dayofyear"] = out["sample_day"].dt.dayofyear
    out["sample_weekday"] = out["sample_day"].dt.weekday
    out["record_idx"] = out.groupby("transformer_id").cumcount()
    first_date = out.groupby("transformer_id")["sample_day"].transform("min")
    out["days_since_first_sample"] = (out["sample_day"] - first_date).dt.days.astype(float)
    prev_date = out.groupby("transformer_id", sort=False)["sample_day"].shift(1)
    out["days_since_prev"] = (out["sample_day"] - prev_date).dt.days.astype(float)
    return out

def _attach_feature_columns(df: pd.DataFrame, features: dict[str, pd.Series]) -> pd.DataFrame:
    """Attach many feature columns in one concat to avoid DataFrame fragmentation."""
    if not features:
        return df
    feature_frame = pd.DataFrame(features, index=df.index)
    return pd.concat([df, feature_frame], axis=1, copy=False)


def add_lag_delta_rate_features(
    df: pd.DataFrame,
    value_cols: List[str],
) -> pd.DataFrame:
    out = df.copy()
    out = out.sort_values(
        ["transformer_id", "sample_day"],
        kind="mergesort",
    ).reset_index(drop=True)

    group = out.groupby("transformer_id", sort=False)
    features: dict[str, pd.Series] = {}

    for col in value_cols:
        if col not in out.columns:
            continue

        lag1 = group[col].shift(1)
        lag2 = group[col].shift(2)
        lag3 = group[col].shift(3)

        features[f"{col}_lag1"] = lag1
        features[f"{col}_lag2"] = lag2
        features[f"{col}_lag3"] = lag3

        delta1 = out[col] - lag1
        delta2 = out[col] - lag2
        delta3 = out[col] - lag3
        features[f"{col}_delta1"] = delta1
        features[f"{col}_delta2"] = delta2
        features[f"{col}_delta3"] = delta3
        features[f"{col}_pct_change1"] = safe_div(delta1, lag1)
        features[f"{col}_rate_per_day"] = safe_div(delta1, out["days_since_prev"])

        previous_value = group[col].shift(MULTIPOINT_RATE_MAX_POINTS - 1)
        previous_date = group["sample_day"].shift(MULTIPOINT_RATE_MAX_POINTS - 1)
        fallback_value = lag1
        fallback_date = group["sample_day"].shift(1)

        start_value = previous_value.combine_first(fallback_value)
        start_date = previous_date.combine_first(fallback_date)
        span_days = (out["sample_day"] - start_date).dt.total_seconds() / 86400.0
        span_months = span_days / 30.4375
        rate_per_year = safe_div(out[col] - start_value, span_days) * 365.25

        valid = (
            start_value.notna()
            & out[col].notna()
            & start_date.notna()
            & out["sample_day"].notna()
            & (span_months >= MULTIPOINT_RATE_MIN_MONTHS)
            & (span_months <= MULTIPOINT_RATE_MAX_MONTHS)
        )
        rate_per_year = rate_per_year.where(valid, np.nan)
        features[f"{col}_rate_per_year"] = rate_per_year
        features[f"{col}_rate_ppm_per_year"] = rate_per_year

        if col == value_cols[0]:
            features["rate_span_months"] = span_months.where(valid, np.nan)
            features["rate_span_days"] = span_days.where(valid, np.nan)
            features["rate_points"] = pd.Series(
                np.where(valid, MULTIPOINT_RATE_MAX_POINTS, 1),
                index=out.index,
                dtype="int8",
            )

    return _attach_feature_columns(out, features)


def add_rolling_features(
    df: pd.DataFrame,
    value_cols: List[str],
) -> pd.DataFrame:
    out = df.copy()
    features: dict[str, pd.Series] = {}
    transformer_groups = out["transformer_id"]

    for col in value_cols:
        if col not in out.columns:
            continue

        history = out.groupby("transformer_id", sort=False)[col].shift(1)
        history_grouped = history.groupby(transformer_groups, sort=False)

        for window in ROLL_WINDOWS:
            rolling = history_grouped.rolling(
                window=window,
                min_periods=1,
            )

            mean_s = rolling.mean().reset_index(level=0, drop=True)
            std_s = rolling.std().reset_index(level=0, drop=True)
            min_s = rolling.min().reset_index(level=0, drop=True)
            max_s = rolling.max().reset_index(level=0, drop=True)

            features[f"{col}_roll{window}_mean"] = mean_s
            features[f"{col}_roll{window}_std"] = std_s
            features[f"{col}_roll{window}_min"] = min_s
            features[f"{col}_roll{window}_max"] = max_s
            features[f"{col}_vs_roll{window}_mean"] = out[col] - mean_s
            features[f"{col}_roll{window}_range"] = max_s - min_s

    return _attach_feature_columns(out, features)


def add_ewm_features(df: pd.DataFrame, value_cols: List[str]) -> pd.DataFrame:
    out = df.copy()
    features: dict[str, pd.Series] = {}
    transformer_groups = out["transformer_id"]

    for col in value_cols:
        if col not in out.columns:
            continue

        history = out.groupby("transformer_id", sort=False)[col].shift(1)
        history_grouped = history.groupby(transformer_groups, sort=False)

        for span in EWMA_SPANS:
            # Vectorized grouped EWM. The previous implementation used
            # groupby(...).transform(lambda ...), which executes Python code
            # once per transformer group and becomes disproportionately slow
            # on production datasets with hundreds of transformers.
            ewm = (
                history_grouped
                .ewm(span=span, adjust=False, min_periods=1)
                .mean()
                .reset_index(level=0, drop=True)
            )
            features[f"{col}_ewm{span}"] = ewm
            features[f"{col}_vs_ewm{span}"] = out[col] - ewm

    return _attach_feature_columns(out, features)


def add_cross_gas_trend_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    delta_cols = [f"{gas}_delta1" for gas in CORE_GASES if f"{gas}_delta1" in out.columns]
    if delta_cols:
        delta_frame = out[delta_cols]
        out["num_gases_increasing"] = (delta_frame > 0).sum(axis=1)
        out["num_gases_decreasing"] = (delta_frame < 0).sum(axis=1)
        out["sum_positive_gas_delta"] = delta_frame.clip(lower=0).sum(axis=1)
        out["sum_negative_gas_delta_abs"] = (-delta_frame.clip(upper=0)).sum(axis=1)
    if all(gas in out.columns for gas in ["h2", "c2h2", "c2h4", "ch4", "c2h6"]):
        out["discharge_gas_index"] = out["h2"] + out["c2h2"]
        out["thermal_gas_index"] = out["c2h4"] + out["ch4"] + out["c2h6"]
        out["discharge_to_thermal"] = safe_div(out["discharge_gas_index"], out["thermal_gas_index"])
    return out

def add_quality_flags(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["is_first_record"] = (out["record_idx"] == 0).astype("int8")
    out["has_prev_record"] = (out["record_idx"] > 0).astype("int8")
    out["short_gap_le_30d"] = (out["days_since_prev"] <= 30).fillna(False).astype("int8")
    out["long_gap_gt_180d"] = (out["days_since_prev"] > 180).fillna(False).astype("int8")
    if "ratio_co2_co" in out.columns:
        out["low_co2_co_flag"] = (out["ratio_co2_co"] < 3).fillna(False).astype("int8")
        out["high_co2_co_flag"] = (out["ratio_co2_co"] > 10).fillna(False).astype("int8")
    return out

MODEL_EXCLUDE_EXACT = {
    "transformer_id", "sample_day", "tested_day", "loc", "name", "ser", "codetx", "mfg",
    "fault_type_label", "fault_rule", "fault_detail_json",
    "keygas_fault", "iec_fault", "rogers_fault", "doernenburg_fault", "duval_triangle_fault",
    "duval_pentagon_fault", "duval_pentagon_p1_fault", "duval_pentagon_p2_fault", "fault_p1", "fault_p2",
    "consensus_fault", "mixed_components", "diagnostic_votes", "diagnostic_confidence",
    "diagnostic_active_methods", "diagnostic_method_count", "diagnostic_coverage",
    "weak_fault_group", "weak_fault_confidence", "weak_fault_is_ABSTAIN",
    "severity_label", "severity_label_text", "severity_score", "severity_gas_score",
    "severity_trend_score", "severity_anomaly_score", "severity_gas_rank", "severity_trend_rank",
    "ieee_dga_status", "ieee_dga_status_label", "ieee_dga_status_reason", "ieee_confirmation_required",
    "ieee_extreme_dga", "fleet_priority_score", "fleet_priority_percent", "fleet_priority_rank",
    "recommended_action", "final_score", "student_fault_label", "student_fault_confidence",
    "consensus_fault_traditional", "gas_below_l1", "diagnostic_unweighted", "diagnostic_disagreement",
    "consensus_fault_group", "student_fault", "student_fault_confidence", "final_fault",
    "final_fault_group", "final_fault_source", "current_severity_level", "current_severity_label",
    "current_severity_score", "history_record_count", "history_span_days", "history_max_severity",
    "history_mean_severity", "history_abnormal_record_count", "history_critical_record_count",
    "history_fault_occurrence_count", "history_fault_recurrence", "history_fault_persistence",
    "history_dominant_fault", "history_fault_entropy", "history_severity_slope",
    "history_positive_severity_slope", "fleet_condition_percentile", "transformer_overall_severity_score",
}

def get_model_feature_columns(df: pd.DataFrame) -> List[str]:
    result = []
    for col in df.columns:
        if col in MODEL_EXCLUDE_EXACT: continue
        if col.startswith(("target_", "weak_prob_", "severity_", "ieee_", "fleet_")): continue
        if pd.api.types.is_numeric_dtype(df[col]): result.append(col)
    return result

def add_nei_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    required = ["ch4", "c2h6", "c2h4", "c2h2"]
    if all(c in out.columns for c in required):
        out["nei_oil"] = (77.7 * out["ch4"] + 93.5 * out["c2h6"] + 104.1 * out["c2h4"] + 278.3 * out["c2h2"]) / 22400.0
    else: out["nei_oil"] = np.nan
    if "co" in out.columns and "co2" in out.columns:
        out["nei_paper"] = (101.4 * out["co"] + 30.2 * out["co2"]) / 22400.0
    else: out["nei_paper"] = np.nan
    return out

def main():
    if not INPUT_PATH.exists(): raise FileNotFoundError(f"Input file not found: {INPUT_PATH}")
    df = pd.read_parquet(INPUT_PATH)
    ensure_required_columns(df)
    df = preprocess_types(df)
    df = sort_and_deduplicate(df)
    df = filter_rows_for_model(df, max_missing_core=3)
    df = add_nb_event_features(df)
    df = add_missingness_flags(df, OPTIONAL_NUMERIC + ["year_energized", "tdcg_raw"])
    df = impute_optional_context_by_transformer(df)
    df = add_tdcg(df)
    df = add_nei_features(df)
    df = add_rating_features(df)
    df = add_metadata_features(df)
    df = add_ratio_features(df)
    df = add_duval_input_features(df)
    df = add_calendar_and_sequence_features(df)
    temporal_value_cols = [col for col in CORE_GASES + ["tdcg"] if col in df.columns]
    for col in ["water", "temp"]:
        if col in df.columns: temporal_value_cols.append(col)
    df = add_lag_delta_rate_features(df, temporal_value_cols)
    df = add_rolling_features(df, temporal_value_cols)
    df = add_ewm_features(df, temporal_value_cols)
    df = add_cross_gas_trend_features(df)
    df = add_quality_flags(df)
    df = df.sort_values(["transformer_id", "sample_day"]).reset_index(drop=True)
    feature_cols = get_model_feature_columns(df)
    meta = {
        "input_path": str(INPUT_PATH), "output_path": str(OUTPUT_PATH),
        "n_rows": int(len(df)), "n_transformers": int(df["transformer_id"].nunique()),
        "date_min": str(df["sample_day"].min()) if len(df) else None,
        "date_max": str(df["sample_day"].max()) if len(df) else None,
        "core_gases": CORE_GASES, "combustible_gases": COMBUSTIBLE_GASES,
        "optional_numeric": OPTIONAL_NUMERIC, "feature_columns": feature_cols,
        "roll_windows": ROLL_WINDOWS, "ewma_spans": EWMA_SPANS, "lag_steps": LAG_STEPS,
        "multipoint_rate": {
            "min_points": MULTIPOINT_RATE_MIN_POINTS, "max_points": MULTIPOINT_RATE_MAX_POINTS,
            "min_months": MULTIPOINT_RATE_MIN_MONTHS, "max_months": MULTIPOINT_RATE_MAX_MONTHS,
        },
        "notes": {
            "no_future_bfill_for_o2_n2": True, "water_temperature_not_imputed": True,
            "tdcg_recalculated_from_combustible_gases": True, "transformer_age_years_added": True,
            "o2_n2_ratio_added": True, "multipoint_rate_added": True,
            "history_only_rolling_features": True, "history_only_ewm_features": True,
        },
    }
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUTPUT_PATH, index=False)
    with open(FEATURE_META_PATH, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    logger.info("Feature engineering complete: %d rows, %d columns, %d model features.", len(df), len(df.columns), len(feature_cols))
    preview_cols = [col for col in ["transformer_id", "sample_day", "h2", "ch4", "c2h6", "c2h4", "c2h2", "co", "co2", "tdcg", "transformer_age_years", "o2_n2_ratio", "h2_delta1", "tdcg_delta1", "h2_rate_per_day", "h2_rate_per_year", "tdcg_rate_per_year", "rate_points", "rate_span_months", "num_gases_increasing", "has_event"] if col in df.columns]
    if preview_cols: logger.info("\n%s", df[preview_cols].head(10).to_string(index=False))

def build_training_features_from_clean(df: pd.DataFrame) -> pd.DataFrame:
    logger.info("Building canonical training/inference features from clean data...")
    out = df.copy()
    out = preprocess_types(out)
    out = sort_and_deduplicate(out)
    out = filter_rows_for_model(out, max_missing_core=3)
    out = add_nb_event_features(out)
    out = add_missingness_flags(out, OPTIONAL_NUMERIC + ["year_energized", "tdcg_raw"])
    out = impute_optional_context_by_transformer(out)
    out = add_tdcg(out)
    out = add_nei_features(out)
    out = add_rating_features(out)
    out = add_metadata_features(out)
    out = add_ratio_features(out)
    out = add_duval_input_features(out)
    out = add_calendar_and_sequence_features(out)
    temporal_value_cols = [col for col in CORE_GASES + ["tdcg"] if col in out.columns]
    for col in ["water", "temp"]:
        if col in out.columns: temporal_value_cols.append(col)
    out = add_lag_delta_rate_features(out, temporal_value_cols)
    out = add_rolling_features(out, temporal_value_cols)
    out = add_ewm_features(out, temporal_value_cols)
    out = add_cross_gas_trend_features(out)
    out = add_quality_flags(out)
    if out.columns.duplicated().any():
        duplicated = out.columns[out.columns.duplicated()].tolist()
        logger.warning(
            "Canonical feature engineering encountered duplicate columns; "
            "keeping first occurrence: %s",
            duplicated,
        )
        out = out.loc[:, ~out.columns.duplicated(keep="first")].copy()
    out = out.sort_values(["transformer_id", "sample_day"], kind="mergesort").reset_index(drop=True)
    logger.info("Canonical feature engineering complete: %d rows x %d columns.", len(out), len(out.columns))
    return out

if __name__ == "__main__":
    main()