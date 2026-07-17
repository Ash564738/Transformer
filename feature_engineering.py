# feature_engineering.py
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Iterable, List

import numpy as np
import pandas as pd


# ============================================================
# Paths
# ============================================================

DATA_DIR = Path("dataset")
INPUT_PATH = DATA_DIR / "processed" / "dga_clean.parquet"
OUTPUT_PATH = DATA_DIR / "processed" / "dga_features.parquet"
FEATURE_META_PATH = DATA_DIR / "processed" / "dga_feature_columns.json"


# ============================================================
# Config
# ============================================================

CORE_GASES = ["h2", "ch4", "c2h6", "c2h4", "c2h2", "co", "co2"]
OPTIONAL_NUMERIC = ["o2", "n2", "water", "temp"]
ROLL_WINDOWS = [3, 5]
EWMA_SPANS = [3, 5]
LAG_STEPS = [1, 2, 3]

EVENT_KEYWORDS = [
    # Protection / Relay
    "trip", "alarm", "buchholz", "bouchholz", "sudden pressure",
    "pressure relief", "oil flow relay", "oil flow", "differential",
    "diff relay", "diff", "tx.diff", "relay", "87k", "87t", "63",
    "50", "51", "51g", "oc-g", "lock out",
    # Transformer fault
    "fault", "flash", "arc", "arcing", "ground fault",
    "single line to ground", "short circuit", "overcurrent",
    "over current", "lightning",
    # Bushing / OLTC / Equipment
    "bushing", "oltc", "off load tap changer", "oil flow relay",
    "neutral bushing", "lead",
    # Mechanical / Oil
    "ระเบิด", "explosion", "burst", "fire", "smoke", "ไหม้",
    "รั่ว", "leak", "leakage", "oil leak", "น้ำมันไหล", "น้ำมันรั่ว", "low oil",
    # Thermal
    "overheat", "over heating", "overheat", "hot spot", "high temp",
    "temperature alarm", "oil temperature alarm", "ร้อนผิดปกติ",
    "เสียงดัง", "มีเสียงดัง",
    # DGA / Gas abnormal
    "c2h2", "acetylene", "hydran detect", "gas alarm", "gassing",
    # Operating condition
    "de-energize", "de energize", "cold standby", "no energize",
    "no-energize", "first energized",
    # External system fault
    "mea", "pea", "กฟน", "กฟภ", "breaker ระเบิด", "bkr. ระเบิด",
    "cvt ระเบิด", "surge arrester",
    # Test because of event
    "after trip", "test after transformer trip",
]

IGNORE_KEYWORDS = [
    "-", "research", "repeat", "ตามผล", "ครั้ง2", "ครั้ง3", "สี",
    "before test", "after dielectric test", "before high voltage test",
    "after high voltage test", "high voltage test", "hv test",
    "impulse", "routine test", "commissioning", "sampling point",
    "ย้ายมาจาก", "ก่อนนำเข้าใช้งาน", "ทดสอบก่อนนำเข้าใช้งาน",
    "หลังทดสอบทางไฟฟ้า", "after oil purify", "hot oil purify",
    "oil purify", "purify", "replace bushing", "replace oltc",
    "overhaul", "oh",
]

def has_event(nb_text):
    if pd.isna(nb_text):
        return False
    text = str(nb_text).lower()
    if any(k in text for k in IGNORE_KEYWORDS):
        return False
    return any(k in text for k in EVENT_KEYWORDS)

def classify_event(nb_text):
    if pd.isna(nb_text):
        return "Other"
    text = str(nb_text).lower()
    if any(w in text for w in ["buchholz", "diff", "flash", "arc", "relay", "trip", "f87", "f63", "f50", "f51", "discharge"]):
        return "Electrical"
    if any(w in text for w in ["overheat", "high temp", "hot spot", "hotspot", "thermal", "ไหม้", "ความร้อนสูง"]):
        return "Thermal"
    if any(w in text for w in ["bushing", "ระเบิด", "burst", "explosion", "leak", "รั่ว", "pressure", "prd", "spr"]):
        return "Bushing/Mechanical"
    if "c2h2" in text:
        return "C2H2_detected"
    if any(w in text for w in ["de-energize", "cold standby", "shutdown", "shut down", "outage"]):
        return "Outage"
    if any(w in text for w in ["repair", "replace", "maintenance", "inspect", "ซ่อม", "เปลี่ยน", "purify"]):
        return "Maintenance"
    return "Other"

# ============================================================
# Helpers
# ============================================================
def ensure_required_columns(df: pd.DataFrame) -> None:
    required = ["transformer_id", "sample_day"] + CORE_GASES
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Thiếu cột bắt buộc trong dga_clean.parquet: {missing}")

def coerce_datetime(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s, errors="coerce")

def coerce_numeric(s: pd.Series) -> pd.Series:
    if pd.api.types.is_numeric_dtype(s):
        return pd.to_numeric(s, errors="coerce")
    x = s.astype(str).str.strip()
    x = x.replace({"": np.nan, "-": np.nan, "--": np.nan, "nan": np.nan, "None": np.nan, "NONE": np.nan, "#VALUE!": np.nan})
    x = x.str.replace(",", "", regex=False)
    x = x.str.replace(r"[^0-9.\-]+", "", regex=True)
    x = x.replace({"": np.nan, "-": np.nan, ".": np.nan, "-.": np.nan})
    return pd.to_numeric(x, errors="coerce")

def safe_div(a: pd.Series, b: pd.Series) -> pd.Series:
    a = pd.to_numeric(a, errors="coerce")
    b = pd.to_numeric(b, errors="coerce")
    out = pd.Series(np.nan, index=a.index, dtype="float64")
    mask = b.notna() & np.isfinite(b) & (b != 0)
    out.loc[mask] = a.loc[mask] / b.loc[mask]
    return out

def slope_from_series(values: np.ndarray) -> float:
    arr = np.asarray(values, dtype=float)
    mask = np.isfinite(arr)
    if mask.sum() < 2:
        return np.nan
    y = arr[mask]
    x = np.arange(len(arr))[mask].astype(float)
    x_mean = x.mean()
    y_mean = y.mean()
    denom = ((x - x_mean) ** 2).sum()
    if denom == 0:
        return np.nan
    return float(((x - x_mean) * (y - y_mean)).sum() / denom)

def extract_numbers_from_rating(x) -> List[float]:
    if pd.isna(x):
        return []
    s = str(x).strip()
    if not s or s.lower() in {"nan", "none", "null"}:
        return []
    nums = re.findall(r"\d+(?:\.\d+)?", s.replace(",", ""))
    out = []
    for n in nums:
        try:
            out.append(float(n))
        except Exception:
            pass
    return out

def rating_stats_series(s: pd.Series, prefix: str) -> pd.DataFrame:
    values = s.apply(extract_numbers_from_rating)
    return pd.DataFrame(
        {
            f"{prefix}_count": values.apply(len).astype(float),
            f"{prefix}_min": values.apply(lambda xs: min(xs) if xs else np.nan),
            f"{prefix}_max": values.apply(lambda xs: max(xs) if xs else np.nan),
            f"{prefix}_mean": values.apply(lambda xs: float(np.mean(xs)) if xs else np.nan),
            f"{prefix}_first": values.apply(lambda xs: xs[0] if xs else np.nan),
        },
        index=s.index,
    )

# ============================================================
# Core preprocessing (giữ nguyên)
# ============================================================
def preprocess_types(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["sample_day"] = coerce_datetime(out["sample_day"])
    out = out[out["sample_day"].notna()].copy()
    numeric_cols = CORE_GASES + OPTIONAL_NUMERIC + ["tdcg_raw", "year_energized"]
    for col in numeric_cols:
        if col in out.columns:
            out[col] = coerce_numeric(out[col])
    return out

def sort_and_deduplicate(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out = out[out["transformer_id"].notna()].copy()
    out = out.sort_values(["transformer_id", "sample_day"]).drop_duplicates(
        subset=["transformer_id", "sample_day"], keep="last"
    ).reset_index(drop=True)
    return out

def filter_rows_for_model(df: pd.DataFrame, max_missing_core: int = 3) -> pd.DataFrame:
    out = df.copy()
    out = out[out["transformer_id"].notna() & out["sample_day"].notna()].copy()
    core_missing_count = out[CORE_GASES].isna().sum(axis=1)
    out = out[core_missing_count <= max_missing_core].copy()
    return out

def impute_optional_context_by_transformer(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    fill_cols = [c for c in ["o2", "n2", "year_energized"] if c in out.columns]
    if not fill_cols:
        return out
    for col in fill_cols:
        out[col] = out.groupby("transformer_id", sort=False)[col].transform(lambda s: s.ffill().bfill())
    return out

def add_missingness_flags(df: pd.DataFrame, cols: Iterable[str]) -> pd.DataFrame:
    out = df.copy()
    for col in cols:
        if col in out.columns and f"{col}_missing" not in out.columns:
            out[f"{col}_missing"] = out[col].isna().astype("int8")
    return out

# ============================================================
# NEW: Extract weak ground truth from NB
# ============================================================
def add_nb_event_features(df: pd.DataFrame) -> pd.DataFrame:
    """Thêm has_event và event_type từ cột nb (đã làm sạch)."""
    out = df.copy()
    if "nb" not in out.columns:
        out["has_event"] = False
        out["event_type"] = "No NB"
    else:
        out["has_event"] = out["nb"].apply(has_event)
        out["event_type"] = out["nb"].apply(classify_event)
    return out

# ============================================================
# Feature blocks (giữ nguyên)
# ============================================================
def add_tdcg(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    combustible = ["h2", "ch4", "c2h6", "c2h4", "c2h2", "co"]
    out["tdcg_recalc"] = out[combustible].sum(axis=1, min_count=1)
    if "tdcg_raw" in out.columns:
        out["tdcg"] = out["tdcg_raw"].fillna(out["tdcg_recalc"])
    else:
        out["tdcg_raw"] = np.nan
        out["tdcg"] = out["tdcg_recalc"]
    out["tdcg_source"] = np.where(out["tdcg_raw"].notna(), "raw", "recalc")
    return out

def add_rating_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "mva" in out.columns:
        mva_stats = rating_stats_series(out["mva"], "mva")
        out = pd.concat([out, mva_stats], axis=1)
    if "kv" in out.columns:
        kv_stats = rating_stats_series(out["kv"], "kv")
        out = pd.concat([out, kv_stats], axis=1)
    return out

def add_metadata_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "year_energized" in out.columns:
        out["transformer_age"] = out["sample_day"].dt.year - out["year_energized"]
        out.loc[out["transformer_age"] < 0, "transformer_age"] = np.nan
    return out

def add_ratio_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["ratio_ch4_h2"] = safe_div(out["ch4"], out["h2"])
    out["ratio_c2h2_c2h4"] = safe_div(out["c2h2"], out["c2h4"])
    out["ratio_c2h4_c2h6"] = safe_div(out["c2h4"], out["c2h6"])
    out["ratio_c2h6_ch4"] = safe_div(out["c2h6"], out["ch4"])
    out["ratio_c2h2_h2"] = safe_div(out["c2h2"], out["h2"])
    out["ratio_c2h2_ch4"] = safe_div(out["c2h2"], out["ch4"])
    out["ratio_co2_co"] = safe_div(out["co2"], out["co"])
    out["ratio_co_co2"] = safe_div(out["co"], out["co2"])
    out["ratio_ch4_tdcg"] = safe_div(out["ch4"], out["tdcg"])
    out["ratio_h2_tdcg"] = safe_div(out["h2"], out["tdcg"])
    out["ratio_c2h2_tdcg"] = safe_div(out["c2h2"], out["tdcg"])
    out["ratio_c2h4_tdcg"] = safe_div(out["c2h4"], out["tdcg"])
    out["ratio_c2h6_tdcg"] = safe_div(out["c2h6"], out["tdcg"])
    out["ratio_co_tdcg"] = safe_div(out["co"], out["tdcg"])
    for col in CORE_GASES + ["tdcg"]:
        out[f"log1p_{col}"] = np.log1p(out[col].clip(lower=0))
    return out

def add_duval_input_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    tri_sum = out[["ch4", "c2h4", "c2h2"]].sum(axis=1, min_count=1)
    out["duval1_sum"] = tri_sum
    out["duval1_pct_ch4"] = safe_div(out["ch4"] * 100.0, tri_sum)
    out["duval1_pct_c2h4"] = safe_div(out["c2h4"] * 100.0, tri_sum)
    out["duval1_pct_c2h2"] = safe_div(out["c2h2"] * 100.0, tri_sum)
    pent_sum = out[["h2", "ch4", "c2h6", "c2h4", "c2h2"]].sum(axis=1, min_count=1)
    out["duval_pent_sum"] = pent_sum
    for g in ["h2", "ch4", "c2h6", "c2h4", "c2h2"]:
        out[f"duval_pent_pct_{g}"] = safe_div(out[g] * 100.0, pent_sum)
    return out

def add_calendar_and_sequence_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["sample_year"] = out["sample_day"].dt.year
    out["sample_month"] = out["sample_day"].dt.month
    out["sample_quarter"] = out["sample_day"].dt.quarter
    out["sample_dayofyear"] = out["sample_day"].dt.dayofyear
    out["sample_weekday"] = out["sample_day"].dt.weekday
    out["record_idx"] = out.groupby("transformer_id").cumcount()
    first_date = out.groupby("transformer_id")["sample_day"].transform("min")
    out["days_since_first_sample"] = (out["sample_day"] - first_date).dt.days.astype(float)
    prev_date = out.groupby("transformer_id")["sample_day"].shift(1)
    out["days_since_prev"] = (out["sample_day"] - prev_date).dt.days.astype(float)
    return out

def add_lag_delta_rate_features(df: pd.DataFrame, value_cols: List[str]) -> pd.DataFrame:
    out = df.copy()
    g = out.groupby("transformer_id", sort=False)
    for col in value_cols:
        for lag in LAG_STEPS:
            out[f"{col}_lag{lag}"] = g[col].shift(lag)
        out[f"{col}_delta1"] = out[col] - out[f"{col}_lag1"]
        out[f"{col}_pct_change1"] = safe_div(out[f"{col}_delta1"], out[f"{col}_lag1"])
        out[f"{col}_rate_per_day"] = safe_div(out[f"{col}_delta1"], out["days_since_prev"])
        out[f"{col}_delta2"] = out[col] - out[f"{col}_lag2"]
        days_lag2 = (out["sample_day"] - g["sample_day"].shift(2)).dt.days.astype(float)
        out[f"{col}_rate_per_day_lag2"] = safe_div(out[f"{col}_delta2"], days_lag2)
    return out

def add_rolling_features(df: pd.DataFrame, value_cols: List[str]) -> pd.DataFrame:
    out = df.copy()
    new_cols = {}
    for col in value_cols:
        hist = out.groupby("transformer_id", sort=False)[col].shift(1)
        hist_grouped = hist.groupby(out["transformer_id"], sort=False)
        for w in ROLL_WINDOWS:
            roll = hist_grouped.rolling(window=w, min_periods=1)
            mean_s = roll.mean().reset_index(level=0, drop=True)
            std_s = roll.std().reset_index(level=0, drop=True)
            min_s = roll.min().reset_index(level=0, drop=True)
            max_s = roll.max().reset_index(level=0, drop=True)
            new_cols[f"{col}_roll{w}_mean"] = mean_s
            new_cols[f"{col}_roll{w}_std"] = std_s
            new_cols[f"{col}_roll{w}_min"] = min_s
            new_cols[f"{col}_roll{w}_max"] = max_s
            new_cols[f"{col}_vs_roll{w}_mean"] = out[col] - mean_s
            new_cols[f"{col}_roll{w}_range"] = max_s - min_s
            slope_s = (
                hist_grouped.rolling(window=w, min_periods=2)
                .apply(slope_from_series, raw=True)
                .reset_index(level=0, drop=True)
            )
            new_cols[f"{col}_roll{w}_slope"] = slope_s
    if new_cols:
        out = pd.concat([out, pd.DataFrame(new_cols, index=out.index)], axis=1)
    return out

def add_ewm_features(df: pd.DataFrame, value_cols: List[str]) -> pd.DataFrame:
    out = df.copy()
    for col in value_cols:
        hist = out.groupby("transformer_id", sort=False)[col].shift(1)
        for span in EWMA_SPANS:
            ewm_col = (
                hist.groupby(out["transformer_id"], sort=False)
                .transform(lambda s: s.ewm(span=span, adjust=False, min_periods=1).mean())
            )
            out[f"{col}_ewm{span}"] = ewm_col
            out[f"{col}_vs_ewm{span}"] = out[col] - out[f"{col}_ewm{span}"]
    return out

def add_cross_gas_trend_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    core_delta_cols_all = [f"{c}_delta1" for c in CORE_GASES if f"{c}_delta1" in out.columns]
    if core_delta_cols_all:
        delta_frame = out[core_delta_cols_all]
        out["num_gases_increasing"] = (delta_frame > 0).sum(axis=1)
        out["num_gases_decreasing"] = (delta_frame < 0).sum(axis=1)
        out["sum_positive_gas_delta"] = delta_frame.clip(lower=0).sum(axis=1)
        out["sum_negative_gas_delta_abs"] = (-delta_frame.clip(upper=0)).sum(axis=1)
    if all(c in out.columns for c in ["h2", "c2h2", "c2h4", "ch4", "c2h6"]):
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

# ============================================================
# Main
# ============================================================
def main():
    if not INPUT_PATH.exists():
        raise FileNotFoundError(f"Không tìm thấy file input: {INPUT_PATH}")

    print(f"[INFO] Reading {INPUT_PATH}")
    df = pd.read_parquet(INPUT_PATH)

    ensure_required_columns(df)

    print("[INFO] Preprocessing dtypes")
    df = preprocess_types(df)

    print("[INFO] Sorting and deduplicating")
    df = sort_and_deduplicate(df)

    print("[INFO] Filtering bad rows")
    df = filter_rows_for_model(df, max_missing_core=3)

    # --- Thêm weak ground truth từ NB (phải làm trước imputation để giữ nguyên text) ---
    print("[INFO] Extracting event flags from NB column")
    df = add_nb_event_features(df)

    print("[INFO] Adding missing flags BEFORE imputation")
    df = add_missingness_flags(df, OPTIONAL_NUMERIC + ["year_energized", "tdcg_raw"])

    print("[INFO] Imputing optional context columns by transformer")
    df = impute_optional_context_by_transformer(df)

    print("[INFO] Creating TDCG")
    df = add_tdcg(df)

    print("[INFO] Adding rating features")
    df = add_rating_features(df)

    print("[INFO] Adding metadata features")
    df = add_metadata_features(df)

    print("[INFO] Adding ratio features")
    df = add_ratio_features(df)

    print("[INFO] Adding Duval input features")
    df = add_duval_input_features(df)

    print("[INFO] Adding calendar/sequence features")
    df = add_calendar_and_sequence_features(df)

    temporal_value_cols = [c for c in CORE_GASES + ["tdcg"] if c in df.columns]
    for c in ["water", "temp"]:
        if c in df.columns:
            temporal_value_cols.append(c)

    print("[INFO] Adding lag/delta/rate features")
    df = add_lag_delta_rate_features(df, temporal_value_cols)

    print("[INFO] Adding rolling features")
    df = add_rolling_features(df, temporal_value_cols)

    print("[INFO] Adding EWMA features")
    df = add_ewm_features(df, temporal_value_cols)

    print("[INFO] Adding cross-gas trend features")
    df = add_cross_gas_trend_features(df)

    print("[INFO] Adding quality flags")
    df = add_quality_flags(df)

    df = df.sort_values(["transformer_id", "sample_day"]).reset_index(drop=True)

    id_cols = [c for c in ["transformer_id", "loc", "name", "ser", "codetx", "mfg", "sample_day"] if c in df.columns]
    non_feature_candidates = set(id_cols)
    feature_cols = [c for c in df.columns if c not in non_feature_candidates]

    meta = {
        "input_path": str(INPUT_PATH),
        "output_path": str(OUTPUT_PATH),
        "n_rows": int(len(df)),
        "n_transformers": int(df["transformer_id"].nunique()),
        "date_min": str(df["sample_day"].min()) if len(df) else None,
        "date_max": str(df["sample_day"].max()) if len(df) else None,
        "core_gases": CORE_GASES,
        "optional_numeric": OPTIONAL_NUMERIC,
        "feature_columns": feature_cols,
        "id_columns": id_cols,
        "roll_windows": ROLL_WINDOWS,
        "ewma_spans": EWMA_SPANS,
        "lag_steps": LAG_STEPS,
        "weak_label_columns": ["has_event", "event_type"],   # đánh dấu đây là nhãn yếu
        "notes": {
            "tdcg_raw_kept": True,
            "tdcg_final_prefers_raw_else_recalc": True,
            "temp_water_not_imputed": True,
            "history_only_roll_ewm": True,
            "has_event_extracted_from_nb": True,
        },
    }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] Writing {OUTPUT_PATH}")
    df.to_parquet(OUTPUT_PATH, index=False)

    print(f"[INFO] Writing {FEATURE_META_PATH}")
    with open(FEATURE_META_PATH, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print("=" * 80)
    print("[DONE] Feature build complete")
    print(f"Rows: {len(df):,}")
    print(f"Transformers: {df['transformer_id'].nunique():,}")
    print(f"Columns: {len(df.columns):,}")
    print(f"Saved features to: {OUTPUT_PATH}")
    print(f"Saved feature registry to: {FEATURE_META_PATH}")
    print("=" * 80)

    preview_cols = [
        c for c in [
            "transformer_id", "sample_day",
            "h2", "ch4", "c2h6", "c2h4", "c2h2", "co", "co2",
            "tdcg_raw", "tdcg_recalc", "tdcg",
            "water", "temp",
            "ratio_ch4_h2", "ratio_c2h2_c2h4", "ratio_c2h4_c2h6", "ratio_co2_co",
            "h2_delta1", "tdcg_delta1", "h2_rate_per_day", "tdcg_rate_per_day",
            "h2_roll3_mean", "tdcg_roll3_mean",
            "num_gases_increasing",
            "transformer_age",
            "mva_mean", "kv_mean",
            "has_event", "event_type"  # thêm vào preview
        ] if c in df.columns
    ]
    print(df[preview_cols].head(10).to_string(index=False))


if __name__ == "__main__":
    main()