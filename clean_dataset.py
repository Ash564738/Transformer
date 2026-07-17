# clean_dataset.py
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd

# ============================================================
# CONFIG
# ============================================================
DATA_DIR = Path("dataset")
INPUT_FILE = DATA_DIR / "DGA of Main Tank only KT 11022026_09062026.xlsx"
OUTPUT_DIR = DATA_DIR / "processed"

KEEP_NB = True

COLUMN_ALIASES = {
    "sample day": "sample_day",
    "sample_day": "sample_day",
    "tested day": "tested_day",
    "tested_day": "tested_day",
    "year energized": "year_energized",
    "year_energized": "year_energized",
    "loc": "loc",
    "name": "name",
    "ser": "ser",
    "nb": "nb",
    "mva": "mva",
    "kv": "kv",
    "temp": "temp",
    "water": "water",
    "wat": "water",
    "h2": "h2",
    "ch4": "ch4",
    "c2h6": "c2h6",
    "c2h4": "c2h4",
    "c2h2": "c2h2",
    "co": "co",
    "co2": "co2",
    "o2": "o2",
    "n2": "n2",
    "tcg": "tcg",
    "codetx": "codetx",
    "mfg": "mfg",
    "c3h6": "c3h6",
    "c3h8": "c3h8",
}

CORE_GASES = ["h2", "ch4", "c2h6", "c2h4", "c2h2", "co", "co2"]
OPTIONAL_GASES = ["o2", "n2", "c3h6", "c3h8"]

# ============================================================
# HELPERS
# ============================================================
def normalize_col_name(col: Any) -> str:
    if col is None:
        return ""
    s = str(col).strip()
    s = re.sub(r"\s+", " ", s)
    return s

def canonicalize_col(col: str) -> str:
    s = normalize_col_name(col).lower()
    s = s.replace("_", " ")
    s = re.sub(r"\s+", " ", s).strip()
    return s

def is_unnamed_or_empty_column(col: Any) -> bool:
    s = normalize_col_name(col)
    if s == "":
        return True
    return bool(re.match(r"^unnamed(?::\s*\d+)?$", s, flags=re.IGNORECASE))

def clean_text_basic(x: Any) -> Optional[str]:
    if pd.isna(x):
        return None
    s = str(x).strip()
    if s == "":
        return None
    if s.lower() in {"nan", "none", "null", "n/a", "na", "-", "--", "#value!"}:
        return None
    return s

def clean_text_field(x: Any) -> Optional[str]:
    s = clean_text_basic(x)
    if s is None:
        return None
    s = re.sub(r"\s+", " ", s).strip()
    return s or None

def clean_ser(x: Any) -> Optional[str]:
    s = clean_text_field(x)
    if s is None:
        return None
    if s.lower() == "xxxx":
        return None
    return s

def parse_date_series(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s, errors="coerce", dayfirst=True)

def parse_numeric_loose(x: Any, allow_negative: bool = True) -> float:
    if pd.isna(x):
        return np.nan
    if isinstance(x, (int, float, np.integer, np.floating)):
        val = float(x)
        if not allow_negative and val < 0:
            return np.nan
        return val
    s = str(x).strip()
    if s == "":
        return np.nan
    s_lower = s.lower()
    if s_lower in {"nan", "none", "null", "n/a", "na", "-", "--", "#value!", "inf", "-inf"}:
        return np.nan
    if "/" in s:
        return np.nan
    s = s.replace(",", "")
    m = re.search(r"-?\d+(?:\.\d+)?", s)
    if not m:
        return np.nan
    try:
        val = float(m.group(0))
    except Exception:
        return np.nan
    if not allow_negative and val < 0:
        return np.nan
    return val

def clean_year_energized(x: Any) -> float:
    val = parse_numeric_loose(x, allow_negative=False)
    if pd.isna(val):
        return np.nan
    val = int(round(val))
    return float(val)

def clean_temp(x: Any) -> float:
    return parse_numeric_loose(x, allow_negative=True)

def clean_water(x: Any) -> float:
    val = parse_numeric_loose(x, allow_negative=True)
    if pd.isna(val):
        return np.nan
    return float(abs(val))

def clean_gas_value(x: Any) -> float:
    val = parse_numeric_loose(x, allow_negative=False)
    if pd.isna(val):
        return np.nan
    return float(val)

def normalize_rating_text(x: Any) -> Optional[str]:
    s = clean_text_field(x)
    if s is None:
        return None
    s = s.replace(",", "")
    s = re.sub(r"(?<=\d)\s*-\s*(?=\d)", "/", s)
    s = re.sub(r"\s*/\s*", "/", s)
    s = re.sub(r"\s+", " ", s).strip()
    if s.lower() in {"nan", "none", "null"}:
        return None
    return s or None

def infer_loc_from_codetx_and_name(row: pd.Series) -> Optional[str]:
    loc = clean_text_field(row.get("loc"))
    if loc is not None:
        return loc
    codetx = clean_text_field(row.get("codetx"))
    name = clean_text_field(row.get("name"))
    if codetx is None or name is None:
        return None
    codetx_s = codetx.strip()
    name_s = name.strip()
    if not codetx_s or not name_s:
        return None
    if codetx_s.endswith(name_s):
        candidate = codetx_s[: len(codetx_s) - len(name_s)].strip()
        return candidate or None
    return None

def fill_missing_ser(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    groups = df.groupby(["codetx", "mfg"], dropna=False)
    for (codetx_val, mfg_val), group in groups:
        if pd.isna(codetx_val) or pd.isna(mfg_val):
            continue
        non_null_ser = group["ser"].dropna()
        if len(non_null_ser) > 0:
            fill_value = non_null_ser.iloc[0]
            mask = (df["codetx"] == codetx_val) & (df["mfg"] == mfg_val) & (df["ser"].isna())
            df.loc[mask, "ser"] = fill_value
    nan_mask = df["ser"].isna()
    if nan_mask.any():
        existing_ser = set(df["ser"].dropna().unique())
        counter = 1
        for idx in df.index[nan_mask]:
            while f"xxxx{counter}" in existing_ser:
                counter += 1
            df.at[idx, "ser"] = f"xxxx{counter}"
            existing_ser.add(f"xxxx{counter}")
            counter += 1
    return df

def build_transformer_id(df: pd.DataFrame) -> pd.Series:
    idx = df.index
    ser = df["ser"].fillna("").astype(str).str.strip() if "ser" in df.columns else pd.Series("", index=idx)
    codetx = df["codetx"].fillna("").astype(str).str.strip() if "codetx" in df.columns else pd.Series("", index=idx)
    loc = df["loc"].fillna("").astype(str).str.strip() if "loc" in df.columns else pd.Series("", index=idx)
    name = df["name"].fillna("").astype(str).str.strip() if "name" in df.columns else pd.Series("", index=idx)
    ser = ser.replace({"nan": "", "None": ""})
    codetx = codetx.replace({"nan": "", "None": ""})
    loc = loc.replace({"nan": "", "None": ""})
    name = name.replace({"nan": "", "None": ""})
    transformer_id = ser.copy()
    use_codetx = transformer_id.eq("")
    transformer_id = transformer_id.mask(use_codetx, codetx)
    use_loc_name = transformer_id.eq("")
    fallback = (loc + " | " + name).str.strip()
    fallback = fallback.str.replace(r"^\|\s*", "", regex=True)
    fallback = fallback.str.replace(r"\s*\|$", "", regex=True)
    transformer_id = transformer_id.mask(use_loc_name, fallback)
    transformer_id = transformer_id.replace({"": np.nan, "nan": np.nan})
    return transformer_id

def report_missing(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame({
        "column": df.columns,
        "missing_count": df.isna().sum().values,
        "missing_ratio": (df.isna().mean() * 100).round(2).values,
        "dtype": [str(df[c].dtype) for c in df.columns],
    })
    return out.sort_values(["missing_ratio", "column"], ascending=[False, True]).reset_index(drop=True)

# ---------------------------
# IMPUTATION FUNCTIONS
# ---------------------------
def impute_by_transformer_timeseries(df: pd.DataFrame, col: str) -> pd.Series:
    """Linear interpolation per transformer; median fallback."""
    s = df[col].copy()
    if s.notna().sum() == len(s):
        return s
    grouped = df.groupby("transformer_id")
    for tid, grp in grouped:
        mask = df["transformer_id"] == tid
        sub = s[mask]
        if sub.notna().sum() >= 2:
            sub_interp = sub.copy()
            sub_interp.index = df.loc[mask, "sample_day"]
            sub_interp = sub_interp.interpolate(method="time", limit_direction="both")
            s[mask] = sub_interp.values
        else:
            med = sub.dropna().median()
            if pd.isna(med):
                med = s.median()
            s[mask] = s[mask].fillna(med)
    return s

# ============================================================
# MAIN
# ============================================================
def clean_dataset(
    input_file: Path = INPUT_FILE,
    output_dir: Path = OUTPUT_DIR,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    if not input_file.exists():
        raise FileNotFoundError(f"Input file not found: {input_file}")
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1) LOAD
    xl = pd.ExcelFile(input_file)
    sheet_name = xl.sheet_names[0]
    raw = xl.parse(sheet_name)
    raw.columns = [normalize_col_name(c) for c in raw.columns]
    df = raw.copy()

    original_shape = df.shape
    original_columns = df.columns.tolist()

    # 2) DROP NOISE COLUMNS
    unnamed_cols = [c for c in df.columns if is_unnamed_or_empty_column(c)]
    if unnamed_cols:
        df = df.drop(columns=unnamed_cols)
    all_nan_cols = [c for c in df.columns if df[c].isna().all()]
    if all_nan_cols:
        df = df.drop(columns=all_nan_cols)
    dropped_noise_columns = unnamed_cols + [c for c in all_nan_cols if c not in unnamed_cols]

    # 3) RENAME COLUMNS
    rename_map = {}
    for c in df.columns:
        canon = canonicalize_col(c)
        if canon in COLUMN_ALIASES:
            rename_map[c] = COLUMN_ALIASES[canon]
    df = df.rename(columns=rename_map)
    df.columns = df.columns.str.lower()

    # 4) DATE PARSING & SWAP
    if "sample_day" not in df.columns:
        raise ValueError("Không tìm thấy cột Sample Day trong dataset.")
    df["sample_day"] = parse_date_series(df["sample_day"])
    if "tested_day" in df.columns:
        df["tested_day"] = parse_date_series(df["tested_day"])
    else:
        df["tested_day"] = pd.NaT
    swap_mask = df["tested_day"].notna() & df["sample_day"].notna() & (df["tested_day"] < df["sample_day"])
    n_swapped = swap_mask.sum()
    if n_swapped > 0:
        df.loc[swap_mask, ["sample_day", "tested_day"]] = df.loc[swap_mask, ["tested_day", "sample_day"]].values

    # 5) TEXT COLUMNS
    for col in ["loc", "name", "codetx", "mfg"]:
        if col not in df.columns:
            df[col] = None
        df[col] = df[col].apply(clean_text_field)
    if "ser" not in df.columns:
        df["ser"] = None
    df["ser"] = df["ser"].apply(clean_ser)

    # 6) FILL LOC
    df["loc"] = df.apply(infer_loc_from_codetx_and_name, axis=1)

    # 7) FILL MISSING SER
    df = fill_missing_ser(df)

    # 8) BUILD TRANSFORMER_ID
    df["transformer_id"] = build_transformer_id(df)

    # 9) CLEAN YEAR / TEMP / WATER
    if "year_energized" in df.columns:
        df["year_energized"] = df["year_energized"].apply(clean_year_energized)
    else:
        df["year_energized"] = np.nan
    if "temp" in df.columns:
        df["temp"] = df["temp"].apply(clean_temp)
    else:
        df["temp"] = np.nan
    if "water" in df.columns:
        df["water"] = df["water"].apply(clean_water)
    else:
        df["water"] = np.nan

    # 10) CLEAN GAS COLUMNS
    for col in CORE_GASES + OPTIONAL_GASES:
        if col in df.columns:
            df[col] = df[col].apply(clean_gas_value)
        else:
            df[col] = np.nan

    # 11) CLEAN TCG (raw) – sẽ bị ghi đè sau khi tính lại, tạm giữ để so sánh
    if "tcg" in df.columns:
        df["tcg"] = df["tcg"].apply(clean_gas_value)
    else:
        df["tcg"] = np.nan
    tcg_raw = df["tcg"].copy()   # lưu tạm để thống kê sai lệch

    # 12) CLEAN MVA / KV
    if "mva" in df.columns:
        df["mva"] = df["mva"].apply(normalize_rating_text)
    else:
        df["mva"] = None
    if "kv" in df.columns:
        df["kv"] = df["kv"].apply(normalize_rating_text)
    else:
        df["kv"] = None

    # 13) NB
    if "nb" in df.columns:
        df["nb"] = df["nb"].apply(clean_text_field)
    else:
        if KEEP_NB:
            df["nb"] = None

    # 14) SORT
    df = df.sort_values(["transformer_id", "sample_day"], kind="mergesort").reset_index(drop=True)

    # 15) DROP DUPLICATES
    before_dedup = len(df)
    df = df.drop_duplicates(keep="first").reset_index(drop=True)
    duplicate_rows_removed = before_dedup - len(df)

    # 16) RECALCULATE TCG (ghi đè cột tcg)
    df["tcg"] = (
        df["h2"].fillna(0)
        + df["ch4"].fillna(0)
        + df["c2h6"].fillna(0)
        + df["c2h4"].fillna(0)
        + df["c2h2"].fillna(0)
        + df["co"].fillna(0)
    )
    # Đếm số dòng có sai lệch > 0.1 so với giá trị gốc (nếu có)
    n_diff = (tcg_raw.notna() & (abs(tcg_raw - df["tcg"]) > 0.1)).sum()

    # 17) IMPUTATION & ROUNDING
    for col in ["temp", "water", "year_energized"]:
        if col in df.columns and df[col].isna().any():
            df[col] = impute_by_transformer_timeseries(df, col)

    # Làm tròn về số nguyên
    for col in ["temp", "water"]:
        if col in df.columns:
            df[col] = df[col].round().astype(int)
    if "year_energized" in df.columns:
        df["year_energized"] = df["year_energized"].round().astype(int)

    # 18) OUTPUT COLUMN ORDER
    preferred_order = [
        "transformer_id",
        "loc", "name", "ser", "codetx", "mfg",
        "sample_day", "tested_day", "year_energized",
        "mva", "kv",
        "temp", "water",
        "h2", "ch4", "c2h6", "c2h4", "c2h2", "co", "co2",
        "o2", "n2", "c3h6", "c3h8",
        "tcg",                 # TCG đã tính lại đúng chuẩn
        "nb",
    ]
    existing_cols = [c for c in preferred_order if c in df.columns]
    extra_cols = [c for c in df.columns if c not in existing_cols]
    df = df[existing_cols + extra_cols].copy()

    # 19) SUMMARY
    missing_summary = report_missing(df)
    summary = {
        "input_file": str(input_file),
        "sheet_name": sheet_name,
        "original_shape": list(original_shape),
        "clean_shape": list(df.shape),
        "original_columns": original_columns,
        "dropped_noise_columns": dropped_noise_columns,
        "clean_columns": df.columns.tolist(),
        "duplicate_rows_removed": int(duplicate_rows_removed),
        "n_unique_transformers": int(df["transformer_id"].nunique(dropna=True)),
        "date_min": None if df["sample_day"].dropna().empty else str(df["sample_day"].min()),
        "date_max": None if df["sample_day"].dropna().empty else str(df["sample_day"].max()),
        "rows_missing_transformer_id": int(df["transformer_id"].isna().sum()),
        "rows_missing_sample_day": int(df["sample_day"].isna().sum()),
        "rows_missing_year_energized": int(df["year_energized"].isna().sum()),
        "rows_missing_temp": int(df["temp"].isna().sum()),
        "rows_missing_water": int(df["water"].isna().sum()),
        "rows_with_tested_before_sample_swapped": int(n_swapped),
        "tcg_discrepancies_count": int(n_diff),
        "core_gases": CORE_GASES,
        "notes": {
            "imputation_method": "linear_interpolation_per_transformer_then_median",
            "imputed_columns": ["temp", "water", "year_energized"],
            "temp_water_year_rounded_to_int": True,
            "tcg_recalculated_directly_inplace": True,
            "tdcg_will_be_added_in_feature_engineering": True,
            "loc_filled_if_codetx_equals_loc_plus_name": True,
            "ser_filled_by_codetx_mfg_rule": True,
            "xxxx_ser_assigned_sequential_numbers": True,
            "sample_tested_swap_applied": True,
        },
    }

    # 20) SAVE
    output_dir.mkdir(parents=True, exist_ok=True)
    clean_parquet = output_dir / "dga_clean.parquet"
    clean_csv = output_dir / "dga_clean.csv"
    missing_csv = output_dir / "missing_summary.csv"
    summary_json = output_dir / "clean_summary.json"

    df.to_parquet(clean_parquet, index=False)
    df.to_csv(clean_csv, index=False, encoding="utf-8-sig")
    missing_summary.to_csv(missing_csv, index=False, encoding="utf-8-sig")
    with open(summary_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    return df, summary

# ============================================================
# CLI
# ============================================================
if __name__ == "__main__":
    pd.set_option("display.max_columns", 300)
    pd.set_option("display.width", 220)
    df_clean, summary = clean_dataset()
    print("=" * 100)
    print("CLEAN DATASET SUMMARY")
    print("=" * 100)
    for k, v in summary.items():
        print(f"{k}: {v}")
    print("\nSample cleaned rows (temp, water, year are integers):")
    print(df_clean.head(10).to_string(index=False))
    print("\nMissing summary (top 40):")
    ms = report_missing(df_clean).head(40)
    print(ms.to_string(index=False))
    print(f"\nSaved cleaned outputs to: {OUTPUT_DIR.resolve()}")