# clean_dataset.py
from __future__ import annotations
import json, logging, re
from pathlib import Path
from typing import Any, Dict, Optional, Tuple
import numpy as np
import pandas as pd
from config import DATASET_DIR
logger = logging.getLogger(__name__)
INPUT_FILE = DATASET_DIR / "DGA of Main Tank only KT 11022026_09062026.xlsx"
OUTPUT_DIR = DATASET_DIR / "processed"
KEEP_NB = True
COLUMN_ALIASES = {"sampling date": "sample_day", "sample date": "sample_day", "sampling day": "sample_day", "sample day": "sample_day", "sample_day": "sample_day", "test date": "tested_day", "tested date": "tested_day", "tested day": "tested_day", "tested_day": "tested_day", "year energized": "year_energized", "year_energized": "year_energized", "loc": "loc", "name": "name", "substation": "loc", "bay": "name", "ser": "ser", "codetx": "codetx", "mfg": "mfg", "kv": "kv", "rated voltage(kv)": "kv", "rated voltage": "kv", "mva": "mva", "temp": "temp", "oil temperature at tested": "temp", "oil temperature": "temp", "water": "water", "wat": "water", "h2o": "water", "h2": "h2", "ch4": "ch4", "c2h6": "c2h6", "c2h4": "c2h4", "c2h2": "c2h2", "co": "co", "co2": "co2", "o2": "o2", "n2": "n2", "c3h6": "c3h6", "c3h8": "c3h8", "tdcg": "tdcg_raw", "tdcg raw": "tdcg_raw", "tcg": "tdcg_raw", "summary and comment": "nb", "summary": "nb", "comment": "nb", "nb": "nb"}
CORE_GASES = ["h2", "ch4", "c2h6", "c2h4", "c2h2", "co", "co2"]
OPTIONAL_GASES = ["o2", "n2", "c3h6", "c3h8"]
def normalize_col_name(col: Any) -> str:
    if col is None: return ""
    try:
        if pd.isna(col): return ""
    except (TypeError, ValueError): pass
    s = str(col).strip()
    s = re.sub(r"\s+", " ", s)
    return s
def canonicalize_col(col: str) -> str:
    s = normalize_col_name(col).lower().replace("_", " ")
    return re.sub(r"\s+", " ", s).strip()
def is_unnamed_or_empty_column(col: Any) -> bool:
    s = normalize_col_name(col)
    if s == "": return True
    return bool(re.match(r"^unnamed(?::\s*\d+)?$", s, flags=re.IGNORECASE))
def clean_text_basic(x: Any) -> Optional[str]:
    if pd.isna(x): return None
    s = str(x).strip()
    if s == "": return None
    if s.lower() in {"nan", "none", "null", "n/a", "na", "-", "--", "#value!"}: return None
    return s
def clean_text_field(x: Any) -> Optional[str]:
    s = clean_text_basic(x)
    if s is None: return None
    return re.sub(r"\s+", " ", s).strip() or None
def clean_ser(x: Any) -> Optional[str]:
    s = clean_text_field(x)
    if s is None: return None
    if s.lower() == "xxxx": return None
    return s
def parse_numeric_loose(x: Any, allow_negative: bool = True) -> float:
    if pd.isna(x): return np.nan
    if isinstance(x, (int, float, np.integer, np.floating)):
        value = float(x)
        if not allow_negative and value < 0: return np.nan
        return value
    s = str(x).strip()
    if s == "" or s.lower() in {"nan", "none", "null", "n/a", "na", "-", "--", "#value!", "inf", "-inf"}: return np.nan
    if "/" in s: return np.nan
    s = s.replace(",", "")
    match = re.search(r"-?\d+(?:\.\d+)?", s)
    if not match: return np.nan
    try: value = float(match.group(0))
    except Exception: return np.nan
    if not allow_negative and value < 0: return np.nan
    return value
def clean_year_energized(x: Any) -> float:
    value = parse_numeric_loose(x, allow_negative=False)
    if pd.isna(value): return np.nan
    year = int(round(value))
    if not (1800 <= year <= 2100): return np.nan
    return float(year)
def clean_temp(x: Any) -> float:
    return parse_numeric_loose(x, allow_negative=True)
def clean_water(x: Any) -> float:
    value = parse_numeric_loose(x, allow_negative=False)
    if pd.isna(value): return np.nan
    return float(value)
def clean_gas_value(x: Any) -> float:
    value = parse_numeric_loose(x, allow_negative=False)
    if pd.isna(value): return np.nan
    return float(value)
def normalize_rating_text(x: Any) -> Optional[str]:
    s = clean_text_field(x)
    if s is None: return None
    s = s.replace(",", "")
    s = re.sub(r"(?<=\d)\s*-\s*(?=\d)", "/", s)
    s = re.sub(r"\s*/\s*", "/", s)
    s = re.sub(r"\s+", " ", s).strip()
    if not s: return None
    if s.lower() in {"nan", "none", "null"}: return None
    return s
def parse_date_series(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s, errors="coerce", dayfirst=True)
def infer_loc_from_codetx_and_name(row: pd.Series) -> Optional[str]:
    loc = clean_text_field(row.get("loc"))
    if loc is not None: return loc
    codetx = clean_text_field(row.get("codetx"))
    name = clean_text_field(row.get("name"))
    if codetx is None or name is None: return None
    if codetx.endswith(name):
        candidate = codetx[:-len(name)].strip()
        return candidate if candidate else None
    return None
def fill_missing_ser(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "ser" not in df.columns: df["ser"] = None
    df["ser"] = df["ser"].apply(clean_ser)
    existing_ser = set(str(value) for value in df["ser"].dropna().unique() if str(value).strip())
    for (codetx, mfg), indices in df.groupby(["codetx", "mfg"], dropna=False).groups.items():
        valid = df.loc[indices, "ser"].dropna()
        if not valid.empty:
            df.loc[indices, "ser"] = df.loc[indices, "ser"].fillna(valid.iloc[0])
    counter = 1
    for (codetx, mfg), indices in df.groupby(["codetx", "mfg"], dropna=False).groups.items():
        if df.loc[indices, "ser"].isna().all():
            while f"xxxx{counter}" in existing_ser:
                counter += 1
            synthetic = f"xxxx{counter}"
            df.loc[indices, "ser"] = synthetic
            existing_ser.add(synthetic)
            counter += 1
    df["ser_is_synthetic"] = (df["ser"].astype(str).str.lower().str.startswith("xxxx").astype("int8"))
    return df
def build_transformer_id(df: pd.DataFrame) -> pd.Series:
    index = df.index
    ser = df["ser"].fillna("").astype(str).str.strip() if "ser" in df.columns else pd.Series("", index=index)
    codetx = df["codetx"].fillna("").astype(str).str.strip() if "codetx" in df.columns else pd.Series("", index=index)
    loc = df["loc"].fillna("").astype(str).str.strip() if "loc" in df.columns else pd.Series("", index=index)
    name = df["name"].fillna("").astype(str).str.strip() if "name" in df.columns else pd.Series("", index=index)
    synthetic = ser.str.lower().str.startswith("xxxx")
    transformer_id = pd.Series(np.nan, index=index, dtype="object")
    real_ser = ser.where(~synthetic, "").replace({"nan": "", "None": ""})
    transformer_id = transformer_id.mask(real_ser.ne(""), real_ser)
    transformer_id = transformer_id.mask(transformer_id.isna() & codetx.ne(""), codetx)
    fallback = (loc + " | " + name).str.strip()
    fallback = fallback.str.replace(r"^\|\s*", "", regex=True)
    fallback = fallback.str.replace(r"\s*\|$", "", regex=True)
    transformer_id = transformer_id.mask(transformer_id.isna() & fallback.ne(""), fallback)
    return transformer_id
def report_missing(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame({"column": df.columns, "missing_count": df.isna().sum().values, "missing_ratio": (df.isna().mean() * 100).round(2).values, "dtype": [str(df[col].dtype) for col in df.columns]})
    return out.sort_values(["missing_ratio", "column"], ascending=[False, True]).reset_index(drop=True)
def _find_header_row(raw: pd.DataFrame) -> int:
    best_idx = 0
    best_score = -1
    for i in range(min(20, len(raw))):
        row_values = [canonicalize_col(v) for v in raw.iloc[i].tolist()]
        score = sum(1 for v in row_values if v in COLUMN_ALIASES)
        if score > best_score:
            best_score = score
            best_idx = i
    return best_idx
def _prepare_dataframe(df: pd.DataFrame) -> Tuple[pd.DataFrame, Tuple[int, int], list, str, int]:
    if df is None: raise ValueError("Input dataframe is None.")
    if df.empty: raise ValueError("Input dataframe is empty.")
    df = df.copy()
    original_shape = df.shape
    original_columns = [normalize_col_name(col) for col in df.columns]
    rename_map = {}
    for col in df.columns:
        canonical = canonicalize_col(col)
        if canonical in COLUMN_ALIASES: rename_map[col] = COLUMN_ALIASES[canonical]
    if rename_map: df = df.rename(columns=rename_map)
    unnamed = [col for col in df.columns if is_unnamed_or_empty_column(col)]
    if unnamed: df = df.drop(columns=unnamed)
    all_nan = [col for col in df.columns if df[col].isna().all()]
    if all_nan: df = df.drop(columns=all_nan)
    df.columns = [str(col).lower().strip() for col in df.columns]
    header_rows_removed = 0
    if "sample_day" not in df.columns:
        header_idx = _find_header_row(df)
        candidate_headers = [normalize_col_name(col) for col in df.iloc[header_idx].tolist()]
        score = sum(1 for col in candidate_headers if canonicalize_col(col) in COLUMN_ALIASES)
        if score > 0:
            df = df.iloc[header_idx + 1:].copy()
            df.columns = candidate_headers
            rename_map = {}
            for col in df.columns:
                canonical = canonicalize_col(col)
                if canonical in COLUMN_ALIASES: rename_map[col] = COLUMN_ALIASES[canonical]
            if rename_map: df = df.rename(columns=rename_map)
            df.columns = [str(col).lower().strip() for col in df.columns]
            header_rows_removed = header_idx + 1
    sheet_name = "uploaded_dataframe"
    return df, original_shape, original_columns, sheet_name, header_rows_removed
def clean_dataset(input_file: Path = INPUT_FILE, output_dir: Path = OUTPUT_DIR, dataframe: Optional[pd.DataFrame] = None) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    output_dir.mkdir(parents=True, exist_ok=True)
    if dataframe is not None:
        df, original_shape, original_columns, sheet_name, header_rows_removed = _prepare_dataframe(dataframe)
        input_description = "uploaded_dataframe"
    else:
        input_file = Path(input_file)
        if not input_file.exists(): raise FileNotFoundError(f"Input file not found: {input_file}")
        logger.info("Reading input Excel: %s", input_file)
        xl = pd.ExcelFile(input_file)
        sheet_name = xl.sheet_names[0]
        raw = xl.parse(sheet_name, header=None)
        header_idx = _find_header_row(raw)
        raw.columns = [normalize_col_name(col) for col in raw.iloc[header_idx].tolist()]
        df = raw.iloc[header_idx + 1:].copy()
        original_shape = df.shape
        original_columns = df.columns.tolist()
        unnamed = [col for col in df.columns if is_unnamed_or_empty_column(col)]
        if unnamed: df = df.drop(columns=unnamed)
        all_nan = [col for col in df.columns if df[col].isna().all()]
        if all_nan: df = df.drop(columns=all_nan)
        rename_map = {}
        for col in df.columns:
            canonical = canonicalize_col(col)
            if canonical in COLUMN_ALIASES: rename_map[col] = COLUMN_ALIASES[canonical]
        if rename_map: df = df.rename(columns=rename_map)
        df.columns = [str(col).lower() for col in df.columns]
        header_rows_removed = header_idx + 1
        input_description = str(input_file)
    if "sample_day" not in df.columns: raise ValueError("Missing sample_day.")
    df["sample_day"] = parse_date_series(df["sample_day"])
    if "tested_day" in df.columns: df["tested_day"] = parse_date_series(df["tested_day"])
    else: df["tested_day"] = pd.NaT
    swap_mask = df["tested_day"].notna() & df["sample_day"].notna() & (df["tested_day"] < df["sample_day"])
    n_swapped = int(swap_mask.sum())
    if n_swapped:
        original_sample = df.loc[swap_mask, "sample_day"].copy()
        df.loc[swap_mask, "sample_day"] = df.loc[swap_mask, "tested_day"].to_numpy()
        df.loc[swap_mask, "tested_day"] = original_sample.to_numpy()
    for col in ["loc", "name", "codetx", "mfg"]:
        if col not in df.columns: df[col] = None
        df[col] = df[col].apply(clean_text_field)
    if "ser" not in df.columns: df["ser"] = None
    df["ser"] = df["ser"].apply(clean_ser)
    inferred_loc = df.apply(infer_loc_from_codetx_and_name, axis=1)
    df["loc"] = df["loc"].fillna(inferred_loc)
    df = fill_missing_ser(df)
    df["transformer_id"] = build_transformer_id(df)
    if "year_energized" in df.columns: df["year_energized"] = df["year_energized"].apply(clean_year_energized)
    else: df["year_energized"] = np.nan
    if "temp" in df.columns: df["temp"] = df["temp"].apply(clean_temp)
    else: df["temp"] = np.nan
    if "water" in df.columns: df["water"] = df["water"].apply(clean_water)
    else: df["water"] = np.nan
    for gas in CORE_GASES + OPTIONAL_GASES:
        if gas in df.columns: df[gas] = df[gas].apply(clean_gas_value)
        else: df[gas] = np.nan
    if "tdcg_raw" in df.columns: df["tdcg_raw"] = df["tdcg_raw"].apply(clean_gas_value)
    else: df["tdcg_raw"] = np.nan
    for col in ["mva", "kv"]:
        if col in df.columns: df[col] = df[col].apply(normalize_rating_text)
        else: df[col] = None
    if KEEP_NB:
        if "nb" in df.columns: df["nb"] = df["nb"].apply(clean_text_field)
        else: df["nb"] = None
    df = df.sort_values(["transformer_id", "sample_day"]).reset_index(drop=True)
    before = len(df)
    df = df.drop_duplicates(keep="last").reset_index(drop=True)
    duplicate_rows_removed = before - len(df)
    tdcg_components = ["h2", "ch4", "c2h6", "c2h4", "c2h2", "co"]
    df["tdcg_recalc"] = df[tdcg_components].sum(axis=1, min_count=1)
    measured_count = df[tdcg_components].notna().sum(axis=1)
    df["tdcg_complete"] = (measured_count == 6).astype("int8")
    df["tdcg"] = df["tdcg_raw"].where(df["tdcg_raw"].notna(), df["tdcg_recalc"])
    df["tdcg_source"] = np.where(df["tdcg_raw"].notna(), "raw_available", "recalculated")
    df["tdcg_difference_raw_recalc"] = df["tdcg_raw"] - df["tdcg_recalc"]
    df["tcg"] = df["tdcg"]
    preferred_order = ["transformer_id", "loc", "name", "ser", "ser_is_synthetic", "codetx", "mfg", "sample_day", "tested_day", "year_energized", "mva", "kv", "temp", "water", "h2", "ch4", "c2h6", "c2h4", "c2h2", "co", "co2", "o2", "n2", "c3h6", "c3h8", "tdcg_raw", "tdcg_recalc", "tdcg", "tcg", "tdcg_complete", "tdcg_source", "tdcg_difference_raw_recalc", "nb"]
    preferred_order = list(dict.fromkeys(preferred_order))
    existing = [col for col in preferred_order if col in df.columns]
    extras = [col for col in df.columns if col not in existing]
    df = df[existing + extras].copy()
    missing_summary = report_missing(df)
    summary = {"input_file": input_description, "sheet_name": sheet_name, "original_shape": list(original_shape), "clean_shape": list(df.shape), "original_columns": original_columns, "dropped_noise_columns": [], "clean_columns": df.columns.tolist(), "duplicate_rows_removed": int(duplicate_rows_removed), "n_unique_transformers": int(df["transformer_id"].nunique(dropna=True)), "date_min": None if df["sample_day"].dropna().empty else str(df["sample_day"].min()), "date_max": None if df["sample_day"].dropna().empty else str(df["sample_day"].max()), "rows_missing_transformer_id": int(df["transformer_id"].isna().sum()), "rows_missing_sample_day": int(df["sample_day"].isna().sum()), "rows_missing_year_energized": int(df["year_energized"].isna().sum()), "rows_missing_temp": int(df["temp"].isna().sum()), "rows_missing_water": int(df["water"].isna().sum()), "rows_with_tested_before_sample_swapped": int(n_swapped), "synthetic_ser_count": int(df["ser_is_synthetic"].sum()), "core_gases": CORE_GASES, "header_rows_removed": int(header_rows_removed), "notes": {"water_negative_values_rejected": True, "temp_water_not_imputed": True, "year_energized_not_globally_imputed": True, "tdcg_recalculated_from_six_combustible_gases": True, "raw_tdcg_preserved": True, "synthetic_ser_not_used_as_transformer_id": True, "deduplicate_exact_rows_only": True, "auto_header_detection": True, "supports_dataframe_input": True}}
    output_dir.mkdir(parents=True, exist_ok=True)
    clean_parquet = output_dir / "dga_unlabeled.parquet"
    clean_csv = output_dir / "dga_unlabeled.csv"
    missing_csv = output_dir / "missing_summary.csv"
    summary_json = output_dir / "clean_summary.json"
    df.to_parquet(clean_parquet, index=False)
    df.to_csv(clean_csv, index=False, encoding="utf-8-sig")
    missing_summary.to_csv(missing_csv, index=False, encoding="utf-8-sig")
    with open(summary_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    logger.info("Cleaning complete: %d rows, %d columns, %d transformers.", len(df), len(df.columns), df["transformer_id"].nunique())
    return df, summary
if __name__ == "__main__":
    pd.set_option("display.max_columns", 300)
    pd.set_option("display.width", 220)
    cleaned, summary = clean_dataset()
    print("=" * 100)
    print("CLEAN DATASET SUMMARY")
    print("=" * 100)
    for key, value in summary.items():
        print(f"{key}: {value}")
    print("\nSample cleaned rows:")
    print(cleaned.head(10).to_string(index=False))
    print("\nMissing summary:")
    print(report_missing(cleaned).head(40).to_string(index=False))