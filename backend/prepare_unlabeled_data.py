# prepare_unlabeled_data.py
"""
Chuẩn bị dữ liệu cho Weak Supervision.
Đọc dữ liệu tích lũy (accumulated_clean.csv) hoặc dga_clean.parquet,
chạy feature engineering + consensus để tạo các cột vote,
lưu thành dga_unlabeled.parquet.
"""
import sys
from pathlib import Path
import pandas as pd
import numpy as np
from logging_config import init_logging
init_logging()
import logging
logger = logging.getLogger(__name__)

from feature_engineering import (
    preprocess_types, sort_and_deduplicate, filter_rows_for_model,
    add_missingness_flags, impute_optional_context_by_transformer,
    add_tdcg, add_rating_features, add_metadata_features,
    add_ratio_features, add_duval_input_features,
    add_calendar_and_sequence_features, add_lag_delta_rate_features,
    add_rolling_features, add_ewm_features, add_cross_gas_trend_features,
    add_quality_flags, CORE_GASES, OPTIONAL_NUMERIC
)
from consensus import apply_consensus
from config import BACKEND_DATA_DIR, BACKEND_ROOT

UNLABELED_PATH = Path(BACKEND_DATA_DIR) / "dga_unlabeled.parquet"
ACCUMULATED_CSV = Path(BACKEND_ROOT) / "dataset" / "accumulated_clean.csv"
CLEAN_PARQUET = Path(BACKEND_DATA_DIR) / "dga_clean.parquet"

def build_features_and_consensus(df: pd.DataFrame) -> pd.DataFrame:
    """Áp dụng toàn bộ pipeline feature engineering + consensus."""
    df = preprocess_types(df)
    df = sort_and_deduplicate(df)
    df = filter_rows_for_model(df, max_missing_core=3)
    df = add_missingness_flags(df, OPTIONAL_NUMERIC + ["year_energized", "tdcg_raw"])
    df = impute_optional_context_by_transformer(df)
    df = add_tdcg(df)
    df = add_rating_features(df)
    df = add_metadata_features(df)
    df = add_ratio_features(df)
    df = add_duval_input_features(df)
    df = add_calendar_and_sequence_features(df)

    temporal_value_cols = [c for c in CORE_GASES + ["tdcg"] if c in df.columns]
    for c in ["water", "temp"]:
        if c in df.columns:
            temporal_value_cols.append(c)

    df = add_lag_delta_rate_features(df, temporal_value_cols)
    df = add_rolling_features(df, temporal_value_cols)
    df = add_ewm_features(df, temporal_value_cols)
    df = add_cross_gas_trend_features(df)
    df = add_quality_flags(df)
    df = apply_consensus(df)
    return df

def main():
    logger.info("Preparing unlabeled dataset for weak supervision...")
    if CLEAN_PARQUET.exists():
        df = pd.read_parquet(CLEAN_PARQUET)
    elif ACCUMULATED_CSV.exists():
        df = pd.read_csv(ACCUMULATED_CSV)
    else:
        raise FileNotFoundError(
            "Không tìm thấy dữ liệu. Hãy chạy predict ít nhất một lần để tạo accumulated_clean.csv "
            "hoặc đặt file dga_clean.parquet trong thư mục processed."
        )
    logger.info(f"Loaded {len(df)} rows from source.")

    df = build_features_and_consensus(df)
    logger.info(f"Saving unlabeled dataset ({len(df)} rows) to {UNLABELED_PATH}")
    UNLABELED_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(UNLABELED_PATH)
    logger.info("Done. Bạn có thể chạy train_models.py --weak-supervision --use-snorkel ngay bây giờ.")

if __name__ == "__main__":
    main()