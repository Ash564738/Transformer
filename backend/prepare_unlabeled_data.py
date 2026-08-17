# prepare_unlabeled_data.py
from __future__ import annotations
import logging
from pathlib import Path
import pandas as pd
from config import DATASET_DIR
from consensus import apply_consensus
from feature_engineering import build_training_features_from_clean
from logging_config import init_logging

init_logging()
logger = logging.getLogger(__name__)

UNLABELED_PATH = Path(DATASET_DIR) / "processed" / "dga_unlabeled.parquet"
ACCUMULATED_PARQUET = Path(DATASET_DIR) / "processed" / "accumulated_clean.parquet"
ACCUMULATED_CSV = Path(DATASET_DIR) / "processed" / "accumulated_clean.csv"
CLEAN_PARQUET = Path(DATASET_DIR) / "processed" / "dga_clean.parquet"

def load_source() -> pd.DataFrame:
    if CLEAN_PARQUET.exists():
        logger.info("Loading clean dataset: %s", CLEAN_PARQUET)
        df = pd.read_parquet(CLEAN_PARQUET)
        logger.info("Loaded %d rows from dga_clean.parquet.", len(df))
        return df
    if ACCUMULATED_PARQUET.exists():
        logger.info("Loading accumulated parquet: %s", ACCUMULATED_PARQUET)
        df = pd.read_parquet(ACCUMULATED_PARQUET)
        logger.info("Loaded %d rows from accumulated_clean.parquet.", len(df))
        return df
    if ACCUMULATED_CSV.exists():
        logger.info("Loading legacy accumulated CSV: %s", ACCUMULATED_CSV)
        df = pd.read_csv(ACCUMULATED_CSV)
        logger.info("Loaded %d rows from accumulated_clean.csv.", len(df))
        return df
    raise FileNotFoundError(
        "No clean dataset was found.\n"
        f"Expected one of:\n"
        f"  {CLEAN_PARQUET}\n"
        f"  {ACCUMULATED_PARQUET}\n"
        f"  {ACCUMULATED_CSV}"
    )

def build_features_and_consensus(df: pd.DataFrame) -> pd.DataFrame:
    logger.info("Starting canonical feature engineering...")
    df = build_training_features_from_clean(df)
    logger.info("Feature engineering complete. Running traditional DGA diagnostics...")
    df = apply_consensus(df)
    logger.info("Consensus complete.")
    return df

def validate_output(df: pd.DataFrame) -> None:
    required_columns = [
        "transformer_id", "sample_day",
        "h2", "ch4", "c2h6", "c2h4", "c2h2", "co", "co2",
        "tdcg",
        "keygas_fault", "iec_fault", "rogers_fault", "doernenburg_fault",
        "duval_triangle_fault", "duval_pentagon_p1_fault", "duval_pentagon_p2_fault",
        "consensus_fault", "diagnostic_confidence"
    ]
    missing = [col for col in required_columns if col not in df.columns]
    if missing:
        raise ValueError("Prepared unlabeled dataset is missing required columns: %s" % missing)
    if df.empty:
        raise ValueError("Prepared unlabeled dataset is empty.")
    if df["transformer_id"].isna().any():
        raise ValueError("transformer_id contains missing values.")
    if df["sample_day"].isna().any():
        raise ValueError("sample_day contains missing values.")
    logger.info("Output validation passed.")
    logger.info("Rows: %d", len(df))
    logger.info("Transformers: %d", df["transformer_id"].nunique())
    logger.info("Consensus distribution:\n%s", df["consensus_fault"].value_counts(dropna=False).to_string())

def main():
    logger.info("=" * 80)
    logger.info("PREPARING UNLABELED DATASET")
    logger.info("=" * 80)
    df = load_source()
    df = build_features_and_consensus(df)
    validate_output(df)
    UNLABELED_PATH.parent.mkdir(parents=True, exist_ok=True)
    logger.info("Saving unlabeled dataset to: %s", UNLABELED_PATH)
    df.to_parquet(UNLABELED_PATH, index=False)
    logger.info("Saved %d rows.", len(df))
    logger.info("=" * 80)
    logger.info("UNLABELED DATA PREPARATION COMPLETE")
    logger.info("Output: %s", UNLABELED_PATH)
    logger.info("=" * 80)

if __name__ == "__main__":
    main()