# dataset_accumulator.py
"""Accumulates cleaned DGA data across /predict calls, instead of each
upload silently replacing everything that came before.

Why this matters: a real user doesn't re-upload their entire historical
dataset every time — they upload whatever new batch of lab readings they
just got, which is often a different file with a different shape than the
last one. clean_dataset() already normalizes column names/formats per file
independently, so by the time a DataFrame reaches merge_with_accumulated()
it's already in the same shape regardless of which raw file it came from —
that's what makes merging safe here rather than in inference_service.py's
raw upload handling.

Dedup key is (transformer_id, sample_day): the same transformer's reading
for the same day is the same lab sample, no matter which upload it arrived
in. On a genuine collision (same key, different gas values — e.g. a
corrected re-upload), the newest upload wins.
"""
import logging
from pathlib import Path

import pandas as pd
from config import DATASET_DIR, DATABASE_DIR, MODEL_DIR, REPORT_DIR

logger = logging.getLogger(__name__)

ACCUMULATED_PATH = DATASET_DIR / "processed" / "accumulated_clean.csv"

_DATE_COLS = ["sample_day", "tested_day"]


def _load_existing() -> pd.DataFrame | None:
    if not ACCUMULATED_PATH.exists():
        logger.debug("No accumulated dataset file found at %s", ACCUMULATED_PATH)
        return None
    try:
        df = pd.read_csv(ACCUMULATED_PATH)
        logger.debug("Loaded existing accumulated dataset: %d rows", len(df))
    except Exception:
        logger.exception("Failed to read accumulated_clean.csv; ignoring previous history.")
        return None
    for col in _DATE_COLS:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")
    return df


def merge_with_accumulated(df_clean_new: pd.DataFrame) -> pd.DataFrame:
    """Merges a freshly-cleaned upload into the accumulated history, drops
    duplicate (transformer_id, sample_day) samples (keeping the newest
    upload's version), and persists the merged result for next time."""
    existing = _load_existing()
    if existing is None or existing.empty:
        logger.info("No existing accumulated data; using new upload as initial dataset.")
        merged = df_clean_new.copy()
    else:
        # Align columns: different uploads can add/omit optional columns
        # (e.g. one file has NB notes, another doesn't) — outer-join the
        # column sets so neither side loses data, missing cells become NaN.
        merged = pd.concat([existing, df_clean_new], ignore_index=True, sort=False)

    before = len(merged)
    if "transformer_id" in merged.columns and "sample_day" in merged.columns:
        merged = merged.drop_duplicates(subset=["transformer_id", "sample_day"], keep="last")
    duplicates_dropped = before - len(merged)

    if "transformer_id" in merged.columns and "sample_day" in merged.columns:
        merged = merged.sort_values(["transformer_id", "sample_day"], kind="mergesort").reset_index(drop=True)

    logger.info(
        "Data accumulation: %d old rows + %d new rows -> %d rows (%d duplicates removed).",
        len(existing) if existing is not None else 0,
        len(df_clean_new),
        len(merged),
        duplicates_dropped,
    )

    ACCUMULATED_PATH.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(ACCUMULATED_PATH, index=False)
    logger.info("Accumulated dataset saved to %s", ACCUMULATED_PATH)
    return merged


def reset_accumulated_dataset() -> None:
    """Wipes accumulated history — used when the user explicitly clears the
    loaded dataset, so 'Clear data' actually starts fresh next upload
    instead of silently merging into whatever was there before."""
    if ACCUMULATED_PATH.exists():
        ACCUMULATED_PATH.unlink()
        logger.info("Deleted accumulated dataset file: %s", ACCUMULATED_PATH)
    else:
        logger.info("No accumulated dataset file to delete.")