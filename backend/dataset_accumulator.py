# dataset_accumulator.py
from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from config import DATASET_DIR

logger = logging.getLogger(
    __name__
)


ACCUMULATED_PATH = (
    DATASET_DIR
    / "processed"
    / "accumulated_clean.parquet"
)

LEGACY_CSV_PATH = (
    DATASET_DIR
    / "processed"
    / "accumulated_clean.csv"
)


_DATE_COLS = [
    "sample_day",
    "tested_day",
]


def _load_existing():

    if ACCUMULATED_PATH.exists():

        try:

            df = pd.read_parquet(
                ACCUMULATED_PATH
            )

            for column in (
                _DATE_COLS
            ):

                if column in df.columns:
                    df[column] = (
                        pd.to_datetime(
                            df[column],
                            errors="coerce",
                        )
                    )

            return df

        except Exception:

            logger.exception(
                "Failed to read accumulated parquet."
            )

    # Backward compatibility.
    if LEGACY_CSV_PATH.exists():

        try:

            df = pd.read_csv(
                LEGACY_CSV_PATH
            )

            for column in (
                _DATE_COLS
            ):

                if column in df.columns:
                    df[column] = (
                        pd.to_datetime(
                            df[column],
                            errors="coerce",
                        )
                    )

            return df

        except Exception:

            logger.exception(
                "Failed to read legacy accumulated CSV."
            )

    return None


def merge_with_accumulated(
    df_clean_new: pd.DataFrame,
) -> pd.DataFrame:

    new = df_clean_new.copy()

    existing = _load_existing()

    if existing is None or existing.empty:

        merged = new

        old_count = 0

    else:

        old_count = len(
            existing
        )

        # Outer union of columns.
        merged = pd.concat(
            [
                existing,
                new,
            ],
            ignore_index=True,
            sort=False,
        )

    required = {
        "transformer_id",
        "sample_day",
    }

    missing = (
        required
        - set(
            merged.columns
        )
    )

    if missing:
        raise ValueError(
            f"Accumulated dataset missing: {sorted(missing)}"
        )

    merged["sample_day"] = (
        pd.to_datetime(
            merged["sample_day"],
            errors="coerce",
        )
    )

    if "tested_day" in merged.columns:

        merged["tested_day"] = (
            pd.to_datetime(
                merged["tested_day"],
                errors="coerce",
            )
        )

    merged = merged.dropna(
        subset=[
            "transformer_id",
            "sample_day",
        ]
    )

    before = len(
        merged
    )

    merged = (
        merged.sort_values(
            [
                "transformer_id",
                "sample_day",
            ],
            kind="mergesort",
        )
        .drop_duplicates(
            subset=[
                "transformer_id",
                "sample_day",
            ],
            keep="last",
        )
        .reset_index(
            drop=True
        )
    )

    duplicates_dropped = (
        before - len(
            merged
        )
    )

    ACCUMULATED_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    # Atomic-ish write.
    temp_path = (
        ACCUMULATED_PATH.with_suffix(
            ".tmp.parquet"
        )
    )

    merged.to_parquet(
        temp_path,
        index=False,
    )

    temp_path.replace(
        ACCUMULATED_PATH
    )

    # Keep CSV only as a backward-compatible mirror.
    merged.to_csv(
        LEGACY_CSV_PATH,
        index=False,
    )

    logger.info(
        "Accumulation: %d old + %d new -> %d rows; %d duplicate keys removed.",
        old_count,
        len(new),
        len(merged),
        duplicates_dropped,
    )

    return merged


def reset_accumulated_dataset():

    for path in [
        ACCUMULATED_PATH,
        LEGACY_CSV_PATH,
    ]:

        if path.exists():

            path.unlink()

            logger.info(
                "Deleted accumulated dataset: %s",
                path,
            )