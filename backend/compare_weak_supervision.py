# compare_weak_supervision.py
from __future__ import annotations

"""
DGA weak-supervision research pipeline.

Production weak supervision backend:
    Snorkel LabelModel ONLY

Pipeline:
    operational unlabeled data
        -> traditional diagnostic labeling functions
        -> Snorkel LabelModel
        -> weak labels
        -> student ML models
        -> evaluation on external labeled benchmark
        -> select best student
        -> optionally promote to production

Important:
- The 4,561 operational records are unlabeled and are NOT used to claim accuracy.
- Accuracy metrics are evaluated only on the external labeled benchmark.
- The locked test split is never used for model/backend selection.
- No manually assigned LF weights are introduced.
- No EM backend exists in this pipeline.
"""

import argparse
import json
import logging
import shutil
import time
from pathlib import Path
from typing import Iterable

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

from config import DATASET_DIR, MODEL_DIR, REPORT_DIR, config as cfg
from clean_dataset import clean_dataset
from consensus import apply_consensus
from feature_engineering import build_training_features_from_clean
from logging_config import init_logging

from train_unsupervised_models import (
    BENCHMARK_DIR,
    _align_feature_frame,
    _decode_class_predictions,
    _evaluate_method_labels_safely,
    _predict_model,
    _prepare_truth,
    _train_weak_students,
    load_labeled_csv_data,
)

from weak_supervision import (
    DEFAULT_WEAK_METHODS,
    SNORKEL_AVAILABLE,
    build_label_matrix,
    fit_label_model_backend,
    predict_from_label_model,
)

init_logging()
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

UNLABELED_PATH = DATASET_DIR / "processed" / "dga_unlabeled.parquet"
RAW_UNLABELED_PATH = (
    DATASET_DIR / "DGA of Main Tank only KT 11022026_09062026.xlsx"
)

RESULT_CSV = REPORT_DIR / "weak_supervision_snorkel_results.csv"
ROBUSTNESS_CSV = REPORT_DIR / "weak_supervision_snorkel_repeated_evaluation.csv"
OPERATIONAL_CSV = REPORT_DIR / "weak_supervision_snorkel_operational_stability.csv"
TIMING_CSV = REPORT_DIR / "weak_supervision_snorkel_timing.csv"
SELECTION_JSON = REPORT_DIR / "weak_supervision_snorkel_selection.json"

WS_MODEL_ROOT = MODEL_DIR / "weak_supervision_backends"
WS_REPORT_ROOT = BENCHMARK_DIR / "weak_supervision_backends"

SNORKEL_MODEL_ROOT = WS_MODEL_ROOT / "snorkel"
SNORKEL_REPORT_ROOT = WS_REPORT_ROOT / "snorkel"

PRODUCTION_COARSE = MODEL_DIR / "weak_label_model_coarse.joblib"
PRODUCTION_FINE = MODEL_DIR / "weak_label_model_fine.joblib"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _dedupe_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Arrow/Parquet requires unique column names.

    Keep the first occurrence of a duplicated column.
    """
    if not df.columns.duplicated().any():
        return df.copy()

    duplicated = df.columns[df.columns.duplicated()].tolist()

    logger.warning(
        "Dropping duplicate dataframe columns before parquet write: %s",
        duplicated,
    )

    return df.loc[
        :,
        ~df.columns.duplicated(keep="first"),
    ].copy()


def _ensure_clean_unlabeled() -> pd.DataFrame:
    """
    Load the complete operational dataset.

    If canonical parquet does not exist, create it from the original
    operational Excel file.
    """
    if UNLABELED_PATH.exists():
        df = pd.read_parquet(UNLABELED_PATH)
        df = _dedupe_columns(df)

        if "transformer_id" not in df.columns:
            raise ValueError(
                f"Canonical operational parquet is missing transformer_id: "
                f"{UNLABELED_PATH}"
            )

        logger.info(
            "Operational dataset loaded: rows=%d transformers=%d columns=%d",
            len(df),
            df["transformer_id"].nunique(),
            len(df.columns),
        )

        return df

    if not RAW_UNLABELED_PATH.exists():
        raise FileNotFoundError(
            f"Operational unlabeled dataset not found: {RAW_UNLABELED_PATH}"
        )

    logger.info(
        "Canonical operational parquet missing; preparing from %s",
        RAW_UNLABELED_PATH,
    )

    cleaned, _meta = clean_dataset(
        input_file=RAW_UNLABELED_PATH,
        output_dir=DATASET_DIR / "processed",
    )

    canonical = build_training_features_from_clean(cleaned)

    canonical = apply_consensus(canonical)

    canonical = _dedupe_columns(canonical)

    UNLABELED_PATH.parent.mkdir(parents=True, exist_ok=True)

    canonical.to_parquet(
        UNLABELED_PATH,
        index=False,
    )

    logger.info(
        "Prepared operational dataset: rows=%d transformers=%d columns=%d",
        len(canonical),
        canonical["transformer_id"].nunique(),
        len(canonical.columns),
    )

    return canonical


def _prepare_external(
    labeled: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """
    Prepare the complete usable external labeled benchmark.

    Conflict rows are excluded from benchmark truth evaluation because a
    single external record should not simultaneously carry conflicting
    canonical benchmark interpretations.
    """
    fine_truth_all, coarse_truth_all = _prepare_truth(labeled)

    conflict = labeled.get(
        "fine_label_conflict",
        pd.Series(False, index=labeled.index),
    ).astype(bool)

    coarse_conflict = labeled.get(
        "coarse_label_conflict",
        pd.Series(False, index=labeled.index),
    ).astype(bool)

    valid = (
        fine_truth_all.isin(cfg.BENCHMARK_FINE_CLASSES)
        & ~conflict
        & ~coarse_conflict
    )

    data = labeled.loc[valid].reset_index(drop=True)
    fine_truth = fine_truth_all.loc[valid].reset_index(drop=True)
    coarse_truth = coarse_truth_all.loc[valid].reset_index(drop=True)

    mapping = {
        label: index
        for index, label in enumerate(cfg.BENCHMARK_FINE_CLASSES)
    }

    encoded = fine_truth.map(mapping)

    keep = encoded.notna().to_numpy(dtype=bool)

    data = data.loc[keep].reset_index(drop=True)
    fine_truth = fine_truth.loc[keep].reset_index(drop=True)
    coarse_truth = coarse_truth.loc[keep].reset_index(drop=True)

    if "evaluation_group" not in data.columns:
        raise ValueError(
            "External benchmark is missing evaluation_group."
        )

    data["evaluation_group"] = (
        data["evaluation_group"]
        .astype(str)
    )

    if data.empty:
        raise ValueError(
            "No usable labeled benchmark rows remain after filtering."
        )

    class_counts = fine_truth.value_counts().to_dict()

    logger.info(
        "External labeled benchmark prepared: rows=%d classes=%s groups=%d",
        len(data),
        class_counts,
        data["evaluation_group"].nunique(),
    )

    return data, fine_truth, coarse_truth


def _make_repeated_dev_test_splits(
    data: pd.DataFrame,
    fine_truth: pd.Series,
    seed: int,
    repeats: int,
    n_splits: int = 5,
) -> list[dict[str, np.ndarray]]:
    """
    Repeated grouped-stratified train/development/test partitions.

    Groups prevent samples belonging to the same external source/evaluation
    group from leaking across train/dev/test.
    """
    if repeats < 2:
        raise ValueError(
            "repeats must be >= 2."
        )

    if n_splits < 3:
        raise ValueError(
            "n_splits must be >= 3."
        )

    label_to_int = {
        label: index
        for index, label in enumerate(cfg.BENCHMARK_FINE_CLASSES)
    }

    y = (
        fine_truth
        .map(label_to_int)
        .to_numpy()
    )

    groups = (
        data["evaluation_group"]
        .astype(str)
        .to_numpy()
    )

    splits: list[dict[str, np.ndarray]] = []

    for repeat in range(repeats):
        repeat_seed = seed + repeat * 1009

        outer = StratifiedGroupKFold(
            n_splits=n_splits,
            shuffle=True,
            random_state=repeat_seed,
        )

        outer_folds = list(
            outer.split(
                data,
                y,
                groups,
            )
        )

        test_fold = repeat % len(outer_folds)

        train_dev_idx, test_idx = outer_folds[test_fold]

        remaining = data.iloc[train_dev_idx]

        y_remaining = y[train_dev_idx]

        groups_remaining = (
            remaining["evaluation_group"]
            .astype(str)
            .to_numpy()
        )

        inner = StratifiedGroupKFold(
            n_splits=4,
            shuffle=True,
            random_state=repeat_seed + 17,
        )

        inner_folds = list(
            inner.split(
                remaining,
                y_remaining,
                groups_remaining,
            )
        )

        inner_fold = repeat % len(inner_folds)

        train_rel, development_rel = inner_folds[inner_fold]

        train_idx = train_dev_idx[train_rel]

        development_idx = train_dev_idx[development_rel]

        splits.append(
            {
                "repeat": int(repeat),
                "seed": int(repeat_seed),
                "train": np.asarray(
                    train_idx,
                    dtype=int,
                ),
                "development": np.asarray(
                    development_idx,
                    dtype=int,
                ),
                "test": np.asarray(
                    test_idx,
                    dtype=int,
                ),
            }
        )

    return splits


def _make_final_locked_split(
    data: pd.DataFrame,
    fine_truth: pd.Series,
    seed: int,
) -> dict[str, np.ndarray]:
    """
    Create the single final train/development/locked-test protocol.

    The locked test is never used for backend/model selection.
    """
    label_to_int = {
        label: index
        for index, label in enumerate(cfg.BENCHMARK_FINE_CLASSES)
    }

    y = (
        fine_truth
        .map(label_to_int)
        .to_numpy()
    )

    groups = (
        data["evaluation_group"]
        .astype(str)
        .to_numpy()
    )

    outer = StratifiedGroupKFold(
        n_splits=5,
        shuffle=True,
        random_state=seed,
    )

    folds = list(
        outer.split(
            data,
            y,
            groups,
        )
    )

    train_dev_idx, test_idx = folds[0]

    remaining = data.iloc[train_dev_idx]

    y_remaining = y[train_dev_idx]

    groups_remaining = (
        remaining["evaluation_group"]
        .astype(str)
        .to_numpy()
    )

    inner = StratifiedGroupKFold(
        n_splits=4,
        shuffle=True,
        random_state=seed + 1,
    )

    inner_folds = list(
        inner.split(
            remaining,
            y_remaining,
            groups_remaining,
        )
    )

    train_rel, development_rel = inner_folds[0]

    train_idx = train_dev_idx[train_rel]

    development_idx = train_dev_idx[development_rel]

    return {
        "train": np.asarray(
            train_idx,
            dtype=int,
        ),
        "development": np.asarray(
            development_idx,
            dtype=int,
        ),
        "test": np.asarray(
            test_idx,
            dtype=int,
        ),
    }


def _evaluate_predictions(
    truth: pd.Series,
    predictions: Iterable,
    allowed: Iterable[str],
) -> dict:
    """
    Positional evaluation wrapper.

    The truth and prediction arrays must be aligned by position, never by
    pandas index labels.
    """
    truth_array = np.asarray(
        list(truth),
        dtype=object,
    )

    prediction_array = np.asarray(
        list(predictions),
        dtype=object,
    )

    if len(truth_array) != len(prediction_array):
        raise ValueError(
            "Prediction evaluation length mismatch: "
            f"truth={len(truth_array)} "
            f"predictions={len(prediction_array)}"
        )

    return _evaluate_method_labels_safely(
        pd.Series(
            truth_array,
            index=pd.RangeIndex(len(truth_array)),
        ),
        prediction_array,
        allowed,
    )


# ---------------------------------------------------------------------------
# Snorkel training
# ---------------------------------------------------------------------------

def _train_snorkel_backend(
    operational: pd.DataFrame,
    seed: int,
    snorkel_epochs: int,
) -> tuple[dict, dict, dict, float]:
    """
    Train Snorkel only.

    No EM model is constructed.
    """
    if not SNORKEL_AVAILABLE:
        raise RuntimeError(
            "Snorkel is unavailable. "
            "Install snorkel before running this pipeline."
        )

    started = time.perf_counter()

    outputs: dict = {
        "coarse": {},
        "fine": {},
    }

    students: dict = {
        "coarse": {},
        "fine": {},
    }

    metadata: dict = {
        "backend": "snorkel",
        "snorkel_epochs": int(snorkel_epochs),
        "seed": int(seed),
        "no_manual_lf_weights": True,
        "em_removed": True,
    }

    SNORKEL_MODEL_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    SNORKEL_REPORT_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    for granularity, group_labels in (
        ("coarse", cfg.WEAK_COARSE_GROUPS),
        ("fine", cfg.WEAK_FINE_GROUPS),
    ):
        logger.info(
            "Snorkel backend START | granularity=%s",
            granularity,
        )

        label_matrix, methods, groups = build_label_matrix(
            operational,
            DEFAULT_WEAK_METHODS,
            group_labels,
            granularity,
        )

        active_count = (
            label_matrix != cfg.WEAK_ABSTAIN_LABEL
        ).sum(axis=1)

        model, probabilities, resolved_backend = (
            fit_label_model_backend(
                label_matrix,
                cardinality=len(groups),
                backend="snorkel",
                random_state=seed,
                snorkel_epochs=snorkel_epochs,
            )
        )

        if resolved_backend != "snorkel":
            raise RuntimeError(
                "Snorkel was explicitly requested, but the weak-supervision "
                f"backend resolved to {resolved_backend!r}."
            )

        output = predict_from_label_model(
            model,
            operational,
            methods,
            groups,
            granularity,
        )

        output = _dedupe_columns(output)

        output_path = (
            SNORKEL_REPORT_ROOT
            / f"operational_weak_labels_{granularity}.parquet"
        )

        output.to_parquet(
            output_path,
            index=False,
        )

        model_metadata = {
            "backend": "snorkel",
            "granularity": granularity,
            "groups": list(groups),
            "methods": list(methods),
            "n_rows": int(len(operational)),
            "n_lfs": int(label_matrix.shape[1]),
            "rows_with_at_least_one_lf": int(
                (active_count > 0).sum()
            ),
            "abstain_rate": float(
                (active_count == 0).mean()
            ),
            "mean_active_lf_count": float(
                active_count.mean()
            ),
            "snorkel_epochs": int(snorkel_epochs),
            "random_state": int(seed),
            "no_manual_lf_weights": True,
            "backend_policy": "snorkel_only",
        }

        artifact = {
            "model": model,
            "groups": list(groups),
            "metadata": model_metadata,
            "output": output,
            "probabilities": probabilities,
        }

        outputs[granularity] = artifact

        model_path = (
            SNORKEL_MODEL_ROOT
            / f"weak_label_model_{granularity}.joblib"
        )

        joblib.dump(
            {
                "model": model,
                "groups": list(groups),
                "metadata": model_metadata,
            },
            model_path,
        )

        students[granularity] = _train_weak_students(
            output.copy(),
            granularity,
            seed,
        )

        student_dir = (
            SNORKEL_MODEL_ROOT
            / "students"
            / granularity
        )

        student_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        for student_key, student_artifact in (
            students[granularity].items()
        ):
            safe_name = (
                student_key
                .replace("/", "_")
                .replace("\\", "_")
            )

            student_path = (
                student_dir
                / f"{safe_name}.joblib"
            )

            joblib.dump(
                student_artifact,
                student_path,
            )

        logger.info(
            "Snorkel backend COMPLETE | granularity=%s",
            granularity,
        )

    elapsed = time.perf_counter() - started

    metadata["elapsed_seconds"] = float(elapsed)

    return (
        outputs,
        students,
        metadata,
        elapsed,
    )


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def _evaluate_direct_snorkel(
    external: pd.DataFrame,
    fine_truth: pd.Series,
    coarse_truth: pd.Series,
    outputs: dict,
    final_split: dict[str, np.ndarray],
) -> pd.DataFrame:
    rows = []

    for granularity in ("coarse", "fine"):
        payload = outputs[granularity]

        result = payload["output"]

        pred_column = (
            f"weak_{granularity}_fault"
        )

        predictions = (
            result[pred_column]
            .to_numpy()
        )

        truth = (
            fine_truth
            if granularity == "fine"
            else coarse_truth
        )

        allowed = (
            cfg.BENCHMARK_FINE_CLASSES
            if granularity == "fine"
            else cfg.COARSE_FAULT_GROUPS
        )

        logger.info(
            "Direct Snorkel external inference | granularity=%s | rows=%d",
            granularity,
            len(external),
        )

        all_metric = _evaluate_predictions(
            truth,
            predictions,
            allowed,
        )

        rows.append(
            {
                "stage": "direct_label_model",
                "backend": "snorkel",
                "granularity": granularity,
                "model": "snorkel_label_model",
                "feature_mode": "labeling_functions_only",
                "split": "all_labeled",
                **all_metric,
            }
        )

        for split_name, indices in (
            (
                "development",
                final_split["development"],
            ),
            (
                "locked_test",
                final_split["test"],
            ),
        ):
            metric = _evaluate_predictions(
                truth.iloc[indices],
                predictions[indices],
                allowed,
            )

            rows.append(
                {
                    "stage": "direct_label_model",
                    "backend": "snorkel",
                    "granularity": granularity,
                    "model": "snorkel_label_model",
                    "feature_mode": "labeling_functions_only",
                    "split": split_name,
                    **metric,
                }
            )

    return pd.DataFrame(rows)


def _evaluate_students(
    external: pd.DataFrame,
    fine_truth: pd.Series,
    coarse_truth: pd.Series,
    students: dict,
    final_split: dict[str, np.ndarray],
) -> pd.DataFrame:
    rows = []

    for granularity in ("coarse", "fine"):
        artifacts = students.get(
            granularity,
            {},
        )

        for student_key, artifact in artifacts.items():
            Xdf = _align_feature_frame(
                external,
                artifact["features"],
                granularity,
            )

            raw_predictions = _predict_model(
                artifact["model"],
                Xdf,
            )

            predictions = _decode_class_predictions(
                raw_predictions,
                list(artifact["labels"]),
            )

            predictions = np.asarray(
                predictions,
                dtype=object,
            )

            truth = (
                fine_truth
                if granularity == "fine"
                else coarse_truth
            )

            allowed = (
                cfg.BENCHMARK_FINE_CLASSES
                if granularity == "fine"
                else cfg.COARSE_FAULT_GROUPS
            )

            model_name = (
                student_key.split(
                    "__",
                    1,
                )[1]
                if "__" in student_key
                else student_key
            )

            all_metric = _evaluate_predictions(
                truth,
                predictions,
                allowed,
            )

            rows.append(
                {
                    "stage": "weak_student_transfer",
                    "backend": "snorkel",
                    "granularity": granularity,
                    "model": model_name,
                    "feature_mode": artifact["feature_mode"],
                    "split": "all_labeled",
                    **all_metric,
                }
            )

            for split_name, indices in (
                (
                    "development",
                    final_split["development"],
                ),
                (
                    "locked_test",
                    final_split["test"],
                ),
            ):
                metric = _evaluate_predictions(
                    truth.iloc[indices],
                    predictions[indices],
                    allowed,
                )

                rows.append(
                    {
                        "stage": "weak_student_transfer",
                        "backend": "snorkel",
                        "granularity": granularity,
                        "model": model_name,
                        "feature_mode": artifact["feature_mode"],
                        "split": split_name,
                        **metric,
                    }
                )

    return pd.DataFrame(rows)


def _repeated_development_evaluation(
    external: pd.DataFrame,
    fine_truth: pd.Series,
    coarse_truth: pd.Series,
    outputs: dict,
    students: dict,
    seed: int,
    repeats: int,
) -> pd.DataFrame:
    """
    Repeated development evaluation.

    This is used to select the best Snorkel student configuration.

    Snorkel itself is NOT compared to EM because EM has been removed.
    """
    splits = _make_repeated_dev_test_splits(
        external,
        fine_truth,
        seed=seed,
        repeats=repeats,
    )

    rows = []

    for split_info in splits:
        repeat = int(split_info["repeat"])
        development_idx = split_info["development"]
        repeat_seed = int(split_info["seed"])

        # Direct Snorkel label model.
        for granularity in (
            "coarse",
            "fine",
        ):
            payload = outputs[granularity]

            result = payload["output"]

            predictions = result[
                f"weak_{granularity}_fault"
            ].to_numpy()

            truth = (
                fine_truth
                if granularity == "fine"
                else coarse_truth
            )

            allowed = (
                cfg.BENCHMARK_FINE_CLASSES
                if granularity == "fine"
                else cfg.COARSE_FAULT_GROUPS
            )

            metric = _evaluate_predictions(
                truth.iloc[development_idx],
                predictions[development_idx],
                allowed,
            )

            rows.append(
                {
                    "repeat": repeat,
                    "seed": repeat_seed,
                    "stage": "direct_label_model",
                    "backend": "snorkel",
                    "granularity": granularity,
                    "model": "snorkel_label_model",
                    "feature_mode": "labeling_functions_only",
                    "split": "development",
                    **metric,
                }
            )

        # Student models.
        for granularity in (
            "coarse",
            "fine",
        ):
            for student_key, artifact in (
                students
                .get(granularity, {})
                .items()
            ):
                Xdf = _align_feature_frame(
                    external,
                    artifact["features"],
                    granularity,
                )

                predictions = _decode_class_predictions(
                    _predict_model(
                        artifact["model"],
                        Xdf,
                    ),
                    list(artifact["labels"]),
                )

                predictions = np.asarray(
                    predictions,
                    dtype=object,
                )

                truth = (
                    fine_truth
                    if granularity == "fine"
                    else coarse_truth
                )

                allowed = (
                    cfg.BENCHMARK_FINE_CLASSES
                    if granularity == "fine"
                    else cfg.COARSE_FAULT_GROUPS
                )

                metric = _evaluate_predictions(
                    truth.iloc[development_idx],
                    predictions[development_idx],
                    allowed,
                )

                model_name = (
                    student_key.split(
                        "__",
                        1,
                    )[1]
                    if "__" in student_key
                    else student_key
                )

                rows.append(
                    {
                        "repeat": repeat,
                        "seed": repeat_seed,
                        "stage": "weak_student_transfer",
                        "backend": "snorkel",
                        "granularity": granularity,
                        "model": model_name,
                        "feature_mode": artifact["feature_mode"],
                        "split": "development",
                        **metric,
                    }
                )

    result = pd.DataFrame(rows)

    logger.info(
        "Repeated development evaluation complete: rows=%d repeats=%d",
        len(result),
        repeats,
    )

    return result


# ---------------------------------------------------------------------------
# Operational diagnostics
# ---------------------------------------------------------------------------

def _operational_stability(
    outputs: dict,
    operational: pd.DataFrame,
) -> pd.DataFrame:
    """
    Report stability/uncertainty diagnostics for the complete operational
    dataset.

    These are NOT accuracy measurements.
    """
    rows = []

    for granularity in (
        "coarse",
        "fine",
    ):
        payload = outputs[granularity]

        result = payload["output"]

        posterior_column = (
            f"weak_{granularity}_posterior_max"
        )

        entropy_column = (
            f"weak_{granularity}_entropy"
        )

        active_lf_column = (
            f"weak_{granularity}_lf_active_count"
        )

        abstain_column = (
            f"weak_{granularity}_is_ABSTAIN"
        )

        rows.append(
            {
                "stage": "operational_stability",
                "backend": "snorkel",
                "granularity": granularity,
                "n_rows": int(len(result)),
                "n_transformers": int(
                    operational["transformer_id"].nunique()
                ),
                "coverage": float(
                    (~result[abstain_column]).mean()
                ),
                "abstain_rate": float(
                    result[abstain_column].mean()
                ),
                "posterior_max_mean": float(
                    result[posterior_column].mean()
                ),
                "posterior_max_median": float(
                    result[posterior_column].median()
                ),
                "entropy_mean": float(
                    result[entropy_column].mean()
                ),
                "active_lf_mean": float(
                    result[active_lf_column].mean()
                ),
            }
        )

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Selection
# ---------------------------------------------------------------------------

def _select_best_student(
    repeated_development: pd.DataFrame,
) -> dict:
    """
    Select the production student using repeated development evidence.

    Primary metric:
        Macro-F1

    Tie breakers:
        Balanced accuracy
        Accuracy

    The locked test is never used.
    """
    candidates = repeated_development[
        (repeated_development["stage"] == "weak_student_transfer")
        & (
            repeated_development["granularity"]
            == "fine"
        )
        & (
            repeated_development["feature_mode"]
            == "gas_plus_traditional"
        )
        & (
            repeated_development["split"]
            == "development"
        )
    ].copy()

    if candidates.empty:
        raise RuntimeError(
            "No fine gas_plus_traditional Snorkel student "
            "development results were found."
        )

    summary_rows = []

    for model_name, group in candidates.groupby("model"):
        summary_rows.append(
            {
                "model": model_name,
                "development_macro_f1_mean": float(
                    group["macro_f1"].mean()
                ),
                "development_macro_f1_std": float(
                    group["macro_f1"].std(ddof=1)
                )
                if len(group) > 1
                else 0.0,
                "development_balanced_accuracy_mean": float(
                    group["balanced_accuracy"].mean()
                ),
                "development_accuracy_mean": float(
                    group["accuracy"].mean()
                ),
                "repeats": int(
                    group["repeat"].nunique()
                ),
            }
        )

    summary = pd.DataFrame(
        summary_rows
    ).sort_values(
        [
            "development_macro_f1_mean",
            "development_balanced_accuracy_mean",
            "development_accuracy_mean",
        ],
        ascending=False,
    ).reset_index(drop=True)

    winner = summary.iloc[0]

    return {
        "backend": "snorkel",
        "selected_student_model": str(
            winner["model"]
        ),
        "selected_feature_mode": "gas_plus_traditional",
        "selection_split": "repeated_development",
        "selection_primary_metric": "macro_f1_mean",
        "selection_tie_breakers": [
            "balanced_accuracy_mean",
            "accuracy_mean",
        ],
        "student_summary": summary.to_dict(
            orient="records"
        ),
        "locked_test_not_used_for_selection": True,
    }


def _promote_snorkel_best_student(
    students: dict,
    selection: dict,
) -> None:
    """
    Promote the selected Snorkel weak-label model and the selected student
    model to the existing production artifact names.
    """
    SNORKEL_MODEL_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    MODEL_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # Promote Snorkel label models.
    for granularity, destination in (
        (
            "coarse",
            PRODUCTION_COARSE,
        ),
        (
            "fine",
            PRODUCTION_FINE,
        ),
    ):
        source = (
            SNORKEL_MODEL_ROOT
            / f"weak_label_model_{granularity}.joblib"
        )

        if not source.exists():
            raise FileNotFoundError(
                f"Missing Snorkel label-model artifact: {source}"
            )

        shutil.copy2(
            source,
            destination,
        )

    selection["production_backend"] = "snorkel"

    marker = {
        "backend": "snorkel",
        "policy": "snorkel_only",
        "selected_student_model": selection[
            "selected_student_model"
        ],
        "selected_feature_mode": selection[
            "selected_feature_mode"
        ],
        "selection_split": selection[
            "selection_split"
        ],
        "selection_primary_metric": selection[
            "selection_primary_metric"
        ],
        "locked_test_not_used_for_selection": True,
        "em_removed": True,
        "production_artifacts": [
            str(PRODUCTION_COARSE),
            str(PRODUCTION_FINE),
        ],
    }

    marker_path = (
        MODEL_DIR
        / "weak_supervision_backend_selection.json"
    )

    marker_path.write_text(
        json.dumps(
            marker,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Snorkel-only weak-supervision comparison/training pipeline "
            "for unlabeled DGA operational data."
        )
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=cfg.RANDOM_STATE,
    )

    parser.add_argument(
        "--repeats",
        type=int,
        default=10,
        help=(
            "Number of repeated grouped-stratified development "
            "evaluations. Default: 10."
        ),
    )

    parser.add_argument(
        "--snorkel-epochs",
        type=int,
        default=500,
    )

    parser.add_argument(
        "--promote",
        action="store_true",
        help=(
            "Promote Snorkel weak-label artifacts to production."
        ),
    )

    args = parser.parse_args(argv)

    if not SNORKEL_AVAILABLE:
        raise SystemExit(
            "Snorkel is not installed. "
            "Install snorkel and rerun."
        )

    if args.repeats < 2:
        raise SystemExit(
            "--repeats must be >= 2."
        )

    REPORT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    BENCHMARK_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    MODEL_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    WS_MODEL_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    WS_REPORT_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    logger.info(
        "SNORKEL-ONLY WEAK SUPERVISION RUN START | "
        "seed=%d | repeats=%d | snorkel_available=%s",
        args.seed,
        args.repeats,
        SNORKEL_AVAILABLE,
    )

    # ---------------------------------------------------------------
    # 1. Complete operational dataset
    # ---------------------------------------------------------------

    operational = _ensure_clean_unlabeled()

    # ---------------------------------------------------------------
    # 2. External labeled benchmark
    # ---------------------------------------------------------------

    labeled_raw = load_labeled_csv_data()

    labeled = apply_consensus(
        labeled_raw
    )

    external, fine_truth, coarse_truth = (
        _prepare_external(labeled)
    )

    # ---------------------------------------------------------------
    # 3. Final locked-test protocol
    # ---------------------------------------------------------------

    final_split = _make_final_locked_split(
        external,
        fine_truth,
        args.seed,
    )

    # ---------------------------------------------------------------
    # 4. Train Snorkel only
    # ---------------------------------------------------------------

    outputs, students, backend_meta, elapsed = (
        _train_snorkel_backend(
            operational=operational,
            seed=args.seed,
            snorkel_epochs=args.snorkel_epochs,
        )
    )

    # ---------------------------------------------------------------
    # 5. Direct Snorkel evaluation
    # ---------------------------------------------------------------

    direct = _evaluate_direct_snorkel(
        external=external,
        fine_truth=fine_truth,
        coarse_truth=coarse_truth,
        outputs=outputs,
        final_split=final_split,
    )

    # ---------------------------------------------------------------
    # 6. Student evaluation
    # ---------------------------------------------------------------

    student_results = _evaluate_students(
        external=external,
        fine_truth=fine_truth,
        coarse_truth=coarse_truth,
        students=students,
        final_split=final_split,
    )

    comparison = pd.concat(
        [
            direct,
            student_results,
        ],
        ignore_index=True,
    )

    comparison.to_csv(
        RESULT_CSV,
        index=False,
        encoding="utf-8-sig",
    )

    # ---------------------------------------------------------------
    # 7. Repeated development evaluation
    # ---------------------------------------------------------------

    repeated_development = (
        _repeated_development_evaluation(
            external=external,
            fine_truth=fine_truth,
            coarse_truth=coarse_truth,
            outputs=outputs,
            students=students,
            seed=args.seed,
            repeats=args.repeats,
        )
    )

    repeated_development.to_csv(
        ROBUSTNESS_CSV,
        index=False,
        encoding="utf-8-sig",
    )

    # ---------------------------------------------------------------
    # 8. Operational stability
    # ---------------------------------------------------------------

    operational_stability = _operational_stability(
        outputs,
        operational,
    )

    operational_stability.to_csv(
        OPERATIONAL_CSV,
        index=False,
        encoding="utf-8-sig",
    )

    # ---------------------------------------------------------------
    # 9. Timing
    # ---------------------------------------------------------------

    timing = pd.DataFrame(
        [
            {
                "backend": "snorkel",
                "elapsed_seconds": float(elapsed),
                "operational_rows": int(
                    len(operational)
                ),
                "operational_transformers": int(
                    operational["transformer_id"]
                    .nunique()
                ),
                "labeled_rows": int(
                    len(external)
                ),
                "snorkel_epochs": int(
                    args.snorkel_epochs
                ),
            }
        ]
    )

    timing.to_csv(
        TIMING_CSV,
        index=False,
        encoding="utf-8-sig",
    )

    # ---------------------------------------------------------------
    # 10. Select best student
    # ---------------------------------------------------------------

    selection = _select_best_student(
        repeated_development
    )

    selection.update(
        {
            "operational_rows": int(
                len(operational)
            ),
            "operational_transformers": int(
                operational["transformer_id"]
                .nunique()
            ),
            "labeled_rows_evaluated": int(
                len(external)
            ),
            "labeled_source_datasets": sorted(
                external.get(
                    "source_dataset",
                    pd.Series(dtype=str),
                )
                .astype(str)
                .unique()
                .tolist()
            ),
            "repeats": int(args.repeats),
            "snorkel_epochs": int(
                args.snorkel_epochs
            ),
            "snorkel_elapsed_seconds": float(
                elapsed
            ),
            "em_removed": True,
            "backend_policy": "snorkel_only",
            "operational_stability": (
                operational_stability
                .to_dict(orient="records")
            ),
        }
    )

    # ---------------------------------------------------------------
    # 11. Locked-test results
    # ---------------------------------------------------------------

    locked = comparison[
        (comparison["split"] == "locked_test")
        & (
            comparison["granularity"]
            == "fine"
        )
        & (
            comparison["feature_mode"]
            == "gas_plus_traditional"
        )
    ].copy()

    selection["locked_test_results"] = (
        locked.to_dict(orient="records")
    )

    selection["final_train_n"] = int(
        len(final_split["train"])
    )

    selection["final_development_n"] = int(
        len(final_split["development"])
    )

    selection["final_locked_test_n"] = int(
        len(final_split["test"])
    )

    # ---------------------------------------------------------------
    # 12. Promotion
    # ---------------------------------------------------------------

    if args.promote:
        _promote_snorkel_best_student(
            students=students,
            selection=selection,
        )

        selection["promoted_to_production"] = True
    else:
        selection["promoted_to_production"] = False

    # ---------------------------------------------------------------
    # 13. Save selection
    # ---------------------------------------------------------------

    SELECTION_JSON.write_text(
        json.dumps(
            selection,
            ensure_ascii=False,
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )

    logger.info(
        "SNORKEL-ONLY WEAK SUPERVISION RUN COMPLETE | "
        "selected_student=%s | promoted=%s | "
        "operational_rows=%d | labeled_rows=%d",
        selection[
            "selected_student_model"
        ],
        selection[
            "promoted_to_production"
        ],
        len(operational),
        len(external),
    )

    print(
        json.dumps(
            selection,
            ensure_ascii=False,
            indent=2,
            default=str,
        )
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())