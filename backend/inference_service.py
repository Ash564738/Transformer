# inference_service.py
from __future__ import annotations

import json
import logging
import shutil
import tempfile
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from clean_dataset import clean_dataset
from config import DATASET_DIR, MODEL_DIR, REPORT_DIR, config as cfg
from consensus import (
    apply_consensus,
    normalize_fault,
    unify_fault,
)
from feature_engineering import build_training_features_from_clean
from logging_config import init_logging
from ranking import (
    build_transformer_ranking,
    classify_fault_criticality,
    fault_criticality_source,
    log_ranking_diagnostics,
)
from severity import apply_severity
from weak_supervision import load_weak_supervision_artifact


init_logging()
logger = logging.getLogger(__name__)

MODEL_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

PROCESSED_DIR = DATASET_DIR / "processed"

# ---------------------------------------------------------------------------
# Production model artifacts
# ---------------------------------------------------------------------------

STUDENT_MODEL_PATH = MODEL_DIR / "fault_classifier.joblib"
ANOMALY_MODEL_PATH = MODEL_DIR / "anomaly_ensemble.joblib"

WEAK_COARSE_MODEL_PATH = MODEL_DIR / "weak_label_model_coarse.joblib"
WEAK_FINE_MODEL_PATH = MODEL_DIR / "weak_label_model_fine.joblib"

# Optional research artifacts. These are NOT used by /predict.
FAULT_MODEL_COARSE_PATH = MODEL_DIR / "fault_model_coarse.joblib"
FAULT_MODEL_FINE_PATH = MODEL_DIR / "fault_model_fine.joblib"

# ---------------------------------------------------------------------------
# Operational outputs
# ---------------------------------------------------------------------------

UNLABELED_PATH = PROCESSED_DIR / "dga_unlabeled.parquet"
RANKING_PATH = PROCESSED_DIR / "transformer_ranking.parquet"
PROCESSED_OUTPUT_PATH = PROCESSED_DIR / "dga_unlabeled_processed.parquet"

STUDENT_COMPARISON_PATH = (
    REPORT_DIR / "student_vs_traditional_by_transformer.csv"
)

INFERENCE_METADATA_PATH = (
    PROCESSED_DIR / "dga_inference_metadata.json"
)


# ---------------------------------------------------------------------------
# Generic helpers
# ---------------------------------------------------------------------------

def _safe_int(value, default=0):
    try:
        if pd.isna(value):
            return int(default)
    except Exception:
        pass

    try:
        return int(value)
    except (TypeError, ValueError):
        return int(default)


def _safe_float(value, default=np.nan):
    try:
        x = float(value)
    except (TypeError, ValueError):
        return default

    return x if np.isfinite(x) else default


def _ui_status(status: int) -> str:
    return {
        0: "Insufficient data",
        1: "Normal",
        2: "Watch",
        3: "High",
    }.get(_safe_int(status), "Insufficient data")


def _ui_severity_label(value):
    if value is None:
        return "INSUFFICIENT_DATA"

    text = str(value).strip().upper()

    if text in cfg.SEVERITY_ORDER:
        return text

    return cfg.ORDINAL_TO_SEVERITY.get(
        _safe_int(value),
        "INSUFFICIENT_DATA",
    )


def _first(row, names, default=None):
    for name in names:
        if name not in row.index:
            continue

        value = row.get(name)

        try:
            if pd.notna(value):
                return value
        except Exception:
            continue

    return default


def _normalize_series(series):
    return (
        series
        .map(normalize_fault)
        .fillna("ABSTAIN")
        .astype(str)
    )


def _save_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(
        path,
        index=False,
        encoding="utf-8-sig",
    )

    logger.info(
        "Saved CSV | %s | rows=%d | columns=%d",
        path,
        len(df),
        len(df.columns),
    )


def _save_parquet(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)

    logger.info(
        "Saved Parquet | %s | rows=%d | columns=%d",
        path,
        len(df),
        len(df.columns),
    )


# ---------------------------------------------------------------------------
# Model artifact validation
# ---------------------------------------------------------------------------

def _require_artifact(path: Path, description: str) -> Path:
    if not path.exists():
        raise FileNotFoundError(
            f"Missing production model artifact: {description}. "
            f"Expected file: {path}"
        )

    return path


def _load_joblib_artifact(path: Path, description: str):
    _require_artifact(path, description)

    try:
        artifact = joblib.load(path)
    except Exception as exc:
        raise RuntimeError(
            f"Unable to load production model artifact "
            f"{description}: {exc}"
        ) from exc

    return artifact


# ---------------------------------------------------------------------------
# Weak-label model inference
# ---------------------------------------------------------------------------

def _build_weak_label_matrix(
    df: pd.DataFrame,
    granularity: str,
    groups: list[str],
):
    """
    Rebuild the same diagnostic label matrix used during weak-label
    model training.

    IMPORTANT:
    This does NOT train anything.

    It only converts the existing traditional diagnostic outputs into the
    integer label matrix expected by the saved weak-label model.
    """
    from weak_supervision import build_label_matrix

    L, methods, resolved_groups = build_label_matrix(
        df,
        label_columns=None,
        groups=groups,
        granularity=granularity,
    )

    logger.debug(
        "Built production weak-label matrix | granularity=%s | rows=%d | "
        "methods=%d | groups=%s",
        granularity,
        len(df),
        len(methods),
        resolved_groups,
    )

    return L, methods, resolved_groups


def _predict_weak_label_model(
    df: pd.DataFrame,
    artifact: dict,
    granularity: str,
):
    """
    Apply a PRE-TRAINED weak-label model.

    No fitting/training occurs here.
    """
    if not isinstance(artifact, dict):
        raise ValueError(
            f"Invalid weak-label artifact for {granularity}."
        )

    model = artifact.get("model")
    groups = artifact.get("groups")

    if model is None:
        raise ValueError(
            f"Weak-label artifact for {granularity} does not contain a model."
        )

    if not groups:
        raise ValueError(
            f"Weak-label artifact for {granularity} does not contain groups."
        )

    groups = [str(value).strip().upper() for value in groups]

    L, _, resolved_groups = _build_weak_label_matrix(
        df,
        granularity,
        groups,
    )

    try:
        probabilities = np.asarray(
            model.predict_proba(L),
            dtype=float,
        )
    except Exception as exc:
        raise RuntimeError(
            f"Weak-label {granularity} model prediction failed: {exc}"
        ) from exc

    if probabilities.ndim != 2:
        raise ValueError(
            f"Weak-label {granularity} probability output must be 2D."
        )

    if probabilities.shape[1] != len(resolved_groups):
        raise ValueError(
            f"Weak-label {granularity} model classes do not match "
            f"artifact groups. probabilities={probabilities.shape[1]}, "
            f"groups={len(resolved_groups)}"
        )

    prediction_indices = np.argmax(
        probabilities,
        axis=1,
    )

    posterior_max = np.max(
        probabilities,
        axis=1,
    )

    safe_probabilities = np.clip(
        probabilities,
        1e-12,
        1.0,
    )

    entropy = -np.sum(
        safe_probabilities * np.log(safe_probabilities),
        axis=1,
    )

    active_count = (
        L != -1
    ).sum(axis=1)

    active_coverage = (
        active_count / float(L.shape[1])
        if L.shape[1] > 0
        else np.zeros(len(df), dtype=float)
    )

    predicted_labels = [
        resolved_groups[int(index)]
        for index in prediction_indices
    ]

    # A weak-label model can technically output a class even when all
    # labeling functions abstain. Preserve ABSTAIN explicitly in that case.
    no_evidence = active_count == 0

    predicted_labels = [
        "ABSTAIN" if no_evidence[i] else predicted_labels[i]
        for i in range(len(predicted_labels))
    ]

    if granularity == "coarse":
        output_prefix = "weak_coarse"
    else:
        output_prefix = "weak_fine"

    result = pd.DataFrame(index=df.index)

    result[f"{output_prefix}_fault"] = predicted_labels

    if granularity == "fine":
        result[f"{output_prefix}_fault_group"] = [
            unify_fault(value)
            for value in predicted_labels
        ]
    else:
        result[f"{output_prefix}_fault_group"] = [
            unify_fault(value)
            for value in predicted_labels
        ]

    result[f"{output_prefix}_posterior_max"] = posterior_max
    result[f"{output_prefix}_entropy"] = entropy
    result[f"{output_prefix}_lf_active_count"] = active_count
    result[f"{output_prefix}_lf_coverage"] = active_coverage
    result[f"{output_prefix}_is_ABSTAIN"] = no_evidence

    logger.info(
        "Applied production weak-label model | granularity=%s | "
        "rows=%d | abstain=%d",
        granularity,
        len(df),
        int(no_evidence.sum()),
    )

    return result


def _apply_pretrained_weak_models(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """
    Apply the saved coarse/fine weak-label artifacts.

    This replaces weak_supervision_pipeline() in production.
    """
    coarse_artifact = _load_joblib_artifact(
        WEAK_COARSE_MODEL_PATH,
        "coarse weak-label model",
    )

    fine_artifact = _load_joblib_artifact(
        WEAK_FINE_MODEL_PATH,
        "fine weak-label model",
    )

    coarse = _predict_weak_label_model(
        df,
        coarse_artifact,
        "coarse",
    )

    fine = _predict_weak_label_model(
        df,
        fine_artifact,
        "fine",
    )

    output = df.copy()

    for column in coarse.columns:
        output[column] = coarse[column]

    for column in fine.columns:
        output[column] = fine[column]

    metadata = {
        "backend": {
            "coarse": (
                coarse_artifact
                .get("metadata", {})
                .get("backend")
            ),
            "fine": (
                fine_artifact
                .get("metadata", {})
                .get("backend")
            ),
        },
        "granularity": "coarse+fine",
        "coarse": coarse_artifact.get("metadata", {}),
        "fine": fine_artifact.get("metadata", {}),
        "uses_manual_lf_weights": False,
        "runtime_training": False,
    }

    return output, metadata


# ---------------------------------------------------------------------------
# Production student model
# ---------------------------------------------------------------------------

def _validate_student_artifact(artifact):
    if not isinstance(artifact, dict):
        return False

    if artifact.get("model") is None:
        return False

    if not artifact.get("labels"):
        return False

    granularity = str(
        artifact.get("granularity", "")
    ).strip().lower()

    if granularity != "fine":
        return False

    features = artifact.get(
        "feature_cols",
        artifact.get(
            "features",
            cfg.COMMON_BENCHMARK_GASES,
        ),
    )

    return bool(features)


def _load_student_model():
    artifact = _load_joblib_artifact(
        STUDENT_MODEL_PATH,
        "production student classifier",
    )

    if not _validate_student_artifact(artifact):
        raise ValueError(
            f"Production student artifact is incompatible: "
            f"{STUDENT_MODEL_PATH}"
        )

    return artifact


def _student_feature_columns():
    return list(cfg.COMMON_BENCHMARK_GASES)


def _prepare_student_matrix(
    df: pd.DataFrame,
    feature_columns,
):
    feature_columns = list(feature_columns)

    missing = [
        column
        for column in feature_columns
        if column not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Missing student-model features: {missing}"
        )

    return (
        df[feature_columns]
        .apply(pd.to_numeric, errors="coerce")
        .replace([np.inf, -np.inf], np.nan)
    )


def _apply_student(
    df: pd.DataFrame,
    artifact: dict,
):
    out = df.copy()

    feature_columns = list(
        artifact.get(
            "feature_cols",
            artifact.get(
                "features",
                cfg.COMMON_BENCHMARK_GASES,
            ),
        )
    )

    X = _prepare_student_matrix(
        out,
        feature_columns,
    )

    model = artifact["model"]

    try:
        proba = np.asarray(
            model.predict_proba(X),
            dtype=float,
        )

        estimator = (
            model.named_steps.get("classifier", model)
            if hasattr(model, "named_steps")
            else model
        )

        model_classes = getattr(
            estimator,
            "classes_",
            None,
        )

        labels = [
            normalize_fault(value)
            for value in (
                model_classes
                if model_classes is not None
                else artifact["labels"]
            )
        ]

        if proba.ndim != 2:
            raise ValueError(
                "Student probability matrix must be 2D."
            )

        if proba.shape[1] != len(labels):
            raise ValueError(
                "Student probability matrix does not match "
                "model classes."
            )

        best = np.argmax(
            proba,
            axis=1,
        )

        confidence = np.max(
            proba,
            axis=1,
        )

        pred = [
            labels[int(index)]
            for index in best
        ]

    except Exception as exc:
        logger.exception(
            "Student predict_proba failed."
        )

        try:
            pred = [
                normalize_fault(value)
                for value in model.predict(X)
            ]
            confidence = np.full(
                len(out),
                np.nan,
                dtype=float,
            )
        except Exception as predict_exc:
            raise RuntimeError(
                "Production student model prediction failed: "
                f"{predict_exc}"
            ) from exc

    out["student_fault_label"] = pred

    out["student_fault_group"] = [
        unify_fault(value)
        for value in pred
    ]

    out["student_fault_confidence"] = confidence

    out["student_model_name"] = artifact.get(
        "model_name",
        "UNKNOWN",
    )

    out["student_training_type"] = artifact.get(
        "training_type",
        "PRETRAINED",
    )

    out["student_feature_set"] = ",".join(
        feature_columns
    )

    return out


# ---------------------------------------------------------------------------
# Traditional + weak + student fusion
# ---------------------------------------------------------------------------

def _combine_consensus_and_student(df):
    out = df.copy()

    weak_fine = (
        _normalize_series(out["weak_fine_fault"])
        if "weak_fine_fault" in out.columns
        else pd.Series(
            "ABSTAIN",
            index=out.index,
        )
    )

    traditional = (
        _normalize_series(out["consensus_fault"])
        if "consensus_fault" in out.columns
        else pd.Series(
            "ABSTAIN",
            index=out.index,
        )
    )

    student = (
        _normalize_series(out["student_fault_label"])
        if "student_fault_label" in out.columns
        else pd.Series(
            "ABSTAIN",
            index=out.index,
        )
    )

    weak_active = weak_fine != "ABSTAIN"
    traditional_active = traditional != "ABSTAIN"
    student_active = student != "ABSTAIN"

    final_fine = weak_fine.copy()

    source = pd.Series(
        "ABSTAIN",
        index=out.index,
        dtype=object,
    )

    use_traditional = (
        ~weak_active
        & traditional_active
    )

    use_student = (
        ~weak_active
        & ~traditional_active
        & student_active
    )

    final_fine.loc[use_traditional] = (
        traditional.loc[use_traditional]
    )

    final_fine.loc[use_student] = (
        student.loc[use_student]
    )

    source.loc[weak_active] = (
        "weak_fine_label_model"
    )

    source.loc[use_traditional] = (
        "traditional_consensus_after_weak_abstain"
    )

    source.loc[use_student] = (
        "student_last_resort"
    )

    weak_group = weak_fine.map(unify_fault)
    traditional_group = traditional.map(unify_fault)
    student_group = student.map(unify_fault)

    physical_conflict = (
        weak_active
        & traditional_active
        & (weak_group != traditional_group)
    )

    same_group_different_fine = (
        weak_active
        & traditional_active
        & ~physical_conflict
        & (weak_fine != traditional)
    )

    out["final_fault"] = final_fine

    out["final_fault_group"] = (
        final_fine
        .map(unify_fault)
        .fillna("ABSTAIN")
        .astype(str)
    )

    out["final_fault_source"] = source

    out["final_fault_conflict"] = physical_conflict

    out["final_fault_same_coarse_different_fine"] = (
        same_group_different_fine
    )

    out["final_fault_conflict_level"] = np.select(
        [
            physical_conflict,
            same_group_different_fine,
            weak_active,
            traditional_active,
            student_active,
        ],
        [
            "PHYSICAL_GROUP_CONFLICT",
            "SAME_GROUP_FINE_DISAGREEMENT",
            "WEAK_FINE_MODEL",
            "TRADITIONAL_CONSENSUS",
            "STUDENT_FALLBACK",
        ],
        default="ABSTAIN",
    )

    out["student_used_as_fallback"] = use_student

    out["final_fault_is_weak_supervision"] = (
        weak_active
    )

    out["student_vs_traditional_coarse_agreement"] = (
        traditional_active
        & student_active
        & traditional_group.eq(student_group)
    )

    out["fault_criticality_class"] = (
        final_fine.map(
            classify_fault_criticality
        )
    )

    out["fault_criticality_source"] = (
        fault_criticality_source()
    )

    return out


# ---------------------------------------------------------------------------
# Anomaly model
# ---------------------------------------------------------------------------

def _load_anomaly_model():
    from anomaly import UnsupervisedEnsemble

    artifact = _load_joblib_artifact(
        ANOMALY_MODEL_PATH,
        "production anomaly ensemble",
    )

    if not isinstance(
        artifact,
        UnsupervisedEnsemble,
    ):
        raise ValueError(
            f"Anomaly artifact is not a valid "
            f"UnsupervisedEnsemble: {ANOMALY_MODEL_PATH}"
        )

    return artifact


def _predict_anomaly(
    df: pd.DataFrame,
):
    model = _load_anomaly_model()

    gases = list(
        cfg.ALL_DGA_GASES
    )

    X = (
        df.reindex(
            columns=gases,
            fill_value=np.nan,
        )
        .apply(
            pd.to_numeric,
            errors="coerce",
        )
        .replace(
            [np.inf, -np.inf],
            np.nan,
        )
        .fillna(0.0)
        .to_numpy(
            dtype=np.float64
        )
    )

    try:
        scores = model.predict(X)
    except Exception as exc:
        raise RuntimeError(
            f"Production anomaly model prediction failed: {exc}"
        ) from exc

    return np.asarray(
        scores,
        dtype=float,
    )


# ---------------------------------------------------------------------------
# Student vs traditional comparison
# ---------------------------------------------------------------------------

def _build_student_comparison(df):
    required = {
        "transformer_id",
        "student_fault_label",
        "consensus_fault",
    }

    if not required.issubset(
        df.columns
    ):
        return pd.DataFrame()

    work = df.copy()

    work["_consensus_fine"] = (
        _normalize_series(
            work["consensus_fault"]
        )
    )

    work["_student_fine"] = (
        _normalize_series(
            work["student_fault_label"]
        )
    )

    work["_consensus_group"] = (
        work["_consensus_fine"]
        .map(unify_fault)
        .fillna("ABSTAIN")
        .astype(str)
    )

    work["_student_group"] = (
        work["_student_fine"]
        .map(unify_fault)
        .fillna("ABSTAIN")
        .astype(str)
    )

    rows = []

    for transformer_id, group in work.groupby(
        "transformer_id",
        sort=False,
    ):
        joint = (
            (group["_consensus_group"] != "ABSTAIN")
            & (group["_student_group"] != "ABSTAIN")
        )

        n_joint = int(
            joint.sum()
        )

        coarse_agreement = (
            float(
                (
                    group.loc[
                        joint,
                        "_consensus_group",
                    ]
                    == group.loc[
                        joint,
                        "_student_group",
                    ]
                ).mean()
            )
            if n_joint
            else np.nan
        )

        fine_agreement = (
            float(
                (
                    group.loc[
                        joint,
                        "_consensus_fine",
                    ]
                    == group.loc[
                        joint,
                        "_student_fine",
                    ]
                ).mean()
            )
            if n_joint
            else np.nan
        )

        rows.append(
            {
                "transformer_id": transformer_id,
                "n_samples": int(len(group)),
                "n_joint_active": n_joint,
                "coarse_agreement_rate": coarse_agreement,
                "fine_agreement_rate": fine_agreement,
                "traditional_abstain_count": int(
                    (
                        group["_consensus_fine"]
                        == "ABSTAIN"
                    ).sum()
                ),
                "student_abstain_count": int(
                    (
                        group["_student_fine"]
                        == "ABSTAIN"
                    ).sum()
                ),
                "student_used_as_fallback_count": int(
                    (
                        (
                            group["_consensus_fine"]
                            == "ABSTAIN"
                        )
                        & (
                            group["_student_fine"]
                            != "ABSTAIN"
                        )
                    ).sum()
                ),
                "physical_conflict_count": int(
                    (
                        joint
                        & (
                            group["_consensus_group"]
                            != group["_student_group"]
                        )
                    ).sum()
                ),
            }
        )

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Metadata
# ---------------------------------------------------------------------------

def _write_inference_metadata(
    df,
    artifact,
    weak_metadata,
    elapsed_seconds,
):
    metadata = {
        "pipeline_type": (
            "operational_unlabeled_inference"
        ),
        "runtime_training": False,
        "runtime_research_benchmark": False,
        "runtime_excel_generation": False,
        "n_rows": int(len(df)),
        "n_transformers": int(
            df["transformer_id"].nunique()
        ),
        "diagnostic_methods": list(
            cfg.DIAGNOSTIC_METHODS
        ),
        "diagnostic_consensus": (
            "unweighted_majority_with_abstain"
        ),
        "diagnostic_method_weights": None,
        "severity_standard": cfg.STANDARD,
        "severity_is_weighted": False,
        "severity_uses_manual_weights": False,
        "severity_uses_nei": False,
        "severity_uses_anomaly": False,
        "severity_type": "IEEE_ORDINAL_STATUS",
        "ranking_policy": list(
            cfg.RANKING_POLICY
        ),
        "ranking_is_weighted": False,
        "ranking_is_health_score": False,
        "ranking_uses_manual_weights": False,
        "ranking_uses_fault_criticality_as_severity": False,
        "ranking_score_type": (
            "UNWEIGHTED_LEXICOGRAPHIC_EVIDENCE_ORDER"
        ),
        "maintenance_priority_extension": (
            "NONE; IEEE_STATUS_1_2_3_ONLY"
        ),
        "critical_rule": "NOT_USED",
        "critical_reference": cfg.CRITICAL_REFERENCE,
        "fault_criticality_source": (
            fault_criticality_source()
        ),
        "student_model_name": artifact.get(
            "model_name",
            "UNKNOWN",
        ),
        "student_training_type": artifact.get(
            "training_type",
            "PRETRAINED",
        ),
        "student_features": artifact.get(
            "feature_cols",
            cfg.COMMON_BENCHMARK_GASES,
        ),
        "weak_supervision_backend": (
            weak_metadata.get("backend")
            if isinstance(
                weak_metadata,
                dict,
            )
            else None
        ),
        "weak_supervision_granularity": (
            weak_metadata.get("granularity")
            if isinstance(
                weak_metadata,
                dict,
            )
            else None
        ),
        "processing_seconds": float(
            elapsed_seconds
        ),
    }

    INFERENCE_METADATA_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    INFERENCE_METADATA_PATH.write_text(
        json.dumps(
            metadata,
            ensure_ascii=False,
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )


# ---------------------------------------------------------------------------
# Payload creation
# ---------------------------------------------------------------------------

def create_payload(
    df,
    ranking_df,
    comparison_df=None,
):
    rows = []

    ordered = df.sort_values(
        [
            "transformer_id",
            "sample_day",
        ],
        ascending=[
            True,
            False,
        ],
        kind="mergesort",
    )

    export_fields = [
        "transformer_id",
        "sample_day",
        "loc",
        "name",
        "ser",
        "codetx",
        "mfg",
        "h2",
        "ch4",
        "c2h6",
        "c2h4",
        "c2h2",
        "co",
        "co2",
        "ieee_dga_status",
        "ieee_dga_status_label",
        "ieee_dga_status_reason",
        "severity_label_text",
        "severity_source",
        "severity_score_type",
        "severity_is_failure_probability",
        "severity_composite_weighted",
        "severity_uses_manual_weights",
        "severity_anomaly_used",
        "severity_nei_used",
        "severity_is_not_a_health_score",
        "ieee_max_standardized_exceedance",
        "ieee_max_status3_standardized_exceedance",
        "ieee_table1_concentration_ratio_all",
        "ieee_table2_concentration_ratio_all",
        "ieee_table3_delta_ratio_all",
        "ieee_table4_rate_ratio_all",
        "ieee_continuous_evidence_ratio",
        "ieee_continuous_evidence_basis",
        "ieee_standard_trigger_count",
        "ieee_table1_max_exceedance_ratio",
        "ieee_table2_max_exceedance_ratio",
        "ieee_table3_max_exceedance_ratio",
        "ieee_table4_max_exceedance_ratio",
        "ieee_maintenance_priority_hint",
        "ieee_maintenance_priority_hint_reason",
        "consensus_fault",
        "consensus_fault_traditional",
        "consensus_fault_group",
        "diagnostic_agreement_ratio",
        "diagnostic_coverage",
        "diagnostic_confidence",
        "weak_fine_fault",
        "weak_fine_fault_group",
        "weak_fine_posterior_max",
        "weak_fine_entropy",
        "weak_fine_lf_active_count",
        "weak_fine_lf_coverage",
        "weak_fine_is_ABSTAIN",
        "weak_coarse_fault",
        "weak_coarse_fault_group",
        "weak_coarse_posterior_max",
        "weak_coarse_entropy",
        "weak_coarse_lf_active_count",
        "weak_coarse_lf_coverage",
        "weak_coarse_is_ABSTAIN",
        "final_fault",
        "final_fault_group",
        "final_fault_source",
        "final_fault_conflict",
        "final_fault_same_coarse_different_fine",
        "final_fault_conflict_level",
        "fault_criticality_class",
        "fault_criticality_source",
        "keygas_fault",
        "iec_fault",
        "rogers_fault",
        "doernenburg_fault",
        "duval_triangle_fault",
        "duval_pentagon_p1_fault",
        "duval_pentagon_p2_fault",
        "fault_p1",
        "fault_p2",
        "student_fault_label",
        "student_fault_group",
        "student_fault_confidence",
        "student_model_name",
        "student_training_type",
        "student_feature_set",
        "student_used_as_fallback",
        "anomaly_percentile",
        "anomaly_is_severity_input",
        "anomaly_interpretation",
        "ieee_confirmation_required",
        "ieee_delta_available",
        "ieee_rate_available",
        "ieee_rate_span_months",
        "ieee_table1_exceeding_gases",
        "ieee_table2_exceeding_gases",
        "ieee_table3_exceeding_gases",
        "ieee_table4_exceeding_gases",
        "ieee_delta",
        "ieee_gas_rate_ppm_per_year",
        "iec_60599_ratios",
        "ieee_o2_n2_ratio",
        "ieee_age_bucket",
    ]

    ranking_lookup = {
        str(r.get("transformer_id")): r
        for _, r in ranking_df.iterrows()
    }

    predictions = []

    for idx, row in ordered.iterrows():
        status = _safe_int(
            row.get(
                "ieee_dga_status",
                0,
            )
        )

        severity_label = _ui_severity_label(
            row.get(
                "severity_label_text",
                "INSUFFICIENT_DATA",
            )
        )

        rank_info = ranking_lookup.get(
            str(
                row.get(
                    "transformer_id"
                )
            ),
            {},
        )

        current_priority = rank_info.get(
            "maintenance_priority",
            "DATA_REVIEW",
        )

        current_critical = bool(
            rank_info.get(
                "critical_front_flag",
                False,
            )
        )

        current_priority_reason = rank_info.get(
            "maintenance_priority_reason",
            "",
        )

        critical_ratio = _safe_float(
            row.get(
                "ieee_max_status3_standardized_exceedance",
                np.nan,
            )
        )

        fault_type = row.get(
            "final_fault",
            row.get(
                "consensus_fault",
                "ABSTAIN",
            ),
        )

        fault_group = row.get(
            "final_fault_group",
            row.get(
                "consensus_fault_group",
                "ABSTAIN",
            ),
        )

        predictions.append(
            {
                "row_index": int(idx),
                "transformer_id": row.get(
                    "transformer_id"
                ),
                "pred_ensemble": status,
                "ieee_status": status,
                "ieee_status_label": severity_label,
                "status": _ui_status(status),
                "severity": _ui_status(status),
                "severity_label": severity_label,
                "maintenance_priority": current_priority,
                "critical_front": current_critical,
                "maintenance_priority_reason": current_priority_reason,
                "critical_evidence_ratio": critical_ratio,
                "continuous_evidence_ratio": _safe_float(
                    row.get(
                        "ieee_continuous_evidence_ratio",
                        np.nan,
                    )
                ),
                "continuous_evidence_basis": row.get(
                    "ieee_continuous_evidence_basis",
                    "NO_CONTINUOUS_EVIDENCE",
                ),
                "fault_type": fault_type,
                "fault_group": fault_group,
                "fault_criticality_class": classify_fault_criticality(
                    fault_type
                ),
                "fault_source": row.get(
                    "final_fault_source",
                    "ABSTAIN",
                ),
                "fault_confidence": _safe_float(
                    row.get(
                        "weak_fine_posterior_max",
                        np.nan,
                    )
                ),
                "fault_entropy": _safe_float(
                    row.get(
                        "weak_fine_entropy",
                        np.nan,
                    )
                ),
                "fault_evidence_level": (
                    "FINE_WEAK_LABEL_MODEL"
                    if row.get(
                        "weak_fine_fault",
                        "ABSTAIN",
                    )
                    != "ABSTAIN"
                    else "FALLBACK"
                ),
                "reason": row.get(
                    "ieee_dga_status_reason",
                    "",
                ),
                "confirmation_required": bool(
                    row.get(
                        "ieee_confirmation_required",
                        False,
                    )
                ),
                "anomaly_percentile": _safe_float(
                    row.get(
                        "anomaly_percentile",
                        np.nan,
                    )
                ),
                "top_features": [],
            }
        )

        rows.append(
            {
                key: row[key]
                for key in export_fields
                if key in row.index
            }
        )

    transformer_summary = []

    for _, rank_row in ranking_df.iterrows():
        status = _safe_int(
            _first(
                rank_row,
                [
                    "transformer_overall_severity_level"
                ],
                0,
            )
        )

        fault_type = _first(
            rank_row,
            [
                "current_fault",
                "history_dominant_fault",
            ],
            "ABSTAIN",
        )

        fault_group = _first(
            rank_row,
            [
                "current_fault_group"
            ],
            "ABSTAIN",
        )

        priority = rank_row.get(
            "maintenance_priority",
            "DATA_REVIEW",
        )

        transformer_summary.append(
            {
                "rank": _safe_int(
                    rank_row.get(
                        "rank",
                        0,
                    )
                ),
                "maintenance_rank": _safe_int(
                    rank_row.get(
                        "maintenance_priority_rank",
                        rank_row.get(
                            "rank",
                            0,
                        ),
                    )
                ),
                "rank_tie": bool(
                    rank_row.get(
                        "rank_tie",
                        False,
                    )
                ),
                "rank_group_size": _safe_int(
                    rank_row.get(
                        "rank_group_size",
                        1,
                    ),
                    1,
                ),
                "transformer_id": rank_row.get(
                    "transformer_id"
                ),
                "latest_sample_day": str(
                    rank_row.get(
                        "sample_day",
                        "",
                    )
                ),
                "maintenance_priority": priority,
                "maintenance_priority_ordinal": _safe_int(
                    rank_row.get(
                        "maintenance_priority_ordinal",
                        0,
                    )
                ),
                "maintenance_priority_reason": rank_row.get(
                    "maintenance_priority_reason",
                    "",
                ),
                "critical_front": bool(
                    rank_row.get(
                        "critical_front_flag",
                        False,
                    )
                ),
                "critical_rule": rank_row.get(
                    "critical_rule",
                    cfg.CRITICAL_RULE,
                ),
                "critical_reference": rank_row.get(
                    "critical_reference",
                    cfg.CRITICAL_REFERENCE,
                ),
                "critical_evidence_table": rank_row.get(
                    "critical_evidence_table"
                ),
                "critical_evidence_gas": rank_row.get(
                    "critical_evidence_gas"
                ),
                "critical_evidence_ratio": _safe_float(
                    rank_row.get(
                        "critical_evidence_ratio",
                        np.nan,
                    )
                ),
                "critical_evidence_scope": (
                    "LATEST_SAMPLE"
                ),
                "historical_max_standardized_exceedance": _safe_float(
                    rank_row.get(
                        "historical_max_standardized_exceedance",
                        np.nan,
                    )
                ),
                "ieee_status": status,
                "ieee_status_label": _ui_severity_label(
                    rank_row.get(
                        "transformer_overall_severity_label",
                        status,
                    )
                ),
                "status": _ui_status(status),
                "severity": _ui_status(status),
                "severity_label": _ui_severity_label(
                    rank_row.get(
                        "transformer_overall_severity_label",
                        status,
                    )
                ),
                "fault_type": fault_type,
                "fault_group": fault_group,
                "fault_criticality_class": rank_row.get(
                    "fault_criticality_class",
                    classify_fault_criticality(
                        fault_type
                    ),
                ),
                "fault_criticality_source": rank_row.get(
                    "fault_criticality_source",
                    fault_criticality_source(),
                ),
                "recommended_action": rank_row.get(
                    "recommended_action",
                    "REVIEW_DATA",
                ),
                "reason": rank_row.get(
                    "maintenance_priority_reason",
                    "",
                ),
                "current_standardized_exceedance": _safe_float(
                    rank_row.get(
                        "current_standardized_exceedance",
                        np.nan,
                    )
                ),
                "current_status3_standardized_exceedance": _safe_float(
                    rank_row.get(
                        "current_status3_standardized_exceedance",
                        np.nan,
                    )
                ),
                "current_delta_exceedance": _safe_int(
                    rank_row.get(
                        "current_delta_exceedance",
                        0,
                    )
                ),
                "current_standard_trigger_count": _safe_int(
                    rank_row.get(
                        "current_standard_trigger_count",
                        0,
                    )
                ),
                "historical_max_standardized_exceedance": _safe_float(
                    rank_row.get(
                        "historical_max_standardized_exceedance",
                        np.nan,
                    )
                ),
                "history_max_status_before_current": _safe_int(
                    rank_row.get(
                        "history_max_status_before_current",
                        0,
                    )
                ),
                "history_record_count": _safe_int(
                    rank_row.get(
                        "history_record_count",
                        0,
                    )
                ),
                "history_worsening_transition_ratio": _safe_float(
                    rank_row.get(
                        "history_worsening_transition_ratio",
                        np.nan,
                    )
                ),
                "history_recurrent_fault_fraction": _safe_float(
                    rank_row.get(
                        "history_current_fault_recurrence_fraction",
                        np.nan,
                    )
                ),
                "pareto_dominance_count": _safe_int(
                    rank_row.get(
                        "pareto_dominance_count",
                        0,
                    )
                ),
                "pareto_front": bool(
                    rank_row.get(
                        "pareto_front",
                        False,
                    )
                ),
                "severity_evidence_vector": rank_row.get(
                    "severity_evidence_vector",
                    "",
                ),
                "maintenance_priority_rank_percentile": _safe_float(
                    rank_row.get(
                        "maintenance_priority_rank_percentile",
                        np.nan,
                    )
                ),
                "priority_score": _safe_float(
                    rank_row.get(
                        "transformer_overall_severity_score",
                        rank_row.get(
                            "priority_score",
                            np.nan,
                        ),
                    )
                ),
                "priority_score_type": rank_row.get(
                    "priority_score_type",
                    "UNWEIGHTED_LEXICOGRAPHIC_EVIDENCE_ORDER",
                ),
                "ranking_policy": rank_row.get(
                    "ranking_policy",
                    "",
                ),
                "ranking_is_weighted": bool(
                    rank_row.get(
                        "ranking_is_weighted",
                        False,
                    )
                ),
                "ranking_is_health_score": bool(
                    rank_row.get(
                        "ranking_is_health_score",
                        False,
                    )
                ),
                "loc": _first(
                    rank_row,
                    ["loc"],
                    "",
                )
                or "",
                "name": _first(
                    rank_row,
                    ["name"],
                    "",
                )
                or "",
                "features": {},
            }
        )

    # -----------------------------------------------------------------------
    # Time series
    # -----------------------------------------------------------------------

    timeseries = {}

    critical_lookup = (
        {
            str(r.get("transformer_id")): r
            for _, r in ranking_df.iterrows()
        }
        if ranking_df is not None
        and not ranking_df.empty
        else {}
    )

    for transformer_id, group in ordered.groupby(
        "transformer_id",
        sort=False,
    ):
        series = []

        group_sorted = group.sort_values(
            "sample_day"
        )

        latest_index = (
            group_sorted.index[-1]
            if len(group_sorted)
            else None
        )

        rank_info = critical_lookup.get(
            str(transformer_id),
            {},
        )

        transformer_is_critical = bool(
            rank_info.get(
                "critical_front_flag",
                False,
            )
        )

        for row_index, row in group_sorted.iterrows():
            status = _safe_int(
                row.get(
                    "ieee_dga_status",
                    0,
                )
            )

            critical_front = bool(
                transformer_is_critical
                and row_index == latest_index
                and status == 3
            )

            critical_ratio = _safe_float(
                row.get(
                    "ieee_max_status3_standardized_exceedance",
                    np.nan,
                )
            )

            continuous_ratio = _safe_float(
                row.get(
                    "ieee_continuous_evidence_ratio",
                    np.nan,
                )
            )

            concentration_ratio = _safe_float(
                row.get(
                    "ieee_table2_concentration_ratio_all",
                    np.nan,
                )
            )

            delta_ratio = _safe_float(
                row.get(
                    "ieee_table3_delta_ratio_all",
                    np.nan,
                )
            )

            rate_ratio = _safe_float(
                row.get(
                    "ieee_table4_rate_ratio_all",
                    np.nan,
                )
            )

            fault = row.get(
                "final_fault",
                row.get(
                    "consensus_fault",
                    "ABSTAIN",
                ),
            )

            series.append(
                {
                    "Sample Day": str(
                        row["sample_day"]
                    ),
                    "H2": _safe_float(
                        row.get(
                            "h2",
                            np.nan,
                        ),
                        0.0,
                    ),
                    "C2H2": _safe_float(
                        row.get(
                            "c2h2",
                            np.nan,
                        ),
                        0.0,
                    ),
                    "TDCG": _safe_float(
                        row.get(
                            "tdcg",
                            row.get(
                                "ieee_tdcg_ppm",
                                np.nan,
                            ),
                        ),
                        0.0,
                    ),
                    "pred_ensemble": status,
                    "ieee_status": status,
                    "ieee_status_label": row.get(
                        "ieee_dga_status_label",
                        "INSUFFICIENT_DATA",
                    ),
                    "status": _ui_status(status),
                    "fault_type": fault,
                    "fault_group": row.get(
                        "final_fault_group",
                        row.get(
                            "consensus_fault_group",
                            "ABSTAIN",
                        ),
                    ),
                    "fault_criticality_class": classify_fault_criticality(
                        fault
                    ),
                    "severity": row.get(
                        "severity_label_text",
                        "INSUFFICIENT_DATA",
                    ),
                    "critical_front": critical_front,
                    "critical_evidence_ratio": critical_ratio,
                    "continuous_evidence_ratio": continuous_ratio,
                    "continuous_evidence_basis": row.get(
                        "ieee_continuous_evidence_basis",
                        "NO_CONTINUOUS_EVIDENCE",
                    ),
                    "continuous_evidence_is_score": False,
                    "continuous_evidence_reference": (
                        "IEEE threshold ratio; per-sample "
                        "diagnostic evidence, not a weighted "
                        "severity score"
                    ),
                    "table2_concentration_ratio": concentration_ratio,
                    "table3_delta_ratio": delta_ratio,
                    "table4_rate_ratio": rate_ratio,
                    "confirmation_required": bool(
                        row.get(
                            "ieee_confirmation_required",
                            False,
                        )
                    ),
                }
            )

        timeseries[str(transformer_id)] = series

    # -----------------------------------------------------------------------
    # Dataset summary
    # -----------------------------------------------------------------------

    status_series = pd.to_numeric(
        df.get(
            "ieee_dga_status",
            pd.Series(
                0,
                index=df.index,
            ),
        ),
        errors="coerce",
    ).fillna(0)

    priority_counts = (
        ranking_df[
            "transformer_overall_severity_label"
        ]
        .value_counts()
        .reindex(
            [
                "STATUS_3",
                "STATUS_2",
                "STATUS_1",
                "INSUFFICIENT_DATA",
            ]
        )
        .fillna(0)
        .astype(int)
        .to_dict()
        if (
            ranking_df is not None
            and "transformer_overall_severity_label"
            in ranking_df.columns
        )
        else {}
    )

    fault_context_counts = (
        ranking_df[
            "fault_criticality_class"
        ]
        .value_counts()
        .to_dict()
        if (
            ranking_df is not None
            and "fault_criticality_class"
            in ranking_df.columns
        )
        else {}
    )

    top_queue = []

    if (
        ranking_df is not None
        and not ranking_df.empty
    ):
        for _, r in ranking_df.head(20).iterrows():
            top_queue.append(
                {
                    "rank": _safe_int(
                        r.get(
                            "rank",
                            0,
                        )
                    ),
                    "transformer_id": r.get(
                        "transformer_id"
                    ),
                    "maintenance_priority": r.get(
                        "maintenance_priority",
                        "DATA_REVIEW",
                    ),
                    "severity_rank_within_class": _safe_int(
                        r.get(
                            "severity_rank_within_class",
                            0,
                        )
                    ),
                    "severity_class_size": _safe_int(
                        r.get(
                            "severity_class_size",
                            0,
                        )
                    ),
                    "ieee_status": _safe_int(
                        r.get(
                            "transformer_overall_severity_level",
                            0,
                        )
                    ),
                    "current_standardized_exceedance": _safe_float(
                        r.get(
                            "current_standardized_exceedance",
                            np.nan,
                        )
                    ),
                    "current_status3_standardized_exceedance": _safe_float(
                        r.get(
                            "current_status3_standardized_exceedance",
                            np.nan,
                        )
                    ),
                    "table2_exceed_count": _safe_int(
                        r.get(
                            "table2_exceed_count",
                            0,
                        )
                    ),
                    "table4_exceed_count": _safe_int(
                        r.get(
                            "table4_exceed_count",
                            0,
                        )
                    ),
                    "table3_exceed_count": _safe_int(
                        r.get(
                            "table3_exceed_count",
                            0,
                        )
                    ),
                    "fault_type": r.get(
                        "current_fault",
                        "ABSTAIN",
                    ),
                    "fault_group": r.get(
                        "current_fault_group",
                        "ABSTAIN",
                    ),
                    "recommended_action": r.get(
                        "recommended_action",
                        "",
                    ),
                }
            )

    dataset_summary = {
        "total_transformers": int(
            df["transformer_id"].nunique()
        ),
        "total_rows": int(len(df)),
        "severity_status_1": int(
            (status_series == 1).sum()
        ),
        "severity_status_2": int(
            (status_series == 2).sum()
        ),
        "severity_status_3": int(
            (status_series == 3).sum()
        ),
        "severity_insufficient_data": int(
            (status_series == 0).sum()
        ),
        "maintenance_priority_counts": priority_counts,
        "high_risk_transformer_count": int(
            priority_counts.get(
                "STATUS_3",
                0,
            )
        ),
        "watch_transformer_count": int(
            priority_counts.get(
                "STATUS_2",
                0,
            )
        ),
        "normal_transformer_count": int(
            priority_counts.get(
                "STATUS_1",
                0,
            )
        ),
        "first_priority_transformer_id": (
            top_queue[0]["transformer_id"]
            if top_queue
            else None
        ),
        "first_priority_rank": (
            top_queue[0]["rank"]
            if top_queue
            else None
        ),
        "maintenance_queue_top20": top_queue,
        "critical_rule": "NOT_USED",
        "critical_reference": (
            "No additional Status-4 severity class is used."
        ),
        "fault_criticality_context_counts": (
            fault_context_counts
        ),
        "fault_criticality_source": (
            fault_criticality_source()
        ),
        "traditional_abstain_rows": int(
            (
                _normalize_series(
                    df.get(
                        "consensus_fault",
                        pd.Series(
                            "ABSTAIN",
                            index=df.index,
                        ),
                    )
                )
                == "ABSTAIN"
            ).sum()
        ),
        "student_fallback_rows": int(
            df.get(
                "student_used_as_fallback",
                pd.Series(
                    False,
                    index=df.index,
                ),
            ).sum()
        ),
        "student_traditional_physical_conflicts": int(
            df.get(
                "final_fault_conflict",
                pd.Series(
                    False,
                    index=df.index,
                ),
            ).sum()
        ),
    }

    return {
        "predictions": predictions,
        "rows": rows,
        "preview_rows": rows[:20],
        "transformer_summary": transformer_summary,
        "transformer_timeseries": timeseries,
        "dataset_summary": dataset_summary,
        "student_traditional_comparison": (
            []
            if comparison_df is None
            or comparison_df.empty
            else comparison_df.to_dict(
                orient="records"
            )
        ),
        "chat_context_payload": {
            "transformer_summary": transformer_summary,
            "dataset_summary": dataset_summary,
        },
    }


# ---------------------------------------------------------------------------
# Production inference
# ---------------------------------------------------------------------------

def process_dataframe(
    uploaded_df: pd.DataFrame,
):
    """
    Production inference only.

    This function MUST NOT:
      - train weak-label models
      - train student models
      - train anomaly models
      - run labeled benchmarks
      - run supervised ML comparison
      - run weak-transfer benchmarks
      - run hybrid research benchmarks
      - generate the research Excel report

    All those operations belong in a separate offline/research pipeline.
    """
    start = time.time()

    tmp_dir = Path(
        tempfile.mkdtemp(
            prefix="dga_inference_"
        )
    )

    PROCESSED_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    REPORT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    try:
        logger.info(
            "\n\n%s\n%s\n%s\n%s",
            "#" * 110,
            "# PRODUCTION DGA INFERENCE START",
            f"# uploaded rows={len(uploaded_df)}",
            "#" * 110,
        )

        # ------------------------------------------------------------------
        # 1. Convert uploaded dataframe to temporary Excel file.
        #
        # clean_dataset() already expects the existing input format.
        # ------------------------------------------------------------------

        input_path = (
            tmp_dir / "uploaded_input.xlsx"
        )

        uploaded_df.to_excel(
            input_path,
            index=False,
            engine="openpyxl",
        )

        # ------------------------------------------------------------------
        # 2. Cleaning
        # ------------------------------------------------------------------

        logger.info(
            "PIPELINE: Cleaning dataset"
        )

        df_clean, _ = clean_dataset(
            input_file=input_path,
            output_dir=PROCESSED_DIR,
        )

        if df_clean.empty:
            raise ValueError(
                "Cleaning produced an empty dataset."
            )

        logger.info(
            "Cleaning complete | rows=%d | transformers=%d | columns=%d",
            len(df_clean),
            (
                df_clean["transformer_id"].nunique()
                if "transformer_id" in df_clean.columns
                else 0
            ),
            len(df_clean.columns),
        )

        # ------------------------------------------------------------------
        # 3. Feature engineering
        # ------------------------------------------------------------------

        logger.info(
            "PIPELINE: Building DGA features"
        )

        df_features = build_training_features_from_clean(
            df_clean
        )

        if df_features.empty:
            raise ValueError(
                "Feature engineering produced an empty dataset."
            )

        logger.info(
            "Feature engineering complete | rows=%d | columns=%d",
            len(df_features),
            len(df_features.columns),
        )

        # ------------------------------------------------------------------
        # 4. Traditional diagnostics + consensus
        # ------------------------------------------------------------------

        logger.info(
            "PIPELINE: Traditional diagnostics + consensus"
        )

        df_labeled = apply_consensus(
            df_features
        )

        if "consensus_fault" in df_labeled.columns:
            logger.info(
                "Traditional consensus distribution | %s",
                df_labeled[
                    "consensus_fault"
                ].value_counts(
                    dropna=False
                ).to_dict(),
            )

        # ------------------------------------------------------------------
        # 5. Apply PRE-TRAINED weak-label models
        # ------------------------------------------------------------------

        logger.info(
            "PIPELINE: Loading pre-trained weak-label models"
        )

        df_labeled, weak_metadata = (
            _apply_pretrained_weak_models(
                df_labeled
            )
        )

        # ------------------------------------------------------------------
        # 6. Apply PRE-TRAINED production student model
        # ------------------------------------------------------------------

        logger.info(
            "PIPELINE: Loading pre-trained student model"
        )

        artifact = _load_student_model()

        df_labeled = _apply_student(
            df_labeled,
            artifact,
        )

        # ------------------------------------------------------------------
        # 7. Combine traditional + weak + student evidence
        # ------------------------------------------------------------------

        logger.info(
            "PIPELINE: Combining traditional + weak + student evidence"
        )

        df_labeled = _combine_consensus_and_student(
            df_labeled
        )

        comparison_df = _build_student_comparison(
            df_labeled
        )

        if not comparison_df.empty:
            _save_csv(
                comparison_df,
                STUDENT_COMPARISON_PATH,
            )

        # ------------------------------------------------------------------
        # 8. Apply PRE-TRAINED anomaly model
        # ------------------------------------------------------------------

        logger.info(
            "PIPELINE: Applying pre-trained anomaly model"
        )

        anomaly_scores = _predict_anomaly(
            df_labeled
        )

        df_labeled[
            "anomaly_percentile"
        ] = anomaly_scores

        df_labeled[
            "anomaly_is_severity_input"
        ] = False

        df_labeled[
            "anomaly_interpretation"
        ] = (
            "Relative anomaly position only; "
            "not IEEE severity or maintenance rank input."
        )

        if len(anomaly_scores):
            logger.info(
                "Anomaly detection complete | min=%.2f | median=%.2f | max=%.2f",
                float(np.nanmin(anomaly_scores)),
                float(np.nanmedian(anomaly_scores)),
                float(np.nanmax(anomaly_scores)),
            )

        # ------------------------------------------------------------------
        # 9. IEEE severity
        # ------------------------------------------------------------------

        logger.info(
            "PIPELINE: IEEE C57.104-2019 severity"
        )

        df_labeled = apply_severity(
            df_labeled,
            nei_reference=None,
        )

        df_labeled[
            "severity_inference_stage"
        ] = "STANDARD_RULE_ENGINE"

        df_labeled[
            "severity_manual_weights"
        ] = False

        df_labeled[
            "severity_weighted_sum_used"
        ] = False

        df_labeled[
            "severity_anomaly_used"
        ] = False

        df_labeled[
            "severity_nei_used"
        ] = False

        logger.info(
            "IEEE severity distribution | %s",
            df_labeled[
                "ieee_dga_status"
            ].value_counts(
                dropna=False
            ).sort_index().to_dict(),
        )

        # ------------------------------------------------------------------
        # 10. Transformer maintenance ranking
        # ------------------------------------------------------------------

        logger.info(
            "PIPELINE: Transformer maintenance ranking"
        )

        ranking_df = build_transformer_ranking(
            df_labeled
        )

        if not ranking_df.empty:
            log_ranking_diagnostics(
                ranking_df,
                top_n=20,
            )

            _save_parquet(
                ranking_df,
                RANKING_PATH,
            )

            _save_csv(
                ranking_df,
                REPORT_DIR
                / "transformer_ranking.csv",
            )

        # ------------------------------------------------------------------
        # 11. Save operational artifacts
        # ------------------------------------------------------------------

        logger.info(
            "PIPELINE: Saving operational artifacts"
        )

        _save_parquet(
            df_labeled,
            UNLABELED_PATH,
        )

        _save_parquet(
            df_labeled,
            PROCESSED_OUTPUT_PATH,
        )

        # ------------------------------------------------------------------
        # 12. Final response
        # ------------------------------------------------------------------

        elapsed = time.time() - start

        _write_inference_metadata(
            df_labeled,
            artifact,
            weak_metadata,
            elapsed,
        )

        payload = create_payload(
            df_labeled,
            ranking_df,
            comparison_df,
        )

        payload["pipeline"] = {
            "status": "completed",
            "mode": "production_inference",
            "runtime_training": False,
            "runtime_research_benchmark": False,
            "runtime_excel_generation": False,
            "rows": int(
                len(df_labeled)
            ),
            "transformers": int(
                df_labeled[
                    "transformer_id"
                ].nunique()
            ),
            "elapsed_seconds": float(
                elapsed
            ),
            "weak_backend": weak_metadata[
                "backend"
            ],
            "models": {
                "weak_coarse": (
                    "pretrained"
                ),
                "weak_fine": (
                    "pretrained"
                ),
                "student": (
                    artifact.get(
                        "model_name",
                        "UNKNOWN",
                    )
                ),
                "anomaly": (
                    "pretrained"
                ),
            },
            "research_benchmark": {
                "status": "precomputed_offline",
                "executed_during_request": False,
            },
            "excel": {
                "status": "precomputed_offline",
                "executed_during_request": False,
            },
            "files": {
                "processed_parquet": str(
                    PROCESSED_OUTPUT_PATH
                ),
                "ranking_parquet": str(
                    RANKING_PATH
                ),
                "ranking_csv": str(
                    REPORT_DIR
                    / "transformer_ranking.csv"
                ),
                "student_comparison": str(
                    STUDENT_COMPARISON_PATH
                ),
                "weak_coarse_model": str(
                    WEAK_COARSE_MODEL_PATH
                ),
                "weak_fine_model": str(
                    WEAK_FINE_MODEL_PATH
                ),
                "student_model": str(
                    STUDENT_MODEL_PATH
                ),
                "anomaly_model": str(
                    ANOMALY_MODEL_PATH
                ),
            },
        }

        # ------------------------------------------------------------------
        # 13. Save payload
        # ------------------------------------------------------------------

        try:
            from data_store import (
                save_payload_to_db,
            )

            save_payload_to_db(
                payload
            )

        except Exception:
            # Database persistence should not destroy an otherwise valid
            # inference response.
            logger.exception(
                "Failed to save inference payload to database"
            )

        # ------------------------------------------------------------------
        # Final logging
        # ------------------------------------------------------------------

        logger.info(
            "=" * 110
        )
        logger.info(
            "PRODUCTION DGA INFERENCE COMPLETED"
        )
        logger.info(
            "Rows           : %d",
            len(df_labeled),
        )
        logger.info(
            "Transformers   : %d",
            df_labeled[
                "transformer_id"
            ].nunique(),
        )
        logger.info(
            "Elapsed seconds: %.2f",
            elapsed,
        )
        logger.info(
            "Weak backend   : %s",
            weak_metadata.get(
                "backend"
            ),
        )
        logger.info(
            "Research       : SKIPPED DURING REQUEST"
        )
        logger.info(
            "Excel report   : SKIPPED DURING REQUEST"
        )
        logger.info(
            "=" * 110
        )

        return payload

    except Exception:
        logger.exception(
            "FATAL ERROR IN PRODUCTION DGA INFERENCE"
        )
        raise

    finally:
        shutil.rmtree(
            tmp_dir,
            ignore_errors=True,
        )


__all__ = [
    "process_dataframe",
    "create_payload",
]