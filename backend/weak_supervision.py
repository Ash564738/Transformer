# weak_supervision.py
from __future__ import annotations

import importlib
import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from consensus import normalize_fault, unify_fault
from logging_config import init_logging

init_logging()
logger = logging.getLogger(__name__)


# ============================================================================
# Snorkel detection
# ============================================================================

SNORKEL_AVAILABLE = False
LabelModel = None
SNORKEL_IMPORT_ERROR = None

try:

    if importlib.util.find_spec(
        "snorkel"
    ) is not None:

        try:
            from snorkel.labeling.model import LabelModel

            SNORKEL_AVAILABLE = True

            logger.debug(
                "Snorkel LabelModel imported successfully."
            )

        except Exception as exc:

            LabelModel = None
            SNORKEL_AVAILABLE = False
            SNORKEL_IMPORT_ERROR = str(exc)

            logger.exception(
                "Snorkel import failed: %s",
                exc,
            )

except Exception as exc:

    SNORKEL_AVAILABLE = False
    SNORKEL_IMPORT_ERROR = str(exc)

    logger.exception(
        "Unexpected Snorkel detection error."
    )


ABSTAIN = -1


# ============================================================================
# Weak-label groups
# ============================================================================

WEAK_GROUPS = [
    "NORMAL",
    "DISCHARGE",
    "THERMAL",
    "CELLULOSE",
    "STRAY_GASSING",
    "MIXED",
]


WEAK_GROUP_TO_INDEX = {
    group: idx
    for idx, group in enumerate(
        WEAK_GROUPS
    )
}


# ============================================================================
# IMPORTANT:
#
# Duval Pentagon P1 and P2 are NOT independent labeling functions.
#
# Both are generated from the same centroid and same five gases.
# Therefore only ONE Pentagon output is included in the weak label matrix.
# ============================================================================

DEFAULT_WEAK_METHODS = {
    "keygas_fault": "keygas_fault",
    "iec_fault": "iec_fault",
    "rogers_fault": "rogers_fault",
    "doernenburg_fault": "doernenburg_fault",
    "duval_triangle_fault": "duval_triangle_fault",
    "duval_pentagon_fault": "duval_pentagon_p2_fault",
}


# ============================================================================
# Fault normalization
# ============================================================================

def _normalize_vote(
    raw_label,
) -> str:

    label = normalize_fault(
        raw_label
    )

    group = unify_fault(
        label
    )

    return group


# ============================================================================
# Label matrix
# ============================================================================

def build_label_matrix(
    df: pd.DataFrame,
    label_columns: Optional[
        Dict[str, str]
    ] = None,
    groups: Optional[
        List[str]
    ] = None,
) -> Tuple[
    np.ndarray,
    List[str],
    List[str],
]:

    label_columns = (
        label_columns
        or DEFAULT_WEAK_METHODS.copy()
    )

    groups = (
        groups
        or WEAK_GROUPS.copy()
    )

    group_to_int = {
        group: idx
        for idx, group in enumerate(
            groups
        )
    }

    all_lfs = list(
        label_columns.keys()
    )

    logger.info(
        "Building weak-label matrix: "
        "%d LFs, %d groups",
        len(all_lfs),
        len(groups),
    )

    labels = []

    for _, row in df.iterrows():

        row_labels = []

        for lf_name in all_lfs:

            source_column = (
                label_columns[lf_name]
            )

            group = _normalize_vote(
                row.get(
                    source_column,
                    None,
                )
            )

            if (
                group in group_to_int
                and group != "ABSTAIN"
            ):
                row_labels.append(
                    group_to_int[group]
                )
            else:
                row_labels.append(
                    ABSTAIN
                )

        labels.append(
            row_labels
        )

    if not labels:

        L = np.empty(
            (
                0,
                len(all_lfs),
            ),
            dtype=np.int64,
        )

    else:

        L = np.asarray(
            labels,
            dtype=np.int64,
        )

    coverage = (
        (L != ABSTAIN)
        .sum(axis=1)
        if len(L)
        else np.array([])
    )

    if len(coverage):

        logger.info(
            "Rows with >=1 weak label: %d / %d (%.1f%%)",
            int(
                (coverage > 0).sum()
            ),
            len(coverage),
            100.0
            * np.mean(
                coverage > 0
            ),
        )

    return (
        L,
        all_lfs,
        groups,
    )


# ============================================================================
# Built-in simple weak supervision model
# ============================================================================

def _simple_label_model(
    L: np.ndarray,
    cardinality: int,
    n_iter: int = 100,
) -> np.ndarray:

    if cardinality <= 0:
        raise ValueError(
            "cardinality must be > 0."
        )

    n, m = L.shape

    if n == 0 or m == 0:
        return np.full(
            (
                n,
                cardinality,
            ),
            1.0 / cardinality,
            dtype=float,
        )

    logger.debug(
        "Running built-in weak-label EM for %d iterations.",
        n_iter,
    )

    # ------------------------------------------------------------------
    # Prior from observed vote frequency.
    # ------------------------------------------------------------------

    priors = np.ones(
        cardinality,
        dtype=float,
    )

    for class_idx in range(
        cardinality
    ):
        priors[class_idx] = np.mean(
            np.any(
                L == class_idx,
                axis=1,
            )
        )

    priors = np.clip(
        priors,
        1e-3,
        None,
    )

    priors /= priors.sum()

    # ------------------------------------------------------------------
    # LF parameters.
    #
    # We model:
    #
    #   accuracy[j, k]
    #   abstain_probability[j]
    # ------------------------------------------------------------------

    lf_accuracy = np.full(
        (
            m,
            cardinality,
        ),
        0.65,
        dtype=float,
    )

    lf_abstain = (
        1.0
        - np.mean(
            L != ABSTAIN,
            axis=0,
        )
    )

    probabilities = np.full(
        (
            n,
            cardinality,
        ),
        1.0 / cardinality,
        dtype=float,
    )

    for _ in range(n_iter):

        log_probability = np.log(
            priors
        )[None, :].repeat(
            n,
            axis=0,
        )

        for j in range(m):

            labels_j = L[:, j]
            observed = (
                labels_j != ABSTAIN
            )

            if not observed.any():
                continue

            # Abstain likelihood.
            if lf_abstain[j] > 0:
                log_probability[
                    ~observed,
                    :,
                ] += np.log(
                    lf_abstain[j]
                    + 1e-12
                )

            observed_indices = np.where(
                observed
            )[0]

            for k in range(
                cardinality
            ):

                accuracy = np.clip(
                    lf_accuracy[j, k],
                    1e-3,
                    0.999,
                )

                correct = (
                    labels_j[
                        observed_indices
                    ]
                    == k
                )

                incorrect_prob = (
                    1.0 - accuracy
                ) / max(
                    cardinality - 1,
                    1,
                )

                for row_idx, is_correct in zip(
                    observed_indices,
                    correct,
                ):

                    if is_correct:
                        log_probability[
                            row_idx,
                            k,
                        ] += np.log(
                            accuracy
                        )
                    else:
                        log_probability[
                            row_idx,
                            k,
                        ] += np.log(
                            incorrect_prob
                        )

        log_probability -= (
            np.max(
                log_probability,
                axis=1,
                keepdims=True,
            )
        )

        probabilities = np.exp(
            log_probability
        )

        probabilities /= np.clip(
            probabilities.sum(
                axis=1,
                keepdims=True,
            ),
            1e-12,
            None,
        )

        priors = probabilities.mean(
            axis=0
        )

        # --------------------------------------------------------------
        # Update LF accuracy
        # --------------------------------------------------------------

        for j in range(m):

            labels_j = L[:, j]
            observed = (
                labels_j != ABSTAIN
            )

            if not observed.any():
                continue

            observed_prob = probabilities[
                observed
            ]

            observed_labels = labels_j[
                observed
            ]

            for k in range(
                cardinality
            ):

                denominator = (
                    observed_prob[
                        :,
                        k,
                    ].sum()
                )

                numerator = (
                    observed_prob[
                        observed_labels == k,
                        k,
                    ].sum()
                )

                lf_accuracy[j, k] = np.clip(
                    numerator
                    / max(
                        denominator,
                        1e-9,
                    ),
                    1e-3,
                    0.999,
                )

            lf_abstain[j] = (
                1.0
                - np.mean(
                    observed
                )
            )

    logger.info(
        "Built-in weak-label model completed. Priors=%s",
        np.round(
            priors,
            4,
        ),
    )

    return probabilities


# ============================================================================
# Label model fitting
# ============================================================================

def fit_label_model(
    L: np.ndarray,
    cardinality: int,
    use_snorkel: bool = True,
    **kwargs,
) -> Tuple[
    Optional[object],
    np.ndarray,
]:

    if use_snorkel and SNORKEL_AVAILABLE:

        label_model = LabelModel(
            cardinality=cardinality,
            verbose=False,
        )

        label_model.fit(
            L_train=L,
            n_epochs=200,
            log_freq=50,
            **kwargs,
        )

        probabilities = (
            label_model.predict_proba(
                L
            )
        )

        logger.info(
            "Snorkel LabelModel fitted."
        )

        return (
            label_model,
            probabilities,
        )

    if use_snorkel:

        if SNORKEL_IMPORT_ERROR:
            logger.warning(
                "Snorkel unavailable: %s. "
                "Using built-in estimator.",
                SNORKEL_IMPORT_ERROR,
            )
        else:
            logger.warning(
                "Snorkel not installed. "
                "Using built-in estimator."
            )

    probabilities = _simple_label_model(
        L,
        cardinality,
    )

    return (
        None,
        probabilities,
    )


# ============================================================================
# Attach probabilities
# ============================================================================

def attach_probabilistic_labels(
    df: pd.DataFrame,
    probs: np.ndarray,
    groups: List[str],
    prefix: str = "weak_prob",
    confidence_threshold: float = 0.70,
) -> pd.DataFrame:

    df = df.copy()

    if probs.shape[0] != len(df):
        raise ValueError(
            "Probability matrix row count does not match DataFrame."
        )

    for idx, group in enumerate(
        groups
    ):

        df[
            f"{prefix}_{group.lower()}"
        ] = probs[:, idx]

    best_idx = np.argmax(
        probs,
        axis=1,
    )

    confidence = np.max(
        probs,
        axis=1,
    )

    df["weak_fault_confidence"] = (
        confidence
    )

    df["weak_fault_is_ABSTAIN"] = (
        confidence
        < confidence_threshold
    )

    df["weak_fault_group"] = [
        groups[idx]
        if confidence[row_idx]
        >= confidence_threshold
        else "ABSTAIN"
        for row_idx, idx in enumerate(
            best_idx
        )
    ]

    logger.info(
        "Weak label distribution:\n%s",
        df[
            "weak_fault_group"
        ]
        .value_counts(
            dropna=False
        )
        .to_string(),
    )

    logger.info(
        "Weak-label confidence: "
        "mean=%.3f, accepted=%.1f%%",
        df[
            "weak_fault_confidence"
        ].mean(),
        100.0
        * (
            ~df[
                "weak_fault_is_ABSTAIN"
            ]
        ).mean(),
    )

    return df


# ============================================================================
# Full pipeline
# ============================================================================

def weak_supervision_pipeline(
    df: pd.DataFrame,
    label_columns: Optional[
        Dict[str, str]
    ] = None,
    groups: Optional[
        List[str]
    ] = None,
    use_snorkel: bool = True,
    confidence_threshold: float = 0.70,
) -> Tuple[
    pd.DataFrame,
    Optional[object],
    List[str],
]:

    logger.info(
        "Starting weak supervision pipeline."
    )

    L, lf_names, groups = build_label_matrix(
        df,
        label_columns=label_columns,
        groups=groups,
    )

    label_model, probabilities = (
        fit_label_model(
            L,
            cardinality=len(groups),
            use_snorkel=use_snorkel,
        )
    )

    df = attach_probabilistic_labels(
        df,
        probabilities,
        groups,
        confidence_threshold=confidence_threshold,
    )

    return (
        df,
        label_model,
        groups,
    )


# ============================================================================
# Student targets
# ============================================================================

def create_student_training_targets(
    df: pd.DataFrame,
    target_group: str = "weak_fault_group",
    weight_column: str = "weak_fault_confidence",
    groups: Optional[
        List[str]
    ] = None,
) -> Tuple[
    np.ndarray,
    np.ndarray,
]:

    groups = (
        groups
        or WEAK_GROUPS
    )

    if target_group not in df.columns:
        raise ValueError(
            f"Missing weak-label target: {target_group}"
        )

    if weight_column not in df.columns:
        raise ValueError(
            f"Missing weak-label weight: {weight_column}"
        )

    group_to_idx = {
        group: idx
        for idx, group in enumerate(
            groups
        )
    }

    clean_df = df[
        df[target_group].isin(
            group_to_idx.keys()
        )
    ].copy()

    if clean_df.empty:
        raise ValueError(
            "No accepted weak-labeled samples remain."
        )

    y = (
        clean_df[
            target_group
        ]
        .map(
            group_to_idx
        )
        .astype(int)
        .to_numpy()
    )

    confidence = (
        pd.to_numeric(
            clean_df[
                weight_column
            ],
            errors="coerce",
        )
        .fillna(0.0)
        .clip(
            0.0,
            1.0,
        )
        .to_numpy()
    )

    # ------------------------------------------------------------------
    # Balanced class weights without requiring external labels.
    # ------------------------------------------------------------------

    class_counts = np.bincount(
        y,
        minlength=len(groups),
    )

    n_samples = len(y)

    class_weights = np.ones(
        len(groups),
        dtype=float,
    )

    nonzero = (
        class_counts > 0
    )

    class_weights[
        nonzero
    ] = (
        n_samples
        / (
            len(
                class_counts[
                    nonzero
                ]
            )
            * class_counts[
                nonzero
            ]
        )
    )

    sample_weights = (
        confidence
        * class_weights[y]
    )

    # Avoid zero-weight samples.
    # Their confidence is still reflected in the relative weight.
    sample_weights = np.clip(
        sample_weights,
        1e-4,
        None,
    )

    return (
        y,
        sample_weights,
    )