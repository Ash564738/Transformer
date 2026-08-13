# weak_supervision.py
from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple
import importlib

import numpy as np
import pandas as pd

from consensus import normalize_fault, unify_fault
from logging_config import init_logging

init_logging()
logger = logging.getLogger(__name__)

# ---------- Snorkel detection ----------
SNORKEL_AVAILABLE = False
LabelModel = None
SNORKEL_IMPORT_ERROR = None
try:
    if importlib.util.find_spec("snorkel") is not None:
        try:
            from snorkel.labeling.model import LabelModel
            SNORKEL_AVAILABLE = True
            logger.debug("Snorkel detected and LabelModel imported successfully.")
        except Exception as ex:
            logger.exception("Snorkel import failed: %s", ex)
            LabelModel = None
            SNORKEL_AVAILABLE = False
            SNORKEL_IMPORT_ERROR = str(ex)
    else:
        logger.debug("Snorkel not found by importlib.util.find_spec")
except Exception:
    logger.exception("Unexpected error while checking snorkel availability")
    SNORKEL_AVAILABLE = False

ABSTAIN = -1

DEFAULT_WEAK_METHODS = {
    "keygas_fault": "keygas_fault",
    "iec_fault": "iec_fault",
    "rogers_fault": "rogers_fault",
    "doernenburg_fault": "doernenburg_fault",
    "duval_triangle_fault": "duval_triangle_fault",
    "duval_pentagon_p1_fault": "fault_p1",
    "duval_pentagon_p2_fault": "duval_pentagon_fault",
}

WEAK_GROUPS = ["NORMAL", "DISCHARGE", "THERMAL", "CELLULOSE", "STRAY_GASSING", "MIXED"]


def _normalize_vote(raw_label: str) -> str:
    label = normalize_fault(raw_label)
    group = unify_fault(label)
    return group


def build_label_matrix(
    df: pd.DataFrame,
    label_columns: Optional[Dict[str, str]] = None,
    groups: Optional[List[str]] = None,
) -> Tuple[np.ndarray, List[str], List[str]]:
    """
    Build a Snorkel-compatible label matrix from traditional DGA votes only.
    No extra rule-based LFs are used.
    """
    label_columns = label_columns or DEFAULT_WEAK_METHODS
    groups = groups or WEAK_GROUPS
    group_to_int = {g: i for i, g in enumerate(groups)}

    all_lfs = list(label_columns.keys())
    logger.info("Building label matrix with %d labeling functions (traditional DGA methods only)", len(all_lfs))
    logger.debug("Labeling functions: %s", all_lfs)

    labels = []
    for _, row in df.iterrows():
        row_labels = []
        for lf in all_lfs:
            group = _normalize_vote(row.get(label_columns[lf], None))
            if group in group_to_int and group != "ABSTAIN":
                row_labels.append(group_to_int[group])
            else:
                row_labels.append(ABSTAIN)
        labels.append(row_labels)

    L = np.asarray(labels, dtype=np.int64)
    total_non_abstain = (L != ABSTAIN).sum(axis=1)
    logger.info("Label matrix built: shape=%s, rows with at least one non-abstain label=%d (%.1f%%)",
                L.shape, (total_non_abstain > 0).sum(), 100 * (total_non_abstain > 0).mean())
    return L, all_lfs, groups


def _simple_label_model(L: np.ndarray, cardinality: int, n_iter: int = 80) -> np.ndarray:
    """Simplified generative EM model."""
    n, m = L.shape
    if n == 0 or m == 0:
        logger.warning("Empty label matrix; returning uniform probabilities.")
        return np.ones((n, cardinality)) / cardinality

    logger.debug("Running simple EM label model for %d iterations...", n_iter)
    priors = np.full(cardinality, 1.0 / cardinality, dtype=float)
    for i in range(cardinality):
        priors[i] = np.mean(np.any(L == i, axis=1))
    priors = np.clip(priors, 1e-3, None)
    priors /= priors.sum()

    lf_acc = np.full((m, cardinality), 0.7, dtype=float)
    lf_abstain = np.full(m, 0.2, dtype=float)

    for iteration in range(n_iter):
        log_p = np.log(priors)[None, :].repeat(n, axis=0)
        for j in range(m):
            label_j = L[:, j]
            obs = label_j != ABSTAIN
            if not np.any(obs):
                continue
            for k in range(cardinality):
                correct = label_j == k
                incorrect = (label_j != k) & obs
                log_p[correct, k] += np.log(lf_acc[j, k] + 1e-9)
                if np.any(incorrect):
                    log_p[incorrect, k] += np.log(
                        (1 - lf_acc[j, k]) / max(cardinality - 1, 1) + 1e-9
                    )
            log_p[~obs, :] += np.log(lf_abstain[j] + 1e-9)

        log_p -= log_p.max(axis=1, keepdims=True)
        p = np.exp(log_p)
        p /= p.sum(axis=1, keepdims=True)

        priors = p.mean(axis=0)
        for j in range(m):
            label_j = L[:, j]
            obs = label_j != ABSTAIN
            if not np.any(obs):
                continue
            for k in range(cardinality):
                weighted_correct = p[obs, k][label_j[obs] == k].sum()
                lf_acc[j, k] = np.clip(
                    weighted_correct / max(p[obs, k].sum(), 1e-6), 1e-3, 0.999
                )
            lf_abstain[j] = 1.0 - np.mean(obs)

    logger.info("Simple EM label model completed. Final priors: %s", priors.round(4))
    return p


def fit_label_model(
    L: np.ndarray,
    cardinality: int,
    use_snorkel: bool = True,
    **kwargs,
) -> Tuple[Optional[object], np.ndarray]:
    logger.info("Fitting label model (use_snorkel=%s, cardinality=%d)", use_snorkel, cardinality)
    if use_snorkel:
        if not SNORKEL_AVAILABLE:
            if SNORKEL_IMPORT_ERROR:
                logger.warning(
                    "Snorkel is installed but LabelModel is unavailable (%s); falling back to the built-in weak supervision estimator.",
                    SNORKEL_IMPORT_ERROR,
                )
            else:
                logger.warning(
                    "Snorkel is not installed; falling back to the built-in weak supervision estimator."
                )
        elif LabelModel is None:
            raise RuntimeError("Snorkel is not importable despite availability flag.")
        else:
            label_model = LabelModel(cardinality=cardinality, verbose=False)
            label_model.fit(L_train=L, n_epochs=200, log_freq=50, **kwargs)
            probs = label_model.predict_proba(L)
            logger.info("Snorkel LabelModel fitted successfully.")
            return label_model, probs

    probs = _simple_label_model(L, cardinality)
    return None, probs


def attach_probabilistic_labels(
    df: pd.DataFrame,
    probs: np.ndarray,
    groups: List[str],
    prefix: str = "weak_prob",
) -> pd.DataFrame:
    logger.info("Attaching probabilistic labels to DataFrame (%d samples)", len(df))
    for idx, group in enumerate(groups):
        df[f"{prefix}_{group.lower()}"] = probs[:, idx]

    df["weak_fault_group"] = [
        groups[i] if i >= 0 else "ABSTAIN" for i in np.argmax(probs, axis=1)
    ]
    df["weak_fault_confidence"] = probs.max(axis=1)
    df["weak_fault_is_ABSTAIN"] = df["weak_fault_confidence"] < 0.5

    group_counts = df["weak_fault_group"].value_counts()
    logger.info("Weak label distribution:\n%s", group_counts.to_string())
    logger.info("Mean confidence: %.3f, ABSTAIN fraction: %.3f",
                df["weak_fault_confidence"].mean(), df["weak_fault_is_ABSTAIN"].mean())
    return df


def weak_supervision_pipeline(
    df: pd.DataFrame,
    label_columns: Optional[Dict[str, str]] = None,
    groups: Optional[List[str]] = None,
    use_snorkel: bool = True,
) -> Tuple[pd.DataFrame, Optional[object], List[str]]:
    logger.info("Starting weak supervision pipeline (use_snorkel=%s)", use_snorkel)
    L, lf_names, groups = build_label_matrix(
        df, label_columns=label_columns, groups=groups
    )
    label_model, probs = fit_label_model(L, len(groups), use_snorkel=use_snorkel)
    df = attach_probabilistic_labels(df, probs, groups)
    logger.info("Weak supervision pipeline completed successfully.")
    return df, label_model, groups


def create_student_training_targets(
    df: pd.DataFrame,
    target_group: str = "weak_fault_group",
    weight_column: str = "weak_fault_confidence",
    groups: Optional[List[str]] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Bổ sung: tính class weights dựa trên phân phối nhãn để giảm lệch.
    """
    if target_group not in df.columns:
        raise ValueError(f"Missing weak supervision target column: {target_group}")

    groups = groups or WEAK_GROUPS
    group_to_idx = {group: idx for idx, group in enumerate(groups)}
    y = df[target_group].map(group_to_idx)
    if y.isna().any():
        unknown = sorted(df.loc[y.isna(), target_group].astype(str).unique())
        logger.error("Unknown weak supervision labels found: %s", unknown)
        raise ValueError(f"Unknown weak supervision labels: {unknown}")

    conf_weights = df[weight_column].fillna(0.0).astype(float).values

    from sklearn.utils.class_weight import compute_class_weight
    classes = np.unique(y)
    class_weights = compute_class_weight('balanced', classes=classes, y=y)
    class_weight_map = {cls: w for cls, w in zip(classes, class_weights)}
    sample_weights = conf_weights * np.array([class_weight_map[label] for label in y])

    logger.info(
        "Student training targets created: %d samples, mean weight=%.3f (balanced). "
        "Class weights: %s",
        len(y), sample_weights.mean(),
        {groups[i]: round(class_weight_map.get(i, 1.0), 3) for i in range(len(groups))}
    )
    return y.astype(int).values, sample_weights