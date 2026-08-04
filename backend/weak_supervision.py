from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from consensus import normalize_fault, unify_fault

logger = logging.getLogger(__name__)

try:
    from snorkel.labeling import LabelModel
    SNORKEL_AVAILABLE = True
except ImportError:  # pragma: no cover
    LabelModel = None
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
    """Build a Snorkel-compatible label matrix from traditional DGA votes."""
    label_columns = label_columns or DEFAULT_WEAK_METHODS
    groups = groups or WEAK_GROUPS
    group_to_int = {g: i for i, g in enumerate(groups)}

    labels = []
    for _, row in df.iterrows():
        row_labels = []
        for _, col in label_columns.items():
            group = _normalize_vote(row.get(col, None))
            if group in group_to_int and group != "ABSTAIN":
                row_labels.append(group_to_int[group])
            else:
                row_labels.append(ABSTAIN)
        labels.append(row_labels)

    return np.asarray(labels, dtype=np.int64), list(label_columns.keys()), groups


def _simple_label_model(L: np.ndarray, cardinality: int, n_iter: int = 80) -> np.ndarray:
    """Simplified generative EM model for label counts when Snorkel is unavailable."""
    n, m = L.shape
    if n == 0 or m == 0:
        return np.zeros((n, cardinality), dtype=float)

    # Initialize priors by majority vote over non-abstains.
    priors = np.full(cardinality, 1.0 / cardinality, dtype=float)
    for i in range(cardinality):
        priors[i] = np.mean(np.any(L == i, axis=1))
    priors = np.clip(priors, 1e-3, None)
    priors /= priors.sum()

    # Each labeling function has a per-class accuracy estimate.
    lf_acc = np.full((m, cardinality), 0.7, dtype=float)
    lf_abstain = np.full(m, 0.2, dtype=float)

    for _ in range(n_iter):
        # E-step: posterior probabilities for each example.
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
                    log_p[incorrect, k] += np.log((1 - lf_acc[j, k]) / max(cardinality - 1, 1) + 1e-9)
            log_p[~obs, :] += np.log(lf_abstain[j] + 1e-9)

        log_p -= log_p.max(axis=1, keepdims=True)
        p = np.exp(log_p)
        p /= p.sum(axis=1, keepdims=True)

        # M-step: update priors and accuracies.
        priors = p.mean(axis=0)
        for j in range(m):
            label_j = L[:, j]
            obs = label_j != ABSTAIN
            if not np.any(obs):
                continue
            for k in range(cardinality):
                weighted_correct = p[obs, k][label_j[obs] == k].sum()
                lf_acc[j, k] = np.clip(weighted_correct / max(p[obs, k].sum(), 1e-6), 1e-3, 0.999)
            lf_abstain[j] = 1.0 - np.mean(obs)

    return p


def fit_label_model(
    L: np.ndarray,
    cardinality: int,
    use_snorkel: bool = True,
    **kwargs,
) -> Tuple[Optional[object], np.ndarray]:
    """Fit a label model and return prediction probabilities."""
    if use_snorkel:
        if not SNORKEL_AVAILABLE:
            logger.warning(
                "Snorkel is not installed; falling back to the built-in weak supervision estimator."
            )
        elif LabelModel is None:
            raise RuntimeError("Snorkel is not importable despite availability flag.")
        else:
            label_model = LabelModel(cardinality=cardinality, verbose=False)
            label_model.fit(L_train=L, n_epochs=200, log_freq=50, **kwargs)
            probs = label_model.predict_proba(L)
            return label_model, probs

    probs = _simple_label_model(L, cardinality)
    return None, probs


def attach_probabilistic_labels(
    df: pd.DataFrame,
    probs: np.ndarray,
    groups: List[str],
    prefix: str = "weak_prob",
) -> pd.DataFrame:
    for idx, group in enumerate(groups):
        df[f"{prefix}_{group.lower()}"] = probs[:, idx]

    df["weak_fault_group"] = [groups[i] if i >= 0 else "ABSTAIN" for i in np.argmax(probs, axis=1)]
    df["weak_fault_confidence"] = probs.max(axis=1)
    df["weak_fault_is_ABSTAIN"] = df["weak_fault_confidence"] < 0.5
    return df


def weak_supervision_pipeline(
    df: pd.DataFrame,
    label_columns: Optional[Dict[str, str]] = None,
    groups: Optional[List[str]] = None,
    use_snorkel: bool = True,
) -> Tuple[pd.DataFrame, Optional[object], List[str]]:
    L, lf_names, groups = build_label_matrix(df, label_columns=label_columns, groups=groups)
    label_model, probs = fit_label_model(L, len(groups), use_snorkel=use_snorkel)
    df = attach_probabilistic_labels(df, probs, groups)
    return df, label_model, groups


def create_student_training_targets(
    df: pd.DataFrame,
    target_group: str = "weak_fault_group",
    weight_column: str = "weak_fault_confidence",
) -> Tuple[np.ndarray, np.ndarray]:
    if target_group not in df.columns:
        raise ValueError(f"Missing weak supervision target column: {target_group}")

    y = df[target_group].astype("category")
    weights = df[weight_column].fillna(0.0).astype(float).values
    return y.cat.codes.values, weights

