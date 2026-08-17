# evaluation.py
from __future__ import annotations
import logging
from typing import Callable, Iterable, Optional, Sequence
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.utils import resample

logger = logging.getLogger(__name__)

def _as_1d_float(values) -> np.ndarray:
    return np.asarray(values, dtype=float).reshape(-1)

def _as_binary_events(events) -> np.ndarray:
    return np.asarray(events).reshape(-1).astype(bool)

def precision_recall_lift(scores, events, k: int):
    scores = _as_1d_float(scores)
    events = _as_binary_events(events)
    if len(scores) != len(events):
        raise ValueError("scores and events must have the same length.")
    n = len(scores)
    if n == 0:
        return (0.0, 0.0, 1.0)
    k = int(np.clip(k, 1, n))
    finite_scores = np.nan_to_num(scores, nan=-np.inf, posinf=np.inf, neginf=-np.inf)
    order = np.argsort(finite_scores)[::-1]
    topk = order[:k]
    tp = int(events[topk].sum())
    precision = tp / k
    total_events = int(events.sum())
    recall = tp / total_events if total_events > 0 else 0.0
    prevalence = float(events.mean())
    lift = precision / prevalence if prevalence > 0 else 1.0
    return float(precision), float(recall), float(lift)

def topk_stability(scores1, scores2, k: int):
    scores1 = _as_1d_float(scores1)
    scores2 = _as_1d_float(scores2)
    n = min(len(scores1), len(scores2))
    if n == 0:
        return 1.0
    k = int(np.clip(k, 1, n))
    idx1 = np.argsort(np.nan_to_num(scores1, nan=-np.inf))[::-1][:k]
    idx2 = np.argsort(np.nan_to_num(scores2, nan=-np.inf))[::-1][:k]
    set1 = set(idx1.tolist())
    set2 = set(idx2.tolist())
    union = len(set1 | set2)
    if union == 0:
        return 1.0
    return float(len(set1 & set2) / union)

def rank_correlation(scores1, scores2):
    scores1 = _as_1d_float(scores1)
    scores2 = _as_1d_float(scores2)
    if len(scores1) != len(scores2):
        raise ValueError("scores1 and scores2 must have equal length.")
    if len(scores1) < 2:
        return 1.0
    rho = spearmanr(scores1, scores2, nan_policy="omit").statistic
    if not np.isfinite(rho):
        return 0.0
    return float(rho)

def temporal_consistency(scores, tdcg, transformer_ids, sample_days: Optional[Sequence] = None):
    scores = _as_1d_float(scores)
    tdcg = _as_1d_float(tdcg)
    ids = np.asarray(transformer_ids)
    if not (len(scores) == len(tdcg) == len(ids)):
        raise ValueError("scores, tdcg and transformer_ids must have equal length.")
    if sample_days is None:
        days = np.arange(len(scores))
    else:
        days = pd.to_datetime(sample_days, errors="coerce")
    frame = pd.DataFrame({"score": scores, "tdcg": tdcg, "id": ids, "day": days})
    consistent = 0
    total = 0
    for _, group in frame.groupby("id"):
        if sample_days is not None:
            group = group.sort_values("day")
        else:
            group = group.sort_index()
        group = group.reset_index(drop=True)
        for i in range(len(group) - 1):
            current_tdcg = group.loc[i, "tdcg"]
            next_tdcg = group.loc[i + 1, "tdcg"]
            current_score = group.loc[i, "score"]
            next_score = group.loc[i + 1, "score"]
            if not (np.isfinite(current_tdcg) and np.isfinite(next_tdcg) and
                    np.isfinite(current_score) and np.isfinite(next_score)):
                continue
            if next_tdcg > current_tdcg:
                total += 1
                if next_score >= current_score:
                    consistent += 1
    return float(consistent / total if total > 0 else 0.0)

def gas_increase_consistency(model, X, gas_columns, n_perturb: int = 100, increase_fraction: float = 0.10, random_state: int = 42):
    X = np.asarray(X, dtype=float)
    if X.ndim != 2:
        raise ValueError("X must be 2-dimensional.")
    n = len(X)
    if n == 0:
        return 0.0
    rng = np.random.default_rng(random_state)
    sample_count = min(n_perturb, n)
    indices = rng.choice(n, size=sample_count, replace=False)
    original_scores = model.predict(X[indices])
    consistent = 0
    for local_index, row_index in enumerate(indices):
        x_original = X[row_index].copy()
        x_perturbed = x_original.copy()
        for column_index in gas_columns:
            if (not isinstance(column_index, (int, np.integer)) or
                column_index < 0 or column_index >= X.shape[1]):
                continue
            value = x_perturbed[column_index]
            if np.isfinite(value) and value >= 0:
                x_perturbed[column_index] = value * (1.0 + increase_fraction)
        perturbed_score = model.predict(x_perturbed.reshape(1, -1))[0]
        if perturbed_score >= original_scores[local_index]:
            consistent += 1
    return float(consistent / sample_count if sample_count > 0 else 0.0)

def evaluate_agreement_with_weak_labels(df: pd.DataFrame, weak_label_col: str, predicted_col: str = "consensus_fault") -> dict:
    from sklearn.metrics import accuracy_score, cohen_kappa_score, f1_score
    if weak_label_col not in df.columns or predicted_col not in df.columns:
        return {"accuracy": None, "macro_f1": None, "cohen_kappa": None, "n": 0}
    mask = df[weak_label_col].notna() & df[predicted_col].notna()
    y_true = df.loc[mask, weak_label_col].astype(str)
    y_pred = df.loc[mask, predicted_col].astype(str)
    if len(y_true) == 0:
        return {"accuracy": None, "macro_f1": None, "cohen_kappa": None, "n": 0}
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "cohen_kappa": float(cohen_kappa_score(y_true, y_pred)),
        "n": int(len(y_true)),
    }

def evaluate_diagnostic_performance(df: pd.DataFrame, ground_truth_col: str, predicted_col: str = "consensus_fault") -> dict:
    from sklearn.metrics import accuracy_score, cohen_kappa_score, f1_score
    if ground_truth_col not in df.columns or predicted_col not in df.columns:
        return {"accuracy": None, "macro_f1": None, "cohen_kappa": None, "n": 0}
    mask = df[ground_truth_col].notna() & df[predicted_col].notna()
    y_true = df.loc[mask, ground_truth_col].astype(str)
    y_pred = df.loc[mask, predicted_col].astype(str)
    if len(y_true) == 0:
        return {"accuracy": None, "macro_f1": None, "cohen_kappa": None, "n": 0}
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "cohen_kappa": float(cohen_kappa_score(y_true, y_pred)),
        "n": int(len(y_true)),
    }

def bootstrap_confidence_interval(data, metric_fn: Callable, n_bootstrap: int = 1000, alpha: float = 0.05, random_state: int = 42):
    data = np.asarray(data)
    if data.size == 0:
        return (np.nan, np.nan, np.nan)
    rng = np.random.default_rng(random_state)
    values = []
    for _ in range(n_bootstrap):
        sample = resample(data, replace=True, n_samples=len(data),
                          random_state=int(rng.integers(0, 2**32 - 1)))
        value = metric_fn(sample)
        if np.isfinite(value):
            values.append(float(value))
    if not values:
        return (np.nan, np.nan, np.nan)
    values = np.asarray(values, dtype=float)
    lower = float(np.percentile(values, 100 * alpha / 2))
    upper = float(np.percentile(values, 100 * (1.0 - alpha / 2)))
    mean = float(np.mean(values))
    return mean, lower, upper