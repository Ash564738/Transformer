# evaluation.py
from __future__ import annotations
import itertools, logging
from typing import Sequence
import numpy as np, pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import StratifiedShuffleSplit
from config import config as cfg
from consensus import normalize_fault, unify_fault
logger = logging.getLogger(__name__)

def harmonize_benchmark_labels(series: pd.Series) -> pd.Series: return series.map(normalize_fault)
def map_to_coarse(series: pd.Series) -> pd.Series: return series.map(unify_fault)
def detect_benchmark_label_column(df: pd.DataFrame) -> str:
    for column in ("fault_type_label", "label", "type"):
        if column in df.columns: return column
    raise ValueError("Cannot find benchmark label column.")

def prepare_benchmark_dataframe(df: pd.DataFrame, source_name: str) -> pd.DataFrame:
    out = df.copy(); label_column = detect_benchmark_label_column(out)
    out["benchmark_source"] = source_name
    out["fault_type_label"] = out[label_column].map(normalize_fault)
    out["fault_group_label"] = out["fault_type_label"].map(unify_fault)
    return out

def combine_labeled_benchmarks(dga_df: pd.DataFrame, iec_df: pd.DataFrame) -> pd.DataFrame:
    dga = prepare_benchmark_dataframe(dga_df, "DGA_dataset"); iec = prepare_benchmark_dataframe(iec_df, "IEC_TC10")
    required = list(cfg.COMMON_BENCHMARK_GASES)
    for frame_name, frame in (("DGA_dataset", dga), ("IEC_TC10", iec)):
        missing = [gas for gas in required if gas not in frame.columns]
        if missing: raise ValueError(f"{frame_name}: missing gases {missing}")
    combined = pd.concat([dga, iec], ignore_index=True)
    dedup_cols = ["benchmark_source", *cfg.COMMON_BENCHMARK_GASES, "fault_type_label"]
    dedup_cols = [c for c in dedup_cols if c in combined.columns]
    return combined.drop_duplicates(subset=dedup_cols).reset_index(drop=True)

def split_benchmark(df: pd.DataFrame):
    work = df.copy()
    stratify_key = work["benchmark_source"].astype(str) + "__" + work["fault_group_label"].astype(str)
    counts = stratify_key.value_counts()
    stratify_key = stratify_key.where(stratify_key.map(counts) >= 2, work["fault_group_label"].astype(str))
    first_split = StratifiedShuffleSplit(n_splits=1, test_size=cfg.TEST_SIZE, random_state=cfg.RANDOM_STATE)
    train_dev_idx, test_idx = next(first_split.split(work, stratify_key))
    train_dev = work.iloc[train_dev_idx].reset_index(drop=True); test = work.iloc[test_idx].reset_index(drop=True)
    dev_fraction = cfg.DEV_SIZE / (1.0 - cfg.TEST_SIZE)
    dev_key = train_dev["benchmark_source"].astype(str) + "__" + train_dev["fault_group_label"].astype(str)
    dev_counts = dev_key.value_counts()
    dev_key = dev_key.where(dev_key.map(dev_counts) >= 2, train_dev["fault_group_label"].astype(str))
    second_split = StratifiedShuffleSplit(n_splits=1, test_size=dev_fraction, random_state=cfg.RANDOM_STATE)
    train_idx, dev_idx = next(second_split.split(train_dev, dev_key))
    train = train_dev.iloc[train_idx].reset_index(drop=True); dev = train_dev.iloc[dev_idx].reset_index(drop=True)
    return train, dev, test

def aggregate_method_combination(df: pd.DataFrame, methods: Sequence[str]) -> pd.Series:
    predictions = []
    for _, row in df.iterrows():
        labels = []
        for method in methods:
            column = cfg.DIAGNOSTIC_METHOD_TO_COLUMN[method]
            value = normalize_fault(row.get(column, "ABSTAIN"))
            if value != "ABSTAIN": labels.append(value)
        if not labels: predictions.append("ABSTAIN"); continue
        coarse = [unify_fault(label) for label in labels]
        counts = pd.Series(coarse).value_counts(); max_count = counts.iloc[0]
        top_groups = counts[counts == max_count].index.tolist()
        if len(top_groups) > 1: predictions.append("ABSTAIN"); continue
        winning_group = top_groups[0]
        fine = [label for label in labels if unify_fault(label) == winning_group]
        fine_counts = pd.Series(fine).value_counts(); max_fine = fine_counts.iloc[0]
        top_fine = fine_counts[fine_counts == max_fine].index.tolist()
        if len(top_fine) == 1: predictions.append(top_fine[0])
        else: predictions.append("ABSTAIN")
    return pd.Series(predictions, index=df.index)

def _clean_metric_labels(y_true, y_pred):
    truth = pd.Series(y_true).reset_index(drop=True).astype(str).str.upper().str.strip()
    pred = pd.Series(y_pred).reset_index(drop=True).astype(str).str.upper().str.strip()
    return truth, pred

def _safe_balanced_accuracy(y_true, y_pred):
    if len(y_true) == 0: return np.nan
    true_labels = sorted(set(y_true))
    if not true_labels: return np.nan
    recalls = recall_score(y_true, y_pred, labels=true_labels, average=None, zero_division=0)
    return float(np.mean(recalls))

def evaluate_predictions(y_true, y_pred, allowed_labels=None, allow_abstain=False):
    truth, pred = _clean_metric_labels(y_true, y_pred)
    if allowed_labels is None: allowed_labels = sorted((set(truth) | set(pred)) - {"ABSTAIN"})
    else: allowed_labels = [str(label).upper() for label in allowed_labels]
    allowed_set = set(allowed_labels)
    valid_truth = truth.isin(allowed_set); truth = truth.loc[valid_truth].reset_index(drop=True); pred = pred.loc[valid_truth].reset_index(drop=True)
    if len(truth) == 0: return {"accuracy": np.nan, "balanced_accuracy": np.nan, "macro_precision": np.nan, "macro_recall": np.nan, "macro_f1": np.nan, "weighted_f1": np.nan, "coverage": 0.0, "selective_accuracy": np.nan, "n": 0, "n_evaluated": 0}
    if allow_abstain: active = np.ones(len(pred), dtype=bool)
    else: active = pred.to_numpy() != "ABSTAIN"
    coverage = float(active.mean())
    if not np.any(active): return {"accuracy": np.nan, "balanced_accuracy": np.nan, "macro_precision": np.nan, "macro_recall": np.nan, "macro_f1": np.nan, "weighted_f1": np.nan, "coverage": coverage, "selective_accuracy": np.nan, "n": int(len(truth)), "n_evaluated": 0}
    indices = np.flatnonzero(active); y_t = truth.iloc[indices].reset_index(drop=True); y_p = pred.iloc[indices].reset_index(drop=True)
    invalid_prediction = ~y_p.isin(allowed_set); invalid_count = int(invalid_prediction.sum()); valid_prediction = ~invalid_prediction
    eval_true = y_t.loc[valid_prediction].reset_index(drop=True); eval_pred = y_p.loc[valid_prediction].reset_index(drop=True)
    correct = int((eval_true == eval_pred).sum())
    if len(eval_true) == 0: return {"accuracy": float(correct / max(len(y_t), 1)), "balanced_accuracy": 0.0, "macro_precision": 0.0, "macro_recall": 0.0, "macro_f1": 0.0, "weighted_f1": 0.0, "coverage": coverage, "selective_accuracy": float(correct / max(len(y_t), 1)), "n": int(len(truth)), "n_evaluated": int(len(y_t))}
    labels = list(allowed_labels); total_correct = int((eval_true == eval_pred).sum())
    return {"accuracy": float(total_correct / max(len(y_t), 1)), "balanced_accuracy": _safe_balanced_accuracy(eval_true, eval_pred), "macro_precision": float(precision_score(eval_true, eval_pred, labels=labels, average="macro", zero_division=0)), "macro_recall": float(recall_score(eval_true, eval_pred, labels=labels, average="macro", zero_division=0)), "macro_f1": float(f1_score(eval_true, eval_pred, labels=labels, average="macro", zero_division=0)), "weighted_f1": float(f1_score(eval_true, eval_pred, labels=labels, average="weighted", zero_division=0)), "coverage": coverage, "selective_accuracy": float(total_correct / max(len(y_t), 1)), "n": int(len(truth)), "n_evaluated": int(len(y_t))}

def generate_method_combinations(methods):
    methods = list(methods)
    for size in range(1, len(methods) + 1): yield from itertools.combinations(methods, size)

def evaluate_traditional_methods(df):
    y_true_coarse = df["fault_type_label"].map(unify_fault); y_true_fine = df["fault_type_label"].map(normalize_fault); rows = []
    for method in cfg.DIAGNOSTIC_METHODS:
        column = cfg.DIAGNOSTIC_METHOD_TO_COLUMN[method]
        if column not in df.columns: continue
        pred = df[column].map(normalize_fault)
        rows.append({"method": method, "level": "coarse", **evaluate_predictions(y_true_coarse, pred.map(unify_fault), allowed_labels=cfg.COARSE_FAULT_GROUPS, allow_abstain=False)})
        rows.append({"method": method, "level": "fine", **evaluate_predictions(y_true_fine, pred, allowed_labels=cfg.BENCHMARK_FINE_CLASSES, allow_abstain=False)})
    return pd.DataFrame(rows)

def evaluate_traditional_combinations(df):
    y_true_coarse = df["fault_type_label"].map(unify_fault); y_true_fine = df["fault_type_label"].map(normalize_fault); rows = []
    for size in range(1, len(cfg.DIAGNOSTIC_METHODS) + 1):
        for combination in itertools.combinations(cfg.DIAGNOSTIC_METHODS, size):
            pred = aggregate_method_combination(df, combination)
            rows.append({"combination": " + ".join(combination), "n_methods": size, "level": "coarse", **evaluate_predictions(y_true_coarse, pred.map(unify_fault), allowed_labels=cfg.COARSE_FAULT_GROUPS, allow_abstain=False)})
            rows.append({"combination": " + ".join(combination), "n_methods": size, "level": "fine", **evaluate_predictions(y_true_fine, pred, allowed_labels=cfg.BENCHMARK_FINE_CLASSES, allow_abstain=False)})
    result = pd.DataFrame(rows)
    if result.empty: return result
    return result.sort_values(["level", "macro_f1", "balanced_accuracy", "coverage"], ascending=[True, False, False, False], na_position="last").reset_index(drop=True)

def empirical_ppm_coverage(df):
    rows = []
    for method in cfg.DIAGNOSTIC_METHODS:
        column = cfg.DIAGNOSTIC_METHOD_TO_COLUMN[method]
        if column not in df.columns: continue
        labels = df[column].map(normalize_fault); active = labels != "ABSTAIN"
        for gas in cfg.COMMON_BENCHMARK_GASES:
            values = pd.to_numeric(df.loc[active, gas], errors="coerce").dropna()
            if values.empty: continue
            rows.append({"method": method, "gas": gas, "active_n": int(len(values)), "coverage_percent": float(active.mean() * 100.0), "min_ppm": float(values.min()), "p05_ppm": float(values.quantile(0.05)), "median_ppm": float(values.median()), "p95_ppm": float(values.quantile(0.95)), "max_ppm": float(values.max())})
    return pd.DataFrame(rows)

def evaluate_weak_transfer(benchmark_df, weak_model, weak_groups, weak_methods, granularity="coarse"):
    from weak_supervision import build_label_matrix
    method_mapping = {method: cfg.DIAGNOSTIC_METHOD_TO_COLUMN[method] for method in weak_methods}
    L, _, _ = build_label_matrix(benchmark_df, method_mapping, weak_groups, granularity)
    probabilities = weak_model.predict_proba(L); labels = np.asarray([weak_groups[int(index)] for index in probabilities.argmax(axis=1)], dtype=object)
    active_count = (L != cfg.WEAK_ABSTAIN_LABEL).sum(axis=1); labels[active_count == 0] = "ABSTAIN"
    if granularity == "coarse": y_true = benchmark_df["fault_group_label"].astype(str); y_pred = pd.Series(labels).map(unify_fault); allowed = cfg.COARSE_FAULT_GROUPS
    elif granularity == "fine": y_true = benchmark_df["fault_type_label"].astype(str); y_pred = pd.Series(labels); allowed = cfg.BENCHMARK_FINE_CLASSES
    else: raise ValueError("granularity must be 'coarse' or 'fine'.")
    metrics = evaluate_predictions(y_true, y_pred, allowed_labels=allowed, allow_abstain=False)
    return {**metrics, "predictions": labels, "posterior_max": probabilities.max(axis=1)}

def bootstrap_metric_ci(y_true, y_pred, allowed_labels, metric="macro_f1", iterations=None, random_state=None):
    iterations = cfg.BOOTSTRAP_ITERATIONS if iterations is None else int(iterations)
    rng = np.random.default_rng(cfg.RANDOM_STATE if random_state is None else int(random_state))
    truth = np.asarray(y_true); pred = np.asarray(y_pred)
    if len(truth) == 0: return np.nan, np.nan
    values = []; labels = list(allowed_labels)
    for _ in range(iterations):
        indices = rng.integers(0, len(truth), size=len(truth)); sample_true = truth[indices]; sample_pred = pred[indices]
        if metric == "accuracy": value = accuracy_score(sample_true, sample_pred)
        elif metric == "balanced_accuracy": value = _safe_balanced_accuracy(sample_true, sample_pred)
        elif metric == "macro_f1": value = f1_score(sample_true, sample_pred, labels=labels, average="macro", zero_division=0)
        elif metric == "macro_precision": value = precision_score(sample_true, sample_pred, labels=labels, average="macro", zero_division=0)
        elif metric == "macro_recall": value = recall_score(sample_true, sample_pred, labels=labels, average="macro", zero_division=0)
        else: raise ValueError(f"Unknown bootstrap metric: {metric}")
        values.append(float(value))
    alpha = (1.0 - cfg.BOOTSTRAP_CONFIDENCE) / 2.0
    return float(np.quantile(values, alpha)), float(np.quantile(values, 1.0 - alpha))