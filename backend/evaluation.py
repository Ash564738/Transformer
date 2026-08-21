# evaluation.py
from __future__ import annotations

import itertools
import logging
from typing import Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import StratifiedGroupKFold, StratifiedShuffleSplit

from config import config as cfg
from consensus import normalize_fault, unify_fault

logger = logging.getLogger(__name__)


def harmonize_benchmark_labels(series: pd.Series) -> pd.Series:
    return series.map(normalize_fault)


def map_to_coarse(series: pd.Series) -> pd.Series:
    return series.map(unify_fault)


def detect_benchmark_label_column(df: pd.DataFrame) -> str:
    for column in ("fault_type_label", "label", "type"):
        if column in df.columns:
            return column
    raise ValueError("Cannot find benchmark label column.")


def prepare_benchmark_dataframe(df: pd.DataFrame, source_name: str) -> pd.DataFrame:
    out = df.copy()
    label_column = detect_benchmark_label_column(out)
    out["benchmark_source"] = source_name
    out["fault_type_label"] = out[label_column].map(normalize_fault)
    out["fault_group_label"] = out["fault_type_label"].map(unify_fault)
    return out


def combine_labeled_benchmarks(dga_df: pd.DataFrame, iec_df: pd.DataFrame) -> pd.DataFrame:
    dga = prepare_benchmark_dataframe(dga_df, "DGA_dataset")
    iec = prepare_benchmark_dataframe(iec_df, "IEC_TC10")
    required = list(cfg.COMMON_BENCHMARK_GASES)
    for name, frame in (("DGA_dataset", dga), ("IEC_TC10", iec)):
        missing = [gas for gas in required if gas not in frame.columns]
        if missing:
            raise ValueError(f"{name}: missing gases {missing}")
    combined = pd.concat([dga, iec], ignore_index=True)
    dedup_cols = ["benchmark_source", *cfg.COMMON_BENCHMARK_GASES, "fault_type_label"]
    dedup_cols = [c for c in dedup_cols if c in combined.columns]
    return combined.drop_duplicates(subset=dedup_cols).reset_index(drop=True)


def _safe_stratified_split(indices_df, y, groups, test_size, seed):
    y = np.asarray(y)
    groups = np.asarray(groups).astype(str)
    if len(np.unique(groups)) >= 3:
        n_splits = min(5, len(np.unique(groups)))
        splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        try:
            folds = list(splitter.split(indices_df, y, groups))
            if folds:
                return folds[0]
        except ValueError:
            pass
    splitter = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    return next(splitter.split(indices_df, y))


def split_benchmark(df: pd.DataFrame):
    work = df.copy().reset_index(drop=True)
    if "evaluation_group" not in work.columns:
        work["evaluation_group"] = (
            work[list(cfg.COMMON_BENCHMARK_GASES)].round(8).fillna(-999999.0).astype(str).agg("|".join, axis=1)
        )
    y = work["fault_group_label"].astype(str).to_numpy()
    groups = work["evaluation_group"].astype(str).to_numpy()
    train_dev_idx, test_idx = _safe_stratified_split(work, y, groups, cfg.TEST_SIZE, cfg.RANDOM_STATE)
    train_dev = work.iloc[train_dev_idx].reset_index(drop=True)
    y_dev = train_dev["fault_group_label"].astype(str).to_numpy()
    groups_dev = train_dev["evaluation_group"].astype(str).to_numpy()
    tr_rel, dev_rel = _safe_stratified_split(train_dev, y_dev, groups_dev, cfg.DEV_SIZE / (1 - cfg.TEST_SIZE), cfg.RANDOM_STATE + 1)
    return (
        train_dev.iloc[tr_rel].reset_index(drop=True),
        train_dev.iloc[dev_rel].reset_index(drop=True),
        work.iloc[test_idx].reset_index(drop=True),
    )


def aggregate_method_combination(df: pd.DataFrame, methods: Sequence[str]) -> pd.Series:
    predictions = []
    for _, row in df.iterrows():
        labels = []
        for method in methods:
            value = normalize_fault(row.get(cfg.DIAGNOSTIC_METHOD_TO_COLUMN[method], "ABSTAIN"))
            if value != "ABSTAIN":
                labels.append(value)
        if not labels:
            predictions.append("ABSTAIN")
            continue
        coarse = [unify_fault(label) for label in labels]
        counts = pd.Series(coarse).value_counts()
        top = counts[counts == counts.max()].index.tolist()
        if len(top) != 1:
            predictions.append("ABSTAIN")
            continue
        winning_group = top[0]
        fine = [label for label in labels if unify_fault(label) == winning_group]
        fine_counts = pd.Series(fine).value_counts()
        top_fine = fine_counts[fine_counts == fine_counts.max()].index.tolist()
        predictions.append(top_fine[0] if len(top_fine) == 1 else "ABSTAIN")
    return pd.Series(predictions, index=df.index)


def _clean_metric_labels(y_true, y_pred):
    truth = pd.Series(y_true).reset_index(drop=True).astype(str).str.upper().str.strip()
    pred = pd.Series(y_pred).reset_index(drop=True).astype(str).str.upper().str.strip()
    return truth, pred


def _safe_balanced_accuracy(y_true, y_pred):
    if len(y_true) == 0:
        return np.nan
    labels = sorted(set(y_true))
    recalls = recall_score(y_true, y_pred, labels=labels, average=None, zero_division=0)
    return float(np.mean(recalls)) if len(recalls) else np.nan


def evaluate_predictions(y_true, y_pred, allowed_labels=None, allow_abstain=False):
    truth, pred = _clean_metric_labels(y_true, y_pred)
    if allowed_labels is None:
        allowed_labels = sorted((set(truth) | set(pred)) - {"ABSTAIN"})
    else:
        allowed_labels = [str(x).upper() for x in allowed_labels]
    allowed = set(allowed_labels)
    valid_truth = truth.isin(allowed)
    truth = truth.loc[valid_truth].reset_index(drop=True)
    pred = pred.loc[valid_truth].reset_index(drop=True)
    n = len(truth)
    if n == 0:
        return {"accuracy": np.nan, "balanced_accuracy": np.nan, "macro_precision": np.nan, "macro_recall": np.nan, "macro_f1": np.nan, "weighted_f1": np.nan, "coverage": 0.0, "selective_accuracy": np.nan, "overall_accuracy_with_abstain_error": 0.0, "n": 0, "n_evaluated": 0}
    active = np.ones(n, dtype=bool) if allow_abstain else (pred.to_numpy() != "ABSTAIN")
    coverage = float(active.mean())
    if not active.any():
        return {"accuracy": 0.0, "balanced_accuracy": 0.0, "macro_precision": 0.0, "macro_recall": 0.0, "macro_f1": 0.0, "weighted_f1": 0.0, "coverage": coverage, "selective_accuracy": np.nan, "overall_accuracy_with_abstain_error": 0.0, "n": n, "n_evaluated": 0}
    yt = truth.loc[active].reset_index(drop=True)
    yp = pred.loc[active].reset_index(drop=True)
    valid_prediction = yp.isin(allowed)
    yt_valid = yt.loc[valid_prediction].reset_index(drop=True)
    yp_valid = yp.loc[valid_prediction].reset_index(drop=True)
    if len(yt_valid) == 0:
        return {"accuracy": 0.0, "balanced_accuracy": 0.0, "macro_precision": 0.0, "macro_recall": 0.0, "macro_f1": 0.0, "weighted_f1": 0.0, "coverage": coverage, "selective_accuracy": 0.0, "overall_accuracy_with_abstain_error": 0.0, "n": n, "n_evaluated": int(len(yt))}
    correct = int((yt_valid == yp_valid).sum())
    return {"accuracy": float(correct / n), "balanced_accuracy": _safe_balanced_accuracy(yt_valid, yp_valid), "macro_precision": float(precision_score(yt_valid, yp_valid, labels=allowed_labels, average="macro", zero_division=0)), "macro_recall": float(recall_score(yt_valid, yp_valid, labels=allowed_labels, average="macro", zero_division=0)), "macro_f1": float(f1_score(yt_valid, yp_valid, labels=allowed_labels, average="macro", zero_division=0)), "weighted_f1": float(f1_score(yt_valid, yp_valid, labels=allowed_labels, average="weighted", zero_division=0)), "coverage": coverage, "selective_accuracy": float(correct / max(len(yt_valid), 1)), "overall_accuracy_with_abstain_error": float(correct / n), "n": n, "n_evaluated": int(len(yt))}


def evaluate_ambiguous_fine_predictions(y_true, y_pred):
    """Evaluate fine labels while treating T1_T2 as a set-valued truth: T1 or T2 is accepted."""
    truth = pd.Series(y_true).reset_index(drop=True).map(normalize_fault)
    pred = pd.Series(y_pred).reset_index(drop=True).map(normalize_fault)
    allowed_truth = set(cfg.BENCHMARK_FINE_CLASSES) | set(cfg.BENCHMARK_AMBIGUOUS_FINE_CLASSES)
    valid = truth.isin(allowed_truth)
    truth = truth.loc[valid].reset_index(drop=True)
    pred = pred.loc[valid].reset_index(drop=True)
    accepted = []
    active = pred != "ABSTAIN"
    for t, p in zip(truth, pred):
        if p == "ABSTAIN": accepted.append(False)
        elif t in cfg.BENCHMARK_AMBIGUOUS_ACCEPTED_PREDICTIONS: accepted.append(p in cfg.BENCHMARK_AMBIGUOUS_ACCEPTED_PREDICTIONS[t])
        else: accepted.append(t == p)
    accepted = np.asarray(accepted, dtype=bool)
    coverage = float(active.mean()) if len(active) else 0.0
    evaluated = active & pred.isin(set(cfg.BENCHMARK_FINE_CLASSES) | set(cfg.BENCHMARK_AMBIGUOUS_FINE_CLASSES))
    n_eval = int(evaluated.sum())
    correct = int(accepted[evaluated].sum()) if n_eval else 0
    return {"accuracy": float(correct / len(truth)) if len(truth) else np.nan, "coverage": coverage, "selective_accuracy": float(correct / n_eval) if n_eval else np.nan, "overall_accuracy_with_abstain_error": float(correct / len(truth)) if len(truth) else np.nan, "n": int(len(truth)), "n_evaluated": n_eval, "ambiguous_truth_count": int((truth == "T1_T2").sum()), "ambiguous_truth_correct_count": int(accepted[truth.eq("T1_T2")].sum()) if (truth == "T1_T2").any() else 0}


def generate_method_combinations(methods):
    methods = list(methods)
    for size in range(1, len(methods) + 1):
        yield from itertools.combinations(methods, size)


def evaluate_traditional_methods(df):
    y_true_coarse = df["fault_type_label"].map(unify_fault)
    y_true_fine = df["fault_type_label"].map(normalize_fault)
    rows = []
    for method in cfg.DIAGNOSTIC_METHODS:
        column = cfg.DIAGNOSTIC_METHOD_TO_COLUMN[method]
        pred = df.get(column, pd.Series("ABSTAIN", index=df.index)).map(normalize_fault)
        rows.append({"method": method, "level": "coarse", **evaluate_predictions(y_true_coarse, pred.map(unify_fault), cfg.COARSE_FAULT_GROUPS)})
        rows.append({"method": method, "level": "fine", **evaluate_predictions(y_true_fine, pred, cfg.BENCHMARK_FINE_CLASSES)})
    return pd.DataFrame(rows)


def _evaluation_valid_mask(df):
    fine = df["fault_type_label"].map(normalize_fault)
    coarse = fine.map(unify_fault)
    conflict = df.get("fine_label_conflict", pd.Series(False, index=df.index)).astype(bool)
    coarse_conflict = df.get("coarse_label_conflict", pd.Series(False, index=df.index)).astype(bool)
    valid_fine = fine.isin(cfg.BENCHMARK_FINE_CLASSES) & ~conflict
    valid_coarse = coarse.isin(cfg.COARSE_FAULT_GROUPS) & ~coarse_conflict
    return fine, coarse, valid_fine, valid_coarse


def make_locked_splits(labeled_df, seed=None):
    seed = cfg.RANDOM_STATE if seed is None else int(seed)
    fine, coarse, valid_fine, valid_coarse = _evaluation_valid_mask(labeled_df)
    valid = valid_fine & valid_coarse
    data = labeled_df.loc[valid].reset_index(drop=True).copy()
    data["evaluation_group"] = data.get("evaluation_group", data[list(cfg.COMMON_BENCHMARK_GASES)].round(8).fillna(-999999.0).astype(str).agg("|".join, axis=1)).astype(str)
    y = fine.loc[valid].map({c: i for i, c in enumerate(cfg.BENCHMARK_FINE_CLASSES)}).astype(int).to_numpy()
    groups = data["evaluation_group"].to_numpy()
    if len(np.unique(groups)) >= 3:
        splitter = StratifiedGroupKFold(n_splits=min(5, len(np.unique(groups))), shuffle=True, random_state=seed)
        folds = list(splitter.split(data, y, groups))
        train_dev_idx, test_idx = folds[0]
        remain = data.iloc[train_dev_idx].reset_index(drop=True)
        y_remain = y[train_dev_idx]
        g_remain = groups[train_dev_idx]
        splitter2 = StratifiedGroupKFold(n_splits=min(4, len(np.unique(g_remain))), shuffle=True, random_state=seed + 1)
        tr_rel, dev_rel = list(splitter2.split(remain, y_remain, g_remain))[0]
        train_idx = train_dev_idx[tr_rel]
        dev_idx = train_dev_idx[dev_rel]
    else:
        split = StratifiedShuffleSplit(n_splits=1, test_size=cfg.TEST_SIZE, random_state=seed)
        train_dev_idx, test_idx = next(split.split(data, y))
        split2 = StratifiedShuffleSplit(n_splits=1, test_size=cfg.DEV_SIZE / (1 - cfg.TEST_SIZE), random_state=seed + 1)
        tr_rel, dev_rel = next(split2.split(data.iloc[train_dev_idx], y[train_dev_idx]))
        train_idx = train_dev_idx[tr_rel]
        dev_idx = train_dev_idx[dev_rel]
    manifest = pd.DataFrame({"source_dataset": data["source_dataset"].to_numpy(), "source_row": data["source_row"].to_numpy(), "fault_type_label": data["fault_type_label"].to_numpy(), "evaluation_group": groups, "split": "train"})
    manifest.loc[dev_idx, "split"] = "development"
    manifest.loc[test_idx, "split"] = "locked_test"
    return data, y, train_idx, dev_idx, test_idx, manifest


def evaluate_traditional_combinations_with_locked_test(labeled_df, seed=None):
    from consensus import apply_consensus_from_existing_diagnostics
    data, y, train_idx, dev_idx, test_idx, manifest = make_locked_splits(labeled_df, seed)
    fine_truth = data["fault_type_label"].map(normalize_fault)
    coarse_truth = fine_truth.map(unify_fault)
    rows = []
    for combo in generate_method_combinations(cfg.DIAGNOSTIC_METHODS):
        combo_text = "+".join(combo)
        predicted = apply_consensus_from_existing_diagnostics(labeled_df, combo)
        pred = predicted.loc[data.index]
        for split_name, idx in (("development", dev_idx), ("locked_test", test_idx)):
            fine_metric = evaluate_predictions(fine_truth.iloc[idx], pred["consensus_fault"].iloc[idx], cfg.BENCHMARK_FINE_CLASSES)
            coarse_metric = evaluate_predictions(coarse_truth.iloc[idx], pred["consensus_fault_group"].iloc[idx], cfg.COARSE_FAULT_GROUPS)
            rows.append({"methods": combo_text, "method_count": len(combo), "split": split_name, "granularity": "fine", **fine_metric})
            rows.append({"methods": combo_text, "method_count": len(combo), "split": split_name, "granularity": "coarse", **coarse_metric})
    result = pd.DataFrame(rows)
    selected = []
    for granularity in ("fine", "coarse"):
        dev = result[(result["granularity"] == granularity) & (result["split"] == "development")].copy()
        if not dev.empty:
            best = dev.sort_values(["macro_f1", "balanced_accuracy", "coverage"], ascending=False, na_position="last").iloc[0]
            selected.append((granularity, best["methods"], int(best["method_count"])))
    result["selected_on_development"] = False
    for granularity, methods, count in selected:
        result.loc[(result["granularity"] == granularity) & (result["methods"] == methods), "selected_on_development"] = True
    result = result.sort_values(["granularity", "split", "macro_f1"], ascending=[True, True, False], na_position="last").reset_index(drop=True)
    return result, manifest


def empirical_ppm_coverage(df):
    rows = []
    for method in cfg.DIAGNOSTIC_METHODS:
        column = cfg.DIAGNOSTIC_METHOD_TO_COLUMN[method]
        labels = df.get(column, pd.Series("ABSTAIN", index=df.index)).map(normalize_fault)
        active = labels != "ABSTAIN"
        for gas in cfg.COMMON_BENCHMARK_GASES:
            values = pd.to_numeric(df.loc[active, gas], errors="coerce").dropna()
            if values.empty:
                continue
            rows.append({"method": method, "gas": gas, "active_n": int(len(values)), "coverage_percent": float(active.mean() * 100.0), "min_ppm": float(values.min()), "p05_ppm": float(values.quantile(0.05)), "median_ppm": float(values.median()), "p95_ppm": float(values.quantile(0.95)), "max_ppm": float(values.max())})
    return pd.DataFrame(rows)


def empirical_fault_class_coverage(df, level="fine"):
    """Report LF activation coverage by benchmark class; this is descriptive coverage, not accuracy."""
    labels = df["fault_type_label"].map(normalize_fault)
    if level == "coarse": labels = labels.map(unify_fault); allowed = cfg.COARSE_FAULT_GROUPS
    else: allowed = list(cfg.BENCHMARK_FINE_CLASSES) + list(cfg.BENCHMARK_AMBIGUOUS_FINE_CLASSES)
    rows = []
    for method in cfg.DIAGNOSTIC_METHODS:
        pred = df.get(cfg.DIAGNOSTIC_METHOD_TO_COLUMN[method], pd.Series("ABSTAIN", index=df.index)).map(normalize_fault)
        if level == "coarse": pred = pred.map(unify_fault)
        for cls in allowed:
            mask = labels == cls
            active = pred != "ABSTAIN"
            n = int(mask.sum())
            rows.append({"method": method, "granularity": level, "fault_class": cls, "class_n": n, "active_n": int((mask & active).sum()), "class_coverage_percent": float(100.0 * (mask & active).sum() / n) if n else np.nan})
    return pd.DataFrame(rows)


def evaluate_weak_transfer(benchmark_df, weak_model, weak_groups, weak_methods, granularity="coarse"):
    from weak_supervision import build_label_matrix
    mapping = {method: cfg.DIAGNOSTIC_METHOD_TO_COLUMN[method] for method in weak_methods}
    L, _, _ = build_label_matrix(benchmark_df, mapping, weak_groups, granularity)
    probabilities = weak_model.predict_proba(L)
    labels = np.asarray([weak_groups[int(i)] for i in probabilities.argmax(axis=1)], dtype=object)
    labels[(L != cfg.WEAK_ABSTAIN_LABEL).sum(axis=1) == 0] = "ABSTAIN"
    if granularity == "coarse":
        y_true = benchmark_df["fault_group_label"].astype(str)
        y_pred = pd.Series(labels).map(unify_fault)
        allowed = cfg.COARSE_FAULT_GROUPS
    elif granularity == "fine":
        y_true = benchmark_df["fault_type_label"].astype(str)
        y_pred = pd.Series(labels)
        allowed = cfg.BENCHMARK_FINE_CLASSES
    else:
        raise ValueError("granularity must be coarse or fine")
    metrics = evaluate_predictions(y_true, y_pred, allowed_labels=allowed)
    return {**metrics, "predictions": labels, "posterior_max": probabilities.max(axis=1)}


def bootstrap_metric_ci(y_true, y_pred, allowed_labels, metric="macro_f1", iterations=None, random_state=None):
    iterations = cfg.BOOTSTRAP_ITERATIONS if iterations is None else int(iterations)
    rng = np.random.default_rng(cfg.RANDOM_STATE if random_state is None else int(random_state))
    truth = np.asarray(y_true)
    pred = np.asarray(y_pred)
    if len(truth) == 0:
        return np.nan, np.nan
    values = []
    labels = list(allowed_labels)
    for _ in range(iterations):
        indices = rng.integers(0, len(truth), size=len(truth))
        yt = truth[indices]
        yp = pred[indices]
        if metric == "accuracy":
            value = accuracy_score(yt, yp)
        elif metric == "balanced_accuracy":
            value = _safe_balanced_accuracy(yt, yp)
        elif metric == "macro_f1":
            value = f1_score(yt, yp, labels=labels, average="macro", zero_division=0)
        elif metric == "macro_precision":
            value = precision_score(yt, yp, labels=labels, average="macro", zero_division=0)
        elif metric == "macro_recall":
            value = recall_score(yt, yp, labels=labels, average="macro", zero_division=0)
        else:
            raise ValueError(f"Unknown bootstrap metric: {metric}")
        values.append(float(value))
    alpha = (1 - cfg.BOOTSTRAP_CONFIDENCE) / 2
    return float(np.quantile(values, alpha)), float(np.quantile(values, 1 - alpha))