# experiment.py
from __future__ import annotations
import json, logging, time
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import LabelEncoder
from anomaly import UnsupervisedEnsemble
from evaluation import (
    precision_recall_lift, topk_stability, rank_correlation,
    temporal_consistency, gas_increase_consistency, bootstrap_confidence_interval,
)
from config import DATASET_DIR, REPORT_DIR
from weak_supervision import _normalize_vote

logger = logging.getLogger(__name__)

GAS_COLS = ["h2", "ch4", "c2h6", "c2h4", "c2h2", "co", "co2"]
GAS_COLUMN_INDICES = list(range(len(GAS_COLS)))
TOP_K_FRACTION = 0.10
SECONDARY_K_VALUES = [20, 50, 100]
OUTER_FOLDS = 5
N_BOOTSTRAP = 1000
N_PERTURB = 200
TRADITIONAL_METHODS = [
    "keygas_fault", "iec_fault", "rogers_fault",
    "doernenburg_fault", "duval_triangle_fault", "duval_pentagon_fault",
]
OUTPUT_DIR = Path(REPORT_DIR)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def _histogram_data(series, bins=30):
    clean = pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    if clean.empty:
        return {"bin_centers": [], "counts": [], "bin_edges": []}
    counts, edges = np.histogram(clean, bins=bins)
    centers = (edges[:-1] + edges[1:]) / 2.0
    return {"bin_centers": centers.tolist(), "counts": counts.tolist(), "bin_edges": edges.tolist()}

def _fold_top_k(n_test: int) -> int:
    if n_test <= 0:
        return 1
    return int(np.clip(np.ceil(TOP_K_FRACTION * n_test), 1, n_test))

def _effective_ks(n_test: int):
    ks = {_fold_top_k(n_test)}
    for k in SECONDARY_K_VALUES:
        if n_test > 0:
            ks.add(min(k, n_test))
    return sorted(ks)

def section_3_1_exploratory_analysis(df):
    logger.info("3.1: exploratory analysis.")
    existing_gases = [gas for gas in GAS_COLS if gas in df.columns]
    if existing_gases:
        nplots = len(existing_gases)
        nrows = int(np.ceil(nplots / 3))
        fig, axes = plt.subplots(nrows, 3, figsize=(14, 4 * nrows))
        axes = np.atleast_1d(axes).flatten()
        for ax, gas in zip(axes, existing_gases):
            values = pd.to_numeric(df[gas], errors="coerce").clip(lower=0)
            ax.hist(np.log1p(values.dropna()), bins=50, color="teal", edgecolor="white", alpha=0.7)
            ax.set_title(f"{gas.upper()} log1p")
        for ax in axes[nplots:]:
            ax.axis("off")
        fig.tight_layout()
        fig.savefig(OUTPUT_DIR / "section_3_1_gas_distributions.png", dpi=150)
        plt.close(fig)
        corr = df[existing_gases].apply(pd.to_numeric, errors="coerce").corr()
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", ax=ax)
        ax.set_title("Gas correlation matrix")
        fig.tight_layout()
        fig.savefig(OUTPUT_DIR / "section_3_1_correlation_heatmap.png", dpi=150)
        plt.close(fig)
    cols = [c for c in [*existing_gases, "tdcg", "water", "temp"] if c in df.columns]
    if cols:
        missing = (df[cols].isna().mean().mul(100).rename("missing_ratio").reset_index().rename(columns={"index": "column"}))
        missing.to_csv(OUTPUT_DIR / "section_3_1_missing_summary.csv", index=False)

def generate_section_3_1_data(df):
    data = {"gas_distributions": {}, "correlation_matrix": {}, "missing_summary": {}}
    existing_gases = [gas for gas in GAS_COLS if gas in df.columns]
    for gas in existing_gases:
        data["gas_distributions"][gas] = _histogram_data(df[gas], bins=50)
    if existing_gases:
        corr = df[existing_gases].apply(pd.to_numeric, errors="coerce").corr()
        data["correlation_matrix"] = {"columns": existing_gases, "values": corr.values.tolist()}
    cols = [c for c in [*existing_gases, "tdcg", "water", "temp"] if c in df.columns]
    if cols:
        missing = df[cols].isna().mean().mul(100)
        data["missing_summary"] = {"columns": missing.index.tolist(), "missing_ratio": missing.round(2).tolist()}
    with open(OUTPUT_DIR / "section_3_1_data.json", "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

def section_3_2_label_distribution(df):
    label_counts = {}
    for method in TRADITIONAL_METHODS:
        if method not in df.columns:
            continue
        values = df[method].fillna("ABSTAIN").astype(str).str.upper()
        values = values[~values.isin(["NORMAL", "ABSTAIN", ""])]
        if values.empty:
            continue
        counts = values.value_counts()
        label_counts[method] = counts.to_dict()
        fig, ax = plt.subplots(figsize=(8, 4))
        counts.plot(kind="bar", ax=ax, color="darkorange")
        ax.set_title(f"Fault labels assigned by {method}")
        ax.set_ylabel("Samples")
        ax.tick_params(axis="x", rotation=45)
        fig.tight_layout()
        fig.savefig(OUTPUT_DIR / f"section_3_2_label_dist_{method}.png", dpi=150)
        plt.close(fig)
    with open(OUTPUT_DIR / "section_3_2_data.json", "w", encoding="utf-8") as f:
        json.dump(label_counts, f, indent=2)

def section_3_3_supervised_plots(supervised_results):
    rows = []
    for method, models in supervised_results.items():
        for model_name, metrics in models.items():
            rows.append({"labeling_method": method, "model": model_name, "accuracy": metrics.get("accuracy"), "f1_macro": metrics.get("f1_macro")})
    results = pd.DataFrame(rows)
    if results.empty:
        return
    results.to_csv(OUTPUT_DIR / "table_3_3_supervised.csv", index=False)
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.barplot(data=results, x="labeling_method", y="accuracy", hue="model", ax=ax)
    ax.set_title("Supervised label-imitation accuracy")
    ax.set_ylabel("Accuracy")
    ax.tick_params(axis="x", rotation=45)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "section_3_3_accuracy_comparison.png", dpi=150)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.barplot(data=results, x="labeling_method", y="f1_macro", hue="model", ax=ax)
    ax.set_title("Supervised label-imitation macro F1")
    ax.set_ylabel("Macro F1")
    ax.tick_params(axis="x", rotation=45)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "section_3_3_f1_comparison.png", dpi=150)
    plt.close(fig)

def section_3_4_risk_score_analysis(df):
    if "severity_score" not in df.columns:
        return
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(pd.to_numeric(df["severity_score"], errors="coerce").dropna(), bins=50, color="navy", alpha=0.7)
    ax.set_title("Severity score distribution")
    ax.set_xlabel("Severity score (0-100)")
    ax.set_ylabel("Samples")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "section_3_4_risk_score_distribution.png", dpi=150)
    plt.close(fig)
    severity_col = "severity_label_text" if "severity_label_text" in df.columns else ("severity_label" if "severity_label" in df.columns else None)
    if severity_col:
        counts = df[severity_col].value_counts()
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.pie(counts, labels=counts.index, autopct="%1.1f%%", startangle=90)
        ax.set_title("DGA severity distribution")
        fig.tight_layout()
        fig.savefig(OUTPUT_DIR / "section_3_4_severity_pie.png", dpi=150)
        plt.close(fig)
    if "transformer_id" in df.columns:
        top5 = df.groupby("transformer_id")["severity_score"].max().nlargest(5)
        fig, ax = plt.subplots(figsize=(8, 4))
        top5.plot(kind="bar", ax=ax, color="firebrick")
        ax.set_title("Top 5 transformers by maximum severity score")
        ax.set_ylabel("Severity score (0-100)")
        fig.tight_layout()
        fig.savefig(OUTPUT_DIR / "section_3_4_top5_transformers.png", dpi=150)
        plt.close(fig)

def generate_section_3_4_data(df):
    data = {}
    if "severity_score" in df.columns:
        data["severity_histogram"] = _histogram_data(df["severity_score"], bins=30)
    severity_col = "severity_label_text" if "severity_label_text" in df.columns else ("severity_label" if "severity_label" in df.columns else None)
    if severity_col:
        counts = df[severity_col].value_counts().to_dict()
        data["severity_label_counts"] = counts
    if "transformer_id" in df.columns and "severity_score" in df.columns:
        top5 = df.groupby("transformer_id")["severity_score"].max().nlargest(5).reset_index()
        data["top5_transformers"] = top5.to_dict(orient="records")
    with open(OUTPUT_DIR / "section_3_4_data.json", "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

def load_data():
    path = DATASET_DIR / "processed" / "dga_unlabeled.parquet"
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")
    df = pd.read_parquet(path)
    gas_cols = [gas for gas in GAS_COLS if gas in df.columns]
    X = df[gas_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0).to_numpy(dtype=float)
    groups = df["transformer_id"].to_numpy()
    if "has_event" in df.columns:
        events = df["has_event"].astype(bool).to_numpy()
    else:
        events = np.zeros(len(df), dtype=bool)
    tdcg = pd.to_numeric(df.get("tdcg", pd.Series(0.0, index=df.index)), errors="coerce").fillna(0.0).to_numpy(dtype=float)
    sample_days = pd.to_datetime(df.get("sample_day", pd.Series(pd.NaT, index=df.index)), errors="coerce").to_numpy()
    method_flags = {}
    for method in TRADITIONAL_METHODS:
        if method not in df.columns:
            method_flags[method] = np.zeros(len(df), dtype=bool)
            continue
        normalized = df[method].fillna("ABSTAIN").astype(str).str.upper()
        method_flags[method] = (~normalized.isin(["NORMAL", "ABSTAIN", ""])).to_numpy()
    logger.info("Experiment data: %d samples, %d gas features, %d transformers, %d proxy events",
                len(df), X.shape[1], df["transformer_id"].nunique(), int(events.sum()))
    return X, groups, events, tdcg, sample_days, method_flags, df

def posthoc_validation(scores, method_flags, ks):
    order = np.argsort(np.nan_to_num(scores, nan=-np.inf))[::-1]
    result = {}
    for method, flags in method_flags.items():
        result[method] = {}
        for k in ks:
            k_eff = min(int(k), len(order))
            if k_eff <= 0:
                continue
            topk = order[:k_eff]
            count = int(flags[topk].sum())
            pct = count / k_eff * 100.0
            result[method][f"Top-{k_eff}"] = {"count": count, "pct": round(pct, 1)}
    return result

def run_anomaly_experiment():
    start_time = time.time()
    X, groups, events, tdcg, sample_days, method_flags, df = load_data()
    methods = ["tdcg", "iforest", "lof", "ocsvm", "autoencoder", "ensemble"]
    fold_results = {name: [] for name in methods}
    n_groups = len(np.unique(groups))
    n_splits = min(OUTER_FOLDS, n_groups)
    if n_splits < 2:
        raise ValueError("Need at least two transformer groups for GroupKFold.")
    cv = GroupKFold(n_splits=n_splits)
    for fold, (train_idx, test_idx) in enumerate(cv.split(X, groups=groups), start=1):
        logger.info("Anomaly fold %d/%d", fold, n_splits)
        X_train = X[train_idx]
        X_test = X[test_idx]
        events_test = events[test_idx]
        tdcg_test = tdcg[test_idx]
        groups_test = groups[test_idx]
        days_test = sample_days[test_idx]
        test_flags = {method: flags[test_idx] for method, flags in method_flags.items()}
        n_test = len(test_idx)
        top_k = _fold_top_k(n_test)
        ks = _effective_ks(n_test)

        max_tdcg = np.nanmax(tdcg_test) if len(tdcg_test) else 0.0
        if not np.isfinite(max_tdcg) or max_tdcg <= 0:
            scores_tdcg = np.zeros(len(tdcg_test))
        else:
            scores_tdcg = tdcg_test / max_tdcg
        precision, recall, lift = precision_recall_lift(scores_tdcg, events_test, top_k)
        fold_results["tdcg"].append({
            "precision": precision, "recall": recall, "lift": lift,
            "gic": 1.0, "stability": 1.0, "spearman": 1.0, "tc": 1.0,
            "top_k": top_k, "posthoc": posthoc_validation(scores_tdcg, test_flags, ks),
        })

        ensemble = UnsupervisedEnsemble()
        ensemble.fit(X_train, feature_names=GAS_COLS)
        scores_ensemble = ensemble.predict(X_test)
        precision, recall, lift = precision_recall_lift(scores_ensemble, events_test, top_k)
        gic = gas_increase_consistency(ensemble, X_test, GAS_COLUMN_INDICES, n_perturb=N_PERTURB)
        ensemble_alt = UnsupervisedEnsemble(random_state=123)
        ensemble_alt.fit(X_train, feature_names=GAS_COLS)
        scores_ensemble_alt = ensemble_alt.predict(X_test)
        stability = topk_stability(scores_ensemble, scores_ensemble_alt, top_k)
        spearman = rank_correlation(scores_ensemble, scores_ensemble_alt)
        tc = temporal_consistency(scores_ensemble, tdcg_test, groups_test, sample_days=days_test)
        fold_results["ensemble"].append({
            "precision": precision, "recall": recall, "lift": lift,
            "gic": gic, "stability": stability, "spearman": spearman, "tc": tc,
            "top_k": top_k, "posthoc": posthoc_validation(scores_ensemble, test_flags, ks),
        })

        for name in ["iforest", "lof", "ocsvm", "autoencoder"]:
            original_weights = ensemble.weights.copy()
            ensemble.weights = {detector_name: 1.0 if detector_name == name else 0.0 for detector_name in ensemble.detectors.keys()}
            scores_single = ensemble.predict(X_test)
            precision, recall, lift = precision_recall_lift(scores_single, events_test, top_k)
            gic_single = gas_increase_consistency(ensemble, X_test, GAS_COLUMN_INDICES, n_perturb=N_PERTURB)
            stability_single = topk_stability(scores_single, scores_ensemble, top_k)
            spearman_single = rank_correlation(scores_single, scores_ensemble)
            tc_single = temporal_consistency(scores_single, tdcg_test, groups_test, sample_days=days_test)
            fold_results[name].append({
                "precision": precision, "recall": recall, "lift": lift,
                "gic": gic_single, "stability": stability_single, "spearman": spearman_single, "tc": tc_single,
                "top_k": top_k, "posthoc": posthoc_validation(scores_single, test_flags, ks),
            })
            ensemble.weights = original_weights

        logger.info("Fold %d: ensemble P@%d=%.4f R@%.0f=%.4f Lift=%.2f",
                    fold, top_k, fold_results["ensemble"][-1]["precision"],
                    top_k, fold_results["ensemble"][-1]["recall"], fold_results["ensemble"][-1]["lift"])

    summary = {}
    for method, values in fold_results.items():
        if not values:
            continue
        precision_values = np.array([item["precision"] for item in values], dtype=float)
        recall_values = np.array([item["recall"] for item in values], dtype=float)
        lift_values = np.array([item["lift"] for item in values], dtype=float)
        gic_values = np.array([item["gic"] for item in values], dtype=float)
        stability_values = np.array([item["stability"] for item in values], dtype=float)
        spearman_values = np.array([item["spearman"] for item in values], dtype=float)
        tc_values = np.array([item["tc"] for item in values], dtype=float)
        _, precision_ci_low, precision_ci_high = bootstrap_confidence_interval(
            precision_values, np.mean, n_bootstrap=N_BOOTSTRAP)

        aggregated_posthoc = {}
        for method_flag in TRADITIONAL_METHODS:
            aggregated_posthoc[method_flag] = {}
            for k in SECONDARY_K_VALUES:
                key = f"Top-{k}"
                percentages = []
                for item in values:
                    posthoc = item["posthoc"].get(method_flag, {})
                    if key in posthoc:
                        percentages.append(posthoc[key]["pct"])
                aggregated_posthoc[method_flag][key] = round(float(np.mean(percentages)), 1) if percentages else None

        summary[method] = {
            "Precision@Top10Percent": float(np.mean(precision_values)),
            "Recall@Top10Percent": float(np.mean(recall_values)),
            "Lift@Top10Percent": float(np.mean(lift_values)),
            "GIC": float(np.mean(gic_values)),
            "Stability": float(np.mean(stability_values)),
            "Spearman": float(np.mean(spearman_values)),
            "TC": float(np.mean(tc_values)),
            "Precision_CI_low": float(precision_ci_low),
            "Precision_CI_high": float(precision_ci_high),
            "Posthoc_proxy_agreement": aggregated_posthoc,
            "note": "has_event and traditional diagnostic labels are proxy/reference signals, not independent ground truth.",
        }

    output_path = OUTPUT_DIR / "experiment_results_full.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    logger.info("Anomaly experiment saved: %s", output_path)
    return df

XGB_AVAILABLE = False
CATBOOST_AVAILABLE = False
SMOTE_AVAILABLE = False

try:
    from xgboost import XGBClassifier
    XGB_AVAILABLE = True
except ImportError:
    pass

try:
    from catboost import CatBoostClassifier
    CATBOOST_AVAILABLE = True
except ImportError:
    pass

try:
    from imblearn.over_sampling import SMOTE
    SMOTE_AVAILABLE = True
except ImportError:
    pass

def prepare_dataset(method):
    df = pd.read_parquet(DATASET_DIR / "processed" / "dga_unlabeled.parquet")
    if method not in df.columns:
        raise ValueError(f"Missing labeling method: {method}")
    X = df[GAS_COLS].apply(pd.to_numeric, errors="coerce").fillna(0.0).to_numpy(dtype=np.float32)
    labels = [_normalize_vote(value) for value in df[method].fillna("ABSTAIN").astype(str)]
    labels = np.asarray(labels)
    mask = labels != "ABSTAIN"
    X = X[mask]
    labels = labels[mask]
    groups = df.loc[mask, "transformer_id"].to_numpy()
    encoder = LabelEncoder()
    y = encoder.fit_transform(labels)
    return X, y, groups, encoder

def _clone_estimator(model):
    from sklearn.base import clone
    return clone(model)

def cross_val_scores(model, X, y, groups, cv):
    accuracy_values = []
    f1_values = []
    for train_idx, test_idx in cv.split(X, y, groups):
        model_clone = _clone_estimator(model)
        model_clone.fit(X[train_idx], y[train_idx])
        prediction = model_clone.predict(X[test_idx])
        accuracy_values.append(accuracy_score(y[test_idx], prediction))
        f1_values.append(f1_score(y[test_idx], prediction, average="macro", zero_division=0))
    return float(np.mean(accuracy_values)), float(np.mean(f1_values))

def cross_val_scores_with_smote(model, X, y, groups, cv):
    if not SMOTE_AVAILABLE:
        return None, None
    accuracy_values = []
    f1_values = []
    for train_idx, test_idx in cv.split(X, y, groups):
        X_train = X[train_idx]
        y_train = y[train_idx]
        minimum_count = np.min(np.bincount(y_train))
        if minimum_count >= 2:
            k_neighbors = min(5, minimum_count - 1)
            sampler = SMOTE(random_state=42, k_neighbors=k_neighbors)
            X_resampled, y_resampled = sampler.fit_resample(X_train, y_train)
        else:
            X_resampled = X_train
            y_resampled = y_train
        model_clone = _clone_estimator(model)
        model_clone.fit(X_resampled, y_resampled)
        prediction = model_clone.predict(X[test_idx])
        accuracy_values.append(accuracy_score(y[test_idx], prediction))
        f1_values.append(f1_score(y[test_idx], prediction, average="macro", zero_division=0))
    return float(np.mean(accuracy_values)), float(np.mean(f1_values))

def run_supervised_experiment():
    start_time = time.time()
    all_results = {}
    for method in TRADITIONAL_METHODS:
        logger.info("Supervised experiment: %s", method)
        X, y, groups, encoder = prepare_dataset(method)
        n_groups = len(np.unique(groups))
        n_splits = min(5, n_groups)
        if n_splits < 2:
            logger.warning("Skipping %s: insufficient groups.", method)
            continue
        cv = GroupKFold(n_splits=n_splits)
        results = {}
        rf = RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1, class_weight="balanced")
        accuracy, macro_f1 = cross_val_scores(rf, X, y, groups, cv)
        results["RF"] = {"accuracy": accuracy, "f1_macro": macro_f1}
        if SMOTE_AVAILABLE:
            accuracy, macro_f1 = cross_val_scores_with_smote(rf, X, y, groups, cv)
            results["RF+SMOTE"] = {"accuracy": accuracy, "f1_macro": macro_f1}
        if XGB_AVAILABLE:
            xgb_model = XGBClassifier(eval_metric="mlogloss", random_state=42, n_estimators=300, max_depth=6, learning_rate=0.05)
            accuracy, macro_f1 = cross_val_scores(xgb_model, X, y, groups, cv)
            results["XGBoost"] = {"accuracy": accuracy, "f1_macro": macro_f1}
        if CATBOOST_AVAILABLE:
            cat_model = CatBoostClassifier(verbose=False, random_seed=42, iterations=300, depth=6, learning_rate=0.05)
            accuracy, macro_f1 = cross_val_scores(cat_model, X, y, groups, cv)
            results["CatBoost"] = {"accuracy": accuracy, "f1_macro": macro_f1}
        all_results[method] = results
    output_path = OUTPUT_DIR / "supervised_results.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)
    logger.info("Supervised results saved to %s", output_path)
    logger.info("Supervised experiment duration: %.1fs", time.time() - start_time)
    return all_results

def run_full_experiments():
    logger.info("Starting full experiment suite.")
    df = run_anomaly_experiment()
    supervised_results = run_supervised_experiment()
    section_3_1_exploratory_analysis(df)
    section_3_2_label_distribution(df)
    section_3_3_supervised_plots(supervised_results)
    section_3_4_risk_score_analysis(df)
    generate_section_3_1_data(df)
    generate_section_3_4_data(df)
    logger.info("All experiment artifacts generated.")
    return df, supervised_results

if __name__ == "__main__":
    run_full_experiments()