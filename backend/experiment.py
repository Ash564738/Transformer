# experiment.py
"""
Run complete experiment comparison:
- Unsupervised anomaly detection (Isolation Forest, LOF, OCSVM, Autoencoder, Ensemble)
- Supervised learning (RF, XGBoost, CatBoost) on each traditional labeling method
- Generate tables and charts for report sections 3.1–3.4

Outputs saved to:
- experiment_results_full.json (anomaly metrics)
- supervised_results.json (supervised metrics)
- reports/section_3_1_data.json, section_3_2_data.json, section_3_4_data.json (for frontend charts)
- reports/section_3_1_*.png, section_3_2_*.png, section_3_3_*.png, section_3_4_*.png (static images)
"""
import json
import logging
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from sklearn.model_selection import GroupKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, f1_score

from anomaly import UnsupervisedEnsemble
from evaluation import (
    precision_recall_lift, topk_stability, rank_correlation,
    temporal_consistency, gas_increase_consistency,
    bootstrap_confidence_interval
)
from config import DATASET_DIR, DATABASE_DIR, MODEL_DIR, REPORT_DIR
from weak_supervision import _normalize_vote

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(name)s: %(message)s', datefmt='%H:%M:%S')
logger = logging.getLogger(__name__)

# ──────────── Common configuration ────────────
GAS_COLS = ['h2','ch4','c2h6','c2h4','c2h2','co','co2']
TDCG_COLS_INDICES = [0,1,2,3,4,5]
K = int(0.1 * 4561)       # Top 10% (~456)
OUTER_FOLDS = 5
N_BOOTSTRAP = 1000
N_PERTURB = 200

TRADITIONAL_METHODS = [
    'keygas_fault',
    'iec_fault',
    'rogers_fault',
    'doernenburg_fault',
    'duval_triangle_fault',
    'duval_pentagon_fault',
]

OUTPUT_DIR = REPORT_DIR
OUTPUT_DIR.mkdir(exist_ok=True)

def _histogram_data(series, bins=30):
    counts, bin_edges = np.histogram(series.dropna(), bins=bins)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    return {
        "bin_centers": bin_centers.tolist(),
        "counts": counts.tolist(),
        "bin_edges": bin_edges.tolist(),
    }

# =============================================================================
# 3.1 Exploratory Analysis
# =============================================================================
def section_3_1_exploratory_analysis(df):
    """Generate gas distribution plots and correlation heatmap."""
    logger.info("3.1: Creating exploratory data analysis charts...")

    # 1. Gas distributions (log scale)
    fig, axes = plt.subplots(3, 3, figsize=(14, 10))
    for ax, gas in zip(axes.flat, GAS_COLS):
        ax.hist(np.log1p(df[gas].clip(lower=0)), bins=50, color='teal', edgecolor='white', alpha=0.7)
        ax.set_title(f'{gas.upper()} (log1p scale)')
        ax.set_xlabel('log1p(ppm)')
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "section_3_1_gas_distributions.png", dpi=150)
    plt.close(fig)

    # 2. Correlation heatmap
    corr = df[GAS_COLS].corr()
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", ax=ax)
    ax.set_title("Gas correlation matrix")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "section_3_1_correlation_heatmap.png", dpi=150)
    plt.close(fig)

    # 3. Missing data summary table
    missing = df[GAS_COLS + ['tdcg', 'water', 'temp']].isnull().mean().reset_index()
    missing.columns = ['column', 'missing_ratio']
    missing.to_csv(OUTPUT_DIR / "section_3_1_missing_summary.csv", index=False)

    logger.info("3.1 PNG/CSV outputs complete.")

def generate_section_3_1_data(df):
    """Export JSON for frontend charts (section 3.1)."""
    logger.info("Exporting section 3.1 data for frontend...")
    data = {}
    # Gas histograms
    gas_hists = {}
    for gas in GAS_COLS:
        gas_hists[gas] = _histogram_data(df[gas], bins=50)
    data["gas_distributions"] = gas_hists

    # Correlation matrix
    corr_df = df[GAS_COLS].corr()
    data["correlation_matrix"] = {
        "columns": GAS_COLS,
        "values": corr_df.values.tolist(),
    }

    # Missing percentages
    missing = df[GAS_COLS + ['tdcg', 'water', 'temp']].isnull().mean() * 100
    data["missing_summary"] = {
        "columns": missing.index.tolist(),
        "missing_ratio": missing.round(2).tolist()
    }

    with open(OUTPUT_DIR / "section_3_1_data.json", "w") as f:
        json.dump(data, f, indent=2)
    logger.info("Section 3.1 frontend data saved.")

# =============================================================================
# 3.2 Label Distribution
# =============================================================================
def section_3_2_label_distribution(df):
    """Compare label allocations across six traditional methods (PNG + JSON)."""
    logger.info("3.2: Label distribution comparison...")
    label_counts_all = {}
    for method in TRADITIONAL_METHODS:
        if method not in df.columns:
            continue
        labels = df[method].dropna().apply(lambda x: str(x).upper())
        labels = labels[~labels.isin(['NORMAL', 'ABSTAIN', ''])]
        if labels.empty:
            continue
        counts = labels.value_counts()
        label_counts_all[method] = counts.to_dict()

        # PNG bar chart
        fig, ax = plt.subplots(figsize=(8, 4))
        counts.plot(kind='bar', ax=ax, color='darkorange')
        ax.set_title(f"Fault labels assigned by {method}")
        ax.set_ylabel("Number of samples")
        fig.tight_layout()
        fig.savefig(OUTPUT_DIR / f"section_3_2_label_dist_{method}.png", dpi=150)
        plt.close(fig)

    # JSON for frontend
    with open(OUTPUT_DIR / "section_3_2_data.json", "w") as f:
        json.dump(label_counts_all, f, indent=2)
    logger.info("3.2 outputs saved.")

# =============================================================================
# 3.3 Supervised ML Performance
# =============================================================================
def section_3_3_supervised_plots(supervised_results):
    """Create bar charts comparing accuracy/F1 across methods and models."""
    logger.info("3.3: Creating supervised learning comparison chart...")

    rows = []
    for method, models in supervised_results.items():
        for model_name, metrics in models.items():
            rows.append({
                "labeling_method": method,
                "model": model_name,
                "accuracy": metrics["accuracy"],
                "f1_macro": metrics["f1_macro"]
            })
    df = pd.DataFrame(rows)
    if df.empty:
        logger.warning("No supervised results to plot.")
        return

    df.to_csv(OUTPUT_DIR / "table_3_3_supervised.csv", index=False)

    # Accuracy bar chart
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.barplot(data=df, x="labeling_method", y="accuracy", hue="model", ax=ax)
    ax.set_title("Supervised model accuracy per labeling method")
    ax.set_ylabel("Accuracy (5-fold CV)")
    ax.tick_params(axis='x', rotation=45)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "section_3_3_accuracy_comparison.png", dpi=150)
    plt.close(fig)

    # F1 macro bar chart
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.barplot(data=df, x="labeling_method", y="f1_macro", hue="model", ax=ax)
    ax.set_title("Supervised model macro F1 per labeling method")
    ax.set_ylabel("Macro F1 (5-fold CV)")
    ax.tick_params(axis='x', rotation=45)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "section_3_3_f1_comparison.png", dpi=150)
    plt.close(fig)

    logger.info("3.3 completed.")

# =============================================================================
# 3.4 Risk Score Formulations
# =============================================================================
def section_3_4_risk_score_analysis(df):
    """PNG charts for risk score, severity pie, top-5 transformers."""
    logger.info("3.4: Risk score evaluation charts...")

    if "severity_score" not in df.columns:
        logger.warning("severity_score column missing.")
        return

    # 1. Histogram
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(df["severity_score"], bins=50, color='navy', alpha=0.7)
    ax.set_title("Distribution of severity scores")
    ax.set_xlabel("Severity score")
    ax.set_ylabel("Number of samples")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "section_3_4_risk_score_distribution.png", dpi=150)
    plt.close(fig)

    # 2. Severity label pie chart
    if "severity_label" in df.columns:
        counts = df["severity_label"].value_counts()
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.pie(counts, labels=counts.index, autopct='%1.1f%%', startangle=90)
        ax.set_title("Severity label distribution")
        fig.tight_layout()
        fig.savefig(OUTPUT_DIR / "section_3_4_severity_pie.png", dpi=150)
        plt.close(fig)

    # 3. Top-5 highest risk transformers
    if "transformer_id" in df.columns and "severity_score" in df.columns:
        top5 = df.groupby("transformer_id")["severity_score"].max().nlargest(5)
        fig, ax = plt.subplots(figsize=(8, 4))
        top5.plot(kind='bar', ax=ax, color='firebrick')
        ax.set_title("Top 5 transformers by maximum severity score")
        ax.set_ylabel("Severity score")
        fig.tight_layout()
        fig.savefig(OUTPUT_DIR / "section_3_4_top5_transformers.png", dpi=150)
        plt.close(fig)

    logger.info("3.4 PNG outputs saved.")

def generate_section_3_4_data(df):
    """Export JSON for frontend charts (section 3.4)."""
    logger.info("Exporting section 3.4 data for frontend...")
    data = {}
    if "severity_score" in df.columns:
        data["severity_histogram"] = _histogram_data(df["severity_score"], bins=30)
    if "severity_label" in df.columns:
        counts = df["severity_label"].value_counts().to_dict()
        data["severity_label_counts"] = counts
    if "transformer_id" in df.columns and "severity_score" in df.columns:
        top5 = df.groupby("transformer_id")["severity_score"].max().nlargest(5).reset_index()
        data["top5_transformers"] = top5.to_dict(orient="records")
    with open(OUTPUT_DIR / "section_3_4_data.json", "w") as f:
        json.dump(data, f, indent=2)
    logger.info("Section 3.4 frontend data saved.")

# =============================================================================
# Anomaly & Supervised experiment functions (unchanged, kept for completeness)
# =============================================================================
def load_data():
    logger.info("Loading unlabeled dataset for anomaly experiment...")
    df = pd.read_parquet(Path(DATASET_DIR) / "processed" / "dga_unlabeled.parquet")
    X = df[GAS_COLS].fillna(0).values
    groups = df['transformer_id'].values
    events = df['has_event'].values if 'has_event' in df.columns else np.zeros(len(df), dtype=bool)
    tdcg = df['tdcg'].values
    method_flags = {}
    for method in TRADITIONAL_METHODS:
        if method in df.columns:
            flag = df[method].notna() & ~df[method].str.upper().isin(['NORMAL', 'ABSTAIN', ''])
            method_flags[method] = flag.values
        else:
            method_flags[method] = np.zeros(len(df), dtype=bool)
    logger.info("Data loaded: %d samples, %d features, %d proxy events, %d unique transformers",
                X.shape[0], X.shape[1], int(events.sum()), len(np.unique(groups)))
    return X, groups, events, tdcg, method_flags, df

def posthoc_validation(scores, method_flags, ks=[20,50,100]):
    order = np.argsort(scores)[::-1]
    result = {}
    for method, flags in method_flags.items():
        method_result = {}
        for k in ks:
            topk = order[:k]
            cnt = flags[topk].sum()
            pct = (cnt / k * 100) if k > 0 else 0.0
            method_result[f"Top-{k}"] = {"count": int(cnt), "pct": round(pct, 1)}
        result[method] = method_result
    return result

def run_anomaly_experiment():
    start_time = time.time()
    X, groups, events, tdcg, method_flags, df = load_data()
    methods = ['tdcg', 'iforest', 'lof', 'ocsvm', 'autoencoder', 'ensemble']
    fold_results = {m: [] for m in methods}

    outer_cv = GroupKFold(n_splits=OUTER_FOLDS)
    logger.info("Starting %d-fold cross-validation for anomaly detection. Top-K=%d", OUTER_FOLDS, K)

    for fold, (train_idx, test_idx) in enumerate(outer_cv.split(X, groups=groups)):
        fold_start = time.time()
        logger.info("=== Anomaly Fold %d/%d ===", fold+1, OUTER_FOLDS)
        X_train, X_test = X[train_idx], X[test_idx]
        events_test = events[test_idx]
        tdcg_test = tdcg[test_idx]
        transformer_ids_test = groups[test_idx]
        test_flags = {m: method_flags[m][test_idx] for m in TRADITIONAL_METHODS}

        # Baseline TDCG
        scores_tdcg = tdcg_test / (tdcg_test.max() + 1e-6)
        prec, rec, lift = precision_recall_lift(scores_tdcg, events_test, K)
        posthoc = posthoc_validation(scores_tdcg, test_flags)
        fold_results['tdcg'].append({
            'precision': prec, 'recall': rec, 'lift': lift,
            'gic': 1.0, 'stability': 1.0, 'spearman': 1.0, 'tc': 1.0,
            'posthoc': posthoc
        })

        # Ensemble
        ensemble = UnsupervisedEnsemble()
        ensemble.fit(X_train)
        scores_ens = ensemble.predict(X_test)
        prec, rec, lift = precision_recall_lift(scores_ens, events_test, K)
        gic = gas_increase_consistency(ensemble, X_test, TDCG_COLS_INDICES, n_perturb=N_PERTURB)
        ensemble2 = UnsupervisedEnsemble(random_state=123)
        ensemble2.fit(X_train)
        scores_ens2 = ensemble2.predict(X_test)
        stability = topk_stability(scores_ens, scores_ens2, K)
        spearman = rank_correlation(scores_ens, scores_ens2)
        tc = temporal_consistency(scores_ens, tdcg_test, transformer_ids_test)
        posthoc = posthoc_validation(scores_ens, test_flags)
        fold_results['ensemble'].append({
            'precision': prec, 'recall': rec, 'lift': lift,
            'gic': gic, 'stability': stability, 'spearman': spearman, 'tc': tc,
            'posthoc': posthoc
        })

        # Individual detectors
        for name in ['iforest','lof','ocsvm','autoencoder']:
            single = UnsupervisedEnsemble()
            single.fit(X_train)
            single.weights = {k: 0.0 for k in single.detectors}
            single.weights[name] = 1.0
            scores = single.predict(X_test)
            prec, rec, lift = precision_recall_lift(scores, events_test, K)
            gic = gas_increase_consistency(single, X_test, TDCG_COLS_INDICES, n_perturb=N_PERTURB)
            stability = topk_stability(scores, scores_ens, K)
            spearman = rank_correlation(scores, scores_ens)
            tc = temporal_consistency(scores, tdcg_test, transformer_ids_test)
            posthoc = posthoc_validation(scores, test_flags)
            fold_results[name].append({
                'precision': prec, 'recall': rec, 'lift': lift,
                'gic': gic, 'stability': stability, 'spearman': spearman, 'tc': tc,
                'posthoc': posthoc
            })
            logger.info("Fold %d %s: P@K=%.4f, R@K=%.4f, Lift=%.2f, GIC=%.3f, Stab=%.3f",
                        fold+1, name, prec, rec, lift, gic, stability)

        logger.info("Fold %d completed in %.1fs", fold+1, time.time()-fold_start)

    # Aggregate results
    summary = {}
    logger.info("\n====== ANOMALY DETECTION RESULTS SUMMARY ======")
    for method in methods:
        vals = fold_results[method]
        if not vals:
            continue
        avg_prec = np.mean([v['precision'] for v in vals])
        avg_rec = np.mean([v['recall'] for v in vals])
        avg_lift = np.mean([v['lift'] for v in vals])
        avg_gic = np.mean([v['gic'] for v in vals])
        avg_stab = np.mean([v['stability'] for v in vals])
        avg_spearman = np.mean([v['spearman'] for v in vals])
        avg_tc = np.mean([v['tc'] for v in vals])

        prec_vals = np.array([v['precision'] for v in vals])
        mean_prec, ci_low, ci_up = bootstrap_confidence_interval(prec_vals, np.mean, n_bootstrap=N_BOOTSTRAP)

        aggregated_posthoc = {}
        for trad in TRADITIONAL_METHODS:
            for k in [20,50,100]:
                key = f"Top-{k}"
                pcts = [v['posthoc'][trad][key]['pct'] for v in vals]
                avg_pct = np.mean(pcts)
                if trad not in aggregated_posthoc:
                    aggregated_posthoc[trad] = {}
                aggregated_posthoc[trad][key] = round(avg_pct, 1)

        summary[method] = {
            'Precision@K': avg_prec,
            'Recall@K': avg_rec,
            'Lift': avg_lift,
            'GIC': avg_gic,
            'Stability': avg_stab,
            'Spearman': avg_spearman,
            'TC': avg_tc,
            'Precision_CI_low': ci_low,
            'Precision_CI_high': ci_up,
            'Posthoc': aggregated_posthoc
        }

        logger.info("%-15s P@K=%.4f [%.4f, %.4f] R@K=%.4f Lift=%.2f GIC=%.3f Stab=%.3f Spearman=%.3f TC=%.3f",
                    method, avg_prec, ci_low, ci_up, avg_rec, avg_lift, avg_gic, avg_stab, avg_spearman, avg_tc)

    output_path = OUTPUT_DIR / 'experiment_results_full.json'
    with open(output_path, 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    logger.info("Anomaly results saved to %s", output_path)
    logger.info("Total anomaly experiment time: %.1fs", time.time()-start_time)
    return df

# ──────────── Supervised Experiment ────────────
XGB_AVAILABLE = False
CATBOOST_AVAILABLE = False
SMOTE_AVAILABLE = False
try:
    from xgboost import XGBClassifier
    XGB_AVAILABLE = True
except ImportError:
    logger.warning("xgboost not installed.")
try:
    from catboost import CatBoostClassifier
    CATBOOST_AVAILABLE = True
except ImportError:
    logger.warning("catboost not installed.")
try:
    from imblearn.over_sampling import SMOTE
    SMOTE_AVAILABLE = True
except ImportError:
    logger.warning("imbalanced-learn not installed.")

def prepare_dataset(method):
    df = pd.read_parquet(Path(DATASET_DIR) / "processed" / "dga_unlabeled.parquet")
    X = df[GAS_COLS].fillna(0).values
    y_str = np.array([_normalize_vote(str(row[method])) for _, row in df.iterrows()])
    mask = y_str != 'ABSTAIN'
    X, y_str = X[mask], y_str[mask]
    groups = df.loc[mask, 'transformer_id'].values
    le = LabelEncoder()
    y = le.fit_transform(y_str)
    return X, y, groups, le

def cross_val_scores(model, X, y, groups, cv):
    accs, f1s = [], []
    for train_idx, test_idx in cv.split(X, y, groups):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        model_clone = model.__class__(**model.get_params())
        model_clone.fit(X_train, y_train)
        preds = model_clone.predict(X_test)
        accs.append(accuracy_score(y_test, preds))
        f1s.append(f1_score(y_test, preds, average='macro', zero_division=0))
    return np.mean(accs), np.mean(f1s)

def cross_val_scores_with_smote(model, X, y, groups, cv):
    if not SMOTE_AVAILABLE:
        return None, None
    accs, f1s = [], []
    for train_idx, test_idx in cv.split(X, y, groups):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        min_class_count = np.min(np.bincount(y_train))
        k_neighbors = min(5, min_class_count - 1)
        try:
            if k_neighbors >= 1:
                sm = SMOTE(random_state=42, k_neighbors=k_neighbors)
                X_res, y_res = sm.fit_resample(X_train, y_train)
            else:
                X_res, y_res = X_train, y_train
        except Exception:
            X_res, y_res = X_train, y_train
        model_clone = model.__class__(**model.get_params())
        model_clone.fit(X_res, y_res)
        preds = model_clone.predict(X_test)
        accs.append(accuracy_score(y_test, preds))
        f1s.append(f1_score(y_test, preds, average='macro', zero_division=0))
    return np.mean(accs), np.mean(f1s)

def run_supervised_experiment():
    start_time = time.time()
    all_results = {}
    for method in TRADITIONAL_METHODS:
        logger.info("--- Processing labeling method: %s ---", method)
        X, y, groups, le = prepare_dataset(method)
        cv = GroupKFold(n_splits=5)
        results = {}

        rf = RandomForestClassifier(n_estimators=100, random_state=42)
        acc, f1 = cross_val_scores(rf, X, y, groups, cv)
        results['RF'] = {'accuracy': acc, 'f1_macro': f1}
        if SMOTE_AVAILABLE:
            acc, f1 = cross_val_scores_with_smote(rf, X, y, groups, cv)
            results['RF+SMOTE'] = {'accuracy': acc, 'f1_macro': f1}

        if XGB_AVAILABLE:
            xgb = XGBClassifier(eval_metric='mlogloss', random_state=42)
            acc, f1 = cross_val_scores(xgb, X, y, groups, cv)
            results['XGBoost'] = {'accuracy': acc, 'f1_macro': f1}
            if SMOTE_AVAILABLE:
                acc, f1 = cross_val_scores_with_smote(xgb, X, y, groups, cv)
                results['XGBoost+SMOTE'] = {'accuracy': acc, 'f1_macro': f1}

        if CATBOOST_AVAILABLE:
            cb = CatBoostClassifier(silent=True, random_state=42)
            acc, f1 = cross_val_scores(cb, X, y, groups, cv)
            results['CatBoost'] = {'accuracy': acc, 'f1_macro': f1}
            if SMOTE_AVAILABLE:
                acc, f1 = cross_val_scores_with_smote(cb, X, y, groups, cv)
                results['CatBoost+SMOTE'] = {'accuracy': acc, 'f1_macro': f1}

        all_results[method] = results

    output_path = OUTPUT_DIR / 'supervised_results.json'
    with open(output_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    logger.info("Supervised results saved to %s", output_path)
    logger.info("Total supervised experiment time: %.1fs", time.time()-start_time)
    return all_results

def run_full_experiments():
    """Run all experiments and generate all report artifacts for frontend."""
    # 1. Run anomaly experiment (returns full df)
    df = run_anomaly_experiment()

    # 2. Run supervised experiment
    sup_results = run_supervised_experiment()

    # 3. Generate PNGs & CSVs (optional but useful)
    section_3_1_exploratory_analysis(df)
    section_3_2_label_distribution(df)
    section_3_3_supervised_plots(sup_results)
    section_3_4_risk_score_analysis(df)

    # 4. Generate JSON data for frontend charts
    generate_section_3_1_data(df)
    # section_3_2_data.json already generated inside section_3_2_label_distribution
    generate_section_3_4_data(df)

    logger.info("All report artifacts generated.")
    return df, sup_results

# =============================================================================
# Main
# =============================================================================
if __name__ == "__main__":
    run_full_experiments()