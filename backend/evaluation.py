# evaluation.py
import logging
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.utils import resample

logger = logging.getLogger(__name__)


def precision_recall_lift(scores, events, k):
    """Compute Precision@k, Recall@k, and Lift for given anomaly scores."""
    order = np.argsort(scores)[::-1]
    topk = order[:k]
    tp = events[topk].sum()
    precision = tp / k
    recall = tp / events.sum() if events.sum() > 0 else 0.0
    prevalence = events.mean()
    lift = precision / prevalence if prevalence > 0 else 1.0
    logger.debug("P@%d=%.4f, R@%d=%.4f, Lift=%.2f (prevalence=%.4f)", k, precision, k, recall, lift, prevalence)
    return precision, recall, lift


def topk_stability(scores1, scores2, k):
    """Jaccard index of Top-K elements between two score vectors."""
    topk1 = set(np.argsort(scores1)[::-1][:k])
    topk2 = set(np.argsort(scores2)[::-1][:k])
    inter = len(topk1 & topk2)
    union = len(topk1 | topk2)
    stability = inter / union if union > 0 else 1.0
    logger.debug("Top-%d stability: %.4f", k, stability)
    return stability


def rank_correlation(scores1, scores2):
    """Spearman rank correlation between two score vectors."""
    rho = spearmanr(scores1, scores2).correlation
    logger.debug("Spearman ρ = %.4f", rho)
    return rho


def temporal_consistency(scores, tdcg, transformer_ids):
    """TC metric: proportion of consecutive samples where score doesn't drop when TDCG increases."""
    df = pd.DataFrame({'score': scores, 'tdcg': tdcg, 'id': transformer_ids})
    consistent = 0
    total = 0
    for _, grp in df.groupby('id'):
        grp = grp.sort_index()
        for i in range(len(grp)-1):
            if grp['tdcg'].iloc[i+1] > grp['tdcg'].iloc[i]:
                if grp['score'].iloc[i+1] >= grp['score'].iloc[i]:
                    consistent += 1
                total += 1
    tc = consistent / total if total > 0 else 0.0
    logger.debug("Temporal consistency (TC): %.4f (%d/%d pairs)", tc, consistent, total)
    return tc


def gas_increase_consistency(model, X, tdcg_cols, n_perturb=100):
    """GIC: fraction of samples where score increases when gas concentrations are increased by 10%."""
    n = X.shape[0]
    idx = np.random.choice(n, min(n_perturb, n), replace=False)
    consistent = 0
    for i in idx:
        x_orig = X[i].copy()
        s_orig = model.predict(x_orig.reshape(1, -1))[0]
        x_pert = x_orig.copy()
        for j in tdcg_cols:
            x_pert[j] *= 1.1
        s_pert = model.predict(x_pert.reshape(1, -1))[0]
        if s_pert >= s_orig:
            consistent += 1
    gic = consistent / len(idx) if len(idx) > 0 else 0.0
    logger.debug("Gas increase consistency (GIC): %.4f (%d/%d perturbations)", gic, consistent, len(idx))
    return gic


def evaluate_agreement_with_weak_labels(
    df: pd.DataFrame,
    weak_label_col: str,
    predicted_col: str = "consensus_fault"
) -> dict:
    """
    Evaluate agreement between model predictions and a weak label (e.g., from IEC method).
    This is NOT diagnostic accuracy; it only measures consistency with the chosen weak label.
    """
    from sklearn.metrics import accuracy_score, f1_score, cohen_kappa_score

    mask = df[weak_label_col].notna() & df[predicted_col].notna()
    y_true = df.loc[mask, weak_label_col]
    y_pred = df.loc[mask, predicted_col]

    if len(y_true) == 0:
        logger.warning("No valid samples for agreement evaluation (column '%s').", weak_label_col)
        return {"accuracy": None, "macro_f1": None, "cohen_kappa": None}

    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
    kappa = cohen_kappa_score(y_true, y_pred)

    logger.info("Weak-label agreement (vs %s): Accuracy=%.3f, Macro F1=%.3f, Cohen’s Kappa=%.3f",
                weak_label_col, acc, f1, kappa)
    return {"accuracy": acc, "macro_f1": f1, "cohen_kappa": kappa}


def evaluate_diagnostic_performance(
    df: pd.DataFrame,
    ground_truth_col: str,
    predicted_col: str = "consensus_fault"
) -> dict:
    """
    Evaluate real-world diagnostic performance against a ground truth column (e.g., maintenance inspection results).
    """
    from sklearn.metrics import accuracy_score, f1_score, cohen_kappa_score

    mask = df[ground_truth_col].notna() & df[predicted_col].notna()
    y_true = df.loc[mask, ground_truth_col]
    y_pred = df.loc[mask, predicted_col]

    if len(y_true) == 0:
        logger.warning("No ground truth samples available for evaluation (column '%s').", ground_truth_col)
        return {"accuracy": None, "macro_f1": None, "cohen_kappa": None}

    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
    kappa = cohen_kappa_score(y_true, y_pred)

    logger.info("Diagnostic performance vs %s: Accuracy=%.3f, Macro F1=%.3f, Cohen’s Kappa=%.3f",
                ground_truth_col, acc, f1, kappa)
    return {"accuracy": acc, "macro_f1": f1, "cohen_kappa": kappa}


def bootstrap_confidence_interval(data, metric_fn, n_bootstrap=1000, alpha=0.05):
    """Compute bootstrap confidence interval for a given metric function."""
    vals = np.array([metric_fn(resample(data)) for _ in range(n_bootstrap)])
    lower = np.percentile(vals, 100 * alpha / 2)
    upper = np.percentile(vals, 100 * (1 - alpha / 2))
    mean = np.mean(vals)
    logger.debug("Bootstrap CI (α=%.2f): mean=%.4f, [%.4f, %.4f]", alpha, mean, lower, upper)
    return mean, lower, upper