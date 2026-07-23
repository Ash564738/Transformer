# evaluation.py
import pandas as pd
import logging

logger = logging.getLogger(__name__)

def evaluate_agreement_with_weak_labels(
    df: pd.DataFrame,
    weak_label_col: str,
    predicted_col: str = "consensus_fault"
) -> dict:
    """
    Đánh giá mức độ đồng thuận giữa kết quả dự đoán và nhãn yếu (ví dụ IEC).
    KHÔNG phải là diagnostic accuracy.
    """
    from sklearn.metrics import accuracy_score, f1_score, cohen_kappa_score  # lazy: only needed when this is called

    mask = df[weak_label_col].notna() & df[predicted_col].notna()
    y_true = df.loc[mask, weak_label_col]
    y_pred = df.loc[mask, predicted_col]

    if len(y_true) == 0:
        logger.warning("Không có mẫu hợp lệ để tính agreement.")
        return {"accuracy": None, "macro_f1": None, "cohen_kappa": None}

    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
    kappa = cohen_kappa_score(y_true, y_pred)

    logger.info(f"Weak-label Agreement (vs {weak_label_col}): "
                f"Acc={acc:.3f}, Macro F1={f1:.3f}, Cohen’s Kappa={kappa:.3f}")
    return {"accuracy": acc, "macro_f1": f1, "cohen_kappa": kappa}

def evaluate_diagnostic_performance(
    df: pd.DataFrame,
    ground_truth_col: str,
    predicted_col: str = "consensus_fault"
) -> dict:
    """
    Đánh giá hiệu năng chẩn đoán thực tế dựa trên ground truth (ví dụ kết quả bảo dưỡng, kiểm tra).
    """
    from sklearn.metrics import accuracy_score, f1_score, cohen_kappa_score  # lazy: only needed when this is called

    mask = df[ground_truth_col].notna() & df[predicted_col].notna()
    y_true = df.loc[mask, ground_truth_col]
    y_pred = df.loc[mask, predicted_col]

    if len(y_true) == 0:
        logger.warning("Không có mẫu ground truth để đánh giá.")
        return {"accuracy": None, "macro_f1": None, "cohen_kappa": None}

    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
    kappa = cohen_kappa_score(y_true, y_pred)

    logger.info(f"Real-world Diagnostic Performance (vs {ground_truth_col}): "
                f"Acc={acc:.3f}, Macro F1={f1:.3f}, Cohen’s Kappa={kappa:.3f}")
    return {"accuracy": acc, "macro_f1": f1, "cohen_kappa": kappa}