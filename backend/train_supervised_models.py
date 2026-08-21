# train_supervised_models.py
from __future__ import annotations
"""Small supervised reference baseline. This file is intentionally separate from the production weak-supervision path. It trains on the labeled benchmark only and reports a holdout metric. It is a reference baseline, not the model used to label the operational dataset."""
import json, logging
from pathlib import Path
import joblib, numpy as np, pandas as pd
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, balanced_accuracy_score, classification_report, f1_score
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.pipeline import Pipeline
from config import DATASET_DIR, MODEL_DIR, REPORT_DIR, config as cfg
from consensus import ABSTAIN, normalize_fault

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
MODEL_OUTPUT_PATH = Path(MODEL_DIR) / "fault_supervised_model.joblib"
GAS_COLS = list(cfg.COMMON_BENCHMARK_GASES)

def load_labeled(path: Path, label_col: str):
    df = pd.read_csv(path, encoding="utf-8-sig")
    df.columns = [str(c).strip().lower().replace("\ufeff", "") for c in df.columns]
    for col in GAS_COLS: df[col] = pd.to_numeric(df[col], errors="coerce") if col in df.columns else np.nan
    df["fault"] = df[label_col].map(normalize_fault)
    df = df[(df["fault"] != ABSTAIN) & df["fault"].isin(cfg.BENCHMARK_FINE_CLASSES)].copy()
    return df

def build_dataset():
    a = load_labeled(Path(DATASET_DIR) / "IEC_TC10_121.csv", "label")
    b = load_labeled(Path(DATASET_DIR) / "DGA dataset.csv", "type")
    return pd.concat([a, b], ignore_index=True)

def train_model(df: pd.DataFrame, seed=42):
    mapping = {c: i for i, c in enumerate(cfg.BENCHMARK_FINE_CLASSES)}
    X = df[GAS_COLS].to_numpy(float)
    y = df["fault"].map(mapping).to_numpy(int)
    counts = pd.Series(y).value_counts()
    if counts.min() < 2: raise ValueError(f"Cannot stratify: {counts.to_dict()}")
    split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=seed)
    train_idx, test_idx = next(split.split(X, y))
    model = Pipeline([
        ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
        ("classifier", ExtraTreesClassifier(n_estimators=500, class_weight="balanced", random_state=seed, n_jobs=-1)),
    ])
    model.fit(X[train_idx], y[train_idx])
    pred = model.predict(X[test_idx])
    labels = list(range(len(cfg.BENCHMARK_FINE_CLASSES)))
    metrics = {
        "accuracy": float(accuracy_score(y[test_idx], pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y[test_idx], pred)),
        "macro_f1": float(f1_score(y[test_idx], pred, labels=labels, average="macro", zero_division=0)),
        "n_train": int(len(train_idx)), "n_test": int(len(test_idx)), "feature_set": "common_5_gases",
    }
    logger.info("Supervised reference metrics: %s", metrics)
    logger.info("\n%s", classification_report(y[test_idx], pred, labels=labels, target_names=cfg.BENCHMARK_FINE_CLASSES, zero_division=0))
    model.fit(X, y)
    Path(MODEL_DIR).mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": model, "labels": list(cfg.BENCHMARK_FINE_CLASSES), "feature_cols": GAS_COLS, "evaluation_metrics": metrics}, MODEL_OUTPUT_PATH)
    Path(REPORT_DIR).mkdir(parents=True, exist_ok=True)
    (Path(REPORT_DIR) / "supervised_reference_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return model, metrics

def main():
    train_model(build_dataset(), cfg.RANDOM_STATE)

if __name__ == "__main__":
    main()