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
    logger.debug("load_labeled: loading %s with label_col='%s'", path, label_col)
    df = pd.read_csv(path, encoding="utf-8-sig")
    df.columns = [str(c).strip().lower().replace("\ufeff", "") for c in df.columns]
    logger.debug("load_labeled: initial shape=%s columns=%s", df.shape, list(df.columns))
    for col in GAS_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        else:
            logger.debug("load_labeled: missing gas column '%s', setting to NaN", col)
            df[col] = np.nan
    logger.debug("load_labeled: label_col present=%s", label_col in df.columns)
    if label_col not in df.columns:
        logger.error("load_labeled: label column '%s' not found in columns %s", label_col, list(df.columns))
        raise KeyError(f"Label column '{label_col}' not found")
    df["fault"] = df[label_col].map(normalize_fault)
    before_filter = len(df)
    df = df[(df["fault"] != ABSTAIN) & df["fault"].isin(cfg.BENCHMARK_FINE_CLASSES)].copy()
    logger.debug("load_labeled: before filter=%d after filter=%d (removed %d rows)", before_filter, len(df), before_filter - len(df))
    logger.debug("load_labeled: unique faults after filter: %s", df["fault"].unique())
    return df

def build_dataset():
    logger.info("build_dataset: constructing supervised reference dataset from labeled benchmarks")
    a = load_labeled(Path(DATASET_DIR) / "IEC_TC10_121.csv", "label")
    b = load_labeled(Path(DATASET_DIR) / "DGA dataset.csv", "type")
    logger.debug("build_dataset: IEC_TC10 rows=%d, DGA dataset rows=%d", len(a), len(b))
    combined = pd.concat([a, b], ignore_index=True)
    logger.info("build_dataset: combined dataset rows=%d", len(combined))
    return combined

def train_model(df: pd.DataFrame, seed=42):
    logger.info("train_model: starting supervised reference training with seed=%d", seed)
    logger.debug("train_model: input df shape=%s columns=%s", df.shape, list(df.columns))
    mapping = {c: i for i, c in enumerate(cfg.BENCHMARK_FINE_CLASSES)}
    X = df[GAS_COLS].to_numpy(float)
    y = df["fault"].map(mapping).to_numpy(int)
    logger.debug("train_model: X shape=%s, y shape=%s", X.shape, y.shape)
    logger.debug("train_model: label counts=%s", pd.Series(y).value_counts().to_dict())
    counts = pd.Series(y).value_counts()
    if counts.min() < 2:
        logger.error("train_model: cannot stratify, class counts=%s", counts.to_dict())
        raise ValueError(f"Cannot stratify: {counts.to_dict()}")
    split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=seed)
    train_idx, test_idx = next(split.split(X, y))
    logger.debug("train_model: train size=%d, test size=%d", len(train_idx), len(test_idx))
    model = Pipeline([
        ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
        ("classifier", ExtraTreesClassifier(n_estimators=500, class_weight="balanced", random_state=seed, n_jobs=-1)),
    ])
    logger.info("train_model: fitting ExtraTrees model on training split")
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
    logger.info("train_model: fitting final model on full dataset")
    model.fit(X, y)
    logger.debug("train_model: saving model to %s", MODEL_OUTPUT_PATH)
    Path(MODEL_DIR).mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": model, "labels": list(cfg.BENCHMARK_FINE_CLASSES), "feature_cols": GAS_COLS, "evaluation_metrics": metrics}, MODEL_OUTPUT_PATH)
    Path(REPORT_DIR).mkdir(parents=True, exist_ok=True)
    metrics_path = Path(REPORT_DIR) / "supervised_reference_metrics.json"
    logger.debug("train_model: writing metrics to %s", metrics_path)
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    logger.info("train_model: supervised reference model saved successfully")
    return model, metrics

def main():
    logger.info("Starting supervised reference training")
    train_model(build_dataset(), cfg.RANDOM_STATE)
    logger.info("Supervised reference training completed")

if __name__ == "__main__":
    main()