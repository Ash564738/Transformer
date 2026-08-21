# train_unsupervised_models.py
from __future__ import annotations
import argparse, json, logging, random, warnings
from itertools import combinations
from pathlib import Path
from typing import Iterable, Sequence, Tuple
import joblib, numpy as np, pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import AdaBoostClassifier, BaggingClassifier, ExtraTreesClassifier, GradientBoostingClassifier, HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.calibration import CalibratedClassifierCV
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    TORCH_AVAILABLE = True
except Exception: TORCH_AVAILABLE = False
try: import xgboost as xgb
except Exception: xgb = None
try: import lightgbm as lgb
except Exception: lgb = None
try: from catboost import CatBoostClassifier
except Exception: CatBoostClassifier = None
from config import DATASET_DIR, MODEL_DIR, REPORT_DIR, config as cfg
from consensus import ABSTAIN, apply_consensus, apply_consensus_from_existing_diagnostics, diagnostic_method_summary, evaluate_method_labels, normalize_fault, pairwise_label_agreement, unify_fault
from evaluation import evaluate_ambiguous_fine_predictions, empirical_fault_class_coverage
from feature_engineering import build_training_features_from_clean
from logging_config import init_logging
from ranking import build_transformer_ranking, log_ranking_diagnostics
from weak_supervision import DEFAULT_WEAK_METHODS, create_student_training_targets, save_weak_supervision_artifacts, weak_supervision_pipeline
init_logging()
logger = logging.getLogger(__name__)
LABELED_CSV_PATH1 = DATASET_DIR / "IEC_TC10_121.csv"
LABELED_CSV_PATH2 = DATASET_DIR / "DGA dataset.csv"
UNLABELED_PATH = DATASET_DIR / "processed" / "dga_unlabeled.parquet"
FAULT_MODEL_COARSE_PATH = MODEL_DIR / "fault_classifiers_coarse.joblib"
FAULT_MODEL_FINE_PATH = MODEL_DIR / "fault_classifiers_fine.joblib"
TRAINING_METADATA_PATH = MODEL_DIR / "training_metadata.json"
BENCHMARK_DIR = REPORT_DIR / "benchmark"
MODEL_FEATURES = list(cfg.COMMON_BENCHMARK_GASES)

def set_global_seed(seed: int):
    logger.debug("set_global_seed: seed=%d torch_available=%s", seed, TORCH_AVAILABLE)
    random.seed(seed); np.random.seed(seed)
    if TORCH_AVAILABLE:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

def _decode_class_predictions(prediction, labels, default=ABSTAIN):
    arr = np.asarray(prediction)
    logger.debug("_decode_class_predictions: input shape=%s labels=%s", arr.shape, labels)
    if arr.ndim == 0: arr = arr.reshape(1)
    elif arr.ndim > 1:
        if arr.size == arr.shape[0]: arr = arr.reshape(-1)
        else: arr = arr.reshape(arr.shape[0], -1)[:, 0]
    decoded = []
    for value in arr:
        try:
            scalar = np.asarray(value).reshape(-1)[0]; index = int(scalar); decoded.append(labels[index] if 0 <= index < len(labels) else default)
        except Exception:
            decoded.append(default)
    logger.debug("_decode_class_predictions: decoded first=%s", decoded[:5])
    return decoded

def _predict_model(model, feature_frame: pd.DataFrame):
    logger.debug("_predict_model: feature_frame shape=%s", feature_frame.shape)
    return model.predict(feature_frame)
def _predict_proba(model, feature_frame: pd.DataFrame):
    logger.debug("_predict_proba: feature_frame shape=%s", feature_frame.shape)
    return model.predict_proba(feature_frame)

def _safe_balanced_accuracy(y_true, y_pred, labels=None):
    y_true = np.asarray(y_true); y_pred = np.asarray(y_pred)
    logger.debug("_safe_balanced_accuracy: y_true=%d y_pred=%d", len(y_true), len(y_pred))
    if y_true.size == 0: return 0.0
    if labels is None: labels = np.unique(y_true)
    labels = list(labels)
    if not labels: return 0.0
    cm = confusion_matrix(y_true, y_pred, labels=labels); support = cm.sum(axis=1); present = support > 0
    if not np.any(present): return 0.0
    recalls = np.divide(np.diag(cm), support, out=np.zeros_like(support, dtype=float), where=support > 0)
    result = float(np.mean(recalls[present]))
    logger.debug("_safe_balanced_accuracy: result=%s", result)
    return result

def _evaluate_method_labels_safely(y_true, y_pred, allowed_labels):
    logger.debug("_evaluate_method_labels_safely: y_true=%d y_pred=%d allowed=%s", len(y_true), len(y_pred), allowed_labels)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="y_pred contains classes not in y_true", category=UserWarning)
        return evaluate_method_labels(y_true, y_pred, allowed_labels)

def _can_stratify(y):
    counts = pd.Series(y).value_counts()
    result = len(counts) >= 2 and counts.min() >= 2
    logger.debug("_can_stratify: counts=%s result=%s", counts.to_dict(), result)
    return result

if TORCH_AVAILABLE:
    class SimpleMLP(nn.Module):
        def __init__(self, input_dim, hidden_dims=(64, 32), output_dim=2, dropout=0.15):
            super().__init__(); layers = []; previous = input_dim
            for hidden in hidden_dims: layers.extend([nn.Linear(previous, hidden), nn.ReLU(), nn.Dropout(dropout)]); previous = hidden
            layers.append(nn.Linear(previous, output_dim)); self.net = nn.Sequential(*layers)
        def forward(self, x): return self.net(x)

    class TorchMLPClassifier(BaseEstimator, ClassifierMixin):
        def __init__(self, hidden_dims=(64, 32), epochs=120, batch_size=32, lr=1e-3, weight_decay=1e-4, random_state=42):
            self.hidden_dims = hidden_dims; self.epochs = epochs; self.batch_size = batch_size; self.lr = lr; self.weight_decay = weight_decay; self.random_state = random_state
        def fit(self, X, y):
            torch.manual_seed(self.random_state); X_np = np.asarray(X); y_np = np.asarray(y); X_tensor = torch.as_tensor(X_np, dtype=torch.float32); y_tensor = torch.as_tensor(y_np, dtype=torch.long); self.classes_ = np.unique(y_np); self.model_ = SimpleMLP(X_tensor.shape[1], self.hidden_dims, len(self.classes_)); optimizer = optim.Adam(self.model_.parameters(), lr=self.lr, weight_decay=self.weight_decay); criterion = nn.CrossEntropyLoss(); dataset = torch.utils.data.TensorDataset(X_tensor, y_tensor); loader = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, shuffle=True); self.model_.train()
            logger.debug("TorchMLPClassifier.fit: classes=%s samples=%d", self.classes_, len(y_np))
            for _ in range(self.epochs):
                for xb, yb in loader: optimizer.zero_grad(); loss = criterion(self.model_(xb), yb); loss.backward(); optimizer.step()
            return self
        def predict_proba(self, X):
            X_tensor = torch.as_tensor(np.asarray(X), dtype=torch.float32); self.model_.eval()
            with torch.no_grad(): probabilities = torch.softmax(self.model_(X_tensor), dim=1).cpu().numpy()
            return probabilities
        def predict(self, X):
            proba = self.predict_proba(X)
            return self.classes_[np.argmax(proba, axis=1)]
else: TorchMLPClassifier = None

def _read_labeled(path: Path, label_col: str, source_name: str) -> pd.DataFrame:
    logger.debug("_read_labeled: path=%s label_col=%s source=%s", path, label_col, source_name)
    if not path.exists():
        logger.error("_read_labeled: path not found %s", path)
        raise FileNotFoundError(path)
    df = pd.read_csv(path, encoding="utf-8-sig"); df.columns = [str(c).strip().lower().replace("\ufeff", "") for c in df.columns]
    logger.debug("_read_labeled: raw shape=%s columns=%s", df.shape, list(df.columns))
    if label_col not in df.columns:
        logger.error("_read_labeled: missing label column %r in %s", label_col, path.name)
        raise ValueError(f"{path.name}: missing label column {label_col!r}")
    for gas in cfg.FULL_EXTERNAL_GASES:
        df[gas] = pd.to_numeric(df[gas], errors="coerce") if gas in df.columns else np.nan
    df["fault_type_label"] = df[label_col].map(normalize_fault); df["source_dataset"] = source_name; df["source_row"] = np.arange(1, len(df) + 1); df["coarse_truth"] = df["fault_type_label"].map(unify_fault)
    logger.debug("_read_labeled: processed shape=%s fault counts=%s", df.shape, df["fault_type_label"].value_counts().to_dict())
    return df

def _external_gas_key(df: pd.DataFrame) -> pd.Series:
    gas = list(cfg.COMMON_BENCHMARK_GASES); x = df[gas].round(8).fillna(-999999.0); result = x.astype(str).agg("|".join, axis=1)
    logger.debug("_external_gas_key: computed key for %d rows", len(result))
    return result

def load_labeled_csv_data() -> pd.DataFrame:
    logger.info("load_labeled_csv_data: loading labeled benchmark CSVs")
    a = _read_labeled(LABELED_CSV_PATH1, "label", "iec_tc10_121"); b = _read_labeled(LABELED_CSV_PATH2, "type", "dga_dataset"); combined = pd.concat([a, b], ignore_index=True); combined["_gas_key"] = _external_gas_key(combined)
    combined["fine_label_conflict"] = combined.groupby("_gas_key")["fault_type_label"].transform("nunique").gt(1); combined["coarse_label_conflict"] = combined.groupby("_gas_key")["coarse_truth"].transform("nunique").gt(1)
    logger.info("load_labeled_csv_data: combined rows=%d fine_conflicts=%d coarse_conflicts=%d", len(combined), int(combined['fine_label_conflict'].sum()), int(combined['coarse_label_conflict'].sum()))
    conflict_df = combined[combined["fine_label_conflict"]].copy(); BENCHMARK_DIR.mkdir(parents=True, exist_ok=True)
    if not conflict_df.empty:
        logger.debug("load_labeled_csv_data: saving %d fine conflict rows", len(conflict_df))
        conflict_df.to_csv(BENCHMARK_DIR / "external_benchmark_label_conflicts.csv", index=False, encoding="utf-8-sig")
    combined["duplicate_group_size"] = combined.groupby("_gas_key")["_gas_key"].transform("size"); combined["evaluation_group"] = combined["_gas_key"]
    result = combined.drop(columns=["_gas_key"]).reset_index(drop=True)
    logger.debug("load_labeled_csv_data: final shape=%s columns=%s", result.shape, list(result.columns))
    return result

def load_unlabeled(path: Path = UNLABELED_PATH) -> pd.DataFrame:
    logger.debug("load_unlabeled: path=%s", path)
    if not path.exists():
        logger.error("load_unlabeled: path not found %s", path)
        raise FileNotFoundError(path)
    df = pd.read_parquet(path); required = {"transformer_id", "sample_day"}; missing = required - set(df.columns)
    if missing:
        logger.error("load_unlabeled: missing required columns %s", sorted(missing))
        raise ValueError(f"{path.name} requires {sorted(missing)}")
    df["sample_day"] = pd.to_datetime(df["sample_day"], errors="coerce"); result = df.dropna(subset=["transformer_id", "sample_day"]).reset_index(drop=True)
    logger.info("load_unlabeled: rows after cleaning=%d", len(result))
    return result

def prepare_unlabeled(df: pd.DataFrame) -> pd.DataFrame:
    logger.debug("prepare_unlabeled: input shape=%s", df.shape)
    out = build_training_features_from_clean(df.copy()); out["sample_day"] = pd.to_datetime(out["sample_day"], errors="coerce"); out = out.sort_values(["transformer_id", "sample_day"], kind="mergesort").reset_index(drop=True)
    logger.debug("prepare_unlabeled: after feature engineering shape=%s", out.shape)
    result = apply_consensus(out)
    logger.debug("prepare_unlabeled: after consensus shape=%s columns=%s", result.shape, list(result.columns))
    return result

def dataframe_to_gas_matrix(df: pd.DataFrame, features: Sequence[str]) -> np.ndarray:
    logger.debug("dataframe_to_gas_matrix: features=%s", list(features))
    x = df.reindex(columns=list(features)).apply(pd.to_numeric, errors="coerce"); result = x.replace([np.inf, -np.inf], np.nan).to_numpy(dtype=np.float32)
    logger.debug("dataframe_to_gas_matrix: result shape=%s", result.shape)
    return result

def _traditional_feature_matrix(df: pd.DataFrame, granularity: str = "fine") -> pd.DataFrame:
    out = pd.DataFrame(index=df.index); vocab = (list(cfg.BENCHMARK_FINE_CLASSES) if granularity == "fine" else list(cfg.WEAK_COARSE_GROUPS)) + [ABSTAIN]
    logger.debug("_traditional_feature_matrix: granularity=%s vocab=%s", granularity, vocab)
    for method, column in cfg.DIAGNOSTIC_METHOD_TO_COLUMN.items():
        vals = df.get(column, pd.Series(ABSTAIN, index=df.index)).map(normalize_fault)
        if granularity == "coarse": vals = vals.map(unify_fault)
        for label in vocab:
            safe = str(label).lower().replace("/", "_").replace(" ", "_"); out[f"{method}__{safe}"] = (vals == label).astype(np.float32)
    logger.debug("_traditional_feature_matrix: output shape=%s", out.shape)
    return out

def build_feature_frame(df: pd.DataFrame, feature_mode: str, granularity: str = "fine") -> pd.DataFrame:
    logger.debug("build_feature_frame: feature_mode=%s granularity=%s df_shape=%s", feature_mode, granularity, df.shape)
    gas = df.reindex(columns=MODEL_FEATURES).apply(pd.to_numeric, errors="coerce").astype(float)
    if feature_mode == "gas_only":
        logger.debug("build_feature_frame: gas_only shape=%s", gas.shape)
        return gas
    if feature_mode == "gas_plus_traditional":
        trad = _traditional_feature_matrix(df, granularity)
        result = pd.concat([gas, trad], axis=1)
        logger.debug("build_feature_frame: gas_plus_traditional shape=%s", result.shape)
        return result
    logger.error("build_feature_frame: unknown feature_mode %s", feature_mode)
    raise ValueError(f"Unknown feature_mode: {feature_mode}")

def _encode_fine_truth(series):
    mapping = {label: idx for idx, label in enumerate(cfg.BENCHMARK_FINE_CLASSES)}; normalized = series.map(normalize_fault).astype(object); encoded = normalized.map(mapping); valid_mask = encoded.notna()
    if not valid_mask.all():
        logger.debug("_encode_fine_truth: dropping %d rows with unsupported fine labels", int((~valid_mask).sum()))
    logger.debug("_encode_fine_truth: valid rows=%d", int(valid_mask.sum()))
    return encoded.loc[valid_mask].astype(int).to_numpy(), valid_mask

def train_dev_test_split(df: pd.DataFrame, y: Sequence, random_state: int):
    y = np.asarray(y); groups = np.asarray(df["evaluation_group"].astype(str))
    logger.debug("train_dev_test_split: n=%d unique_groups=%d", len(y), len(np.unique(groups)))
    if len(np.unique(groups)) < 3:
        logger.error("train_dev_test_split: only %d unique groups", len(np.unique(groups)))
        raise ValueError("Need at least three unique groups for grouped train/dev/test split")
    if not _can_stratify(y):
        logger.error("train_dev_test_split: cannot stratify classes: %s", pd.Series(y).value_counts().to_dict())
        raise ValueError(f"Cannot stratify classes: {pd.Series(y).value_counts().to_dict()}")
    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=random_state); folds = list(sgkf.split(df, y, groups)); train_dev_idx, test_idx = folds[0]
    remain = df.iloc[train_dev_idx].copy(); y_remain = y[train_dev_idx]; remain_groups = remain["evaluation_group"].astype(str).to_numpy()
    sgkf2 = StratifiedGroupKFold(n_splits=4, shuffle=True, random_state=random_state + 1); inner = list(sgkf2.split(remain, y_remain, remain_groups)); tr_rel, dev_rel = inner[0]
    result = train_dev_idx[tr_rel], train_dev_idx[dev_rel], test_idx
    logger.debug("train_dev_test_split: train=%d dev=%d test=%d", len(result[0]), len(result[1]), len(result[2]))
    return result

def _build_pipeline(estimator, scaled=False):
    steps = [("imputer", SimpleImputer(strategy="median", add_indicator=True))]
    if scaled: steps.append(("scaler", StandardScaler()))
    steps.append(("classifier", estimator))
    logger.debug("_build_pipeline: estimator=%s scaled=%s", type(estimator).__name__, scaled)
    return Pipeline(steps)

def build_models(seed: int):
    models = {
        "logistic_regression": _build_pipeline(LogisticRegression(max_iter=3000, class_weight="balanced", random_state=seed), True),
        "random_forest": _build_pipeline(RandomForestClassifier(n_estimators=500, class_weight="balanced", random_state=seed, n_jobs=-1)),
        "extra_trees": _build_pipeline(ExtraTreesClassifier(n_estimators=500, class_weight="balanced", random_state=seed, n_jobs=-1)),
        "svm_rbf": _build_pipeline(CalibratedClassifierCV(SVC(C=1.0, kernel="rbf", class_weight="balanced", probability=False, random_state=seed), ensemble=False), True),
        "sklearn_mlp": _build_pipeline(MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=800, early_stopping=True, random_state=seed), True),
        "knn": _build_pipeline(KNeighborsClassifier(n_neighbors=5, weights="distance"), True),
        "gaussian_nb": _build_pipeline(GaussianNB(), True),
        "decision_tree": _build_pipeline(DecisionTreeClassifier(class_weight="balanced", random_state=seed)),
        "gradient_boosting": _build_pipeline(GradientBoostingClassifier(n_estimators=200, learning_rate=0.05, max_depth=3, random_state=seed)),
        "hist_gradient_boosting": _build_pipeline(HistGradientBoostingClassifier(max_iter=150, learning_rate=0.05, max_depth=3, min_samples_leaf=10, random_state=seed)),
        "adaboost": _build_pipeline(AdaBoostClassifier(n_estimators=200, learning_rate=0.3, random_state=seed)),
        "bagging": _build_pipeline(BaggingClassifier(n_estimators=100, random_state=seed, n_jobs=-1)),
        "lda": _build_pipeline(LinearDiscriminantAnalysis(), True),
    }
    if lgb is not None: models["lightgbm"] = _build_pipeline(lgb.LGBMClassifier(n_estimators=300, learning_rate=0.03, num_leaves=31, random_state=seed, verbosity=-1))
    if xgb is not None: models["xgboost"] = _build_pipeline(xgb.XGBClassifier(n_estimators=300, learning_rate=0.03, max_depth=5, subsample=0.85, colsample_bytree=0.85, objective="multi:softprob", eval_metric="mlogloss", tree_method="hist", random_state=seed))
    if CatBoostClassifier is not None: models["catboost"] = _build_pipeline(CatBoostClassifier(iterations=300, learning_rate=0.05, depth=6, verbose=0, random_seed=seed, auto_class_weights="Balanced"))
    if TorchMLPClassifier is not None: models["torch_mlp"] = _build_pipeline(TorchMLPClassifier(random_state=seed), True)
    logger.debug("build_models: number of models=%d", len(models))
    return models

def evaluate_numeric(y_true, y_pred, labels):
    y_true = np.asarray(y_true, dtype=int); y_pred = np.asarray(y_pred, dtype=int); ids = list(range(len(labels)))
    logger.debug("evaluate_numeric: y_true=%d y_pred=%d labels=%d", len(y_true), len(y_pred), len(labels))
    result = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": _safe_balanced_accuracy(y_true, y_pred, ids),
        "macro_precision": float(precision_score(y_true, y_pred, labels=ids, average="macro", zero_division=0)),
        "macro_recall": float(recall_score(y_true, y_pred, labels=ids, average="macro", zero_division=0)),
        "macro_f1": float(f1_score(y_true, y_pred, labels=ids, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, labels=ids, average="weighted", zero_division=0)),
        "coverage": 1.0,
        "selective_accuracy": float(accuracy_score(y_true, y_pred)),
        "overall_accuracy_with_abstain_error": float(accuracy_score(y_true, y_pred)),
    }
    logger.debug("evaluate_numeric: metrics=%s", result)
    return result

def iter_nonempty_method_combinations(methods: Sequence[str]) -> Iterable[Tuple[str, Tuple[str, ...]]]:
    names = {1: "INDIVIDUAL", 2: "PAIR", 3: "TRIPLE", 4: "QUADRUPLE", 5: "QUINTUPLE", 6: "SEXTUPLE", 7: "SEPTUPLE"}
    logger.debug("iter_nonempty_method_combinations: methods=%s", list(methods))
    for n in range(1, len(methods) + 1):
        for combo in combinations(tuple(methods), n):
            yield names[n], combo

def _prepare_truth(labeled_df):
    fine = labeled_df["fault_type_label"].map(normalize_fault)
    fine = fine.where(fine.isin(cfg.BENCHMARK_FINE_CLASSES), ABSTAIN)
    fine_conflict = labeled_df.get("fine_label_conflict", pd.Series(False, index=labeled_df.index)).astype(bool)
    fine = fine.where(~fine_conflict, ABSTAIN)
    coarse = fine.map(unify_fault)
    coarse_conflict = labeled_df.get("coarse_label_conflict", pd.Series(False, index=labeled_df.index)).astype(bool)
    coarse = coarse.where(~coarse_conflict, ABSTAIN)
    logger.debug("_prepare_truth: valid fine=%d coarse=%d", int((fine != ABSTAIN).sum()), int((coarse != ABSTAIN).sum()))
    return fine, coarse

def _make_ambiguity_eval_split(labeled_df, seed):
    logger.debug("_make_ambiguity_eval_split: input rows=%d seed=%d", len(labeled_df), seed)
    fine = labeled_df["fault_type_label"].map(normalize_fault)
    conflict = labeled_df.get("fine_label_conflict", pd.Series(False, index=labeled_df.index)).astype(bool)
    coarse_conflict = labeled_df.get("coarse_label_conflict", pd.Series(False, index=labeled_df.index)).astype(bool)
    valid = fine.isin(set(cfg.BENCHMARK_FINE_CLASSES) | set(cfg.BENCHMARK_AMBIGUOUS_FINE_CLASSES)) & ~conflict & ~coarse_conflict
    data = labeled_df.loc[valid].reset_index(drop=True).copy()
    if data.empty:
        logger.debug("_make_ambiguity_eval_split: empty")
        return data, np.array([], dtype=int), np.array([], dtype=int), np.array([], dtype=int)
    data["evaluation_group"] = data["evaluation_group"].astype(str)
    coarse = data["fault_type_label"].map(normalize_fault).map(unify_fault)
    coarse_codes = coarse.map({c: i for i, c in enumerate(cfg.COARSE_FAULT_GROUPS)}).astype(int).to_numpy()
    tr, dev, test = train_dev_test_split(data, coarse_codes, seed)
    logger.debug("_make_ambiguity_eval_split: data=%d train=%d dev=%d test=%d", len(data), len(tr), len(dev), len(test))
    return data, tr, dev, test


def benchmark_traditional_individual(labeled_df):
    """Evaluate individual traditional methods on DEV and LOCKED TEST."""
    logger.info("benchmark_traditional_individual: start rows=%d", len(labeled_df))
    eval_df = labeled_df.copy().reset_index(drop=True)
    fine_truth_all, coarse_truth_all = _prepare_truth(eval_df)
    conflict = eval_df.get("fine_label_conflict", pd.Series(False, index=eval_df.index)).astype(bool)
    coarse_conflict = eval_df.get("coarse_label_conflict", pd.Series(False, index=eval_df.index)).astype(bool)
    valid = fine_truth_all.isin(cfg.BENCHMARK_FINE_CLASSES) & ~conflict & ~coarse_conflict
    eval_df = eval_df.loc[valid].reset_index(drop=True)
    fine_truth = fine_truth_all.loc[valid].reset_index(drop=True)
    coarse_truth = coarse_truth_all.loc[valid].reset_index(drop=True)
    logger.debug("benchmark_traditional_individual: valid rows after filter=%d", len(eval_df))
    eval_df["evaluation_group"] = eval_df.get("evaluation_group", eval_df[list(cfg.COMMON_BENCHMARK_GASES)].round(8).fillna(-999999.0).astype(str).agg("|".join, axis=1)).astype(str)
    mapping = {c: i for i, c in enumerate(cfg.BENCHMARK_FINE_CLASSES)}
    y = fine_truth.map(mapping).astype(int).to_numpy()
    tr, dev, test = train_dev_test_split(eval_df, y, cfg.RANDOM_STATE)
    rows = []
    for method, column in cfg.DIAGNOSTIC_METHOD_TO_COLUMN.items():
        logger.debug("benchmark_traditional_individual: method=%s column=%s", method, column)
        pred = labeled_df.get(column, pd.Series(ABSTAIN, index=labeled_df.index)).map(normalize_fault).loc[valid].reset_index(drop=True)
        for split_name, idx in (("development", dev), ("locked_test", test)):
            fine_metric = _evaluate_method_labels_safely(fine_truth.iloc[idx], pred.iloc[idx], cfg.BENCHMARK_FINE_CLASSES)
            coarse_metric = _evaluate_method_labels_safely(coarse_truth.iloc[idx], pred.iloc[idx].map(unify_fault), cfg.COARSE_FAULT_GROUPS)
            rows.append({"method": method, "granularity": "fine", "split": split_name, **fine_metric})
            rows.append({"method": method, "granularity": "coarse", "split": split_name, **coarse_metric})
        amb_data, _, amb_dev, amb_test = _make_ambiguity_eval_split(labeled_df, cfg.RANDOM_STATE)
        if not amb_data.empty:
            amb_truth = amb_data["fault_type_label"].map(normalize_fault); amb_pred = amb_data.get(column, pd.Series(ABSTAIN, index=amb_data.index)).map(normalize_fault).reset_index(drop=True)
            for split_name, idx in (("development", amb_dev), ("locked_test", amb_test)):
                metric = evaluate_ambiguous_fine_predictions(amb_truth.iloc[idx], amb_pred.iloc[idx])
                rows.append({"method": method, "granularity": "fine_ambiguous_tolerant", "split": split_name, **metric})
    result = pd.DataFrame(rows)
    result["selected_on_development"] = False
    for granularity in ("fine", "coarse"):
        dev_rows = result[(result["granularity"] == granularity) & (result["split"] == "development")]
        if not dev_rows.empty:
            best = dev_rows.sort_values(["macro_f1", "balanced_accuracy", "coverage"], ascending=False, na_position="last").iloc[0]
            result.loc[(result["granularity"] == granularity) & (result["method"] == best["method"]), "selected_on_development"] = True
    logger.info("benchmark_traditional_individual: generated %d rows", len(result))
    result.to_csv(BENCHMARK_DIR / "traditional_individual_benchmark.csv", index=False, encoding="utf-8-sig")
    return result

def benchmark_traditional_combinations(labeled_df, split=None):
    """Evaluate all combinations on DEV and LOCKED TEST; selection is based only on DEV."""
    logger.info("benchmark_traditional_combinations: start rows=%d", len(labeled_df))
    eval_df = labeled_df.copy().reset_index(drop=True)
    fine_truth_all, coarse_truth_all = _prepare_truth(eval_df)
    conflict = eval_df.get("fine_label_conflict", pd.Series(False, index=eval_df.index)).astype(bool)
    coarse_conflict = eval_df.get("coarse_label_conflict", pd.Series(False, index=eval_df.index)).astype(bool)
    valid_mask = fine_truth_all.isin(cfg.BENCHMARK_FINE_CLASSES) & ~conflict & ~coarse_conflict
    eval_df = eval_df.loc[valid_mask].reset_index(drop=True)
    fine_truth = fine_truth_all.loc[valid_mask].reset_index(drop=True)
    coarse_truth = coarse_truth_all.loc[valid_mask].reset_index(drop=True)
    if eval_df.empty:
        logger.error("benchmark_traditional_combinations: no valid rows after conflict filtering")
        raise ValueError("No valid benchmark rows remain after conflict filtering.")
    if "evaluation_group" not in eval_df.columns:
        eval_df["evaluation_group"] = eval_df[list(cfg.COMMON_BENCHMARK_GASES)].round(8).fillna(-999999.0).astype(str).agg("|".join, axis=1)
    eval_df["evaluation_group"] = eval_df["evaluation_group"].astype(str)
    mapping = {c: i for i, c in enumerate(cfg.BENCHMARK_FINE_CLASSES)}
    y = fine_truth.map(mapping).astype(int).to_numpy()
    tr, dev, test = train_dev_test_split(eval_df, y, cfg.RANDOM_STATE)
    logger.debug("benchmark_traditional_combinations: valid rows=%d train=%d dev=%d test=%d", len(eval_df), len(tr), len(dev), len(test))
    rows = []
    for level, combo in iter_nonempty_method_combinations(cfg.DIAGNOSTIC_METHODS):
        logger.debug("benchmark_traditional_combinations: evaluating %s %s", level, combo)
        predicted = apply_consensus_from_existing_diagnostics(eval_df, combo)
        for split_name, idx in (("development", dev), ("locked_test", test)):
            fine_metric = _evaluate_method_labels_safely(fine_truth.iloc[idx], predicted["consensus_fault"].iloc[idx], cfg.BENCHMARK_FINE_CLASSES)
            coarse_metric = _evaluate_method_labels_safely(coarse_truth.iloc[idx], predicted["consensus_fault_group"].iloc[idx], cfg.COARSE_FAULT_GROUPS)
            rows.append({"combination_level": level, "methods": "+".join(combo), "method_count": len(combo), "split": split_name, "granularity": "fine", **fine_metric})
            rows.append({"combination_level": level, "methods": "+".join(combo), "method_count": len(combo), "split": split_name, "granularity": "coarse", **coarse_metric})
    amb_data, _, amb_dev, amb_test = _make_ambiguity_eval_split(labeled_df, cfg.RANDOM_STATE)
    if not amb_data.empty:
        logger.debug("benchmark_traditional_combinations: ambiguity data rows=%d", len(amb_data))
        for level, combo in iter_nonempty_method_combinations(cfg.DIAGNOSTIC_METHODS):
            predicted_amb = apply_consensus_from_existing_diagnostics(amb_data, combo)
            for split_name, idx in (("development", amb_dev), ("locked_test", amb_test)):
                metric = evaluate_ambiguous_fine_predictions(amb_data["fault_type_label"].map(normalize_fault).iloc[idx], predicted_amb["consensus_fault"].iloc[idx])
                rows.append({"combination_level": level, "methods": "+".join(combo), "method_count": len(combo), "split": split_name, "granularity": "fine_ambiguous_tolerant", **metric})
    result = pd.DataFrame(rows)
    result["selected_on_development"] = False
    for granularity in ("fine", "coarse"):
        dev_rows = result[(result["granularity"] == granularity) & (result["split"] == "development")]
        if not dev_rows.empty:
            best = dev_rows.sort_values(["macro_f1", "balanced_accuracy", "coverage"], ascending=False, na_position="last").iloc[0]
            result.loc[(result["granularity"] == granularity) & (result["methods"] == best["methods"]), "selected_on_development"] = True
    result.to_csv(BENCHMARK_DIR / "traditional_combinations_benchmark.csv", index=False, encoding="utf-8-sig")
    logger.info("Traditional combination benchmark complete | rows=%d", len(result))
    return result

def benchmark_traditional_ppm_coverage(labeled_df):
    logger.info("benchmark_traditional_ppm_coverage: start rows=%d", len(labeled_df))
    rows = []
    for method, column in cfg.DIAGNOSTIC_METHOD_TO_COLUMN.items():
        active = labeled_df.get(column, pd.Series(ABSTAIN, index=labeled_df.index)).map(normalize_fault) != ABSTAIN
        logger.debug("benchmark_traditional_ppm_coverage: method=%s active=%d", method, int(active.sum()))
        for gas in cfg.COMMON_BENCHMARK_GASES:
            x = pd.to_numeric(labeled_df[gas], errors="coerce"); valid = active & x.notna(); values = x[valid].to_numpy(float)
            rows.append({"method": method, "gas": gas, "active_count": int(len(values)), "coverage": float(len(values) / max(len(labeled_df), 1)), "min_ppm": float(np.min(values)) if len(values) else np.nan, "p05_ppm": float(np.quantile(values, 0.05)) if len(values) else np.nan, "median_ppm": float(np.median(values)) if len(values) else np.nan, "p95_ppm": float(np.quantile(values, 0.95)) if len(values) else np.nan, "max_ppm": float(np.max(values)) if len(values) else np.nan, "observed_range_ppm": float(np.max(values) - np.min(values)) if len(values) else np.nan})
    result = pd.DataFrame(rows)
    result.to_csv(BENCHMARK_DIR / "traditional_ppm_coverage.csv", index=False, encoding="utf-8-sig")
    class_coverage = empirical_fault_class_coverage(labeled_df, "fine")
    class_coverage.to_csv(BENCHMARK_DIR / "traditional_fault_class_coverage.csv", index=False, encoding="utf-8-sig")
    logger.info("benchmark_traditional_ppm_coverage: generated %d rows", len(result))
    return result


def _benchmark_supervised_feature_mode(labeled_df, seed, feature_mode, models=None):
    logger.info("_benchmark_supervised_feature_mode: feature_mode=%s seed=%d", feature_mode, seed)
    labels, _ = _prepare_truth(labeled_df); mask = labels != ABSTAIN; data = labeled_df.loc[mask].reset_index(drop=True); y_names = labels.loc[mask].reset_index(drop=True)
    if "fine_label_conflict" in data.columns:
        keep = ~data["fine_label_conflict"].astype(bool); data = data.loc[keep].reset_index(drop=True); y_names = y_names.loc[keep].reset_index(drop=True)
    mapping = {c: i for i, c in enumerate(cfg.BENCHMARK_FINE_CLASSES)}; encoded_y = y_names.map(mapping); valid = encoded_y.notna()
    data = data.loc[valid].reset_index(drop=True); y_names = y_names.loc[valid].reset_index(drop=True); encoded_y = encoded_y.loc[valid]
    if data.empty:
        logger.error("_benchmark_supervised_feature_mode: no valid rows")
        raise ValueError("No valid labeled benchmark rows remain for supervised training.")
    y = encoded_y.astype(int).to_numpy(); data = data.assign(evaluation_group=data["evaluation_group"].astype(str)); tr, dev, test = train_dev_test_split(data, y, seed)
    logger.debug("_benchmark_supervised_feature_mode: train=%d dev=%d test=%d", len(tr), len(dev), len(test))
    Xdf = build_feature_frame(data, feature_mode, "fine"); model_map = models or build_models(seed); dev_rows = []
    for name, model in model_map.items():
        logger.debug("_benchmark_supervised_feature_mode: fitting %s", name)
        try:
            model.fit(Xdf.iloc[tr], y[tr]); pred = _predict_model(model, Xdf.iloc[dev]); pred = np.asarray(pred).reshape(-1); dev_rows.append({"model": name, "feature_mode": feature_mode, "dataset": "external_labeled", "split": "development", "granularity": "fine", "selected_on_dev": False, **evaluate_numeric(y[dev], pred, cfg.BENCHMARK_FINE_CLASSES)})
        except Exception as exc:
            logger.warning("Supervised dev failed: %s: %s", name, exc)
    dev_df = pd.DataFrame(dev_rows)
    if dev_df.empty: return pd.DataFrame()
    best = dev_df.sort_values(["macro_f1", "balanced_accuracy", "accuracy"], ascending=False).iloc[0]["model"]
    dev_df.loc[dev_df["model"] == best, "selected_on_dev"] = True
    logger.debug("_benchmark_supervised_feature_mode: best dev model=%s", best)
    train_dev = np.concatenate([tr, dev]); test_rows = []
    for name in model_map:
        logger.debug("_benchmark_supervised_feature_mode: retraining %s on train+dev", name)
        try:
            model = build_models(seed)[name]; model.fit(Xdf.iloc[train_dev], y[train_dev]); pred = _predict_model(model, Xdf.iloc[test]); pred = np.asarray(pred).reshape(-1); test_rows.append({"model": name, "feature_mode": feature_mode, "dataset": "external_labeled", "split": "locked_test", "granularity": "fine", "selected_on_dev": name == best, **evaluate_numeric(y[test], pred, cfg.BENCHMARK_FINE_CLASSES)})
        except Exception as exc:
            logger.warning("Supervised test failed: %s: %s", name, exc)
    result = pd.concat([dev_df, pd.DataFrame(test_rows)], ignore_index=True)
    logger.debug("_benchmark_supervised_feature_mode: result rows=%d", len(result))
    return result

def benchmark_supervised_models(labeled_df, seed):
    logger.info("benchmark_supervised_models: start rows=%d", len(labeled_df))
    parts = []
    for mode in ("gas_only", "gas_plus_traditional"):
        part = _benchmark_supervised_feature_mode(labeled_df, seed, mode)
        if not part.empty: parts.append(part)
    result = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
    result.to_csv(BENCHMARK_DIR / "supervised_fault_benchmark.csv", index=False, encoding="utf-8-sig")
    logger.info("benchmark_supervised_models: generated %d rows", len(result))
    return result

def _train_weak_students(df, granularity, seed):
    logger.info("_train_weak_students: granularity=%s seed=%d", granularity, seed)
    target = f"weak_{granularity}_fault_group" if granularity == "coarse" else f"weak_{granularity}_fault"; abstain = f"weak_{granularity}_is_ABSTAIN"; groups = list(cfg.WEAK_COARSE_GROUPS if granularity == "coarse" else cfg.BENCHMARK_FINE_CLASSES)
    clean, y, present = create_student_training_targets(df, target, abstain, groups)
    logger.debug("_train_weak_students: clean shape=%s y size=%d present labels=%s", clean.shape, len(y), present)
    artifacts = {}
    for feature_mode in ("gas_only", "gas_plus_traditional"):
        Xdf = build_feature_frame(clean, feature_mode, granularity); models = build_models(seed)
        logger.debug("_train_weak_students: feature_mode=%s models=%d", feature_mode, len(models))
        for name, model in models.items():
            try:
                model.fit(Xdf, y); artifacts[f"{feature_mode}__{name}"] = {"model": model, "feature_mode": feature_mode, "granularity": granularity, "features": list(Xdf.columns), "labels": list(present), "n_training_rows": int(len(clean)), "class_counts": {str(k): int(v) for k, v in pd.Series(y).value_counts().items()}}
                logger.debug("_train_weak_students: trained %s__%s", feature_mode, name)
            except Exception as exc:
                logger.warning("Weak student failed: %s/%s: %s", feature_mode, name, exc)
    if not artifacts:
        logger.error("_train_weak_students: no weak students trained for %s", granularity)
        raise RuntimeError(f"No weak students trained for {granularity}")
    logger.debug("_train_weak_students: artifacts count=%d", len(artifacts))
    return artifacts

def _align_feature_frame(df, feature_names, granularity):
    feature_names = list(feature_names); mode = "gas_plus_traditional" if any("__" in c for c in feature_names) else "gas_only"; frame = build_feature_frame(df, mode, granularity); result = frame.reindex(columns=feature_names, fill_value=0.0)
    logger.debug("_align_feature_frame: mode=%s granularity=%s input=%s output=%s", mode, granularity, frame.shape, result.shape)
    return result

def benchmark_weak_transfer(labeled_df, weak_students, seed):
    logger.info("benchmark_weak_transfer: start rows=%d", len(labeled_df))
    fine_truth, coarse_truth = _prepare_truth(labeled_df); conflict = labeled_df.get("fine_label_conflict", pd.Series(False, index=labeled_df.index)).astype(bool); valid = (fine_truth != ABSTAIN) & ~conflict
    data = labeled_df.loc[valid].reset_index(drop=True); fine_truth = fine_truth.loc[valid].reset_index(drop=True); coarse_truth = coarse_truth.loc[valid].reset_index(drop=True)
    mapping = {c: i for i, c in enumerate(cfg.BENCHMARK_FINE_CLASSES)}; encoded = fine_truth.map(mapping); valid_encoded = encoded.notna()
    keep = valid_encoded.to_numpy(dtype=bool); data = data.loc[keep].reset_index(drop=True); fine_truth = fine_truth.loc[keep].reset_index(drop=True); coarse_truth = coarse_truth.loc[keep].reset_index(drop=True); y_split = encoded.loc[keep].astype(int).to_numpy()
    if data.empty:
        logger.error("benchmark_weak_transfer: no valid rows")
        raise ValueError("No valid rows remain for weak-transfer benchmark.")
    data = data.assign(evaluation_group=data["evaluation_group"].astype(str)); _, dev, test = train_dev_test_split(data, y_split, seed)
    logger.debug("benchmark_weak_transfer: valid rows=%d dev=%d test=%d", len(data), len(dev), len(test))
    rows = []
    for granularity in ("coarse", "fine"):
        artifacts = weak_students.get(granularity, {})
        logger.debug("benchmark_weak_transfer: granularity=%s artifacts=%d", granularity, len(artifacts))
        for key, artifact in artifacts.items():
            feature_names = artifact["features"]; Xdf = _align_feature_frame(data, feature_names, granularity); model = artifact["model"]; labels = list(artifact["labels"])
            pred_dev = _decode_class_predictions(_predict_model(model, Xdf.iloc[dev]), labels); pred_test = _decode_class_predictions(_predict_model(model, Xdf.iloc[test]), labels)
            pred_dev_labels = pd.Series(pred_dev, index=dev); pred_test_labels = pd.Series(pred_test, index=test)
            if granularity == "fine": td = fine_truth.iloc[dev]; tt = fine_truth.iloc[test]; allowed = cfg.BENCHMARK_FINE_CLASSES
            else: td = coarse_truth.iloc[dev]; tt = coarse_truth.iloc[test]; allowed = cfg.COARSE_FAULT_GROUPS
            dev_metric = _evaluate_method_labels_safely(td.reset_index(drop=True), pred_dev_labels.to_numpy(), allowed); test_metric = _evaluate_method_labels_safely(tt.reset_index(drop=True), pred_test_labels.to_numpy(), allowed)
            base = {"granularity": granularity, "model": key.split("__", 1)[1], "feature_mode": artifact["feature_mode"], "training_dataset": "unlabeled_operational_weak", "evaluation_dataset": "external_labeled", "selected_on_dev": False}
            rows.append({**base, "split": "development", **dev_metric}); rows.append({**base, "split": "locked_test", **test_metric})
    amb_data, _, amb_dev, amb_test = _make_ambiguity_eval_split(labeled_df, seed)
    if not amb_data.empty and "fine" in weak_students:
        amb_truth = amb_data["fault_type_label"].map(normalize_fault)
        logger.debug("benchmark_weak_transfer: ambiguity data rows=%d", len(amb_data))
        for key, artifact in weak_students.get("fine", {}).items():
            feature_names = artifact["features"]; Xdf_amb = _align_feature_frame(amb_data, feature_names, "fine"); pred_amb = pd.Series(_decode_class_predictions(_predict_model(artifact["model"], Xdf_amb), list(artifact["labels"])), index=amb_data.index)
            for split_name, idx in (("development", amb_dev), ("locked_test", amb_test)):
                metric = evaluate_ambiguous_fine_predictions(amb_truth.iloc[idx], pred_amb.iloc[idx])
                rows.append({"granularity": "fine_ambiguous_tolerant", "model": key.split("__", 1)[1], "feature_mode": artifact["feature_mode"], "training_dataset": "unlabeled_operational_weak", "evaluation_dataset": "external_labeled", "selected_on_dev": False, "split": split_name, "macro_f1": np.nan, "balanced_accuracy": np.nan, "macro_precision": np.nan, "macro_recall": np.nan, "weighted_f1": np.nan, **metric})
    result = pd.DataFrame(rows)
    if not result.empty:
        for granularity in ("fine", "coarse"):
            sub = result[(result["granularity"] == granularity) & (result["split"] == "development")]
            if not sub.empty:
                best_idx = sub.sort_values(["macro_f1", "balanced_accuracy", "accuracy"], ascending=False).index[0]; best_model = result.loc[best_idx, "model"]; best_mode = result.loc[best_idx, "feature_mode"]; mask = (result["granularity"] == granularity) & (result["model"] == best_model) & (result["feature_mode"] == best_mode); result.loc[mask, "selected_on_dev"] = True
    result.to_csv(BENCHMARK_DIR / "weak_transfer_fault_benchmark.csv", index=False, encoding="utf-8-sig")
    logger.info("benchmark_weak_transfer: generated %d rows", len(result))
    return result

def benchmark_weak_traditional_hybrids(labeled_df, weak_students, seed):
    """Evaluate non-weighted hybrid policies on the independent labeled benchmark.

    agreement_only keeps a prediction only when the weak student and the unweighted
    traditional consensus agree exactly. Disagreement becomes ABSTAIN; no arbitrary
    numeric fusion weight is introduced.
    """
    logger.info("benchmark_weak_traditional_hybrids: start rows=%d", len(labeled_df))
    fine_truth, coarse_truth = _prepare_truth(labeled_df)
    conflict = labeled_df.get("fine_label_conflict", pd.Series(False, index=labeled_df.index)).astype(bool)
    valid = (fine_truth != ABSTAIN) & ~conflict
    data = labeled_df.loc[valid].reset_index(drop=True)
    fine_truth = fine_truth.loc[valid].reset_index(drop=True); coarse_truth = coarse_truth.loc[valid].reset_index(drop=True)
    encoded = fine_truth.map({c: i for i, c in enumerate(cfg.BENCHMARK_FINE_CLASSES)})
    keep = encoded.notna().to_numpy(dtype=bool)
    data = data.loc[keep].reset_index(drop=True); fine_truth = fine_truth.loc[keep].reset_index(drop=True); coarse_truth = coarse_truth.loc[keep].reset_index(drop=True); y = encoded.loc[keep].astype(int).to_numpy()
    if data.empty:
        logger.debug("benchmark_weak_traditional_hybrids: no valid rows")
        return pd.DataFrame()
    data = data.assign(evaluation_group=data["evaluation_group"].astype(str))
    _, dev, test = train_dev_test_split(data, y, seed)
    logger.debug("benchmark_weak_traditional_hybrids: valid rows=%d dev=%d test=%d", len(data), len(dev), len(test))
    traditional = apply_consensus(data)
    rows = []
    for granularity in ("coarse", "fine"):
        artifacts = weak_students.get(granularity, {})
        logger.debug("benchmark_weak_traditional_hybrids: granularity=%s artifacts=%d", granularity, len(artifacts))
        for key, artifact in artifacts.items():
            feature_names = artifact["features"]; Xdf = _align_feature_frame(data, feature_names, granularity); model = artifact["model"]; labels = list(artifact["labels"])
            student = pd.Series(_decode_class_predictions(_predict_model(model, Xdf), labels), index=data.index)
            trad = traditional["consensus_fault_group" if granularity == "coarse" else "consensus_fault"].map(unify_fault if granularity == "coarse" else normalize_fault)
            hybrid = student.where(student == trad, ABSTAIN)
            if granularity == "fine": td, tt, allowed = fine_truth, fine_truth, cfg.BENCHMARK_FINE_CLASSES
            else: td, tt, allowed = coarse_truth, coarse_truth, cfg.COARSE_FAULT_GROUPS
            for split_name, idx in (("development", dev), ("locked_test", test)):
                yt = td.iloc[idx]; yp = hybrid.iloc[idx]
                metric = _evaluate_method_labels_safely(yt.reset_index(drop=True), yp.to_numpy(), allowed)
                ambiguous = evaluate_ambiguous_fine_predictions(yt, yp) if granularity == "fine" else {}
                rows.append({"granularity": granularity, "model": key.split("__", 1)[1], "feature_mode": artifact["feature_mode"], "hybrid_policy": "agreement_only", "training_dataset": "unlabeled_operational_weak", "evaluation_dataset": "external_labeled", "split": split_name, "selected_on_dev": False, **metric, "ambiguous_tolerant_accuracy": ambiguous.get("accuracy", np.nan), "ambiguous_truth_count": ambiguous.get("ambiguous_truth_count", 0)})
    result = pd.DataFrame(rows)
    if not result.empty:
        for granularity in result["granularity"].unique():
            sub = result[(result["granularity"] == granularity) & (result["split"] == "development")]
            if not sub.empty:
                best_idx = sub.sort_values(["macro_f1", "balanced_accuracy", "coverage"], ascending=False, na_position="last").index[0]
                best_model = result.loc[best_idx, "model"]; best_mode = result.loc[best_idx, "feature_mode"]
                mask = (result["granularity"] == granularity) & (result["model"] == best_model) & (result["feature_mode"] == best_mode)
                result.loc[mask, "selected_on_dev"] = True
    result.to_csv(BENCHMARK_DIR / "weak_traditional_hybrid_benchmark.csv", index=False, encoding="utf-8-sig")
    logger.info("benchmark_weak_traditional_hybrids: generated %d rows", len(result))
    return result


def benchmark_direct_supervised_transfer(unlabeled_df, labeled_df, seed):
    logger.debug("benchmark_direct_supervised_transfer: unlabeled_df shape=%s labeled_df shape=%s", unlabeled_df.shape, labeled_df.shape)
    result = benchmark_supervised_models(labeled_df, seed)
    candidates = result[(result["split"] == "locked_test") & result["selected_on_dev"].astype(bool)] if not result.empty else pd.DataFrame()
    candidates.to_csv(BENCHMARK_DIR / "direct_supervised_reference_selected_on_dev.csv", index=False, encoding="utf-8-sig")
    return candidates

def run_unlabeled_pipeline(seed, use_snorkel, save_model=True):
    logger.info("run_unlabeled_pipeline: start seed=%d use_snorkel=%s", seed, use_snorkel)
    raw = load_unlabeled(); logger.info("UNLABELED START | rows=%d | transformers=%d", len(raw), raw["transformer_id"].nunique()); df = prepare_unlabeled(raw)
    outputs = {}; weak_students = {"coarse": {}, "fine": {}}
    for granularity, groups in (("coarse", cfg.COARSE_FAULT_GROUPS), ("fine", cfg.BENCHMARK_FINE_CLASSES)):
        logger.info("run_unlabeled_pipeline: weak supervision for %s groups=%s", granularity, groups)
        out, model, out_groups, meta, L, probabilities, pairwise = weak_supervision_pipeline(df, DEFAULT_WEAK_METHODS, groups, use_snorkel=use_snorkel, random_state=seed, granularity=granularity)
        outputs[granularity] = out; pairwise.to_csv(BENCHMARK_DIR / f"weak_lf_pairwise_agreement_{granularity}.csv", index=False, encoding="utf-8-sig"); save_weak_supervision_artifacts(out, model, out_groups, meta, granularity=granularity); weak_students[granularity] = _train_weak_students(out, granularity, seed); del L, probabilities
    merge_keys = ["transformer_id", "sample_day"]; df = outputs["coarse"].copy(); fine_cols = [c for c in outputs["fine"].columns if c.startswith("weak_fine_")]; df = df.merge(outputs["fine"][merge_keys + fine_cols], on=merge_keys, how="left", suffixes=("", "_fine"))
    logger.debug("run_unlabeled_pipeline: after merge shape=%s", df.shape)
    for granularity, artifacts in weak_students.items():
        for key, artifact in artifacts.items():
            Xdf = _align_feature_frame(df, artifact["features"], granularity); pred = _decode_class_predictions(_predict_model(artifact["model"], Xdf), list(artifact["labels"])); df[f"weak_student_{granularity}_{key}"] = pred
            if hasattr(artifact["model"], "predict_proba"):
                try:
                    proba = np.asarray(_predict_proba(artifact["model"], Xdf), dtype=float)
                    if proba.ndim == 2: confidence = np.max(proba, axis=1)
                    elif proba.ndim == 1: confidence = np.abs(proba)
                    else: confidence = np.full(len(df), np.nan, dtype=float)
                    df[f"weak_student_{granularity}_{key}_confidence"] = confidence
                except Exception:
                    logger.warning("Weak student confidence failed | %s | %s", granularity, key, exc_info=True)
    from severity import apply_severity
    logger.debug("run_unlabeled_pipeline: applying severity")
    df = apply_severity(df, nei_reference=None); ranking = build_transformer_ranking(df); log_ranking_diagnostics(ranking, 20)
    processed = DATASET_DIR / "processed"; processed.mkdir(parents=True, exist_ok=True); df.to_parquet(processed / "dga_unlabeled_processed.parquet", index=False); ranking.to_parquet(processed / "transformer_ranking.parquet", index=False); ranking.to_csv(REPORT_DIR / "transformer_ranking.csv", index=False, encoding="utf-8-sig")
    if save_model:
        MODEL_DIR.mkdir(parents=True, exist_ok=True); joblib.dump({"models": weak_students["coarse"], "training_type": "weak_supervision_plus_discriminative_ml", "training_dataset": str(UNLABELED_PATH), "features": MODEL_FEATURES}, FAULT_MODEL_COARSE_PATH); joblib.dump({"models": weak_students["fine"], "training_type": "weak_supervision_plus_discriminative_ml", "training_dataset": str(UNLABELED_PATH), "features": MODEL_FEATURES}, FAULT_MODEL_FINE_PATH); metadata = {"seed": seed, "unlabeled_dataset": str(UNLABELED_PATH), "weak_supervision": "Snorkel LabelModel or EM fallback", "student_feature_modes": ["gas_only", "gas_plus_traditional"], "student_model_count_coarse": len(weak_students["coarse"]), "student_model_count_fine": len(weak_students["fine"]), "severity_source": cfg.STANDARD, "severity_is_weighted": False, "severity_is_failure_probability": False, "ranking_policy": list(cfg.RANKING_POLICY), "ranking_is_weighted": False, "ranking_is_health_score": False, "benchmark_policy": "Operational unlabeled data are used for weak labels and student training only; labeled benchmark is reserved for external evaluation and locked test reporting."}; TRAINING_METADATA_PATH.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info("run_unlabeled_pipeline: complete final df shape=%s", df.shape)
    return df, ranking, weak_students

def run_labeled_benchmark(seed):
    logger.info("run_labeled_benchmark: start seed=%d", seed)
    labeled = apply_consensus(load_labeled_csv_data()); individual = benchmark_traditional_individual(labeled); combinations_result = benchmark_traditional_combinations(labeled, None); ppm = benchmark_traditional_ppm_coverage(labeled); pairwise = pairwise_label_agreement(labeled); pairwise.to_csv(BENCHMARK_DIR / "traditional_pairwise_agreement.csv", index=False, encoding="utf-8-sig"); method_summary = diagnostic_method_summary(labeled); method_summary.to_csv(BENCHMARK_DIR / "traditional_method_summary.csv", index=False, encoding="utf-8-sig"); supervised = benchmark_supervised_models(labeled, seed)
    class_coverage = empirical_fault_class_coverage(labeled, "fine"); class_coverage.to_csv(BENCHMARK_DIR / "traditional_fault_class_coverage.csv", index=False, encoding="utf-8-sig")
    benchmark = {"individual": individual, "combinations": combinations_result, "ppm_coverage": ppm, "class_coverage": class_coverage, "pairwise": pairwise, "method_summary": method_summary, "supervised": supervised}; benchmark["split_manifest"] = _write_split_manifest(labeled, seed)
    logger.info("run_labeled_benchmark: complete")
    return benchmark

def _write_split_manifest(labeled, seed):
    labels, _ = _prepare_truth(labeled); conflict = labeled.get("fine_label_conflict", pd.Series(False, index=labeled.index)).astype(bool); valid = (labels != ABSTAIN) & ~conflict; data = labeled.loc[valid].reset_index(drop=True); y = labels.loc[valid].map({c: i for i, c in enumerate(cfg.BENCHMARK_FINE_CLASSES)}).to_numpy(int); data = data.assign(evaluation_group=data["evaluation_group"].astype(str)); _, dev, test = train_dev_test_split(data, y, seed)
    manifest = data[["source_dataset", "source_row", "fault_type_label", "evaluation_group"]].copy(); split = np.full(len(data), "train", dtype=object); split[dev] = "development"; split[test] = "locked_test"; manifest["split"] = split; manifest.to_csv(BENCHMARK_DIR / "benchmark_split_manifest.csv", index=False, encoding="utf-8-sig")
    logger.debug("_write_split_manifest: rows=%d train=%d dev=%d test=%d", len(manifest), int((split=="train").sum()), len(dev), len(test))
    return manifest

def write_confusion_matrices(weak_transfer, supervised, labeled, seed):
    logger.debug("write_confusion_matrices: start")
    labels = _prepare_truth(labeled)[0]; conflict = labeled.get("fine_label_conflict", pd.Series(False, index=labeled.index)).astype(bool); valid = (labels != ABSTAIN) & ~conflict; data = labeled.loc[valid].reset_index(drop=True); y_names = labels.loc[valid].reset_index(drop=True); y = y_names.map({c: i for i, c in enumerate(cfg.BENCHMARK_FINE_CLASSES)}).to_numpy(int); _, _, test = train_dev_test_split(data.assign(evaluation_group=data["evaluation_group"].astype(str)), y, seed)
    for mode in supervised["feature_mode"].unique():
        sub = supervised[(supervised["feature_mode"] == mode) & (supervised["split"] == "locked_test")]
        if sub.empty: continue
        best = sub.sort_values(["macro_f1", "balanced_accuracy", "accuracy"], ascending=False).iloc[0]; model = build_models(seed)[best["model"]]; Xdf = build_feature_frame(data, mode, "fine"); train_mask = np.ones(len(data), dtype=bool); train_mask[test] = False; model.fit(Xdf.iloc[train_mask], y[train_mask]); pred = np.asarray(model.predict(Xdf.iloc[test])).reshape(-1); cm = confusion_matrix(y[test], pred, labels=list(range(len(cfg.BENCHMARK_FINE_CLASSES)))); out = pd.DataFrame(cm, index=cfg.BENCHMARK_FINE_CLASSES, columns=cfg.BENCHMARK_FINE_CLASSES); out.to_csv(BENCHMARK_DIR / f"confusion_supervised_{mode}.csv", encoding="utf-8-sig")
        logger.debug("write_confusion_matrices: saved confusion matrix for mode=%s", mode)

def main(args=None):
    parser = argparse.ArgumentParser(); parser.add_argument("--mode", choices=["benchmark", "unlabeled", "transfer", "all"], default="all"); parser.add_argument("--use-snorkel", action="store_true"); parser.add_argument("--seed", type=int, default=cfg.RANDOM_STATE); parsed = parser.parse_args(args)
    logger.info("main: mode=%s seed=%d use_snorkel=%s", parsed.mode, parsed.seed, parsed.use_snorkel)
    set_global_seed(parsed.seed); unlabeled_result = None
    if parsed.mode in {"unlabeled", "transfer", "all"}:
        unlabeled_result = run_unlabeled_pipeline(parsed.seed, parsed.use_snorkel)
    benchmark = None
    if parsed.mode in {"benchmark", "all"}:
        benchmark = run_labeled_benchmark(parsed.seed); print("\n=== Traditional individual ==="); print(benchmark["individual"].sort_values(["granularity", "macro_f1"], ascending=[True, False]).to_string(index=False)); print("\n=== Best traditional combinations on locked test ==="); print(benchmark["combinations"].query("split == 'locked_test'").sort_values(["granularity", "macro_f1"], ascending=[True, False]).groupby("granularity").head(10).to_string(index=False)); print("\n=== Supervised reference ==="); print(benchmark["supervised"].query("split == 'locked_test'").sort_values("macro_f1", ascending=False).head(20).to_string(index=False))
    if parsed.mode in {"transfer", "all"}:
        if unlabeled_result is None: unlabeled_result = run_unlabeled_pipeline(parsed.seed, parsed.use_snorkel)
        labeled = apply_consensus(load_labeled_csv_data()); weak_transfer = benchmark_weak_transfer(labeled, unlabeled_result[2], parsed.seed); print("\n=== Weak-supervision transfer ==="); print(weak_transfer.query("split == 'locked_test'").sort_values(["granularity", "macro_f1"], ascending=[True, False]).head(30).to_string(index=False))
    try:
        from experiment import build_excel_report
        build_excel_report(REPORT_DIR, DATASET_DIR / "processed", REPORT_DIR / "dga_research_report.xlsx"); logger.info("Excel report saved to %s", REPORT_DIR / "dga_research_report.xlsx")
    except Exception:
        logger.exception("Excel report generation failed")

if __name__ == "__main__":
    main()