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
    random.seed(seed); np.random.seed(seed)
    if TORCH_AVAILABLE:
        torch.manual_seed(seed)
        if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

def _decode_class_predictions(prediction, labels, default=ABSTAIN):
    arr = np.asarray(prediction)
    if arr.ndim == 0: arr = arr.reshape(1)
    elif arr.ndim > 1:
        if arr.size == arr.shape[0]: arr = arr.reshape(-1)
        else: arr = arr.reshape(arr.shape[0], -1)[:, 0]
    decoded = []
    for value in arr:
        try:
            scalar = np.asarray(value).reshape(-1)[0]; index = int(scalar); decoded.append(labels[index] if 0 <= index < len(labels) else default)
        except Exception: decoded.append(default)
    return decoded

def _predict_model(model, feature_frame: pd.DataFrame): return model.predict(feature_frame)
def _predict_proba(model, feature_frame: pd.DataFrame): return model.predict_proba(feature_frame)

def _safe_balanced_accuracy(y_true, y_pred, labels=None):
    y_true = np.asarray(y_true); y_pred = np.asarray(y_pred)
    if y_true.size == 0: return 0.0
    if labels is None: labels = np.unique(y_true)
    labels = list(labels)
    if not labels: return 0.0
    cm = confusion_matrix(y_true, y_pred, labels=labels); support = cm.sum(axis=1); present = support > 0
    if not np.any(present): return 0.0
    recalls = np.divide(np.diag(cm), support, out=np.zeros_like(support, dtype=float), where=support > 0)
    return float(np.mean(recalls[present]))

def _evaluate_method_labels_safely(y_true, y_pred, allowed_labels):
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="y_pred contains classes not in y_true", category=UserWarning)
        return evaluate_method_labels(y_true, y_pred, allowed_labels)

def _can_stratify(y):
    counts = pd.Series(y).value_counts(); return len(counts) >= 2 and counts.min() >= 2

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
            for _ in range(self.epochs):
                for xb, yb in loader: optimizer.zero_grad(); loss = criterion(self.model_(xb), yb); loss.backward(); optimizer.step()
            return self
        def predict_proba(self, X):
            X_tensor = torch.as_tensor(np.asarray(X), dtype=torch.float32); self.model_.eval()
            with torch.no_grad(): probabilities = torch.softmax(self.model_(X_tensor), dim=1).cpu().numpy()
            return probabilities
        def predict(self, X): return self.classes_[np.argmax(self.predict_proba(X), axis=1)]
else: TorchMLPClassifier = None

def _read_labeled(path: Path, label_col: str, source_name: str) -> pd.DataFrame:
    if not path.exists(): raise FileNotFoundError(path)
    df = pd.read_csv(path, encoding="utf-8-sig"); df.columns = [str(c).strip().lower().replace("\ufeff", "") for c in df.columns]
    if label_col not in df.columns: raise ValueError(f"{path.name}: missing label column {label_col!r}")
    for gas in cfg.FULL_EXTERNAL_GASES: df[gas] = pd.to_numeric(df[gas], errors="coerce") if gas in df.columns else np.nan
    df["fault_type_label"] = df[label_col].map(normalize_fault); df["source_dataset"] = source_name; df["source_row"] = np.arange(1, len(df) + 1); df["coarse_truth"] = df["fault_type_label"].map(unify_fault)
    return df

def _external_gas_key(df: pd.DataFrame) -> pd.Series:
    gas = list(cfg.COMMON_BENCHMARK_GASES); x = df[gas].round(8).fillna(-999999.0); return x.astype(str).agg("|".join, axis=1)

def load_labeled_csv_data() -> pd.DataFrame:
    a = _read_labeled(LABELED_CSV_PATH1, "label", "iec_tc10_121"); b = _read_labeled(LABELED_CSV_PATH2, "type", "dga_dataset"); combined = pd.concat([a, b], ignore_index=True); combined["_gas_key"] = _external_gas_key(combined)
    combined["fine_label_conflict"] = combined.groupby("_gas_key")["fault_type_label"].transform("nunique").gt(1); combined["coarse_label_conflict"] = combined.groupby("_gas_key")["coarse_truth"].transform("nunique").gt(1)
    conflict_df = combined[combined["fine_label_conflict"]].copy(); BENCHMARK_DIR.mkdir(parents=True, exist_ok=True)
    if not conflict_df.empty: conflict_df.to_csv(BENCHMARK_DIR / "external_benchmark_label_conflicts.csv", index=False, encoding="utf-8-sig")
    combined["duplicate_group_size"] = combined.groupby("_gas_key")["_gas_key"].transform("size"); combined["evaluation_group"] = combined["_gas_key"]
    return combined.drop(columns=["_gas_key"]).reset_index(drop=True)

def load_unlabeled(path: Path = UNLABELED_PATH) -> pd.DataFrame:
    if not path.exists(): raise FileNotFoundError(path)
    df = pd.read_parquet(path); required = {"transformer_id", "sample_day"}; missing = required - set(df.columns)
    if missing: raise ValueError(f"{path.name} requires {sorted(missing)}")
    df["sample_day"] = pd.to_datetime(df["sample_day"], errors="coerce"); return df.dropna(subset=["transformer_id", "sample_day"]).reset_index(drop=True)

def prepare_unlabeled(df: pd.DataFrame) -> pd.DataFrame:
    out = build_training_features_from_clean(df.copy()); out["sample_day"] = pd.to_datetime(out["sample_day"], errors="coerce"); out = out.sort_values(["transformer_id", "sample_day"], kind="mergesort").reset_index(drop=True); return apply_consensus(out)

def dataframe_to_gas_matrix(df: pd.DataFrame, features: Sequence[str]) -> np.ndarray:
    x = df.reindex(columns=list(features)).apply(pd.to_numeric, errors="coerce"); return x.replace([np.inf, -np.inf], np.nan).to_numpy(dtype=np.float32)

def _traditional_feature_matrix(df: pd.DataFrame, granularity: str = "fine") -> pd.DataFrame:
    out = pd.DataFrame(index=df.index); vocab = (list(cfg.BENCHMARK_FINE_CLASSES) if granularity == "fine" else list(cfg.WEAK_COARSE_GROUPS)) + [ABSTAIN]
    for method, column in cfg.DIAGNOSTIC_METHOD_TO_COLUMN.items():
        vals = df.get(column, pd.Series(ABSTAIN, index=df.index)).map(normalize_fault)
        if granularity == "coarse": vals = vals.map(unify_fault)
        for label in vocab:
            safe = str(label).lower().replace("/", "_").replace(" ", "_"); out[f"{method}__{safe}"] = (vals == label).astype(np.float32)
    return out

def build_feature_frame(df: pd.DataFrame, feature_mode: str, granularity: str = "fine") -> pd.DataFrame:
    gas = df.reindex(columns=MODEL_FEATURES).apply(pd.to_numeric, errors="coerce").astype(float)
    if feature_mode == "gas_only": return gas
    if feature_mode == "gas_plus_traditional": trad = _traditional_feature_matrix(df, granularity); return pd.concat([gas, trad], axis=1)
    raise ValueError(f"Unknown feature_mode: {feature_mode}")

def _encode_fine_truth(series):
    mapping = {label: idx for idx, label in enumerate(cfg.BENCHMARK_FINE_CLASSES)}; normalized = series.map(normalize_fault).astype(object); encoded = normalized.map(mapping); valid_mask = encoded.notna()
    if not valid_mask.all(): logger.warning("Dropped %d benchmark rows with unsupported fine labels.", int((~valid_mask).sum()))
    return encoded.loc[valid_mask].astype(int).to_numpy(), valid_mask

def train_dev_test_split(df: pd.DataFrame, y: Sequence, random_state: int):
    y = np.asarray(y); groups = np.asarray(df["evaluation_group"].astype(str))
    if len(np.unique(groups)) < 3: raise ValueError("Need at least three unique groups for grouped train/dev/test split")
    if not _can_stratify(y): raise ValueError(f"Cannot stratify classes: {pd.Series(y).value_counts().to_dict()}")
    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=random_state); folds = list(sgkf.split(df, y, groups)); train_dev_idx, test_idx = folds[0]
    remain = df.iloc[train_dev_idx].copy(); y_remain = y[train_dev_idx]; remain_groups = remain["evaluation_group"].astype(str).to_numpy()
    sgkf2 = StratifiedGroupKFold(n_splits=4, shuffle=True, random_state=random_state + 1); inner = list(sgkf2.split(remain, y_remain, remain_groups)); tr_rel, dev_rel = inner[0]
    return train_dev_idx[tr_rel], train_dev_idx[dev_rel], test_idx

def _build_pipeline(estimator, scaled=False):
    steps = [("imputer", SimpleImputer(strategy="median", add_indicator=True))]
    if scaled: steps.append(("scaler", StandardScaler()))
    steps.append(("classifier", estimator)); return Pipeline(steps)

def build_models(seed: int):
    models = {
        "logistic_regression": _build_pipeline(LogisticRegression(max_iter=3000, class_weight="balanced", random_state=seed), True),
        "random_forest": _build_pipeline(RandomForestClassifier(n_estimators=500, class_weight="balanced", random_state=seed, n_jobs=-1)),
        "extra_trees": _build_pipeline(ExtraTreesClassifier(n_estimators=500, class_weight="balanced", random_state=seed, n_jobs=-1)),
        "svm_rbf": _build_pipeline(SVC(C=1.0, kernel="rbf", class_weight="balanced", probability=True, random_state=seed), True),
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
    return models

def evaluate_numeric(y_true, y_pred, labels):
    y_true = np.asarray(y_true, dtype=int); y_pred = np.asarray(y_pred, dtype=int); ids = list(range(len(labels)))
    return {
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

def iter_nonempty_method_combinations(methods: Sequence[str]) -> Iterable[Tuple[str, Tuple[str, ...]]]:
    names = {1: "INDIVIDUAL", 2: "PAIR", 3: "TRIPLE", 4: "QUADRUPLE", 5: "QUINTUPLE", 6: "SEXTUPLE", 7: "SEPTUPLE"}
    for n in range(1, len(methods) + 1):
        for combo in combinations(tuple(methods), n): yield names[n], combo

def _prepare_truth(labeled_df):
    fine = labeled_df["fault_type_label"].map(normalize_fault); fine = fine.where(fine.isin(cfg.BENCHMARK_FINE_CLASSES), ABSTAIN)
    if "fine_label_conflict" in labeled_df.columns: fine = fine.where(~labeled_df["fine_label_conflict"].astype(bool), ABSTAIN)
    coarse = labeled_df["fault_type_label"].map(unify_fault); coarse = coarse.where(coarse.isin(cfg.COARSE_FAULT_GROUPS), ABSTAIN)
    return fine, coarse

def benchmark_traditional_individual(labeled_df):
    fine_truth, coarse_truth = _prepare_truth(labeled_df); rows = []
    for method, column in cfg.DIAGNOSTIC_METHOD_TO_COLUMN.items():
        pred_fine = labeled_df.get(column, pd.Series(ABSTAIN, index=labeled_df.index)).map(normalize_fault); pred_coarse = pred_fine.map(unify_fault)
        rows.append({"method": method, "granularity": "fine", **_evaluate_method_labels_safely(fine_truth, pred_fine, cfg.BENCHMARK_FINE_CLASSES)})
        rows.append({"method": method, "granularity": "coarse", **_evaluate_method_labels_safely(coarse_truth, pred_coarse, cfg.COARSE_FAULT_GROUPS)})
    result = pd.DataFrame(rows); result.to_csv(BENCHMARK_DIR / "traditional_individual_benchmark.csv", index=False, encoding="utf-8-sig"); return result

def benchmark_traditional_combinations(labeled_df, split=None):
    eval_df = labeled_df.copy().reset_index(drop=True)
    if "fine_label_conflict" in eval_df.columns: eval_df = eval_df[~eval_df["fine_label_conflict"].astype(bool)].reset_index(drop=True)
    fine_truth, coarse_truth = _prepare_truth(eval_df); _, valid_encoding = _encode_fine_truth(fine_truth); valid_mask = (fine_truth != ABSTAIN) & valid_encoding
    split_frame = eval_df.loc[valid_mask].reset_index(drop=True); split_truth = fine_truth.loc[valid_mask].reset_index(drop=True)
    if split_frame.empty: raise ValueError("No valid benchmark rows remain after fine-label filtering.")
    split_frame = split_frame.assign(evaluation_group=split_frame["evaluation_group"].astype(str)); mapping = {c: i for i, c in enumerate(cfg.BENCHMARK_FINE_CLASSES)}; split_y = split_truth.map(mapping).astype(int).to_numpy(); _, dev, test = train_dev_test_split(split_frame, split_y, cfg.RANDOM_STATE)
    rows = []
    for level, combo in iter_nonempty_method_combinations(cfg.DIAGNOSTIC_METHODS):
        pred = apply_consensus_from_existing_diagnostics(eval_df, combo); pred_valid = pred.loc[valid_mask].reset_index(drop=True); coarse_valid = coarse_truth.loc[valid_mask].reset_index(drop=True)
        for split_name, idx in (("development", dev), ("locked_test", test)):
            fine = _evaluate_method_labels_safely(split_truth.iloc[idx], pred_valid["consensus_fault"].iloc[idx], cfg.BENCHMARK_FINE_CLASSES); coarse = _evaluate_method_labels_safely(coarse_valid.iloc[idx], pred_valid["consensus_fault_group"].iloc[idx], cfg.COARSE_FAULT_GROUPS)
            rows.extend([
                {"combination_level": level, "methods": "+".join(combo), "method_count": len(combo), "split": split_name, "granularity": "fine", **fine},
                {"combination_level": level, "methods": "+".join(combo), "method_count": len(combo), "split": split_name, "granularity": "coarse", **coarse},
            ])
    result = pd.DataFrame(rows); result.to_csv(BENCHMARK_DIR / "traditional_combinations_benchmark.csv", index=False, encoding="utf-8-sig"); logger.info("Traditional combination benchmark complete | rows=%d", len(result)); return result

def benchmark_traditional_ppm_coverage(labeled_df):
    rows = []
    for method, column in cfg.DIAGNOSTIC_METHOD_TO_COLUMN.items():
        active = labeled_df.get(column, pd.Series(ABSTAIN, index=labeled_df.index)).map(normalize_fault) != ABSTAIN
        for gas in cfg.COMMON_BENCHMARK_GASES:
            x = pd.to_numeric(labeled_df[gas], errors="coerce"); valid = active & x.notna(); values = x[valid].to_numpy(float)
            rows.append({"method": method, "gas": gas, "active_count": int(len(values)), "coverage": float(len(values) / max(len(labeled_df), 1)), "min_ppm": float(np.min(values)) if len(values) else np.nan, "p05_ppm": float(np.quantile(values, 0.05)) if len(values) else np.nan, "median_ppm": float(np.median(values)) if len(values) else np.nan, "p95_ppm": float(np.quantile(values, 0.95)) if len(values) else np.nan, "max_ppm": float(np.max(values)) if len(values) else np.nan})
    result = pd.DataFrame(rows); result.to_csv(BENCHMARK_DIR / "traditional_ppm_coverage.csv", index=False, encoding="utf-8-sig"); return result

def _benchmark_supervised_feature_mode(labeled_df, seed, feature_mode, models=None):
    labels, _ = _prepare_truth(labeled_df); mask = labels != ABSTAIN; data = labeled_df.loc[mask].reset_index(drop=True); y_names = labels.loc[mask].reset_index(drop=True)
    if "fine_label_conflict" in data.columns:
        keep = ~data["fine_label_conflict"].astype(bool); data = data.loc[keep].reset_index(drop=True); y_names = y_names.loc[keep].reset_index(drop=True)
    mapping = {c: i for i, c in enumerate(cfg.BENCHMARK_FINE_CLASSES)}; encoded_y = y_names.map(mapping); valid = encoded_y.notna()
    data = data.loc[valid].reset_index(drop=True); y_names = y_names.loc[valid].reset_index(drop=True); encoded_y = encoded_y.loc[valid]
    if data.empty: raise ValueError("No valid labeled benchmark rows remain for supervised training.")
    y = encoded_y.astype(int).to_numpy(); data = data.assign(evaluation_group=data["evaluation_group"].astype(str)); tr, dev, test = train_dev_test_split(data, y, seed)
    Xdf = build_feature_frame(data, feature_mode, "fine"); model_map = models or build_models(seed); dev_rows = []
    for name, model in model_map.items():
        try:
            model.fit(Xdf.iloc[tr], y[tr]); pred = _predict_model(model, Xdf.iloc[dev]); pred = np.asarray(pred).reshape(-1); dev_rows.append({"model": name, "feature_mode": feature_mode, "dataset": "external_labeled", "split": "development", "granularity": "fine", **evaluate_numeric(y[dev], pred, cfg.BENCHMARK_FINE_CLASSES)})
        except Exception as exc: logger.warning("Supervised dev failed: %s: %s", name, exc)
    dev_df = pd.DataFrame(dev_rows)
    if dev_df.empty: return pd.DataFrame()
    best = dev_df.sort_values(["macro_f1", "balanced_accuracy", "accuracy"], ascending=False).iloc[0]["model"]
    train_dev = np.concatenate([tr, dev]); test_rows = []
    for name in model_map:
        try:
            model = build_models(seed)[name]; model.fit(Xdf.iloc[train_dev], y[train_dev]); pred = _predict_model(model, Xdf.iloc[test]); pred = np.asarray(pred).reshape(-1); test_rows.append({"model": name, "feature_mode": feature_mode, "dataset": "external_labeled", "split": "locked_test", "granularity": "fine", "selected_on_dev": name == best, **evaluate_numeric(y[test], pred, cfg.BENCHMARK_FINE_CLASSES)})
        except Exception as exc: logger.warning("Supervised test failed: %s: %s", name, exc)
    return pd.concat([dev_df, pd.DataFrame(test_rows)], ignore_index=True)

def benchmark_supervised_models(labeled_df, seed):
    parts = []
    for mode in ("gas_only", "gas_plus_traditional"):
        part = _benchmark_supervised_feature_mode(labeled_df, seed, mode)
        if not part.empty: parts.append(part)
    result = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(); result.to_csv(BENCHMARK_DIR / "supervised_fault_benchmark.csv", index=False, encoding="utf-8-sig"); return result

def _train_weak_students(df, granularity, seed):
    target = f"weak_{granularity}_fault_group" if granularity == "coarse" else f"weak_{granularity}_fault"; abstain = f"weak_{granularity}_is_ABSTAIN"; groups = list(cfg.WEAK_COARSE_GROUPS if granularity == "coarse" else cfg.BENCHMARK_FINE_CLASSES)
    clean, y, present = create_student_training_targets(df, target, abstain, groups)
    artifacts = {}
    for feature_mode in ("gas_only", "gas_plus_traditional"):
        Xdf = build_feature_frame(clean, feature_mode, granularity); models = build_models(seed)
        for name, model in models.items():
            try:
                model.fit(Xdf, y); artifacts[f"{feature_mode}__{name}"] = {"model": model, "feature_mode": feature_mode, "granularity": granularity, "features": list(Xdf.columns), "labels": list(present), "n_training_rows": int(len(clean)), "class_counts": {str(k): int(v) for k, v in pd.Series(y).value_counts().items()}}
            except Exception as exc: logger.warning("Weak student failed: %s/%s: %s", feature_mode, name, exc)
    if not artifacts: raise RuntimeError(f"No weak students trained for {granularity}")
    return artifacts

def _align_feature_frame(df, feature_names, granularity):
    feature_names = list(feature_names); mode = "gas_plus_traditional" if any("__" in c for c in feature_names) else "gas_only"; frame = build_feature_frame(df, mode, granularity); return frame.reindex(columns=feature_names, fill_value=0.0)

def benchmark_weak_transfer(labeled_df, weak_students, seed):
    fine_truth, coarse_truth = _prepare_truth(labeled_df); conflict = labeled_df.get("fine_label_conflict", pd.Series(False, index=labeled_df.index)).astype(bool); valid = (fine_truth != ABSTAIN) & ~conflict
    data = labeled_df.loc[valid].reset_index(drop=True); fine_truth = fine_truth.loc[valid].reset_index(drop=True); coarse_truth = coarse_truth.loc[valid].reset_index(drop=True)
    mapping = {c: i for i, c in enumerate(cfg.BENCHMARK_FINE_CLASSES)}; encoded = fine_truth.map(mapping); valid_encoded = encoded.notna()
    keep = valid_encoded.to_numpy(dtype=bool); data = data.loc[keep].reset_index(drop=True); fine_truth = fine_truth.loc[keep].reset_index(drop=True); coarse_truth = coarse_truth.loc[keep].reset_index(drop=True); y_split = encoded.loc[keep].astype(int).to_numpy()
    if data.empty: raise ValueError("No valid rows remain for weak-transfer benchmark.")
    data = data.assign(evaluation_group=data["evaluation_group"].astype(str)); _, dev, test = train_dev_test_split(data, y_split, seed)
    rows = []
    for granularity in ("coarse", "fine"):
        for key, artifact in weak_students.get(granularity, {}).items():
            feature_names = artifact["features"]; Xdf = _align_feature_frame(data, feature_names, granularity); model = artifact["model"]; labels = list(artifact["labels"])
            pred_dev = _decode_class_predictions(_predict_model(model, Xdf.iloc[dev]), labels); pred_test = _decode_class_predictions(_predict_model(model, Xdf.iloc[test]), labels)
            pred_dev_labels = pd.Series(pred_dev, index=dev); pred_test_labels = pd.Series(pred_test, index=test)
            if granularity == "fine": td = fine_truth.iloc[dev]; tt = fine_truth.iloc[test]; allowed = cfg.BENCHMARK_FINE_CLASSES
            else: td = coarse_truth.iloc[dev]; tt = coarse_truth.iloc[test]; allowed = cfg.COARSE_FAULT_GROUPS
            dev_metric = _evaluate_method_labels_safely(td.reset_index(drop=True), pred_dev_labels.to_numpy(), allowed); test_metric = _evaluate_method_labels_safely(tt.reset_index(drop=True), pred_test_labels.to_numpy(), allowed)
            base = {"granularity": granularity, "model": key.split("__", 1)[1], "feature_mode": artifact["feature_mode"], "training_dataset": "unlabeled_operational_weak", "evaluation_dataset": "external_labeled", "selected_on_dev": False}
            rows.append({**base, "split": "development", **dev_metric}); rows.append({**base, "split": "locked_test", **test_metric})
    result = pd.DataFrame(rows)
    if not result.empty:
        for granularity in result["granularity"].unique():
            sub = result[(result["granularity"] == granularity) & (result["split"] == "development")]
            if not sub.empty:
                best_idx = sub.sort_values(["macro_f1", "balanced_accuracy", "accuracy"], ascending=False).index[0]; best_model = result.loc[best_idx, "model"]; best_mode = result.loc[best_idx, "feature_mode"]; mask = (result["granularity"] == granularity) & (result["model"] == best_model) & (result["feature_mode"] == best_mode); result.loc[mask, "selected_on_dev"] = True
    result.to_csv(BENCHMARK_DIR / "weak_transfer_fault_benchmark.csv", index=False, encoding="utf-8-sig"); return result

def benchmark_direct_supervised_transfer(unlabeled_df, labeled_df, seed):
    result = benchmark_supervised_models(labeled_df, seed); best = result[result["split"] == "locked_test"].sort_values(["macro_f1", "balanced_accuracy", "accuracy"], ascending=False).head(1); best.to_csv(BENCHMARK_DIR / "direct_supervised_reference_best.csv", index=False, encoding="utf-8-sig"); return best

def run_unlabeled_pipeline(seed, use_snorkel, save_model=True):
    raw = load_unlabeled(); logger.info("UNLABELED START | rows=%d | transformers=%d", len(raw), raw["transformer_id"].nunique()); df = prepare_unlabeled(raw)
    outputs = {}; weak_students = {"coarse": {}, "fine": {}}
    for granularity, groups in (("coarse", cfg.COARSE_FAULT_GROUPS), ("fine", cfg.BENCHMARK_FINE_CLASSES)):
        out, model, out_groups, meta, L, probabilities, pairwise = weak_supervision_pipeline(df, DEFAULT_WEAK_METHODS, groups, use_snorkel=use_snorkel, random_state=seed, granularity=granularity)
        outputs[granularity] = out; pairwise.to_csv(BENCHMARK_DIR / f"weak_lf_pairwise_agreement_{granularity}.csv", index=False, encoding="utf-8-sig"); save_weak_supervision_artifacts(out, model, out_groups, meta, granularity=granularity); weak_students[granularity] = _train_weak_students(out, granularity, seed); del L, probabilities
    merge_keys = ["transformer_id", "sample_day"]; df = outputs["coarse"].copy(); fine_cols = [c for c in outputs["fine"].columns if c.startswith("weak_fine_")]; df = df.merge(outputs["fine"][merge_keys + fine_cols], on=merge_keys, how="left", suffixes=("", "_fine"))
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
                except Exception: logger.warning("Weak student confidence failed | %s | %s", granularity, key, exc_info=True)
    from severity import apply_severity
    df = apply_severity(df, nei_reference=None); ranking = build_transformer_ranking(df); log_ranking_diagnostics(ranking, 20)
    processed = DATASET_DIR / "processed"; processed.mkdir(parents=True, exist_ok=True); df.to_parquet(processed / "dga_unlabeled_processed.parquet", index=False); ranking.to_parquet(processed / "transformer_ranking.parquet", index=False); ranking.to_csv(REPORT_DIR / "transformer_ranking.csv", index=False, encoding="utf-8-sig")
    if save_model:
        MODEL_DIR.mkdir(parents=True, exist_ok=True); joblib.dump({"models": weak_students["coarse"], "training_type": "weak_supervision_plus_discriminative_ml", "training_dataset": str(UNLABELED_PATH), "features": MODEL_FEATURES}, FAULT_MODEL_COARSE_PATH); joblib.dump({"models": weak_students["fine"], "training_type": "weak_supervision_plus_discriminative_ml", "training_dataset": str(UNLABELED_PATH), "features": MODEL_FEATURES}, FAULT_MODEL_FINE_PATH); metadata = {"seed": seed, "unlabeled_dataset": str(UNLABELED_PATH), "weak_supervision": "Snorkel LabelModel or EM fallback", "student_feature_modes": ["gas_only", "gas_plus_traditional"], "student_model_count_coarse": len(weak_students["coarse"]), "student_model_count_fine": len(weak_students["fine"]), "severity_source": cfg.STANDARD, "severity_is_weighted": False, "severity_is_failure_probability": False, "ranking_policy": list(cfg.RANKING_POLICY), "ranking_is_weighted": False, "ranking_is_health_score": False, "benchmark_policy": "Operational unlabeled data are used for weak labels and student training only; labeled benchmark is reserved for external evaluation and locked test reporting."}; TRAINING_METADATA_PATH.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    return df, ranking, weak_students

def run_labeled_benchmark(seed):
    labeled = apply_consensus(load_labeled_csv_data()); individual = benchmark_traditional_individual(labeled); combinations_result = benchmark_traditional_combinations(labeled, None); ppm = benchmark_traditional_ppm_coverage(labeled); pairwise = pairwise_label_agreement(labeled); pairwise.to_csv(BENCHMARK_DIR / "traditional_pairwise_agreement.csv", index=False, encoding="utf-8-sig"); method_summary = diagnostic_method_summary(labeled); method_summary.to_csv(BENCHMARK_DIR / "traditional_method_summary.csv", index=False, encoding="utf-8-sig"); supervised = benchmark_supervised_models(labeled, seed)
    benchmark = {"individual": individual, "combinations": combinations_result, "ppm_coverage": ppm, "pairwise": pairwise, "method_summary": method_summary, "supervised": supervised}; benchmark["split_manifest"] = _write_split_manifest(labeled, seed); return benchmark

def _write_split_manifest(labeled, seed):
    labels, _ = _prepare_truth(labeled); conflict = labeled.get("fine_label_conflict", pd.Series(False, index=labeled.index)).astype(bool); valid = (labels != ABSTAIN) & ~conflict; data = labeled.loc[valid].reset_index(drop=True); y = labels.loc[valid].map({c: i for i, c in enumerate(cfg.BENCHMARK_FINE_CLASSES)}).to_numpy(int); data = data.assign(evaluation_group=data["evaluation_group"].astype(str)); _, dev, test = train_dev_test_split(data, y, seed)
    manifest = data[["source_dataset", "source_row", "fault_type_label", "evaluation_group"]].copy(); split = np.full(len(data), "train", dtype=object); split[dev] = "development"; split[test] = "locked_test"; manifest["split"] = split; manifest.to_csv(BENCHMARK_DIR / "benchmark_split_manifest.csv", index=False, encoding="utf-8-sig"); return manifest

def write_confusion_matrices(weak_transfer, supervised, labeled, seed):
    labels = _prepare_truth(labeled)[0]; conflict = labeled.get("fine_label_conflict", pd.Series(False, index=labeled.index)).astype(bool); valid = (labels != ABSTAIN) & ~conflict; data = labeled.loc[valid].reset_index(drop=True); y_names = labels.loc[valid].reset_index(drop=True); y = y_names.map({c: i for i, c in enumerate(cfg.BENCHMARK_FINE_CLASSES)}).to_numpy(int); _, _, test = train_dev_test_split(data.assign(evaluation_group=data["evaluation_group"].astype(str)), y, seed)
    for mode in supervised["feature_mode"].unique():
        sub = supervised[(supervised["feature_mode"] == mode) & (supervised["split"] == "locked_test")]
        if sub.empty: continue
        best = sub.sort_values(["macro_f1", "balanced_accuracy", "accuracy"], ascending=False).iloc[0]; model = build_models(seed)[best["model"]]; Xdf = build_feature_frame(data, mode, "fine"); train_mask = np.ones(len(data), dtype=bool); train_mask[test] = False; model.fit(Xdf.iloc[train_mask], y[train_mask]); pred = np.asarray(model.predict(Xdf.iloc[test])).reshape(-1); cm = confusion_matrix(y[test], pred, labels=list(range(len(cfg.BENCHMARK_FINE_CLASSES)))); out = pd.DataFrame(cm, index=cfg.BENCHMARK_FINE_CLASSES, columns=cfg.BENCHMARK_FINE_CLASSES); out.to_csv(BENCHMARK_DIR / f"confusion_supervised_{mode}.csv", encoding="utf-8-sig")

def main(args=None):
    parser = argparse.ArgumentParser(); parser.add_argument("--mode", choices=["benchmark", "unlabeled", "transfer", "all"], default="all"); parser.add_argument("--use-snorkel", action="store_true"); parser.add_argument("--seed", type=int, default=cfg.RANDOM_STATE); parsed = parser.parse_args(args)
    set_global_seed(parsed.seed); unlabeled_result = None
    if parsed.mode in {"unlabeled", "transfer", "all"}: unlabeled_result = run_unlabeled_pipeline(parsed.seed, parsed.use_snorkel)
    benchmark = None
    if parsed.mode in {"benchmark", "all"}:
        benchmark = run_labeled_benchmark(parsed.seed); print("\n=== Traditional individual ==="); print(benchmark["individual"].sort_values(["granularity", "macro_f1"], ascending=[True, False]).to_string(index=False)); print("\n=== Best traditional combinations on locked test ==="); print(benchmark["combinations"].query("split == 'locked_test'").sort_values(["granularity", "macro_f1"], ascending=[True, False]).groupby("granularity").head(10).to_string(index=False)); print("\n=== Supervised reference ==="); print(benchmark["supervised"].query("split == 'locked_test'").sort_values("macro_f1", ascending=False).head(20).to_string(index=False))
    if parsed.mode in {"transfer", "all"}:
        if unlabeled_result is None: unlabeled_result = run_unlabeled_pipeline(parsed.seed, parsed.use_snorkel)
        labeled = apply_consensus(load_labeled_csv_data()); weak_transfer = benchmark_weak_transfer(labeled, unlabeled_result[2], parsed.seed); print("\n=== Weak-supervision transfer ==="); print(weak_transfer.query("split == 'locked_test'").sort_values(["granularity", "macro_f1"], ascending=[True, False]).head(30).to_string(index=False))
    try:
        from experiment import build_excel_report
        build_excel_report(REPORT_DIR, DATASET_DIR / "processed", REPORT_DIR / "dga_research_report.xlsx"); logger.info("Excel report saved to %s", REPORT_DIR / "dga_research_report.xlsx")
    except Exception: logger.exception("Excel report generation failed")

if __name__ == "__main__":
    main()