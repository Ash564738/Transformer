# train_unsupervised_models.py
from __future__ import annotations
import json
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
from dga_features import FEATS as SCALE_INVARIANT_DGA_FEATURES, build_scale_invariant_features
from logging_config import init_logging
from ranking import build_transformer_ranking, log_ranking_diagnostics
from weak_supervision import DEFAULT_WEAK_METHODS, create_student_training_targets, load_weak_supervision_artifact, save_weak_supervision_artifacts, weak_supervision_pipeline
init_logging()
logger = logging.getLogger(__name__)
LABELED_CSV_PATH1 = DATASET_DIR / "IEC_TC10_121.csv"
LABELED_CSV_PATH2 = DATASET_DIR / "DGA dataset.csv"
UNLABELED_PATH = DATASET_DIR / "processed" / "dga_unlabeled.parquet"
FAULT_MODEL_COARSE_PATH = MODEL_DIR / "fault_classifiers_coarse.joblib"
FAULT_MODEL_FINE_PATH = MODEL_DIR / "fault_classifiers_fine.joblib"
TRAINING_METADATA_PATH = MODEL_DIR / "training_metadata.json"
PRODUCTION_SELECTION_PATH = MODEL_DIR / "production_fault_selection.joblib"
BENCHMARK_DIR = REPORT_DIR / "benchmark"
MODEL_FEATURES = list(cfg.COMMON_BENCHMARK_GASES)

def set_global_seed(seed: int):
    random.seed(seed); np.random.seed(seed)
    if TORCH_AVAILABLE:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

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
        except Exception:
            decoded.append(default)
    return decoded

def _predict_model(model, feature_frame: pd.DataFrame):
    return model.predict(feature_frame)

def _predict_proba(model, feature_frame: pd.DataFrame):
    return model.predict_proba(feature_frame)

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
    counts = pd.Series(y).value_counts()
    return len(counts) >= 2 and counts.min() >= 2


class CatBoostSklearnAdapter(BaseEstimator, ClassifierMixin):
    """Small sklearn-compatible adapter for CatBoost versions whose native
    estimator does not expose the current sklearn tag protocol.

    The wrapper does not alter the CatBoost algorithm or hyperparameters; it
    only makes fit/predict/predict_proba usable inside sklearn Pipeline.
    """

    def __init__(self, iterations=300, learning_rate=0.05, depth=6,
                 random_seed=42, auto_class_weights="Balanced"):
        self.iterations = iterations
        self.learning_rate = learning_rate
        self.depth = depth
        self.random_seed = random_seed
        self.auto_class_weights = auto_class_weights

    def fit(self, X, y, **fit_params):
        if CatBoostClassifier is None:
            raise ImportError("catboost is not installed")
        self.model_ = CatBoostClassifier(
            iterations=self.iterations,
            learning_rate=self.learning_rate,
            depth=self.depth,
            verbose=0,
            random_seed=self.random_seed,
            auto_class_weights=self.auto_class_weights,
        )
        self.model_.fit(X, y, **fit_params)
        self.classes_ = np.unique(np.asarray(y))
        return self

    def predict(self, X):
        return np.asarray(self.model_.predict(X)).reshape(-1)

    def predict_proba(self, X):
        return np.asarray(self.model_.predict_proba(X))


class AdaptiveCalibratedSVC(BaseEstimator, ClassifierMixin):
    """SVC with calibration CV chosen from the available class support.

    Recent scikit-learn versions warn or fail when the requested calibration
    folds exceed the number of samples in the rarest class. This wrapper
    chooses a valid calibration fold count when possible and falls back to
    an uncalibrated SVC when a class has fewer than two samples.
    """

    def __init__(
        self,
        C=1.0,
        kernel="rbf",
        class_weight="balanced",
        random_state=42,
        calibration_cv=5,
    ):
        self.C = C
        self.kernel = kernel
        self.class_weight = class_weight
        self.random_state = random_state
        self.calibration_cv = calibration_cv

    def fit(self, X, y):
        y_array = np.asarray(y)
        counts = pd.Series(y_array).value_counts()
        min_count = int(counts.min()) if not counts.empty else 0

        base = SVC(
            C=self.C,
            kernel=self.kernel,
            class_weight=self.class_weight,
            random_state=self.random_state,
        )

        self.classes_ = np.unique(y_array)
        self.calibrated_ = False

        if min_count >= 2 and len(self.classes_) >= 2:
            cv = min(int(self.calibration_cv), min_count)
            try:
                self.model_ = CalibratedClassifierCV(
                    base,
                    cv=cv,
                    ensemble=False,
                )
                self.model_.fit(X, y_array)
                self.calibrated_ = True
                return self
            except ValueError as exc:
                logger.debug(
                    "SVC calibration unavailable; falling back to uncalibrated SVC | min_class=%d | error=%s",
                    min_count,
                    exc,
                )

        self.model_ = base.fit(X, y_array)
        return self

    def predict(self, X):
        return self.model_.predict(X)

    def predict_proba(self, X):
        if self.calibrated_ and hasattr(self.model_, "predict_proba"):
            return self.model_.predict_proba(X)

        decision = self.model_.decision_function(X)
        decision = np.asarray(decision, dtype=float)

        if decision.ndim == 1:
            if len(self.classes_) == 2:
                decision = np.column_stack([-decision, decision])
            else:
                decision = decision.reshape(-1, 1)

        # Softmax over decision scores provides a monotonic confidence fallback
        # when calibration is impossible. It is not claimed to be calibrated
        # probability.
        decision = decision - np.max(decision, axis=1, keepdims=True)
        exp_scores = np.exp(np.clip(decision, -50.0, 50.0))
        denominator = np.sum(exp_scores, axis=1, keepdims=True)
        return exp_scores / np.maximum(denominator, np.finfo(float).eps)

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
        def predict(self, X):
            proba = self.predict_proba(X)
            return self.classes_[np.argmax(proba, axis=1)]
else: TorchMLPClassifier = None

def _read_labeled(path: Path, label_col: str, source_name: str) -> pd.DataFrame:
    if not path.exists():
        logger.error("_read_labeled: path not found %s", path)
        raise FileNotFoundError(path)
    df = pd.read_csv(path, encoding="utf-8-sig"); df.columns = [str(c).strip().lower().replace("\ufeff", "") for c in df.columns]
    if label_col not in df.columns:
        logger.error("_read_labeled: missing label column %r in %s", label_col, path.name)
        raise ValueError(f"{path.name}: missing label column {label_col!r}")
    for gas in cfg.FULL_EXTERNAL_GASES:
        df[gas] = pd.to_numeric(df[gas], errors="coerce") if gas in df.columns else np.nan
    df["fault_type_label"] = df[label_col].map(normalize_fault); df["source_dataset"] = source_name; df["source_row"] = np.arange(1, len(df) + 1); df["coarse_truth"] = df["fault_type_label"].map(unify_fault)
    return df

def _external_gas_key(df: pd.DataFrame) -> pd.Series:
    gas = list(cfg.COMMON_BENCHMARK_GASES); x = df[gas].round(8).fillna(-999999.0); return x.astype(str).agg("|".join, axis=1)

def load_labeled_csv_data() -> pd.DataFrame:
    logger.debug("load_labeled_csv_data: start")
    a = _read_labeled(LABELED_CSV_PATH1, "label", "iec_tc10_121"); b = _read_labeled(LABELED_CSV_PATH2, "type", "dga_dataset"); combined = pd.concat([a, b], ignore_index=True); combined["_gas_key"] = _external_gas_key(combined)
    combined["fine_label_conflict"] = combined.groupby("_gas_key")["fault_type_label"].transform("nunique").gt(1); combined["coarse_label_conflict"] = combined.groupby("_gas_key")["coarse_truth"].transform("nunique").gt(1)
    logger.debug("load_labeled_csv_data: combined rows=%d fine_conflicts=%d coarse_conflicts=%d", len(combined), int(combined['fine_label_conflict'].sum()), int(combined['coarse_label_conflict'].sum()))
    conflict_df = combined[combined["fine_label_conflict"]].copy(); BENCHMARK_DIR.mkdir(parents=True, exist_ok=True)
    if not conflict_df.empty:
        conflict_df.to_csv(BENCHMARK_DIR / "external_benchmark_label_conflicts.csv", index=False, encoding="utf-8-sig")
    combined["duplicate_group_size"] = combined.groupby("_gas_key")["_gas_key"].transform("size"); combined["evaluation_group"] = combined["_gas_key"]
    result = combined.drop(columns=["_gas_key"]).reset_index(drop=True)
    logger.debug("load_labeled_csv_data: final rows=%d", len(result))
    return result

def load_unlabeled(path: Path = UNLABELED_PATH) -> pd.DataFrame:
    """Load canonical operational data, preparing it automatically when absent.

    The offline research runner is self-contained: a fresh checkout can run
    `python train_unsupervised_models.py --mode all` without first running a
    separate preparation command.

    A canonical parquet is treated as authoritative and is never passed through
    feature engineering a second time.
    """
    if path.exists():
        logger.info("Loading canonical unlabeled parquet: %s", path)
        df = pd.read_parquet(path)
    else:
        source_candidates = [
            DATASET_DIR / "DGA of Main Tank only KT 11022026_09062026.xlsx",
            DATASET_DIR / "dga_unlabeled.xlsx",
            DATASET_DIR / "dga_unlabeled.csv",
        ]
        source = next((candidate for candidate in source_candidates if candidate.exists()), None)
        if source is None:
            raise FileNotFoundError(
                "Operational unlabeled dataset not found. Expected either "
                f"{path} or one of: "
                + ", ".join(str(x) for x in source_candidates)
            )

        logger.info("Canonical parquet missing; preparing from %s", source)

        if source.suffix.lower() in {".xlsx", ".xls"}:
            raw = pd.read_excel(source)
        else:
            raw = pd.read_csv(source, encoding="utf-8-sig")

        raw.columns = [
            str(c).strip().lower().replace("\ufeff", "").replace(" ", "_")
            for c in raw.columns
        ]

        # Canonicalize raw operational columns first (LOC/NAME/CODETX/Sample Day
        # -> transformer_id/sample_day), then build features exactly once.
        from clean_dataset import clean_dataset
        cleaned, _summary = clean_dataset(
            dataframe=raw,
            output_dir=DATASET_DIR / "processed",
        )
        df = prepare_unlabeled(cleaned)
        _assert_unique_columns(df, "load_unlabeled prepared output")

        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(path, index=False)

        logger.info(
            "Prepared canonical operational parquet: rows=%d transformers=%d columns=%d",
            len(df),
            df["transformer_id"].nunique() if "transformer_id" in df.columns else 0,
            len(df.columns),
        )

    _assert_unique_columns(df, f"load_unlabeled input {path.name}")

    required = {"transformer_id", "sample_day"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{path.name} requires {sorted(missing)}")

    result = df.copy()
    result["sample_day"] = pd.to_datetime(result["sample_day"], errors="coerce")
    result = (
        result.dropna(subset=["transformer_id", "sample_day"])
        .sort_values(["transformer_id", "sample_day"], kind="mergesort")
        .reset_index(drop=True)
    )

    logger.info(
        "load_unlabeled: rows=%d transformers=%d columns=%d",
        len(result),
        result["transformer_id"].nunique(),
        len(result.columns),
    )
    return result


def _assert_unique_columns(df: pd.DataFrame, context: str) -> None:
    if df.columns.duplicated().any():
        duplicated = df.columns[df.columns.duplicated()].tolist()
        raise ValueError(
            f"{context} produced duplicate columns: "
            + ", ".join(map(str, duplicated))
        )


def prepare_unlabeled(df: pd.DataFrame) -> pd.DataFrame:
    """Return canonical unlabeled data without re-engineering an already prepared frame.

    The canonical parquet written by prepare_unlabeled_data.py already contains
    feature-engineering and traditional-diagnostic columns. Re-running feature
    engineering on that frame creates duplicate column names (e.g. mva_count,
    h2_lag1, rate_span_days), which pyarrow correctly rejects when saving parquet.
    """
    logger.debug("prepare_unlabeled: start shape=%s", df.shape)
    required_prepared = {
        "transformer_id", "sample_day", "h2", "ch4", "c2h6", "c2h4", "c2h2",
        "keygas_fault", "iec_fault", "rogers_fault", "doernenburg_fault",
        "duval_triangle_fault", "consensus_fault", "diagnostic_confidence",
    }
    if required_prepared.issubset(set(df.columns)):
        result = df.copy()
        if result.columns.duplicated().any():
            duplicated = result.columns[result.columns.duplicated()].tolist()
            logger.warning("prepare_unlabeled: dropping duplicate columns from already-prepared data: %s", duplicated)
            result = result.loc[:, ~result.columns.duplicated(keep="first")].copy()
        _assert_unique_columns(result, "prepare_unlabeled canonical input")
        result["sample_day"] = pd.to_datetime(result["sample_day"], errors="coerce")
        result = result.dropna(subset=["transformer_id", "sample_day"]).sort_values(
            ["transformer_id", "sample_day"], kind="mergesort"
        ).reset_index(drop=True)
        logger.debug("prepare_unlabeled: input already canonical; skipped feature engineering, shape=%s", result.shape)
        return result

    out = build_training_features_from_clean(df.copy())
    out["sample_day"] = pd.to_datetime(out["sample_day"], errors="coerce")
    out = out.sort_values(["transformer_id", "sample_day"], kind="mergesort").reset_index(drop=True)
    result = apply_consensus(out)
    _assert_unique_columns(result, "prepare_unlabeled final output")
    logger.debug("prepare_unlabeled: final shape=%s", result.shape)
    return result

def dataframe_to_gas_matrix(df: pd.DataFrame, features: Sequence[str]) -> np.ndarray:
    x = df.reindex(columns=list(features)).apply(pd.to_numeric, errors="coerce")
    return x.replace([np.inf, -np.inf], np.nan).to_numpy(dtype=np.float32)

def _traditional_feature_matrix(df: pd.DataFrame, granularity: str = "fine") -> pd.DataFrame:
    out = pd.DataFrame(index=df.index); vocab = (list(cfg.BENCHMARK_FINE_CLASSES) if granularity == "fine" else list(cfg.WEAK_COARSE_GROUPS)) + [ABSTAIN]
    for method, column in cfg.DIAGNOSTIC_METHOD_TO_COLUMN.items():
        vals = df.get(column, pd.Series(ABSTAIN, index=df.index)).map(normalize_fault)
        if granularity == "coarse": vals = vals.map(unify_fault)
        for label in vocab:
            safe = str(label).lower().replace("/", "_").replace(" ", "_"); out[f"{method}__{safe}"] = (vals == label).astype(np.float32)
    return out

def _align_feature_frame(df: pd.DataFrame, feature_names, granularity: str = "fine") -> pd.DataFrame:
    """Rebuild the exact feature schema stored with a weak student artifact.

    Feature mode is inferred from the artifact schema, including the merged
    scale-invariant representation introduced from the teammate code. This is
    essential: treating ratio features as traditional one-hot diagnostics would
    silently change the model input at inference time.
    """
    names = [str(x) for x in feature_names]
    gas_names = set(MODEL_FEATURES)
    ratio_names = set(SCALE_INVARIANT_DGA_FEATURES)

    if set(names).issubset(gas_names):
        feature_mode = "gas_only"
    elif set(names).issubset(ratio_names):
        feature_mode = "ratio_only"
    elif set(names).issubset(gas_names | ratio_names):
        feature_mode = "gas_plus_ratio"
    else:
        feature_mode = "gas_plus_traditional"

    X = build_feature_frame(df, feature_mode, granularity)

    missing = [name for name in names if name not in X.columns]
    if missing:
        X = X.copy()
        for name in missing:
            X[name] = np.nan

    return X.reindex(columns=names)


def _ratio_feature_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """Canonical scale-invariant DGA representation (13 deterministic features).

    This is the useful part of the teammate implementation: the model can
    compare gas composition independently of absolute concentration scale.
    It is an ablation/feature representation only; traditional diagnosis and
    severity/ranking logic are unchanged.
    """
    return build_scale_invariant_features(df).reindex(columns=SCALE_INVARIANT_DGA_FEATURES)

def build_feature_frame(df: pd.DataFrame, feature_mode: str, granularity: str = "fine") -> pd.DataFrame:
    gas = df.reindex(columns=MODEL_FEATURES).apply(pd.to_numeric, errors="coerce").astype(float)

    if feature_mode == "gas_only":
        return gas

    if feature_mode == "ratio_only":
        return _ratio_feature_matrix(df)

    if feature_mode == "gas_plus_ratio":
        return pd.concat([gas, _ratio_feature_matrix(df)], axis=1)

    if feature_mode == "gas_plus_traditional":
        trad = _traditional_feature_matrix(df, granularity)
        return pd.concat([gas, trad], axis=1)

    logger.error("build_feature_frame: unknown feature_mode %s", feature_mode)
    raise ValueError(f"Unknown feature_mode: {feature_mode}")

def _encode_fine_truth(series):
    mapping = {label: idx for idx, label in enumerate(cfg.BENCHMARK_FINE_CLASSES)}; normalized = series.map(normalize_fault).astype(object); encoded = normalized.map(mapping); valid_mask = encoded.notna()
    return encoded.loc[valid_mask].astype(int).to_numpy(), valid_mask

def train_dev_test_split(df: pd.DataFrame, y: Sequence, random_state: int):
    y = np.asarray(y); groups = np.asarray(df["evaluation_group"].astype(str))
    if len(np.unique(groups)) < 3:
        logger.error("train_dev_test_split: only %d unique groups", len(np.unique(groups)))
        raise ValueError("Need at least three unique groups for grouped train/dev/test split")
    if not _can_stratify(y):
        logger.error("train_dev_test_split: cannot stratify classes: %s", pd.Series(y).value_counts().to_dict())
        raise ValueError(f"Cannot stratify classes: {pd.Series(y).value_counts().to_dict()}")
    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=random_state); folds = list(sgkf.split(df, y, groups)); train_dev_idx, test_idx = folds[0]
    remain = df.iloc[train_dev_idx].copy(); y_remain = y[train_dev_idx]; remain_groups = remain["evaluation_group"].astype(str).to_numpy()
    sgkf2 = StratifiedGroupKFold(n_splits=4, shuffle=True, random_state=random_state + 1); inner = list(sgkf2.split(remain, y_remain, remain_groups)); tr_rel, dev_rel = inner[0]
    return train_dev_idx[tr_rel], train_dev_idx[dev_rel], test_idx

def _build_pipeline(estimator, scaled=False):
    steps = [("imputer", SimpleImputer(strategy="median", add_indicator=True))]
    if scaled:
        steps.append(("scaler", StandardScaler()))
    steps.append(("classifier", estimator))
    pipeline = Pipeline(steps)

    # Keep transformed feature names when the downstream estimator supports
    # pandas input. This prevents LightGBM's "valid feature names" warning
    # after SimpleImputer / StandardScaler transform the DataFrame into an
    # unnamed ndarray. Older sklearn versions may not provide set_output, so
    # this is intentionally guarded.
    try:
        pipeline.set_output(transform="pandas")
    except (AttributeError, ValueError):
        pass

    return pipeline

def build_models(seed: int):
    models = {
        "logistic_regression": _build_pipeline(LogisticRegression(max_iter=3000, class_weight="balanced", random_state=seed), True),
        "random_forest": _build_pipeline(RandomForestClassifier(n_estimators=500, class_weight="balanced", random_state=seed, n_jobs=-1)),
        "extra_trees": _build_pipeline(ExtraTreesClassifier(n_estimators=500, class_weight="balanced", random_state=seed, n_jobs=-1)),
        "svm_rbf": _build_pipeline(AdaptiveCalibratedSVC(random_state=seed), True),
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
    if CatBoostClassifier is not None: models["catboost"] = _build_pipeline(CatBoostSklearnAdapter(iterations=300, learning_rate=0.05, depth=6, random_seed=seed, auto_class_weights="Balanced"))
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
    return fine, coarse

def _make_ambiguity_eval_split(labeled_df, seed):
    fine = labeled_df["fault_type_label"].map(normalize_fault)
    conflict = labeled_df.get("fine_label_conflict", pd.Series(False, index=labeled_df.index)).astype(bool)
    coarse_conflict = labeled_df.get("coarse_label_conflict", pd.Series(False, index=labeled_df.index)).astype(bool)
    valid = fine.isin(set(cfg.BENCHMARK_FINE_CLASSES) | set(cfg.BENCHMARK_AMBIGUOUS_FINE_CLASSES)) & ~conflict & ~coarse_conflict
    data = labeled_df.loc[valid].reset_index(drop=True).copy()
    if data.empty:
        return data, np.array([], dtype=int), np.array([], dtype=int), np.array([], dtype=int)
    data["evaluation_group"] = data["evaluation_group"].astype(str)
    coarse = data["fault_type_label"].map(normalize_fault).map(unify_fault)
    coarse_codes = coarse.map({c: i for i, c in enumerate(cfg.COARSE_FAULT_GROUPS)}).astype(int).to_numpy()
    tr, dev, test = train_dev_test_split(data, coarse_codes, seed)
    return data, tr, dev, test

def benchmark_traditional_individual(labeled_df):
    """Evaluate individual traditional methods on DEV and LOCKED TEST."""
    logger.debug("benchmark_traditional_individual: start rows=%d", len(labeled_df))
    eval_df = labeled_df.copy().reset_index(drop=True)
    fine_truth_all, coarse_truth_all = _prepare_truth(eval_df)
    conflict = eval_df.get("fine_label_conflict", pd.Series(False, index=eval_df.index)).astype(bool)
    coarse_conflict = eval_df.get("coarse_label_conflict", pd.Series(False, index=eval_df.index)).astype(bool)
    valid = fine_truth_all.isin(cfg.BENCHMARK_FINE_CLASSES) & ~conflict & ~coarse_conflict
    eval_df = eval_df.loc[valid].reset_index(drop=True)
    fine_truth = fine_truth_all.loc[valid].reset_index(drop=True)
    coarse_truth = coarse_truth_all.loc[valid].reset_index(drop=True)
    eval_df["evaluation_group"] = eval_df.get("evaluation_group", eval_df[list(cfg.COMMON_BENCHMARK_GASES)].round(8).fillna(-999999.0).astype(str).agg("|".join, axis=1)).astype(str)
    mapping = {c: i for i, c in enumerate(cfg.BENCHMARK_FINE_CLASSES)}
    y = fine_truth.map(mapping).astype(int).to_numpy()
    tr, dev, test = train_dev_test_split(eval_df, y, cfg.RANDOM_STATE)
    rows = []
    for method, column in cfg.DIAGNOSTIC_METHOD_TO_COLUMN.items():
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
    result.to_csv(BENCHMARK_DIR / "traditional_individual_benchmark.csv", index=False, encoding="utf-8-sig")
    logger.debug("benchmark_traditional_individual: generated %d rows", len(result))
    return result

def benchmark_traditional_combinations(labeled_df, split=None):
    """Evaluate all combinations on DEV and LOCKED TEST; selection is based only on DEV."""
    logger.debug("benchmark_traditional_combinations: start rows=%d", len(labeled_df))
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
    rows = []
    for level, combo in iter_nonempty_method_combinations(cfg.DIAGNOSTIC_METHODS):
        predicted = apply_consensus_from_existing_diagnostics(eval_df, combo)
        for split_name, idx in (("development", dev), ("locked_test", test)):
            fine_metric = _evaluate_method_labels_safely(fine_truth.iloc[idx], predicted["consensus_fault"].iloc[idx], cfg.BENCHMARK_FINE_CLASSES)
            coarse_metric = _evaluate_method_labels_safely(coarse_truth.iloc[idx], predicted["consensus_fault_group"].iloc[idx], cfg.COARSE_FAULT_GROUPS)
            rows.append({"combination_level": level, "methods": "+".join(combo), "method_count": len(combo), "split": split_name, "granularity": "fine", **fine_metric})
            rows.append({"combination_level": level, "methods": "+".join(combo), "method_count": len(combo), "split": split_name, "granularity": "coarse", **coarse_metric})
    amb_data, _, amb_dev, amb_test = _make_ambiguity_eval_split(labeled_df, cfg.RANDOM_STATE)
    if not amb_data.empty:
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
    logger.debug("benchmark_traditional_combinations: generated %d rows", len(result))
    return result

def benchmark_traditional_ppm_bins(labeled_df):
    """Export empirical ppm-bin coverage for each diagnostic method.

    The bins are descriptive reporting bins, not diagnostic thresholds and not
    learned weights.  They make the requested "coverage over ppm" comparison
    reproducible in Excel without claiming a physical operating range.
    """
    bins = [0.0, 1.0, 5.0, 10.0, 20.0, 50.0, 100.0, 250.0, 500.0, 1000.0, np.inf]
    labels = ["0-<1", "1-<5", "5-<10", "10-<20", "20-<50", "50-<100", "100-<250", "250-<500", "500-<1000", "1000+"]
    rows = []
    for method, column in cfg.DIAGNOSTIC_METHOD_TO_COLUMN.items():
        active = labeled_df.get(column, pd.Series(ABSTAIN, index=labeled_df.index)).map(normalize_fault) != ABSTAIN
        for gas in cfg.COMMON_BENCHMARK_GASES:
            x = pd.to_numeric(labeled_df[gas], errors="coerce")
            valid = active & x.notna() & (x >= 0)
            bucket = pd.cut(x.where(valid), bins=bins, labels=labels, right=False, include_lowest=True)
            counts = bucket.value_counts(sort=False)
            for bin_label, count in counts.items():
                rows.append({
                    "method": method, "gas": gas, "ppm_bin": str(bin_label),
                    "active_count": int(count),
                    "active_percent_of_all": float(count / max(len(labeled_df), 1) * 100.0),
                    "active_percent_of_method_coverage": float(count / max(int(valid.sum()), 1) * 100.0),
                })
    result = pd.DataFrame(rows)
    result.to_csv(BENCHMARK_DIR / "traditional_ppm_bins.csv", index=False, encoding="utf-8-sig")
    return result


def benchmark_traditional_ppm_coverage(labeled_df):
    logger.debug("benchmark_traditional_ppm_coverage: start rows=%d", len(labeled_df))
    rows = []
    for method, column in cfg.DIAGNOSTIC_METHOD_TO_COLUMN.items():
        active = labeled_df.get(column, pd.Series(ABSTAIN, index=labeled_df.index)).map(normalize_fault) != ABSTAIN
        for gas in cfg.COMMON_BENCHMARK_GASES:
            x = pd.to_numeric(labeled_df[gas], errors="coerce"); valid = active & x.notna(); values = x[valid].to_numpy(float)
            rows.append({"method": method, "gas": gas, "active_count": int(len(values)), "coverage": float(len(values) / max(len(labeled_df), 1)), "min_ppm": float(np.min(values)) if len(values) else np.nan, "p05_ppm": float(np.quantile(values, 0.05)) if len(values) else np.nan, "median_ppm": float(np.median(values)) if len(values) else np.nan, "p95_ppm": float(np.quantile(values, 0.95)) if len(values) else np.nan, "max_ppm": float(np.max(values)) if len(values) else np.nan, "observed_range_ppm": float(np.max(values) - np.min(values)) if len(values) else np.nan})
    result = pd.DataFrame(rows)
    result.to_csv(BENCHMARK_DIR / "traditional_ppm_coverage.csv", index=False, encoding="utf-8-sig")
    class_coverage = empirical_fault_class_coverage(labeled_df, "fine")
    class_coverage.to_csv(BENCHMARK_DIR / "traditional_fault_class_coverage.csv", index=False, encoding="utf-8-sig")
    logger.debug("benchmark_traditional_ppm_coverage: generated %d rows", len(result))
    return result

def _benchmark_supervised_feature_mode(labeled_df, seed, feature_mode, models=None):
    logger.debug("_benchmark_supervised_feature_mode: feature_mode=%s", feature_mode)
    labels, _ = _prepare_truth(labeled_df); mask = labels != ABSTAIN; data = labeled_df.loc[mask].reset_index(drop=True); y_names = labels.loc[mask].reset_index(drop=True)
    if "fine_label_conflict" in data.columns:
        keep = ~data["fine_label_conflict"].astype(bool); data = data.loc[keep].reset_index(drop=True); y_names = y_names.loc[keep].reset_index(drop=True)
    mapping = {c: i for i, c in enumerate(cfg.BENCHMARK_FINE_CLASSES)}; encoded_y = y_names.map(mapping); valid = encoded_y.notna()
    data = data.loc[valid].reset_index(drop=True); y_names = y_names.loc[valid].reset_index(drop=True); encoded_y = encoded_y.loc[valid]
    if data.empty:
        logger.error("_benchmark_supervised_feature_mode: no valid rows")
        raise ValueError("No valid labeled benchmark rows remain for supervised training.")
    y = encoded_y.astype(int).to_numpy(); data = data.assign(evaluation_group=data["evaluation_group"].astype(str)); tr, dev, test = train_dev_test_split(data, y, seed)
    Xdf = build_feature_frame(data, feature_mode, "fine"); model_map = models or build_models(seed); dev_rows = []
    for name, model in model_map.items():
        try:
            model.fit(Xdf.iloc[tr], y[tr]); pred = _predict_model(model, Xdf.iloc[dev]); pred = np.asarray(pred).reshape(-1); dev_rows.append({"model": name, "feature_mode": feature_mode, "dataset": "external_labeled", "split": "development", "granularity": "fine", "selected_on_dev": False, **evaluate_numeric(y[dev], pred, cfg.BENCHMARK_FINE_CLASSES)})
        except Exception as exc:
            logger.warning("Supervised dev failed: %s: %s", name, exc)
    dev_df = pd.DataFrame(dev_rows)
    if dev_df.empty: return pd.DataFrame()
    best = dev_df.sort_values(["macro_f1", "balanced_accuracy", "accuracy"], ascending=False).iloc[0]["model"]
    dev_df.loc[dev_df["model"] == best, "selected_on_dev"] = True
    train_dev = np.concatenate([tr, dev]); test_rows = []
    for name in model_map:
        try:
            model = build_models(seed)[name]; model.fit(Xdf.iloc[train_dev], y[train_dev]); pred = _predict_model(model, Xdf.iloc[test]); pred = np.asarray(pred).reshape(-1); test_rows.append({"model": name, "feature_mode": feature_mode, "dataset": "external_labeled", "split": "locked_test", "granularity": "fine", "selected_on_dev": name == best, **evaluate_numeric(y[test], pred, cfg.BENCHMARK_FINE_CLASSES)})
        except Exception as exc:
            logger.warning("Supervised test failed: %s: %s", name, exc)
    result = pd.concat([dev_df, pd.DataFrame(test_rows)], ignore_index=True)
    logger.debug("_benchmark_supervised_feature_mode: result rows=%d", len(result))
    return result

def benchmark_supervised_models(labeled_df, seed):
    logger.debug("benchmark_supervised_models: start rows=%d", len(labeled_df))
    parts = []
    for mode in ("gas_only", "ratio_only", "gas_plus_ratio", "gas_plus_traditional"):
        part = _benchmark_supervised_feature_mode(labeled_df, seed, mode)
        if not part.empty: parts.append(part)
    result = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
    result.to_csv(BENCHMARK_DIR / "supervised_fault_benchmark.csv", index=False, encoding="utf-8-sig")
    logger.debug("benchmark_supervised_models: generated %d rows", len(result))
    return result

def _group_balanced_sample_weights(frame):
    """Give each transformer equal total training weight via inverse cluster size.

    This is an inverse-cluster-size weighting scheme: a transformer with n records
    contributes total weight 1, so transformers with many records cannot dominate
    the weak-student fit. The weights are normalized to mean 1 only for numerical stability.
    """
    if "transformer_id" not in frame.columns:
        return np.ones(len(frame), dtype=float)
    counts = frame["transformer_id"].astype(str).value_counts()
    weights = frame["transformer_id"].astype(str).map(lambda x: 1.0 / float(counts.get(x, 1))).to_numpy(dtype=float)
    mean_w = float(np.mean(weights)) if len(weights) else 1.0
    return weights / mean_w if mean_w > 0 else np.ones(len(frame), dtype=float)


def _fit_student_model(model, Xdf, y, sample_weight):
    """Fit with inverse-cluster-size weights when the estimator exposes sample_weight."""
    try:
        model.fit(Xdf, y, classifier__sample_weight=sample_weight)
        return True
    except (TypeError, ValueError):
        # Some estimators (notably the Torch wrapper) do not expose sample_weight through Pipeline.
        model.fit(Xdf, y)
        return False


def _train_weak_students(df, granularity, seed):
    """Train the operational weak-label student set.

    The operational set is unlabeled, so this stage exists to learn from weak
    targets. It intentionally uses a smaller, reproducible model set than the
    broad external benchmark. The broad ML comparison remains in the labeled
    benchmark stage.
    """
    logger.info("_train_weak_students: granularity=%s", granularity)

    target = (
        f"weak_{granularity}_fault_group"
        if granularity == "coarse"
        else f"weak_{granularity}_fault"
    )
    abstain = f"weak_{granularity}_is_ABSTAIN"
    groups = list(
        cfg.WEAK_COARSE_GROUPS
        if granularity == "coarse"
        else cfg.BENCHMARK_FINE_CLASSES
    )

    clean, y, present = create_student_training_targets(
        df,
        target,
        abstain,
        groups,
    )

    if clean.empty:
        raise ValueError(
            f"No non-abstaining weak targets available for granularity={granularity}"
        )

    sample_weight = _group_balanced_sample_weights(clean)
    artifacts = {}

    # Deliberately smaller than the broad external benchmark.
    weak_model_names = tuple(
        getattr(
            cfg,
            "WEAK_STUDENT_MODELS",
            (
                "logistic_regression",
                "random_forest",
                "extra_trees",
                "hist_gradient_boosting",
                "sklearn_mlp",
            ),
        )
    )

    for feature_mode in cfg.STUDENT_FEATURE_MODES:
        Xdf = build_feature_frame(clean, feature_mode, granularity)
        models = build_models(seed)

        for name in weak_model_names:
            model = models.get(name)
            if model is None:
                logger.warning(
                    "Weak student model unavailable | granularity=%s feature_mode=%s model=%s",
                    granularity,
                    feature_mode,
                    name,
                )
                continue

            try:
                used_weights = _fit_student_model(
                    model,
                    Xdf,
                    y,
                    sample_weight,
                )

                artifacts[f"{feature_mode}__{name}"] = {
                    "model": model,
                    "feature_mode": feature_mode,
                    "granularity": granularity,
                    "features": list(Xdf.columns),
                    "labels": list(present),
                    "n_training_rows": int(len(clean)),
                    "n_training_transformers": (
                        int(clean["transformer_id"].nunique())
                        if "transformer_id" in clean.columns
                        else None
                    ),
                    "cluster_weighting": "inverse_transformer_record_count",
                    "sample_weight_used": bool(used_weights),
                    "class_counts": {
                        str(k): int(v)
                        for k, v in pd.Series(y).value_counts().to_dict().items()
                    },
                    "model_source": "operational_unlabeled_weak_targets",
                }

                logger.info(
                    "Weak student trained | granularity=%s feature_mode=%s model=%s rows=%d transformers=%d",
                    granularity,
                    feature_mode,
                    name,
                    len(clean),
                    clean["transformer_id"].nunique()
                    if "transformer_id" in clean.columns
                    else 0,
                )
            except Exception as exc:
                logger.exception(
                    "Weak student training failed | granularity=%s feature_mode=%s model=%s | %s",
                    granularity,
                    feature_mode,
                    name,
                    exc,
                )

    if not artifacts:
        raise RuntimeError(
            f"No weak student model could be trained for granularity={granularity}"
        )

    logger.info(
        "_train_weak_students: trained=%d artifacts | granularity=%s",
        len(artifacts),
        granularity,
    )
    return artifacts

def benchmark_weak_transfer(labeled_df, weak_students, seed):
    logger.debug("benchmark_weak_transfer: start rows=%d", len(labeled_df))
    fine_truth, coarse_truth = _prepare_truth(labeled_df); conflict = labeled_df.get("fine_label_conflict", pd.Series(False, index=labeled_df.index)).astype(bool); valid = (fine_truth != ABSTAIN) & ~conflict
    data = labeled_df.loc[valid].reset_index(drop=True); fine_truth = fine_truth.loc[valid].reset_index(drop=True); coarse_truth = coarse_truth.loc[valid].reset_index(drop=True)
    mapping = {c: i for i, c in enumerate(cfg.BENCHMARK_FINE_CLASSES)}; encoded = fine_truth.map(mapping); valid_encoded = encoded.notna()
    keep = valid_encoded.to_numpy(dtype=bool); data = data.loc[keep].reset_index(drop=True); fine_truth = fine_truth.loc[keep].reset_index(drop=True); coarse_truth = coarse_truth.loc[keep].reset_index(drop=True); y_split = encoded.loc[keep].astype(int).to_numpy()
    if data.empty:
        logger.error("benchmark_weak_transfer: no valid rows")
        raise ValueError("No valid rows remain for weak-transfer benchmark.")
    data = data.assign(evaluation_group=data["evaluation_group"].astype(str)); _, dev, test = train_dev_test_split(data, y_split, seed)
    rows = []
    for granularity in ("coarse", "fine"):
        artifacts = weak_students.get(granularity, {})
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
    logger.debug("benchmark_weak_transfer: generated %d rows", len(result))
    return result

def benchmark_weak_label_model_transfer(labeled_df, weak_models=None):
    """Transfer the operationally fitted weak-label models directly to the labeled benchmark.

    The weak-label artifacts are persisted separately by save_weak_supervision_artifacts().
    Earlier code accidentally passed the unlabeled prediction dataframe here and then
    attempted payload["model"], causing KeyError("model").  This implementation accepts
    the explicit artifact mapping; when omitted, it loads both persisted weak-label
    model artifacts from MODEL_DIR.
    """
    if weak_models is None or not isinstance(weak_models, dict):
        weak_models = {}
        for granularity in ("coarse", "fine"):
            try:
                weak_models[granularity] = load_weak_supervision_artifact(granularity)
            except FileNotFoundError:
                logger.warning(
                    "Missing persisted weak-label artifact for granularity=%s; "
                    "direct transfer will skip this granularity.",
                    granularity,
                )

    fine_truth, coarse_truth = _prepare_truth(labeled_df)
    conflict = labeled_df.get("fine_label_conflict", pd.Series(False, index=labeled_df.index)).astype(bool)
    valid = (fine_truth != ABSTAIN) & ~conflict
    data = labeled_df.loc[valid].reset_index(drop=True)
    fine_truth = fine_truth.loc[valid].reset_index(drop=True); coarse_truth = coarse_truth.loc[valid].reset_index(drop=True)
    mapping = {c:i for i,c in enumerate(cfg.BENCHMARK_FINE_CLASSES)}
    y = fine_truth.map(mapping); keep = y.notna().to_numpy(dtype=bool)
    data = data.loc[keep].reset_index(drop=True); fine_truth = fine_truth.loc[keep].reset_index(drop=True); coarse_truth = coarse_truth.loc[keep].reset_index(drop=True)
    data = data.reset_index(drop=True)
    _, dev, test = train_dev_test_split(data.assign(evaluation_group=data["evaluation_group"].astype(str)), y.loc[keep].astype(int).to_numpy(), cfg.RANDOM_STATE)
    rows=[]
    for granularity, payload in weak_models.items():
        if payload is None: continue
        model = payload["model"]; groups=list(payload["groups"]); methods=list(payload.get("metadata",{}).get("methods", DEFAULT_WEAK_METHODS.keys()))
        # Rebuild exactly the LF columns used during operational fitting.
        from weak_supervision import predict_from_label_model
        pred_frame = predict_from_label_model(model, data, methods, groups, granularity)
        pred = pred_frame[f"weak_{granularity}_fault" if granularity == "fine" else f"weak_{granularity}_fault_group"]
        truth = fine_truth if granularity == "fine" else coarse_truth
        allowed = cfg.BENCHMARK_FINE_CLASSES if granularity == "fine" else cfg.COARSE_FAULT_GROUPS
        for split_name, idx in (("development", dev), ("locked_test", test)):
            metric = _evaluate_method_labels_safely(truth.iloc[idx].reset_index(drop=True), pred.iloc[idx].to_numpy(), allowed)
            rows.append({"granularity":granularity,"model":"weak_label_model","backend":payload.get("metadata",{}).get("backend","unknown"),"training_dataset":"unlabeled_operational_weak","evaluation_dataset":"external_labeled","split":split_name,"selected_on_dev":False, **metric})
    result=pd.DataFrame(rows)
    if not result.empty:
        result.to_csv(BENCHMARK_DIR / "weak_label_model_transfer_fault_benchmark.csv", index=False, encoding="utf-8-sig")
    return result


def benchmark_weak_traditional_hybrids(labeled_df, weak_students, seed):
    """Exhaustively compare each weak student against every non-empty traditional LF subset.

    Hybrid policy is exact agreement: student prediction == traditional subset consensus;
    disagreement becomes ABSTAIN. No hand-tuned fusion weight is introduced. Development
    chooses the configuration; locked test only evaluates the chosen configuration.
    """
    fine_truth, coarse_truth = _prepare_truth(labeled_df)
    conflict = labeled_df.get("fine_label_conflict", pd.Series(False, index=labeled_df.index)).astype(bool)
    coarse_conflict = labeled_df.get("coarse_label_conflict", pd.Series(False, index=labeled_df.index)).astype(bool)
    valid = (fine_truth != ABSTAIN) & ~conflict & ~coarse_conflict
    data = labeled_df.loc[valid].reset_index(drop=True)
    fine_truth=fine_truth.loc[valid].reset_index(drop=True); coarse_truth=coarse_truth.loc[valid].reset_index(drop=True)
    enc=fine_truth.map({c:i for i,c in enumerate(cfg.BENCHMARK_FINE_CLASSES)}); keep=enc.notna().to_numpy(dtype=bool)
    data=data.loc[keep].reset_index(drop=True); fine_truth=fine_truth.loc[keep].reset_index(drop=True); coarse_truth=coarse_truth.loc[keep].reset_index(drop=True); y=enc.loc[keep].astype(int).to_numpy()
    if data.empty: return pd.DataFrame()
    data=data.assign(evaluation_group=data["evaluation_group"].astype(str)); _,dev,test=train_dev_test_split(data,y,seed)
    diagnostic_df=apply_consensus(data)
    rows=[]
    combo_items=list(iter_nonempty_method_combinations(cfg.DIAGNOSTIC_METHODS))
    for granularity in ("coarse","fine"):
        artifacts=weak_students.get(granularity,{})
        trad_col="consensus_fault_group" if granularity=="coarse" else "consensus_fault"
        allowed=cfg.COARSE_FAULT_GROUPS if granularity=="coarse" else cfg.BENCHMARK_FINE_CLASSES
        truth=coarse_truth if granularity=="coarse" else fine_truth
        normalizer=unify_fault if granularity=="coarse" else normalize_fault
        for key,artifact in artifacts.items():
            Xdf=_align_feature_frame(data,artifact["features"],granularity); student=pd.Series(_decode_class_predictions(_predict_model(artifact["model"],Xdf),list(artifact["labels"])),index=data.index).map(normalizer)
            for level,combo in combo_items:
                # Compute the traditional consensus using only this subset.
                trad=apply_consensus_from_existing_diagnostics(data, combo)[trad_col].map(normalizer)
                hybrid=student.where(student.eq(trad), ABSTAIN)
                for split_name,idx in (("development",dev),("locked_test",test)):
                    metric=_evaluate_method_labels_safely(truth.iloc[idx].reset_index(drop=True),hybrid.iloc[idx].to_numpy(),allowed)
                    rows.append({"granularity":granularity,"student_model":key.split("__",1)[1],"feature_mode":artifact["feature_mode"],"methods":"+".join(combo),"method_count":len(combo),"combination_level":level,"hybrid_policy":"exact_agreement_only","training_dataset":"unlabeled_operational_weak","evaluation_dataset":"external_labeled","split":split_name,"selected_on_dev":False,**metric})
    result=pd.DataFrame(rows)
    if not result.empty:
        for granularity in ("coarse","fine"):
            sub=result[(result.granularity==granularity)&(result.split=="development")]
            if not sub.empty:
                best_idx=sub.sort_values(["macro_f1","balanced_accuracy","coverage"],ascending=False,na_position="last").index[0]
                best=result.loc[best_idx]; mask=(result.granularity==granularity)&(result.student_model==best.student_model)&(result.feature_mode==best.feature_mode)&(result.methods==best.methods)
                result.loc[mask,"selected_on_dev"]=True
        result.to_csv(BENCHMARK_DIR / "weak_traditional_hybrid_benchmark.csv", index=False, encoding="utf-8-sig")
    return result

def benchmark_direct_supervised_transfer(unlabeled_df, labeled_df, seed):
    result = benchmark_supervised_models(labeled_df, seed)
    candidates = result[(result["split"] == "locked_test") & result["selected_on_dev"].astype(bool)] if not result.empty else pd.DataFrame()
    candidates.to_csv(BENCHMARK_DIR / "direct_supervised_reference_selected_on_dev.csv", index=False, encoding="utf-8-sig")
    return candidates

def run_unlabeled_pipeline(seed, use_snorkel, save_model=True):
    logger.debug("run_unlabeled_pipeline: start seed=%d", seed)
    raw = load_unlabeled(); df = prepare_unlabeled(raw)
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
                except Exception:
                    logger.warning("Weak student confidence failed | %s | %s", granularity, key, exc_info=True)
    from severity import apply_severity
    df = apply_severity(df, nei_reference=None)
    ranking = build_transformer_ranking(df)

    # Add the latest TDCG as a descriptive baseline for research rank-correlation
    # analysis. It is NOT used by the maintenance ranking itself.
    latest_tdcg = (
        df.sort_values(["transformer_id", "sample_day"], kind="mergesort")
        .groupby("transformer_id", as_index=False)
        .tail(1)[["transformer_id"] + [c for c in ["tdcg", "tdcg_raw", "tdcg_recalc"] if c in df.columns]]
        .copy()
    )
    if not latest_tdcg.empty:
        ranking = ranking.merge(latest_tdcg, on="transformer_id", how="left")

    log_ranking_diagnostics(ranking, 20)

    processed = DATASET_DIR / "processed"
    processed.mkdir(parents=True, exist_ok=True)
    _assert_unique_columns(df, "run_unlabeled_pipeline processed dataframe")
    _assert_unique_columns(ranking, "run_unlabeled_pipeline ranking")
    df.to_parquet(processed / "dga_unlabeled_processed.parquet", index=False)
    ranking.to_parquet(processed / "transformer_ranking.parquet", index=False)
    ranking.to_csv(REPORT_DIR / "transformer_ranking.csv", index=False, encoding="utf-8-sig")
    if save_model:
        MODEL_DIR.mkdir(parents=True, exist_ok=True); joblib.dump({"models": weak_students["coarse"], "training_type": "weak_supervision_plus_discriminative_ml", "training_dataset": str(UNLABELED_PATH), "features": MODEL_FEATURES}, FAULT_MODEL_COARSE_PATH); joblib.dump({"models": weak_students["fine"], "training_type": "weak_supervision_plus_discriminative_ml", "training_dataset": str(UNLABELED_PATH), "features": MODEL_FEATURES}, FAULT_MODEL_FINE_PATH); metadata = {"seed": seed, "unlabeled_dataset": str(UNLABELED_PATH), "weak_supervision": "Snorkel LabelModel", "weak_labeling_methods": list(cfg.WEAK_LABELING_METHODS), "student_feature_modes": list(cfg.STUDENT_FEATURE_MODES), "student_model_count_coarse": len(weak_students["coarse"]), "student_model_count_fine": len(weak_students["fine"]), "severity_source": cfg.STANDARD, "severity_is_weighted": False, "severity_is_failure_probability": False, "ranking_policy": list(cfg.RANKING_POLICY), "ranking_is_weighted": False, "ranking_is_health_score": False, "benchmark_policy": "Operational unlabeled data are used for weak labels and student training only; labeled benchmark is reserved for external evaluation and locked test reporting."}; TRAINING_METADATA_PATH.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.debug("run_unlabeled_pipeline: complete final df shape=%s", df.shape)
    return df, ranking, weak_students

def run_labeled_benchmark(seed):
    logger.debug("run_labeled_benchmark: start seed=%d", seed)
    labeled = apply_consensus(load_labeled_csv_data()); individual = benchmark_traditional_individual(labeled); combinations_result = benchmark_traditional_combinations(labeled, None); ppm = benchmark_traditional_ppm_coverage(labeled); pairwise = pairwise_label_agreement(labeled); pairwise.to_csv(BENCHMARK_DIR / "traditional_pairwise_agreement.csv", index=False, encoding="utf-8-sig"); method_summary = diagnostic_method_summary(labeled); method_summary.to_csv(BENCHMARK_DIR / "traditional_method_summary.csv", index=False, encoding="utf-8-sig"); supervised = benchmark_supervised_models(labeled, seed)
    class_coverage = empirical_fault_class_coverage(labeled, "fine"); class_coverage.to_csv(BENCHMARK_DIR / "traditional_fault_class_coverage.csv", index=False, encoding="utf-8-sig")
    benchmark = {"individual": individual, "combinations": combinations_result, "ppm_coverage": ppm, "class_coverage": class_coverage, "pairwise": pairwise, "method_summary": method_summary, "supervised": supervised}; benchmark["split_manifest"] = _write_split_manifest(labeled, seed)
    logger.debug("run_labeled_benchmark: complete")
    return benchmark

def _write_split_manifest(labeled, seed):
    labels, _ = _prepare_truth(labeled); conflict = labeled.get("fine_label_conflict", pd.Series(False, index=labeled.index)).astype(bool); valid = labels.isin(cfg.BENCHMARK_FINE_CLASSES) & ~conflict; data = labeled.loc[valid].reset_index(drop=True); y = labels.loc[valid].map({c: i for i, c in enumerate(cfg.BENCHMARK_FINE_CLASSES)}).to_numpy(int); data = data.assign(evaluation_group=data["evaluation_group"].astype(str)); _, dev, test = train_dev_test_split(data, y, seed)
    manifest = data[["source_dataset", "source_row", "fault_type_label", "evaluation_group"]].copy(); split = np.full(len(data), "train", dtype=object); split[dev] = "development"; split[test] = "locked_test"; manifest["split"] = split; manifest.to_csv(BENCHMARK_DIR / "benchmark_split_manifest.csv", index=False, encoding="utf-8-sig")
    return manifest

def write_confusion_matrices(weak_transfer, supervised, labeled, seed):
    labels = _prepare_truth(labeled)[0]; conflict = labeled.get("fine_label_conflict", pd.Series(False, index=labeled.index)).astype(bool); valid = (labels != ABSTAIN) & ~conflict; data = labeled.loc[valid].reset_index(drop=True); y_names = labels.loc[valid].reset_index(drop=True); y = y_names.map({c: i for i, c in enumerate(cfg.BENCHMARK_FINE_CLASSES)}).to_numpy(int); _, _, test = train_dev_test_split(data.assign(evaluation_group=data["evaluation_group"].astype(str)), y, seed)
    for mode in supervised["feature_mode"].unique():
        sub = supervised[(supervised["feature_mode"] == mode) & (supervised["split"] == "locked_test")]
        if sub.empty: continue
        best = sub.sort_values(["macro_f1", "balanced_accuracy", "accuracy"], ascending=False).iloc[0]; model = build_models(seed)[best["model"]]; Xdf = build_feature_frame(data, mode, "fine"); train_mask = np.ones(len(data), dtype=bool); train_mask[test] = False; model.fit(Xdf.iloc[train_mask], y[train_mask]); pred = np.asarray(model.predict(Xdf.iloc[test])).reshape(-1); cm = confusion_matrix(y[test], pred, labels=list(range(len(cfg.BENCHMARK_FINE_CLASSES)))); out = pd.DataFrame(cm, index=cfg.BENCHMARK_FINE_CLASSES, columns=cfg.BENCHMARK_FINE_CLASSES); out.to_csv(BENCHMARK_DIR / f"confusion_supervised_{mode}.csv", encoding="utf-8-sig")


def _select_production_fault_pipeline(traditional_result: pd.DataFrame, weak_transfer: pd.DataFrame, weak_students: dict, seed: int):
    """Select one production fault pipeline using DEVELOPMENT only, then freeze it.

    The locked test is never consulted for deployment selection. This keeps the
    paper evaluation honest while allowing the web app to load exactly one
    already-trained artifact.
    """
    candidates = []

    if traditional_result is not None and not traditional_result.empty:
        sub = traditional_result[(traditional_result["granularity"] == "fine") & (traditional_result["split"] == "development")]
        if not sub.empty:
            best = sub.sort_values(["macro_f1", "balanced_accuracy", "coverage"], ascending=False, na_position="last").iloc[0]
            candidates.append({
                "source": "traditional",
                "methods": str(best["methods"]).split("+") if pd.notna(best.get("methods")) else [],
                "macro_f1_dev": float(best["macro_f1"]),
                "balanced_accuracy_dev": float(best["balanced_accuracy"]),
                "coverage_dev": float(best.get("coverage", np.nan)),
                "model_name": None,
                "feature_mode": None,
            })

    if weak_transfer is not None and not weak_transfer.empty:
        sub = weak_transfer[(weak_transfer["granularity"] == "fine") & (weak_transfer["split"] == "development")]
        if not sub.empty:
            best = sub.sort_values(["macro_f1", "balanced_accuracy", "coverage"], ascending=False, na_position="last").iloc[0]
            model_key = f"{best['feature_mode']}__{best['model']}"
            artifact = weak_students.get("fine", {}).get(model_key)
            if artifact is not None:
                candidates.append({
                    "source": "student",
                    "methods": [],
                    "macro_f1_dev": float(best["macro_f1"]),
                    "balanced_accuracy_dev": float(best["balanced_accuracy"]),
                    "coverage_dev": float(best.get("coverage", np.nan)),
                    "model_name": str(best["model"]),
                    "feature_mode": str(best["feature_mode"]),
                    "artifact": artifact,
                })

    if not candidates:
        selection = {"source": "traditional", "methods": [], "selection_metric": "macro_f1", "selection_split": "development", "selection_note": "No trained student candidate was available."}
        joblib.dump(selection, PRODUCTION_SELECTION_PATH)
        return selection

    # Development-only selection. This is not a hand-tuned numerical fusion.
    chosen = sorted(candidates, key=lambda x: (x["macro_f1_dev"], x["balanced_accuracy_dev"], x["coverage_dev"] if np.isfinite(x["coverage_dev"]) else -1.0), reverse=True)[0]
    selection = {
        "source": chosen["source"],
        "methods": chosen.get("methods", []),
        "model_name": chosen.get("model_name"),
        "feature_mode": chosen.get("feature_mode"),
        "selection_metric": "macro_f1",
        "selection_split": "development",
        "macro_f1_dev": chosen["macro_f1_dev"],
        "balanced_accuracy_dev": chosen["balanced_accuracy_dev"],
        "coverage_dev": chosen["coverage_dev"],
        "random_state": int(seed),
        "runtime_training": False,
        "selection_candidates": [
            {k: v for k, v in item.items() if k != "artifact"}
            for item in candidates
        ],
    }
    if chosen["source"] == "student":
        # Keep selection metadata separate from the estimator so the joblib does
        # not serialize the same model object twice. Render needs this one file.
        joblib.dump({"selection": selection, "student_artifact": chosen["artifact"]}, PRODUCTION_SELECTION_PATH)
    else:
        joblib.dump(selection, PRODUCTION_SELECTION_PATH)
    return selection

def _run_single_seed(args=None):
    parser = argparse.ArgumentParser(
        description="DGA offline research pipeline: weak supervision, ML comparison, transfer evaluation and Excel report."
    )
    parser.add_argument(
        "--mode",
        choices=["benchmark", "unlabeled", "transfer", "all"],
        default="all",
    )
    parser.add_argument(
        "--use-snorkel",
        action="store_true",
        default=True,
        help="Use Snorkel LabelModel (the only supported weak-supervision backend).",
    )
    parser.add_argument("--seed", type=int, default=cfg.RANDOM_STATE)
    parsed = parser.parse_args(args)

    logger.info(
        "DGA OFFLINE RESEARCH RUN START | mode=%s | seed=%d | snorkel=%s",
        parsed.mode,
        parsed.seed,
        parsed.use_snorkel,
    )
    set_global_seed(parsed.seed)

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    BENCHMARK_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    run_id = __import__("datetime").datetime.now(
        __import__("datetime").timezone.utc
    ).strftime("%Y%m%dT%H%M%SZ")
    manifest_path = REPORT_DIR / "experiment_run_manifest.json"
    failed_manifest_path = REPORT_DIR / "experiment_run_failed.json"

    # Never advertise an old completed run while a new run is executing.
    if manifest_path.exists():
        try:
            manifest_path.unlink()
        except OSError:
            logger.warning("Could not remove previous experiment manifest.", exc_info=True)

    stages = []

    def stage(name, fn):
        logger.info("Experiment stage START: %s", name)
        try:
            result = fn()
            stages.append({"stage": name, "status": "COMPLETE"})
            logger.info("Experiment stage COMPLETE: %s", name)
            return result
        except Exception as exc:
            stages.append({
                "stage": name,
                "status": "FAILED",
                "error": repr(exc),
            })
            logger.exception("Experiment stage FAILED: %s", name)

            failed_manifest = {
                "run_id": run_id,
                "started_at_utc": run_id,
                "finished_at_utc": __import__("datetime").datetime.now(
                    __import__("datetime").timezone.utc
                ).isoformat(),
                "status": "FAILED",
                "mode": parsed.mode,
                "seed": int(parsed.seed),
                "use_snorkel": bool(parsed.use_snorkel),
                "failed_stage": name,
                "stages": stages,
            }
            failed_manifest_path.write_text(
                json.dumps(failed_manifest, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            raise

    unlabeled_result = None
    benchmark = None
    weak_transfer = pd.DataFrame()

    if parsed.mode in {"unlabeled", "transfer", "all"}:
        unlabeled_result = stage(
            "unlabeled_weak_supervision_and_students",
            lambda: run_unlabeled_pipeline(
                parsed.seed,
                parsed.use_snorkel,
                save_model=True,
            ),
        )

    if parsed.mode in {"benchmark", "all"}:
        benchmark = stage(
            "external_labeled_benchmark",
            lambda: run_labeled_benchmark(parsed.seed),
        )

    if parsed.mode in {"transfer", "all"}:
        if unlabeled_result is None:
            unlabeled_result = stage(
                "unlabeled_weak_supervision_and_students",
                lambda: run_unlabeled_pipeline(
                    parsed.seed,
                    parsed.use_snorkel,
                    save_model=True,
                ),
            )

        labeled = apply_consensus(load_labeled_csv_data())

        weak_transfer = stage(
            "weak_student_transfer_to_external_labels",
            lambda: benchmark_weak_transfer(
                labeled,
                unlabeled_result[2],
                parsed.seed,
            ),
        )

        stage(
            "weak_label_model_direct_transfer",
            lambda: benchmark_weak_label_model_transfer(
                labeled,
                None,
            ),
        )

        stage(
            "weak_student_traditional_hybrid_comparison",
            lambda: benchmark_weak_traditional_hybrids(
                labeled,
                unlabeled_result[2],
                parsed.seed,
            ),
        )

        if benchmark is None:
            traditional = benchmark_traditional_combinations(
                labeled,
                None,
            )
        else:
            traditional = benchmark["combinations"]

        stage(
            "production_fault_pipeline_selection",
            lambda: _select_production_fault_pipeline(
                traditional,
                weak_transfer,
                unlabeled_result[2],
                parsed.seed,
            ),
        )

    # Teammate-inspired but scientifically separated analyses.
    #
    # These analyses are descriptive/reporting layers only:
    # - absolute concentration vs scale-invariant ratio/percentage domain gap
    # - fleet rank correlation against TDCG and other independent evidence
    if parsed.mode == "all":
        from research_analysis import (
            build_domain_gap_analysis,
            build_rank_correlation_analysis,
            cross_dataset_transfer_grid,
        )

        labeled_all = load_labeled_csv_data()
        source_frames = []
        for source_name, source_df in labeled_all.groupby(
            "source_dataset",
            sort=False,
        ):
            source_frames.append((source_name, source_df))

        if len(source_frames) >= 2:
            stage(
                "domain_gap_absolute_vs_ratio_analysis",
                lambda: build_domain_gap_analysis(
                    source_frames[:2],
                    BENCHMARK_DIR,
                ),
            )

            stage(
                "cross_dataset_transfer_grid",
                lambda: cross_dataset_transfer_grid(
                    source_frames[:2],
                    build_feature_frame,
                    build_models,
                    BENCHMARK_DIR,
                    parsed.seed,
                ),
            )

        if unlabeled_result is not None:
            stage(
                "fleet_rank_correlation_baseline_analysis",
                lambda: build_rank_correlation_analysis(
                    unlabeled_result[1],
                    BENCHMARK_DIR,
                ),
            )

    # Build Excel only after all requested computational stages succeeded.
    stage(
        "excel_research_report",
        lambda: __import__("experiment").build_excel_report(
            REPORT_DIR,
            DATASET_DIR / "processed",
            REPORT_DIR / "dga_research_report.xlsx",
        ),
    )

    # Compute the exact manifest requirements from the requested run mode.
    required_artifacts = []

    if parsed.mode in {"benchmark", "all"}:
        required_artifacts.extend(
            [
                "reports/benchmark/benchmark_split_manifest.csv",
                "reports/benchmark/traditional_individual_benchmark.csv",
                "reports/benchmark/traditional_combinations_benchmark.csv",
                "reports/benchmark/traditional_ppm_coverage.csv",
                "reports/benchmark/traditional_fault_class_coverage.csv",
                "reports/benchmark/traditional_pairwise_agreement.csv",
                "reports/benchmark/traditional_method_summary.csv",
                "reports/benchmark/supervised_fault_benchmark.csv",
            ]
        )

    if parsed.mode in {"unlabeled", "transfer", "all"}:
        required_artifacts.extend(
            [
                "models/fault_classifiers_coarse.joblib",
                "models/fault_classifiers_fine.joblib",
                "models/training_metadata.json",
                "reports/transformer_ranking.csv",
                "dataset/processed/dga_unlabeled_processed.parquet",
                "dataset/processed/transformer_ranking.parquet",
            ]
        )

    if parsed.mode in {"transfer", "all"}:
        required_artifacts.extend(
            [
                "reports/benchmark/weak_transfer_fault_benchmark.csv",
                "reports/benchmark/weak_label_model_transfer_fault_benchmark.csv",
                "reports/benchmark/weak_traditional_hybrid_benchmark.csv",
                "models/production_fault_selection.joblib",
            ]
        )

    if parsed.mode == "all":
        required_artifacts.extend(
            [
                "reports/benchmark/domain_gap_absolute_vs_ratio.csv",
                "reports/benchmark/domain_gap_representation_summary.csv",
                "reports/benchmark/rank_correlation_spearman.csv",
                "reports/benchmark/rank_correlation_kendall.csv",
                "reports/benchmark/cross_dataset_transfer_grid.csv",
            ]
        )

    required_artifacts.append("reports/dga_research_report.xlsx")

    missing = [
        rel
        for rel in required_artifacts
        if not (cfg.BACKEND_ROOT / rel).exists()
    ]
    if missing:
        error = (
            "Experiment stages completed but required artifacts are missing: "
            + ", ".join(missing)
        )
        failed_manifest = {
            "run_id": run_id,
            "status": "FAILED",
            "mode": parsed.mode,
            "seed": int(parsed.seed),
            "use_snorkel": bool(parsed.use_snorkel),
            "failed_stage": "artifact_verification",
            "error": error,
            "stages": stages,
        }
        failed_manifest_path.write_text(
            json.dumps(failed_manifest, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        raise RuntimeError(error)

    completed_at = __import__("datetime").datetime.now(
        __import__("datetime").timezone.utc
    ).isoformat()

    manifest = {
        "run_id": run_id,
        "completed_at_utc": completed_at,
        "status": "COMPLETE",
        "mode": parsed.mode,
        "seed": int(parsed.seed),
        "use_snorkel": bool(parsed.use_snorkel),
        "operational_dataset_rows": (
            int(len(unlabeled_result[0]))
            if unlabeled_result is not None
            else None
        ),
        "operational_transformers": (
            int(unlabeled_result[0]["transformer_id"].nunique())
            if unlabeled_result is not None
            else None
        ),
        "required_artifacts": required_artifacts,
        "stages": stages,
        "methodology": {
            "operational_training_is_unlabeled": True,
            "external_labels_are_used_for_evaluation_only": True,
            "development_selects_candidates": True,
            "locked_test_is_not_used_for_selection": True,
            "ranking_is_weighted": False,
            "ranking_is_health_score": False,
            "severity_is_weighted": False,
            "severity_is_failure_probability": False,
            "teammate_inspired_domain_gap_analysis": "descriptive_only",
            "teammate_inspired_rank_correlation": "descriptive_only",
        },
    }

    # Atomic publish: the web sees a COMPLETE manifest only after every
    # required artifact exists.
    temporary_manifest = REPORT_DIR / f".experiment_run_manifest.{run_id}.tmp"
    temporary_manifest.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    temporary_manifest.replace(manifest_path)

    if failed_manifest_path.exists():
        try:
            failed_manifest_path.unlink()
        except OSError:
            pass

    logger.info(
        "DGA OFFLINE RESEARCH RUN COMPLETE | run_id=%s | mode=%s",
        run_id,
        parsed.mode,
    )


DEFAULT_EVALUATION_SEEDS = (42, 43, 44, 45, 46)


def _snapshot_seed_outputs(seed: int):
    """Copy current run outputs into an immutable per-seed archive."""
    import shutil

    seed_root = REPORT_DIR / "seeds" / f"seed_{int(seed)}"
    seed_benchmark = seed_root / "benchmark"
    seed_models = seed_root / "models"
    seed_processed = seed_root / "processed"
    for directory in (seed_benchmark, seed_models, seed_processed):
        directory.mkdir(parents=True, exist_ok=True)

    for source_dir, destination_dir in (
        (BENCHMARK_DIR, seed_benchmark),
        (MODEL_DIR, seed_models),
        (DATASET_DIR / "processed", seed_processed),
    ):
        if not source_dir.exists():
            continue
        for source in source_dir.iterdir():
            if source.is_file():
                shutil.copy2(source, destination_dir / source.name)

    for source in (
        REPORT_DIR / "transformer_ranking.csv",
        REPORT_DIR / "experiment_run_manifest.json",
    ):
        if source.exists():
            shutil.copy2(source, seed_root / source.name)

    logger.info("Seed outputs archived: seed=%d path=%s", int(seed), seed_root)
    return seed_root


def _aggregate_seed_csvs(seed_roots):
    """Create long-form and mean/std summaries from archived seed CSVs."""
    import re

    aggregate_root = BENCHMARK_DIR / "multiseed"
    aggregate_root.mkdir(parents=True, exist_ok=True)
    collected = {}
    completed_seeds = []

    for root in seed_roots:
        match = re.search(r"seed_(\d+)$", str(root))
        if not match:
            continue
        seed = int(match.group(1))
        completed_seeds.append(seed)
        bench = root / "benchmark"
        if not bench.exists():
            continue
        for path in bench.glob("*.csv"):
            try:
                frame = pd.read_csv(path)
            except Exception as exc:
                logger.warning("Could not read seed CSV %s: %s", path, exc)
                continue
            frame.insert(0, "seed", seed)
            collected.setdefault(path.name, []).append(frame)

    long_form_paths = []
    for filename, frames in collected.items():
        combined = pd.concat(frames, ignore_index=True, sort=False)
        target = aggregate_root / filename
        combined.to_csv(target, index=False, encoding="utf-8-sig")
        long_form_paths.append(target)

    metric_targets = []
    preferred = {
        "traditional_individual_benchmark.csv",
        "traditional_combinations_benchmark.csv",
        "supervised_fault_benchmark.csv",
        "weak_transfer_fault_benchmark.csv",
        "weak_label_model_transfer_fault_benchmark.csv",
        "weak_traditional_hybrid_benchmark.csv",
    }
    metric_cols_all = [
        "accuracy", "balanced_accuracy", "macro_f1", "weighted_f1",
        "precision", "recall", "coverage", "abstention_rate",
    ]
    key_cols_all = [
        "model", "feature_mode", "methods", "granularity", "split", "dataset",
    ]

    for filename in preferred:
        path = aggregate_root / filename
        if not path.exists():
            continue
        frame = pd.read_csv(path)
        metric_cols = [c for c in metric_cols_all if c in frame.columns]
        key_cols = [c for c in key_cols_all if c in frame.columns]
        if "seed" not in frame.columns or not metric_cols or not key_cols:
            continue
        aggregate = (
            frame.groupby(key_cols, dropna=False)[metric_cols]
            .agg(["mean", "std", "count"])
            .reset_index()
        )
        aggregate.columns = [
            "_".join(str(part) for part in col if str(part) not in {"", "None"}).rstrip("_")
            if isinstance(col, tuple) else str(col)
            for col in aggregate.columns
        ]
        out = aggregate_root / f"multiseed_summary_{Path(filename).stem}.csv"
        aggregate.to_csv(out, index=False, encoding="utf-8-sig")
        metric_targets.append(out)

    manifest = {
        "seed_count": len(completed_seeds),
        "seeds": sorted(set(completed_seeds)),
        "long_form_csv_count": len(long_form_paths),
        "metric_summary_csvs": [str(p.relative_to(REPORT_DIR)) for p in metric_targets],
        "severity_accuracy_note": (
            "Not computed: supplied labeled datasets contain fault labels but no independent severity ground truth."
        ),
    }
    (aggregate_root / "multiseed_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return aggregate_root, long_form_paths, metric_targets


def main(args=None):
    """Run one seed or an explicit repeated-seed evaluation grid."""
    parser = argparse.ArgumentParser(
        description="DGA offline research pipeline with optional multi-seed evaluation."
    )
    parser.add_argument(
        "--mode",
        choices=["benchmark", "unlabeled", "transfer", "all"],
        default="all",
    )
    parser.add_argument(
        "--use-snorkel",
        action="store_true",
        default=True,
        help="Use Snorkel LabelModel (the only supported weak-supervision backend).",
    )
    parser.add_argument("--seed", type=int, default=cfg.RANDOM_STATE)
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=None,
        help="Explicit seed list, e.g. --seeds 42 43 44 45 46. Overrides --seed.",
    )
    parser.add_argument(
        "--all-seeds",
        action="store_true",
        help="Run the predefined evaluation grid: 42 43 44 45 46.",
    )
    parsed = parser.parse_args(args)

    if parsed.all_seeds and parsed.seeds:
        parser.error("Use either --all-seeds or --seeds, not both.")

    if parsed.all_seeds:
        seeds = list(DEFAULT_EVALUATION_SEEDS)
    elif parsed.seeds:
        seeds = list(dict.fromkeys(int(seed) for seed in parsed.seeds))
    else:
        seeds = [int(parsed.seed)]

    logger.info(
        "DGA OFFLINE RESEARCH REQUEST | mode=%s | seeds=%s | snorkel=%s",
        parsed.mode,
        seeds,
        parsed.use_snorkel,
    )

    if len(seeds) == 1:
        single_args = ["--mode", parsed.mode, "--seed", str(seeds[0])]
        if parsed.use_snorkel:
            single_args.append("--use-snorkel")
        return _run_single_seed(single_args)

    seed_roots = []
    failures = []
    for seed in seeds:
        logger.info("MULTI-SEED RUN START | seed=%d", seed)
        single_args = ["--mode", parsed.mode, "--seed", str(seed)]
        if parsed.use_snorkel:
            single_args.append("--use-snorkel")
        try:
            _run_single_seed(single_args)
            seed_roots.append(_snapshot_seed_outputs(seed))
            logger.info("MULTI-SEED RUN COMPLETE | seed=%d", seed)
        except Exception as exc:
            failures.append({"seed": int(seed), "error": repr(exc)})
            logger.exception("MULTI-SEED RUN FAILED | seed=%d", seed)

    aggregate_root, _, metric_targets = _aggregate_seed_csvs(seed_roots)
    manifest = {
        "status": "COMPLETE" if not failures else "PARTIAL",
        "mode": parsed.mode,
        "seeds_requested": seeds,
        "seeds_completed": sorted(int(seed) for seed in [
            int(item.name.replace("seed_", "")) for item in seed_roots
        ]),
        "failures": failures,
        "aggregate_root": str(aggregate_root),
        "metric_summary_csvs": [str(p) for p in metric_targets],
        "severity_accuracy": (
            "NOT_COMPUTED: independent severity ground truth is absent from the supplied labeled datasets."
        ),
        "methodology": {
            "operational_data_used_for_training": True,
            "external_labeled_data_used_for_evaluation": True,
            "locked_test_used_for_selection": False,
            "manual_lf_weights": False,
            "arbitrary_severity_weights": False,
        },
    }
    multi_manifest = REPORT_DIR / "multiseed_experiment_manifest.json"
    multi_manifest.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    if failures:
        raise RuntimeError(
            "Multi-seed run completed with failures: "
            + ", ".join(f"seed {item['seed']}" for item in failures)
        )

    logger.info(
        "MULTI-SEED RUN COMPLETE | seeds=%s | aggregate=%s",
        seeds,
        aggregate_root,
    )
    return manifest


if __name__ == "__main__":
    main()