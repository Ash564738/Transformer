# train_models.py
from pathlib import Path
import argparse
import logging
import numpy as np
import pandas as pd
import joblib
import lightgbm as lgb
import xgboost as xgb
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import accuracy_score, f1_score, classification_report, mean_squared_error
from logging_config import init_logging
init_logging()
logger = logging.getLogger(__name__)

from config import FAULT_LABELS, SEVERITY_LABELS
from weak_supervision import WEAK_GROUPS, weak_supervision_pipeline, create_student_training_targets
from consensus import apply_consensus
from feature_engineering import (
    preprocess_types, sort_and_deduplicate, filter_rows_for_model,
    add_missingness_flags, impute_optional_context_by_transformer,
    add_tdcg, add_rating_features, add_metadata_features,
    add_ratio_features, add_duval_input_features,
    add_calendar_and_sequence_features, add_lag_delta_rate_features,
    add_rolling_features, add_ewm_features, add_cross_gas_trend_features,
    add_quality_flags
)
# ------------------------------
# Config
# ------------------------------
from config import FAULT_LABELS, SEVERITY_LABELS, BACKEND_DATA_DIR, BACKEND_ROOT

LABELED_PATH = Path(BACKEND_DATA_DIR) / "dga_labeled.parquet"
UNLABELED_PATH = Path(BACKEND_DATA_DIR) / "dga_unlabeled.parquet"
WEAK_LABEL_PATH = Path(BACKEND_DATA_DIR) / "dga_weak_labels.parquet"
MODEL_DIR = Path(BACKEND_ROOT) / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)
# Ensure backend dataset dir exists
Path(BACKEND_DATA_DIR).mkdir(parents=True, exist_ok=True)

ID_LIKE_COLS = [
    "transformer_id", "sample_day", "loc", "name", "ser", "codetx", "mfg",
    "sample_year", "sample_month", "sample_quarter", "sample_dayofyear",
    "sample_weekday", "record_idx", "tested_day", "tdcg_source",
    "fault_type_label", "fault_rule",
    "fault_ieee_key_gas", "fault_iec_ratio", "fault_duval_triangle_1",
    "fault_duval_pentagon", "fault_rogers_ratio", "fault_detail_json",
    "severity_label", "severity_gas_score", "severity_trend_score",
    "severity_fault_score", "severity_aging_score", "severity_score",
    "severity_gas_detail", "severity_trend_detail", "severity_aging_detail",
    "fleet_priority_rank", "target_fault_type", "target_severity",
    "target_severity_score"
]

CORE_GASES = ["h2", "ch4", "c2h6", "c2h4", "c2h2", "co", "co2"]
OPTIONAL_NUMERIC = ["o2", "n2", "water", "temp"]
REQUIRED_WEAK_VOTE_COLS = [
    "keygas_fault", "iec_fault", "rogers_fault", "doernenburg_fault",
    "duval_triangle_fault", "fault_p1", "duval_pentagon_fault",
]


def build_training_features_from_clean(df: pd.DataFrame) -> pd.DataFrame:
    """Mirror inference feature engineering before weak-label generation."""
    df = preprocess_types(df)
    df = sort_and_deduplicate(df)
    df = filter_rows_for_model(df, max_missing_core=3)
    df = add_missingness_flags(df, OPTIONAL_NUMERIC + ["year_energized", "tdcg_raw"])
    df = impute_optional_context_by_transformer(df)
    df = add_tdcg(df)
    df = add_rating_features(df)
    df = add_metadata_features(df)
    df = add_ratio_features(df)
    df = add_duval_input_features(df)
    df = add_calendar_and_sequence_features(df)

    temporal_value_cols = [c for c in CORE_GASES + ["tdcg"] if c in df.columns]
    for c in ["water", "temp"]:
        if c in df.columns:
            temporal_value_cols.append(c)

    df = add_lag_delta_rate_features(df, temporal_value_cols)
    df = add_rolling_features(df, temporal_value_cols)
    df = add_ewm_features(df, temporal_value_cols)
    df = add_cross_gas_trend_features(df)
    df = add_quality_flags(df)
    return df


def ensure_consensus_inputs(df: pd.DataFrame) -> pd.DataFrame:
    if all(c in df.columns for c in REQUIRED_WEAK_VOTE_COLS):
        return df
    logger.info("Weak-label input has no traditional DGA votes; building features and consensus first.")
    return apply_consensus(build_training_features_from_clean(df))

# ------------------------------
# Data loading
# ------------------------------
def load_data(conf_threshold=70):
    """Load labeled data if available, otherwise fall back to weak-labeled pseudo labels.
    The weak label file produced by the weak_supervision pipeline contains
    probabilistic labels and argmax/confidence fields that can be used to train
    the student when no hand-labeled dataset is present.
    """
    if LABELED_PATH.exists():
        df = pd.read_parquet(LABELED_PATH)
    elif WEAK_LABEL_PATH.exists():
        df = pd.read_parquet(WEAK_LABEL_PATH)
    else:
        raise FileNotFoundError(f"Labeled dataset not found: {LABELED_PATH}. If you intended to train from weak labels, run the weak supervision step first to generate {WEAK_LABEL_PATH}.")
    df["sample_day"] = pd.to_datetime(df["sample_day"])
    df = df.dropna(subset=["transformer_id", "sample_day"])
    if "diagnostic_confidence" in df.columns:
        df = df[df["diagnostic_confidence"] >= conf_threshold]
    return df

def prepare_static_data(df):
    # If the DataFrame only has weak supervision outputs (weak_prob_*, weak_fault_group),
    # synthesize the standard target columns expected by the training pipeline.
    from config import FAULT_LABELS, SEVERITY_LABELS, config as cfg
    if "fault_type_label" not in df.columns and "weak_fault_group" in df.columns:
        logger.info("Synthesizing fault_type_label and severity targets from weak labels...")
        # Map weak_fault_group string to FAULT_LABELS index (fallback to ABSTAIN if missing)
        grp_to_idx = {v: i for i, v in enumerate(FAULT_LABELS)}
        df["fault_type_label"] = df["weak_fault_group"].map(lambda x: grp_to_idx.get(x, grp_to_idx.get("ABSTAIN"))).astype(int)
        # Map group to severity score via config.SEVERITY_BY_GROUP
        df["severity_score"] = df["weak_fault_group"].map(lambda x: cfg.SEVERITY_BY_GROUP.get(x, cfg.SEVERITY_BY_GROUP.get("ABSTAIN", 1))).astype(float)
        # Convert numeric severity_score to severity_label using boundaries
        b = cfg.SEVERITY_CLASS_BOUNDARIES
        def score_to_label(s):
            if s < b[0]:
                return 0
            if s < b[1]:
                return 1
            if s < b[2]:
                return 2
            return 3
        df["severity_label"] = df["severity_score"].map(score_to_label).astype(int)

    feature_cols = [c for c in df.columns if c not in ID_LIKE_COLS
                    and not c.startswith("target_")
                    and df[c].dtype in ['float64','int64','int32','float32']]
    X = (
        df[feature_cols]
        .apply(pd.to_numeric, errors='coerce')
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0)
    )

    # Dòng cũ gây lỗi:
    # y_fault = df["fault_type_label"].map({v:i for i,v in enumerate(FAULT_LABELS)}).values.astype(int)
    # Sửa thành: dùng trực tiếp cột đã là số
    y_fault = df["fault_type_label"].astype(int).values

    # Dòng cũ:
    # y_sev_cls = df["severity_label"].map({v:i for i,v in enumerate(SEVERITY_LABELS)}).values.astype(int)
    # Sửa thành:
    y_sev_cls = df["severity_label"].astype(int).values

    y_sev_score = df["severity_score"].values.astype(float)
    groups = df["transformer_id"].values
    return X, y_fault, y_sev_cls, y_sev_score, groups, feature_cols

# ------------------------------
# Sequence dataset
# ------------------------------
class SeqDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
    def __len__(self): return len(self.y)
    def __getitem__(self, idx): return self.X[idx], self.y[idx]

class TemporalModel(nn.Module):
    def __init__(self, input_size, hidden=64, layers=2, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden, layers, batch_first=True, dropout=dropout)
        self.attn = nn.Linear(hidden, 1)
        self.fc = nn.Sequential(
            nn.Linear(hidden, hidden//2),
            nn.ReLU(),
            nn.Linear(hidden//2, 1)
        )
    def forward(self, x):
        out, _ = self.lstm(x)
        w = torch.softmax(self.attn(out).squeeze(-1), dim=1).unsqueeze(-1)
        ctx = (out * w).sum(dim=1)
        return self.fc(ctx).squeeze(-1)

def build_sequences(df, feature_cols, seq_len=5, target_col="severity_score"):
    seqs, tgts, grps = [], [], []
    for tid, grp in df.groupby("transformer_id"):
        grp = grp.sort_values("sample_day")
        feat = (
            grp[feature_cols]
            .replace([np.inf, -np.inf], np.nan)
            .fillna(0)
            .values
            .astype(np.float32)
        )
        targ = grp[target_col].values.astype(np.float32)
        for i in range(len(grp)):
            start = max(0, i - seq_len + 1)
            win = feat[start:i+1]
            padded = np.zeros((seq_len, feat.shape[1]), dtype=np.float32)
            padded[-win.shape[0]:] = win
            seqs.append(padded)
            tgts.append(targ[i])
            grps.append(tid)
    return np.stack(seqs), np.array(tgts), np.array(grps)

# ------------------------------
# Training functions
# ------------------------------
def train_fault_cls(X_train, y_train, X_val, y_val):
    dtrain = lgb.Dataset(X_train, label=y_train)
    dval = lgb.Dataset(X_val, label=y_val, reference=dtrain)
    params = {
        "objective": "multiclass", "num_class": len(FAULT_LABELS),
        "metric": "multi_logloss", "learning_rate": 0.03,
        "num_leaves": 63, "min_data_in_leaf": 20,
        "feature_fraction": 0.85, "bagging_fraction": 0.85,
        "bagging_freq": 5, "verbosity": -1, "seed": 42,
    }
    model = lgb.train(params, dtrain, num_boost_round=600,
                      valid_sets=[dtrain, dval], valid_names=["train","val"],
                      callbacks=[lgb.early_stopping(40), lgb.log_evaluation(False)])
    return model

def train_severity_cls(X_train, y_train, X_val, y_val):
    # similar to fault but for severity labels
    dtrain = lgb.Dataset(X_train, label=y_train)
    dval = lgb.Dataset(X_val, label=y_val, reference=dtrain)
    params = {
        "objective": "multiclass", "num_class": len(SEVERITY_LABELS),
        "metric": "multi_logloss", "learning_rate": 0.03,
        "num_leaves": 63, "min_data_in_leaf": 20,
        "feature_fraction": 0.85, "bagging_fraction": 0.85,
        "bagging_freq": 5, "verbosity": -1, "seed": 42,
    }
    model = lgb.train(params, dtrain, num_boost_round=600,
                      valid_sets=[dtrain, dval], callbacks=[lgb.early_stopping(40), lgb.log_evaluation(False)])
    return model

def train_severity_reg(X_train, y_train, X_val, y_val):
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dval = xgb.DMatrix(X_val, label=y_val)
    params = {
        "objective": "reg:squarederror", "eval_metric": "rmse",
        "eta": 0.05, "max_depth": 6, "subsample": 0.8,
        "colsample_bytree": 0.8, "seed": 42,
    }
    model = xgb.train(params, dtrain, num_boost_round=500,
                      evals=[(dtrain,"train"), (dval,"val")],
                      early_stopping_rounds=40, verbose_eval=False)
    return model

def train_temporal(X_seq, y_seq, train_idx, val_idx, feature_dim):
    scaler = StandardScaler()
    X_seq_2d = X_seq.reshape(-1, feature_dim)
    scaler.fit(X_seq_2d)
    X_scaled = scaler.transform(X_seq_2d).reshape(X_seq.shape)
    train_ds = SeqDataset(X_scaled[train_idx], y_seq[train_idx])
    val_ds = SeqDataset(X_scaled[val_idx], y_seq[val_idx])
    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=64)

    model = TemporalModel(feature_dim)
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
    criterion = nn.MSELoss()
    best_loss = float("inf")
    best_state = None
    for epoch in range(35):
        model.train()
        for xb, yb in train_loader:
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for xb, yb in val_loader:
                val_loss += criterion(model(xb), yb).item() * xb.size(0)
        val_loss /= len(val_ds)
        if val_loss < best_loss:
            best_loss = val_loss
            best_state = model.state_dict()
    model.load_state_dict(best_state)
    return model, scaler

# ------------------------------
# Weak supervision helpers
# ------------------------------

def load_unlabeled_data() -> pd.DataFrame:
    if not UNLABELED_PATH.exists():
        raise FileNotFoundError(
            f"Unlabeled dataset không tìm thấy tại {UNLABELED_PATH}.\n"
            "Vui lòng chạy 'python backend/prepare_unlabeled_data.py' để tạo file này từ dữ liệu tích lũy."
        )
    df = pd.read_parquet(UNLABELED_PATH)
    df["sample_day"] = pd.to_datetime(df["sample_day"])
    return df


def generate_weak_labels(df: pd.DataFrame, use_snorkel: bool = True) -> pd.DataFrame:
    df = ensure_consensus_inputs(df)
    df, label_model, groups = weak_supervision_pipeline(df, use_snorkel=use_snorkel)
    logger = logging.getLogger(__name__)
    logger.info("Weak supervision labels generated: %s", ", ".join(groups))
    return df


def prepare_weak_labeled_data(df: pd.DataFrame):
    if "weak_fault_group" not in df.columns:
        raise ValueError("Weak supervision labels are missing. Run generate_weak_labels first.")
    # Keep only samples where weak label didn't abstain
    df = df[df["weak_fault_group"] != "ABSTAIN"].copy()
    y, weights = create_student_training_targets(df)
    feature_cols = [c for c in df.columns if c not in ID_LIKE_COLS
                    and not c.startswith("target_")
                    and not c.startswith("weak_prob_")
                    and df[c].dtype in ['float64','int64','int32','float32']]
    X = (
        df[feature_cols]
        .apply(pd.to_numeric, errors='coerce')
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0)
    )
    return X, y, weights, feature_cols


# ------------------------------
# Main
# ------------------------------
def train_student_fault_classifier(
    X, y, weights, X_val, y_val, w_val, labels=None,
):
    labels = labels or FAULT_LABELS
    dtrain = lgb.Dataset(X, label=y, weight=weights)
    dval = lgb.Dataset(X_val, label=y_val, weight=w_val, reference=dtrain)
    params = {
        "objective": "multiclass",
        "num_class": len(labels),
        "metric": "multi_logloss",
        "learning_rate": 0.03,
        "num_leaves": 63,
        "min_data_in_leaf": 20,
        "feature_fraction": 0.85,
        "bagging_fraction": 0.85,
        "bagging_freq": 5,
        "verbosity": -1,
        "seed": 42,
    }
    model = lgb.train(
        params,
        dtrain,
        num_boost_round=600,
        valid_sets=[dtrain, dval],
        valid_names=["train", "val"],
        callbacks=[lgb.early_stopping(40), lgb.log_evaluation(False)],
    )
    return model

def main(args=None):
    parser = argparse.ArgumentParser(description="Train DGA models with optional weak supervision.")
    parser.add_argument(
        "--weak-supervision",
        action="store_true",
        help="Generate probabilistic target labels from traditional diagnostics before training.",
    )
    parser.add_argument(
        "--use-snorkel",
        action="store_true",
        help="Use Snorkel's LabelModel to estimate weak supervision probabilistic labels.",
    )
    parser.add_argument(
        "--weak-only",
        action="store_true",
        help="Create weak supervision labels only; do not train the standard models.",
    )
    parsed = parser.parse_args(args=args)

    generated_weak_df = None
    if parsed.weak_supervision:
        logger.info("Loading unlabeled data for weak supervision...")
        unlabeled = load_unlabeled_data()
        weak_df = generate_weak_labels(unlabeled, use_snorkel=parsed.use_snorkel)
        generated_weak_df = weak_df
        WEAK_LABEL_PATH.parent.mkdir(parents=True, exist_ok=True)
        weak_df.to_parquet(WEAK_LABEL_PATH)
        logger.info("Weak supervision labels saved to %s", WEAK_LABEL_PATH)
        if parsed.weak_only:
            return

    if parsed.weak_supervision and generated_weak_df is not None and not LABELED_PATH.exists():
        logger.info("Using in-memory weak-labeled dataframe for training (no labeled dataset found at %s).", LABELED_PATH)
        df = generated_weak_df.copy()
    else:
        df = load_data()
    X, y_fault, y_sev_cls, y_sev_score, groups, feature_cols = prepare_static_data(df)

    weak_training = None
    fault_labels_for_artifact = FAULT_LABELS
    if parsed.weak_supervision:
        weak_df = generated_weak_df if generated_weak_df is not None else pd.read_parquet(WEAK_LABEL_PATH)
        if "weak_fault_group" in weak_df.columns:
            X_weak, y_fault_weak, sample_weights, weak_feature_cols = prepare_weak_labeled_data(weak_df)
            weak_training = (X_weak, y_fault_weak, sample_weights, weak_feature_cols)
            fault_labels_for_artifact = WEAK_GROUPS
            logger.info("Training fault classifier on weak supervision pseudo-labels with sample weights.")
        else:
            raise RuntimeError("Weak supervision file exists but weak fault labels are missing.")

    fault_feature_names = feature_cols
    if weak_training is not None:
        X_weak, y_fault_weak, sample_weights, weak_feature_cols = weak_training
        fault_feature_names = weak_feature_cols
        weak_splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
        # Use transformer_id as groups for a group-aware split when available
        try:
            groups_weak = weak_df.loc[X_weak.index, "transformer_id"].values
        except Exception:
            groups_weak = None
        if groups_weak is None:
            # Fallback to stratified split on pseudo-labels if transformer groups aren't available
            from sklearn.model_selection import StratifiedShuffleSplit
            ssplit = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
            weak_train_idx, weak_test_idx = next(ssplit.split(X_weak, y_fault_weak))
        else:
            weak_train_idx, weak_test_idx = next(weak_splitter.split(X_weak, y_fault_weak, groups=groups_weak))
        X_train_fault, X_test_fault = X_weak.iloc[weak_train_idx], X_weak.iloc[weak_test_idx]
        y_fault_train, y_fault_test = y_fault_weak[weak_train_idx], y_fault_weak[weak_test_idx]
        weight_train, weight_test = sample_weights[weak_train_idx], sample_weights[weak_test_idx]
        fault_model = train_student_fault_classifier(
            X_train_fault, y_fault_train, weight_train, X_test_fault, y_fault_test, weight_test,
            labels=fault_labels_for_artifact,
        )
    else:
        splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
        train_idx, test_idx = next(splitter.split(X, y_sev_score, groups))
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_fault_train, y_fault_test = y_fault[train_idx], y_fault[test_idx]
        weight_train = None
        weight_test = None
        fault_model = train_fault_cls(X_train, y_fault_train, X_test, y_fault_test)

    # 2. Severity classifier
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, test_idx = next(splitter.split(X, y_sev_score, groups))
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_sev_cls_train, y_sev_cls_test = y_sev_cls[train_idx], y_sev_cls[test_idx]
    y_score_train, y_score_test = y_sev_score[train_idx], y_sev_score[test_idx]
    sev_cls_model = train_severity_cls(X_train, y_sev_cls_train, X_test, y_sev_cls_test)
    # 3. Severity regressor
    sev_reg_model = train_severity_reg(X_train, y_score_train, X_test, y_score_test)

    # 4. Temporal model
    X_seq, y_seq, groups_seq = build_sequences(df, feature_cols, seq_len=5, target_col="severity_score")
    train_idx_seq, test_idx_seq = next(splitter.split(X_seq, y_seq, groups_seq))
    temp_model, scaler = train_temporal(X_seq, y_seq, train_idx_seq, test_idx_seq, X_seq.shape[2])

    # Save
    joblib.dump({
        "model": fault_model,
        "features": fault_feature_names,
        "labels": fault_labels_for_artifact,
        "target_type": "weak_group" if fault_labels_for_artifact == WEAK_GROUPS else "fault_detail",
    }, MODEL_DIR / "fault_classifier.joblib")
    joblib.dump({"model": sev_cls_model, "features": feature_cols, "labels": SEVERITY_LABELS}, MODEL_DIR / "severity_classifier.joblib")
    joblib.dump({"model": sev_reg_model, "features": feature_cols}, MODEL_DIR / "severity_regressor.joblib")
    torch.save(temp_model.state_dict(), MODEL_DIR / "temporal_model_state_dict.pt")
    joblib.dump(scaler, MODEL_DIR / "temporal_scaler.joblib")
    logger.info("Models saved to %s", MODEL_DIR)


if __name__ == "__main__":
    main()
