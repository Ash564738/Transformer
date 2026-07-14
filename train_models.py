# train_models.py
from pathlib import Path
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

# ------------------------------
# Config
# ------------------------------
LABELED_PATH = Path("dataset/processed/dga_labeled.parquet")
MODEL_DIR = Path("models")
MODEL_DIR.mkdir(exist_ok=True)

FAULT_LABELS = ["Normal", "PD", "D1", "D2", "T1", "T2", "T3", "Cellulose", "MIXED", "UNCERTAIN"]
SEVERITY_LABELS = ["Normal", "Watchlist", "Warning", "Critical"]

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

# ------------------------------
# Data loading
# ------------------------------
def load_data():
    df = pd.read_parquet(LABELED_PATH)
    df["sample_day"] = pd.to_datetime(df["sample_day"])
    df = df.dropna(subset=["transformer_id", "sample_day"])
    return df

def prepare_static_data(df):
    # Feature columns
    feature_cols = [c for c in df.columns if c not in ID_LIKE_COLS
                    and not c.startswith("target_")
                    and df[c].dtype in ['float64','int64','int32','float32']]
    X = df[feature_cols].apply(pd.to_numeric, errors='coerce').fillna(0)
    y_fault = df["fault_type_label"].map({v:i for i,v in enumerate(FAULT_LABELS)}).values.astype(int)
    y_sev_cls = df["severity_label"].map({v:i for i,v in enumerate(SEVERITY_LABELS)}).values.astype(int)
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
        feat = grp[feature_cols].fillna(0).values.astype(np.float32)
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
# Main
# ------------------------------
def main():
    df = load_data()
    X, y_fault, y_sev_cls, y_sev_score, groups, feature_cols = prepare_static_data(df)

    splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, test_idx = next(splitter.split(X, y_sev_score, groups))

    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_fault_train, y_fault_test = y_fault[train_idx], y_fault[test_idx]
    y_sev_cls_train, y_sev_cls_test = y_sev_cls[train_idx], y_sev_cls[test_idx]
    y_score_train, y_score_test = y_sev_score[train_idx], y_sev_score[test_idx]

    # 1. Fault classifier
    fault_model = train_fault_cls(X_train, y_fault_train, X_test, y_fault_test)
    # 2. Severity classifier
    sev_cls_model = train_severity_cls(X_train, y_sev_cls_train, X_test, y_sev_cls_test)
    # 3. Severity regressor
    sev_reg_model = train_severity_reg(X_train, y_score_train, X_test, y_score_test)

    # 4. Temporal model
    X_seq, y_seq, groups_seq = build_sequences(df, feature_cols, seq_len=5, target_col="severity_score")
    train_idx_seq, test_idx_seq = next(splitter.split(X_seq, y_seq, groups_seq))
    temp_model, scaler = train_temporal(X_seq, y_seq, train_idx_seq, test_idx_seq, X_seq.shape[2])

    # Save
    joblib.dump({"model": fault_model, "features": feature_cols, "labels": FAULT_LABELS}, MODEL_DIR / "fault_classifier.joblib")
    joblib.dump({"model": sev_cls_model, "features": feature_cols, "labels": SEVERITY_LABELS}, MODEL_DIR / "severity_classifier.joblib")
    joblib.dump({"model": sev_reg_model, "features": feature_cols}, MODEL_DIR / "severity_regressor.joblib")
    torch.save(temp_model.state_dict(), MODEL_DIR / "temporal_model_state_dict.pt")
    joblib.dump(scaler, MODEL_DIR / "temporal_scaler.joblib")
    print("Models saved to", MODEL_DIR)

if __name__ == "__main__":
    main()