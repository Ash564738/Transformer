# train_unsupervised_models.py
from __future__ import annotations

import argparse
import copy
import logging
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import xgboost as xgb
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset

from logging_config import init_logging

init_logging()
logger = logging.getLogger(__name__)


from config import (
    FAULT_LABELS,
    SEVERITY_LABELS,
    DATASET_DIR,
    DATABASE_DIR,
    MODEL_DIR,
    REPORT_DIR,
)

from weak_supervision import (
    WEAK_GROUPS,
    weak_supervision_pipeline,
    create_student_training_targets,
)

from consensus import apply_consensus

from severity import apply_severity

from feature_engineering import (
    build_training_features_from_clean,
    get_model_feature_columns,
)


# ============================================================================
# Paths
# ============================================================================

LABELED_PATH = (
    Path(DATASET_DIR)
    / "processed"
    / "dga_labeled.parquet"
)

UNLABELED_PATH = (
    Path(DATASET_DIR)
    / "processed"
    / "dga_unlabeled.parquet"
)

WEAK_LABEL_PATH = (
    Path(DATASET_DIR)
    / "processed"
    / "dga_weak_labels.parquet"
)


# ============================================================================
# Training configuration
# ============================================================================

RANDOM_STATE = 42
TEST_SIZE = 0.20

SEQ_LEN = 5

AE_EPOCHS = 30
AE_LATENT_DIM = 32

TEMPORAL_EPOCHS = 35

WEAK_CONFIDENCE_THRESHOLD = 0.70

BATCH_SIZE = 64

DEVICE = (
    torch.device("cuda")
    if torch.cuda.is_available()
    else torch.device("cpu")
)


# ============================================================================
# Feature groups
# ============================================================================

CORE_GASES = [
    "h2",
    "ch4",
    "c2h6",
    "c2h4",
    "c2h2",
    "co",
    "co2",
]

OPTIONAL_NUMERIC = [
    "o2",
    "n2",
    "water",
    "temp",
]


# ============================================================================
# Columns that must never become ML features
#
# These contain:
#
#   identifiers
#   labels
#   consensus outputs
#   weak supervision targets
#   severity targets
#   target leakage
# ============================================================================

NON_FEATURE_COLS = {
    # IDs / metadata
    "transformer_id",
    "sample_day",
    "loc",
    "name",
    "ser",
    "codetx",
    "mfg",
    "record_idx",
    "tested_day",
    "tdcg_source",

    # Raw labels
    "fault_type_label",
    "fault_rule",
    "target_fault_type",
    "target_severity",
    "target_severity_score",

    # Legacy diagnostic labels
    "fault_ieee_key_gas",
    "fault_iec_ratio",
    "fault_duval_triangle_1",
    "fault_duval_pentagon",
    "fault_rogers_ratio",
    "fault_detail_json",

    # Method outputs
    "keygas_fault",
    "iec_fault",
    "rogers_fault",
    "doernenburg_fault",
    "duval_triangle_fault",
    "duval_pentagon_fault",
    "duval_pentagon_p1_fault",
    "duval_pentagon_p2_fault",
    "fault_p1",
    "fault_p2",

    # Consensus output
    "consensus_fault",
    "mixed_components",
    "diagnostic_votes",
    "diagnostic_confidence",
    "diagnostic_active_methods",
    "diagnostic_method_count",
    "diagnostic_coverage",

    # Weak supervision output
    "weak_fault_group",
    "weak_fault_confidence",
    "weak_fault_is_ABSTAIN",

    # Severity targets / derived values
    "severity_label",
    "severity_label_text",
    "severity_score",
    "severity_gas_score",
    "severity_trend_score",
    "severity_anomaly_score",
    "severity_gas_rank",
    "severity_trend_rank",

    # IEEE status outputs
    "ieee_dga_status",
    "ieee_dga_status_label",
    "ieee_dga_status_reason",
    "ieee_table1_exceeding_gases",
    "ieee_table2_exceeding_gases",
    "ieee_table3_exceeding_gases",
    "ieee_table4_exceeding_gases",
    "ieee_confirmation_required",
    "ieee_extreme_dga",

    # Fleet ranking
    "fleet_priority_rank",
    "fleet_priority_score",
    "fleet_priority_percent",
    "recommended_action",
    "final_score",
}


# Additional prefixes to prevent leakage.
NON_FEATURE_PREFIXES = (
    "target_",
    "weak_prob_",
    "fault_",
    "severity_",
    "ieee_",
    "fleet_",
)


# ============================================================================
# Reproducibility
# ============================================================================

def set_global_seed(
    seed: int = RANDOM_STATE,
) -> None:

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


set_global_seed()

# ============================================================================
# Consensus input preparation
# ============================================================================

REQUIRED_WEAK_VOTE_COLS = [
    "keygas_fault",
    "iec_fault",
    "rogers_fault",
    "doernenburg_fault",
    "duval_triangle_fault",
    "duval_pentagon_p2_fault",
]


def ensure_consensus_inputs(
    df: pd.DataFrame,
) -> pd.DataFrame:

    if all(
        col in df.columns
        for col in REQUIRED_WEAK_VOTE_COLS
    ):
        return df

    logger.info(
        "Diagnostic votes missing; running DGA consensus pipeline."
    )

    return apply_consensus(
        build_training_features_from_clean(
            df
        )
    )


# ============================================================================
# Data loading
# ============================================================================

def load_data(
    conf_threshold: float = 70.0,
) -> pd.DataFrame:

    if LABELED_PATH.exists():

        logger.info(
            "Loading labeled dataset: %s",
            LABELED_PATH,
        )

        df = pd.read_parquet(
            LABELED_PATH
        )

    elif WEAK_LABEL_PATH.exists():

        logger.info(
            "Loading weak-labeled dataset: %s",
            WEAK_LABEL_PATH,
        )

        df = pd.read_parquet(
            WEAK_LABEL_PATH
        )

    else:

        raise FileNotFoundError(
            f"Dataset not found:\n"
            f"  {LABELED_PATH}\n"
            f"  {WEAK_LABEL_PATH}"
        )

    if "sample_day" not in df.columns:
        raise ValueError(
            "Dataset is missing sample_day."
        )

    df["sample_day"] = pd.to_datetime(
        df["sample_day"],
        errors="coerce",
    )

    df = df.dropna(
        subset=[
            "transformer_id",
            "sample_day",
        ]
    )

    if (
        "diagnostic_confidence"
        in df.columns
    ):

        before = len(df)

        confidence = pd.to_numeric(
            df["diagnostic_confidence"],
            errors="coerce",
        )

        df = df[
            confidence >= conf_threshold
        ].copy()

        logger.info(
            "Consensus-confidence filter: "
            "%d -> %d rows",
            before,
            len(df),
        )

    return df.reset_index(
        drop=True
    )


def load_unlabeled_data() -> pd.DataFrame:

    if not UNLABELED_PATH.exists():
        raise FileNotFoundError(
            f"Unlabeled dataset not found: "
            f"{UNLABELED_PATH}"
        )

    df = pd.read_parquet(
        UNLABELED_PATH
    )

    df["sample_day"] = pd.to_datetime(
        df["sample_day"],
        errors="coerce",
    )

    df = df.dropna(
        subset=[
            "transformer_id",
            "sample_day",
        ]
    ).reset_index(
        drop=True
    )

    logger.info(
        "Loaded unlabeled data: %d rows",
        len(df),
    )

    return df


# ============================================================================
# Feature column selection
# ============================================================================

def select_feature_columns(
    df: pd.DataFrame,
) -> List[str]:

    feature_cols = []

    for col in df.columns:

        if col in NON_FEATURE_COLS:
            continue

        if any(
            col.startswith(prefix)
            for prefix in NON_FEATURE_PREFIXES
        ):
            continue

        if col in {
            "sample_year",
            "sample_month",
            "sample_quarter",
            "sample_dayofyear",
            "sample_weekday",
        }:
            # Calendar variables are allowed as numeric features.
            pass

        if not pd.api.types.is_numeric_dtype(
            df[col]
        ):
            continue

        feature_cols.append(
            col
        )

    if not feature_cols:
        raise ValueError(
            "No usable numeric features remain after leakage filtering."
        )

    return feature_cols


def dataframe_to_matrix(
    df: pd.DataFrame,
    feature_cols: List[str],
) -> pd.DataFrame:

    X = (
        df[feature_cols]
        .apply(
            pd.to_numeric,
            errors="coerce",
        )
        .replace(
            [
                np.inf,
                -np.inf,
            ],
            np.nan,
        )
        .fillna(0.0)
    )

    return X.astype(
        np.float32
    )


# ============================================================================
# Target preparation
# ============================================================================

def prepare_fault_target(
    df: pd.DataFrame,
) -> Tuple[
    Optional[np.ndarray],
    Optional[List[str]],
]:

    if "fault_type_label" in df.columns:

        y = pd.to_numeric(
            df["fault_type_label"],
            errors="coerce",
        )

        valid = y.notna()

        if not valid.all():
            logger.warning(
                "Dropping %d rows with invalid fault targets.",
                (~valid).sum(),
            )

            df.drop(
                index=df.index[~valid],
                inplace=True,
            )

        y = y.loc[
            df.index
        ].astype(
            int
        ).to_numpy()

        return (
            y,
            list(FAULT_LABELS),
        )

    return (
        None,
        None,
    )


def prepare_severity_target(
    df: pd.DataFrame,
) -> Tuple[
    np.ndarray,
    np.ndarray,
]:

    if "ieee_dga_status" not in df.columns:
        raise ValueError(
            "IEEE severity must be computed before training."
        )

    status = pd.to_numeric(
        df["ieee_dga_status"],
        errors="coerce",
    ).fillna(1)

    status = status.clip(
        1,
        3,
    ).astype(
        int
    )

    # 0,1,2 target for three DGA statuses.
    y_cls = (
        status - 1
    ).to_numpy(
        dtype=np.int64
    )

    score = pd.to_numeric(
        df["severity_score"],
        errors="coerce",
    ).fillna(
        20.0
    ).to_numpy(
        dtype=np.float32
    )

    return (
        y_cls,
        score,
    )


# ============================================================================
# Sequence builders
# ============================================================================

def build_sequences(
    df: pd.DataFrame,
    feature_cols: List[str],
    seq_len: int = SEQ_LEN,
) -> Tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:

    seqs = []
    groups = []
    indices = []

    for transformer_id, group in df.groupby(
        "transformer_id",
        sort=False,
    ):

        group = group.sort_values(
            "sample_day"
        )

        values = (
            dataframe_to_matrix(
                group,
                feature_cols,
            )
            .to_numpy(
                dtype=np.float32
            )
        )

        for i in range(
            len(group)
        ):

            start = max(
                0,
                i - seq_len + 1,
            )

            window = values[
                start:i + 1
            ]

            padded = np.zeros(
                (
                    seq_len,
                    values.shape[1],
                ),
                dtype=np.float32,
            )

            padded[
                -len(window):
            ] = window

            seqs.append(
                padded
            )

            groups.append(
                transformer_id
            )

            indices.append(
                group.index[i]
            )

    if not seqs:

        raise ValueError(
            "No temporal sequences could be built."
        )

    X_seq = np.stack(
        seqs
    )

    groups = np.asarray(
        groups
    )

    indices = np.asarray(
        indices
    )

    return (
        X_seq,
        groups,
        indices,
        None,
    )


# ============================================================================
# Torch datasets
# ============================================================================

class SeqDataset(Dataset):

    def __init__(
        self,
        X,
        y=None,
    ):

        self.X = torch.as_tensor(
            X,
            dtype=torch.float32,
        )

        self.y = (
            None
            if y is None
            else torch.as_tensor(
                y,
                dtype=torch.float32,
            )
        )

    def __len__(self):
        return len(
            self.X
        )

    def __getitem__(
        self,
        idx,
    ):

        if self.y is None:
            return self.X[idx]

        return (
            self.X[idx],
            self.y[idx],
        )


# ============================================================================
# Temporal supervised model
# ============================================================================

class TemporalModel(nn.Module):

    def __init__(
        self,
        input_size: int,
        hidden: int = 64,
        layers: int = 2,
        dropout: float = 0.2,
    ):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden,
            num_layers=layers,
            batch_first=True,
            dropout=(
                dropout
                if layers > 1
                else 0.0
            ),
        )

        self.attn = nn.Linear(
            hidden,
            1,
        )

        self.fc = nn.Sequential(
            nn.Linear(
                hidden,
                hidden // 2,
            ),
            nn.ReLU(),
            nn.Linear(
                hidden // 2,
                1,
            ),
        )

    def forward(
        self,
        x,
    ):

        out, _ = self.lstm(
            x
        )

        weights = torch.softmax(
            self.attn(
                out
            ).squeeze(-1),
            dim=1,
        ).unsqueeze(-1)

        context = (
            out
            * weights
        ).sum(
            dim=1
        )

        return self.fc(
            context
        ).squeeze(-1)


# ============================================================================
# Correct temporal autoencoder
# ============================================================================

class TemporalAutoencoder(nn.Module):

    def __init__(
        self,
        input_size: int,
        hidden: int = 64,
        layers: int = 2,
        dropout: float = 0.2,
        latent_dim: int = 32,
    ):
        super().__init__()

        self.input_size = input_size
        self.hidden = hidden
        self.layers = layers
        self.latent_dim = latent_dim

        self.encoder = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden,
            num_layers=layers,
            batch_first=True,
            dropout=(
                dropout
                if layers > 1
                else 0.0
            ),
        )

        self.to_latent = nn.Linear(
            hidden,
            latent_dim,
        )

        self.from_latent = nn.Linear(
            latent_dim,
            hidden,
        )

        self.decoder = nn.LSTM(
            input_size=hidden,
            hidden_size=hidden,
            num_layers=layers,
            batch_first=True,
            dropout=(
                dropout
                if layers > 1
                else 0.0
            ),
        )

        self.output_projection = nn.Linear(
            hidden,
            input_size,
        )

    def encode(
        self,
        x,
    ):

        _, (
            hidden_state,
            _,
        ) = self.encoder(
            x
        )

        latent = self.to_latent(
            hidden_state[-1]
        )

        return latent

    def forward(
        self,
        x,
    ):

        latent = self.encode(
            x
        )

        decoder_state = (
            self.from_latent(
                latent
            )
        )

        h0 = (
            decoder_state
            .unsqueeze(0)
            .repeat(
                self.layers,
                1,
                1,
            )
        )

        c0 = torch.zeros_like(
            h0
        )

        # Constant decoder input.
        decoder_input = torch.zeros(
            (
                x.size(0),
                x.size(1),
                self.hidden,
            ),
            device=x.device,
        )

        decoded, _ = self.decoder(
            decoder_input,
            (
                h0,
                c0,
            ),
        )

        reconstructed = (
            self.output_projection(
                decoded
            )
        )

        return (
            reconstructed,
            latent,
        )


# ============================================================================
# Autoencoder training
# ============================================================================

def train_autoencoder(
    X_seq: np.ndarray,
    train_idx: np.ndarray,
    feature_dim: int,
    epochs: int = AE_EPOCHS,
    latent_dim: int = AE_LATENT_DIM,
):
    logger.info(
        "Training temporal autoencoder on %d training sequences.",
        len(train_idx),
    )

    scaler = StandardScaler()

    X_train = X_seq[
        train_idx
    ]

    scaler.fit(
        X_train.reshape(
            -1,
            feature_dim,
        )
    )

    X_train_scaled = scaler.transform(
        X_train.reshape(
            -1,
            feature_dim,
        )
    ).reshape(
        X_train.shape
    )

    dataset = SeqDataset(
        X_train_scaled
    )

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
    )

    model = TemporalAutoencoder(
        input_size=feature_dim,
        latent_dim=latent_dim,
    ).to(
        DEVICE
    )

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=3e-4,
    )

    criterion = nn.MSELoss()

    best_loss = float(
        "inf"
    )

    best_state = None

    for epoch in range(
        epochs
    ):

        model.train()

        running_loss = 0.0

        for batch in loader:

            batch = batch.to(
                DEVICE
            )

            optimizer.zero_grad()

            reconstructed, _ = (
                model(
                    batch
                )
            )

            loss = criterion(
                reconstructed,
                batch,
            )

            loss.backward()

            optimizer.step()

            running_loss += (
                loss.item()
                * batch.size(0)
            )

        average_loss = (
            running_loss
            / len(dataset)
        )

        if average_loss < best_loss:

            best_loss = average_loss

            # IMPORTANT:
            # state_dict() must be copied.
            best_state = copy.deepcopy(
                model.state_dict()
            )

        if (
            epoch + 1
        ) % 10 == 0:

            logger.info(
                "AE epoch %d/%d loss=%.6f",
                epoch + 1,
                epochs,
                average_loss,
            )

    if best_state is None:
        raise RuntimeError(
            "Autoencoder failed to produce a best state."
        )

    model.load_state_dict(
        best_state
    )

    logger.info(
        "Temporal autoencoder complete. "
        "Best loss=%.6f",
        best_loss,
    )

    return (
        model,
        scaler,
    )


def extract_embeddings(
    model: TemporalAutoencoder,
    scaler: StandardScaler,
    X_seq: np.ndarray,
    feature_dim: int,
) -> np.ndarray:

    X_scaled = scaler.transform(
        X_seq.reshape(
            -1,
            feature_dim,
        )
    ).reshape(
        X_seq.shape
    )

    model.eval()

    batches = []

    with torch.no_grad():

        for start in range(
            0,
            len(X_scaled),
            BATCH_SIZE,
        ):

            batch = torch.as_tensor(
                X_scaled[
                    start:start
                    + BATCH_SIZE
                ],
                dtype=torch.float32,
                device=DEVICE,
            )

            latent = model.encode(
                batch
            )

            batches.append(
                latent.cpu().numpy()
            )

    return np.vstack(
        batches
    )


# ============================================================================
# Model training helpers
# ============================================================================

def train_fault_cls(
    X_train,
    y_train,
    X_val,
    y_val,
):

    num_classes = len(
        FAULT_LABELS
    )

    dtrain = lgb.Dataset(
        X_train,
        label=y_train,
    )

    dval = lgb.Dataset(
        X_val,
        label=y_val,
        reference=dtrain,
    )

    params = {
        "objective": "multiclass",
        "num_class": num_classes,
        "metric": "multi_logloss",
        "learning_rate": 0.03,
        "num_leaves": 63,
        "min_data_in_leaf": 20,
        "feature_fraction": 0.85,
        "bagging_fraction": 0.85,
        "bagging_freq": 5,
        "verbosity": -1,
        "seed": RANDOM_STATE,
    }

    return lgb.train(
        params,
        dtrain,
        num_boost_round=600,
        valid_sets=[dval],
        valid_names=["val"],
        callbacks=[
            lgb.early_stopping(
                40,
                first_metric_only=True,
                verbose=False,
            ),
            lgb.log_evaluation(
                False
            ),
        ],
    )


def train_severity_cls(
    X_train,
    y_train,
    X_val,
    y_val,
):

    dtrain = lgb.Dataset(
        X_train,
        label=y_train,
    )

    dval = lgb.Dataset(
        X_val,
        label=y_val,
        reference=dtrain,
    )

    params = {
        "objective": "multiclass",
        "num_class": 3,
        "metric": "multi_logloss",
        "learning_rate": 0.03,
        "num_leaves": 31,
        "min_data_in_leaf": 20,
        "feature_fraction": 0.85,
        "bagging_fraction": 0.85,
        "bagging_freq": 5,
        "verbosity": -1,
        "seed": RANDOM_STATE,
    }

    return lgb.train(
        params,
        dtrain,
        num_boost_round=500,
        valid_sets=[dval],
        valid_names=["val"],
        callbacks=[
            lgb.early_stopping(
                40,
                first_metric_only=True,
                verbose=False,
            ),
            lgb.log_evaluation(
                False
            ),
        ],
    )


def train_severity_reg(
    X_train,
    y_train,
    X_val,
    y_val,
):

    dtrain = xgb.DMatrix(
        X_train,
        label=y_train,
    )

    dval = xgb.DMatrix(
        X_val,
        label=y_val,
    )

    params = {
        "objective": "reg:squarederror",
        "eval_metric": "rmse",
        "eta": 0.05,
        "max_depth": 6,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "seed": RANDOM_STATE,
    }

    callbacks = [
        xgb.callback.EarlyStopping(
            rounds=40,
            save_best=True,
            maximize=False,
        )
    ]

    return xgb.train(
        params,
        dtrain,
        num_boost_round=500,
        evals=[
            (
                dval,
                "validation",
            )
        ],
        callbacks=callbacks,
        verbose_eval=False,
    )


# ============================================================================
# Temporal severity model
# ============================================================================

def train_temporal(
    X_seq: np.ndarray,
    y_seq: np.ndarray,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    feature_dim: int,
):

    # ------------------------------------------------------------------
    # IMPORTANT:
    # scaler fitted ONLY on training data.
    # ------------------------------------------------------------------

    scaler = StandardScaler()

    X_train = X_seq[
        train_idx
    ]

    X_val = X_seq[
        val_idx
    ]

    scaler.fit(
        X_train.reshape(
            -1,
            feature_dim,
        )
    )

    X_train_scaled = scaler.transform(
        X_train.reshape(
            -1,
            feature_dim,
        )
    ).reshape(
        X_train.shape
    )

    X_val_scaled = scaler.transform(
        X_val.reshape(
            -1,
            feature_dim,
        )
    ).reshape(
        X_val.shape
    )

    train_ds = SeqDataset(
        X_train_scaled,
        y_seq[train_idx],
    )

    val_ds = SeqDataset(
        X_val_scaled,
        y_seq[val_idx],
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=BATCH_SIZE,
    )

    model = TemporalModel(
        feature_dim
    ).to(
        DEVICE
    )

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=3e-4,
    )

    criterion = nn.MSELoss()

    best_loss = float(
        "inf"
    )

    best_state = None

    for epoch in range(
        TEMPORAL_EPOCHS
    ):

        model.train()

        for xb, yb in train_loader:

            xb = xb.to(
                DEVICE
            )

            yb = yb.to(
                DEVICE
            )

            optimizer.zero_grad()

            prediction = model(
                xb
            )

            loss = criterion(
                prediction,
                yb,
            )

            loss.backward()

            optimizer.step()

        model.eval()

        validation_total = 0.0

        with torch.no_grad():

            for xb, yb in val_loader:

                xb = xb.to(
                    DEVICE
                )

                yb = yb.to(
                    DEVICE
                )

                prediction = model(
                    xb
                )

                loss = criterion(
                    prediction,
                    yb,
                )

                validation_total += (
                    loss.item()
                    * xb.size(0)
                )

        val_loss = (
            validation_total
            / len(val_ds)
        )

        if val_loss < best_loss:

            best_loss = val_loss

            best_state = copy.deepcopy(
                model.state_dict()
            )

    if best_state is None:
        raise RuntimeError(
            "Temporal model failed to produce a best state."
        )

    model.load_state_dict(
        best_state
    )

    return (
        model,
        scaler,
    )


# ============================================================================
# Weak supervision
# ============================================================================

def generate_weak_labels(
    df: pd.DataFrame,
    use_snorkel: bool = True,
) -> pd.DataFrame:

    df = ensure_consensus_inputs(
        df
    )

    df, label_model, groups = (
        weak_supervision_pipeline(
            df,
            use_snorkel=use_snorkel,
            confidence_threshold=(
                WEAK_CONFIDENCE_THRESHOLD
            ),
        )
    )

    return df


def prepare_weak_labeled_data(
    df: pd.DataFrame,
):

    if "weak_fault_group" not in df.columns:
        raise ValueError(
            "Missing weak_fault_group."
        )

    # --------------------------------------------------------
    # Keep only confident pseudo-labels.
    # --------------------------------------------------------

    confidence = pd.to_numeric(
        df.get(
            "weak_fault_confidence",
            0.0,
        ),
        errors="coerce",
    ).fillna(
        0.0
    )

    df = df[
        (
            df["weak_fault_group"]
            != "ABSTAIN"
        )
        & (
            confidence
            >= WEAK_CONFIDENCE_THRESHOLD
        )
    ].copy()

    if df.empty:
        raise ValueError(
            "No weak-labeled samples meet the confidence threshold."
        )

    y, sample_weights = (
        create_student_training_targets(
            df
        )
    )

    feature_cols = select_feature_columns(
        df
    )

    X = dataframe_to_matrix(
        df,
        feature_cols,
    )

    return (
        X,
        y,
        sample_weights,
        feature_cols,
        df,
    )


def train_student_fault_classifier(
    X,
    y,
    weights,
    X_val,
    y_val,
    w_val,
    labels,
):

    dtrain = lgb.Dataset(
        X,
        label=y,
        weight=weights,
    )

    dval = lgb.Dataset(
        X_val,
        label=y_val,
        weight=w_val,
        reference=dtrain,
    )

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
        "seed": RANDOM_STATE,
    }

    return lgb.train(
        params,
        dtrain,
        num_boost_round=600,
        valid_sets=[dval],
        valid_names=["validation"],
        callbacks=[
            lgb.early_stopping(
                40,
                first_metric_only=True,
                verbose=False,
            ),
            lgb.log_evaluation(
                False
            ),
        ],
    )


# ============================================================================
# Group split helper
# ============================================================================

def group_split(
    X,
    groups,
    test_size=TEST_SIZE,
):

    splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=test_size,
        random_state=RANDOM_STATE,
    )

    train_idx, val_idx = next(
        splitter.split(
            X,
            groups=groups,
        )
    )

    return (
        train_idx,
        val_idx,
    )


# ============================================================================
# Main
# ============================================================================

def main(
    args=None,
):

    parser = argparse.ArgumentParser(
        description=(
            "Train DGA diagnostic ML models "
            "with IEEE C57.104-2019 severity handling."
        )
    )

    parser.add_argument(
        "--weak-supervision",
        action="store_true",
        help=(
            "Generate weak labels and train the student "
            "fault classifier."
        ),
    )

    parser.add_argument(
        "--use-snorkel",
        action="store_true",
        help=(
            "Use Snorkel LabelModel if available."
        ),
    )

    parser.add_argument(
        "--weak-only",
        action="store_true",
        help=(
            "Generate and save weak labels only."
        ),
    )

    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=70.0,
        help=(
            "Minimum consensus confidence for labeled data."
        ),
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=RANDOM_STATE,
    )

    parsed = parser.parse_args(
        args=args
    )

    set_global_seed(
        parsed.seed
    )

    generated_weak_df = None
    weak_training = None

    # ==================================================================
    # Weak supervision
    # ==================================================================

    if parsed.weak_supervision:

        logger.info(
            "Weak supervision enabled."
        )

        unlabeled = load_unlabeled_data()

        generated_weak_df = (
            generate_weak_labels(
                unlabeled,
                use_snorkel=(
                    parsed.use_snorkel
                ),
            )
        )

        WEAK_LABEL_PATH.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        generated_weak_df.to_parquet(
            WEAK_LABEL_PATH,
            index=False,
        )

        logger.info(
            "Weak labels saved to %s",
            WEAK_LABEL_PATH,
        )

        if parsed.weak_only:

            logger.info(
                "weak-only requested; stopping after weak-label generation."
            )

            return

        (
            X_weak,
            y_weak,
            w_weak,
            weak_feature_cols,
            weak_training_df,
        ) = prepare_weak_labeled_data(
            generated_weak_df
        )

        weak_training = (
            X_weak,
            y_weak,
            w_weak,
            weak_feature_cols,
            weak_training_df,
        )

    # ==================================================================
    # Select main training dataframe
    # ==================================================================

    if generated_weak_df is not None:

        df = generated_weak_df.copy()

    else:

        df = load_data(
            conf_threshold=(
                parsed.confidence_threshold
            )
        )

    # ==================================================================
    # Feature engineering
    # ==================================================================

    df = build_training_features_from_clean(
        df
    )

    # ==================================================================
    # Sort and reset indexes BEFORE sequence/index mapping.
    # ==================================================================

    df = df.sort_values(
        [
            "transformer_id",
            "sample_day",
        ]
    ).reset_index(
        drop=True
    )

    # ==================================================================
    # Targets
    # ==================================================================

    if weak_training is None:

        if "fault_type_label" not in df.columns:

            logger.warning(
                "fault_type_label missing; "
                "standard fault classifier will not be trained."
            )

            y_fault = None
            fault_labels = None

        else:

            y_fault, fault_labels = (
                prepare_fault_target(
                    df
                )
            )

    else:

        y_fault = None
        fault_labels = None

    df = apply_severity(df)
    
    y_severity_cls, y_severity_score = (
        prepare_severity_target(
            df
        )
    )

    groups = df[
        "transformer_id"
    ].to_numpy()

    # ==================================================================
    # Select leakage-safe static features.
    # ==================================================================

    feature_cols = select_feature_columns(
        df
    )

    X = dataframe_to_matrix(
        df,
        feature_cols,
    )

    logger.info(
        "Static feature matrix: %d samples x %d features",
        X.shape[0],
        X.shape[1],
    )

    # ==================================================================
    # Group split BEFORE fitting scalers/models.
    # ==================================================================

    train_idx, val_idx = group_split(
        X,
        groups,
    )

    # ==================================================================
    # Temporal sequences
    # ==================================================================

    (
        X_seq,
        groups_seq,
        seq_indices,
        _,
    ) = build_sequences(
        df,
        feature_cols,
        seq_len=SEQ_LEN,
    )

    train_group_set = set(
        groups[
            train_idx
        ].tolist()
    )

    val_group_set = set(
        groups[
            val_idx
        ].tolist()
    )

    seq_train_idx = np.array(
        [
            i
            for i, group in enumerate(
                groups_seq
            )
            if group in train_group_set
        ],
        dtype=np.int64,
    )

    seq_val_idx = np.array(
        [
            i
            for i, group in enumerate(
                groups_seq
            )
            if group in val_group_set
        ],
        dtype=np.int64,
    )

    logger.info(
        "Temporal split: train=%d, validation=%d",
        len(seq_train_idx),
        len(seq_val_idx),
    )

    # ==================================================================
    # Train temporal autoencoder ONLY on training transformers.
    # ==================================================================

    ae_model, ae_scaler = (
        train_autoencoder(
            X_seq,
            seq_train_idx,
            X_seq.shape[2],
            epochs=AE_EPOCHS,
            latent_dim=AE_LATENT_DIM,
        )
    )

    embeddings = extract_embeddings(
        ae_model,
        ae_scaler,
        X_seq,
        X_seq.shape[2],
    )

    # ==================================================================
    # Map embeddings back to dataframe.
    # ==================================================================

    emb_cols = [
        f"emb_{i}"
        for i in range(
            embeddings.shape[1]
        )
    ]

    for col in emb_cols:
        df[col] = np.nan

    for row_idx, embedding in zip(
        seq_indices,
        embeddings,
    ):
        df.loc[
            row_idx,
            emb_cols,
        ] = embedding

    # Replace NaN embeddings only if necessary.
    for col in emb_cols:
        df[col] = pd.to_numeric(
            df[col],
            errors="coerce",
        ).fillna(0.0)

    # Static feature matrix after embeddings.
    feature_cols_with_embeddings = (
        feature_cols
        + emb_cols
    )

    X = dataframe_to_matrix(
        df,
        feature_cols_with_embeddings,
    )

    # ==================================================================
    # Fault classifier
    # ==================================================================

    fault_model = None
    fault_feature_names = None
    fault_labels_for_artifact = (
        FAULT_LABELS
    )

    if weak_training is not None:

        (
            X_weak,
            y_weak,
            w_weak,
            weak_feature_cols,
            weak_training_df,
        ) = weak_training

        weak_groups = (
            weak_training_df[
                "transformer_id"
            ]
            .to_numpy()
        )

        weak_train_idx, weak_val_idx = (
            group_split(
                X_weak,
                weak_groups,
            )
        )

        fault_model = (
            train_student_fault_classifier(
                X_weak.iloc[
                    weak_train_idx
                ],
                y_weak[
                    weak_train_idx
                ],
                w_weak[
                    weak_train_idx
                ],
                X_weak.iloc[
                    weak_val_idx
                ],
                y_weak[
                    weak_val_idx
                ],
                w_weak[
                    weak_val_idx
                ],
                labels=WEAK_GROUPS,
            )
        )

        fault_feature_names = (
            weak_feature_cols
        )

        fault_labels_for_artifact = (
            WEAK_GROUPS
        )

    elif y_fault is not None:

        fault_train_X = X.iloc[
            train_idx
        ]

        fault_val_X = X.iloc[
            val_idx
        ]

        fault_model = train_fault_cls(
            fault_train_X,
            y_fault[
                train_idx
            ],
            fault_val_X,
            y_fault[
                val_idx
            ],
        )

        fault_feature_names = (
            feature_cols_with_embeddings
        )

        fault_labels_for_artifact = (
            FAULT_LABELS
        )

    else:

        logger.warning(
            "No fault targets available; "
            "fault classifier will not be trained."
        )

    # ==================================================================
    # Severity classification/regression
    #
    # NOTE:
    # These models are SURROGATE models of the rule-based IEEE status.
    # The rule-based status remains authoritative.
    #
    # Features do not include:
    #   severity_score
    #   ieee_dga_status
    #   severity_label
    # etc.
    # ==================================================================

    severity_feature_names = (
        feature_cols_with_embeddings
    )

    sev_train_X = X.iloc[
        train_idx
    ]

    sev_val_X = X.iloc[
        val_idx
    ]

    sev_cls_model = train_severity_cls(
        sev_train_X,
        y_severity_cls[
            train_idx
        ],
        sev_val_X,
        y_severity_cls[
            val_idx
        ],
    )

    sev_reg_model = train_severity_reg(
        sev_train_X,
        y_severity_score[
            train_idx
        ],
        sev_val_X,
        y_severity_score[
            val_idx
        ],
    )

    # ==================================================================
    # Temporal supervised severity surrogate
    # ==================================================================

    temp_model, temp_scaler = train_temporal(
        X_seq,
        y_severity_score[
            np.array(
                [
                    df.index.get_loc(
                        idx
                    )
                    for idx in seq_indices
                ]
            )
        ],
        seq_train_idx,
        seq_val_idx,
        X_seq.shape[2],
    )

    # ==================================================================
    # Save autoencoder
    # ==================================================================

    MODEL_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    torch.save(
        ae_model.state_dict(),
        MODEL_DIR
        / "temporal_autoencoder.pt",
    )

    joblib.dump(
        ae_scaler,
        MODEL_DIR
        / "temporal_ae_scaler.joblib",
    )

    joblib.dump(
        {
            "feature_names": feature_cols,
            "embedding_features": emb_cols,
            "seq_len": SEQ_LEN,
            "latent_dim": AE_LATENT_DIM,
            "device": str(DEVICE),
        },
        MODEL_DIR
        / "temporal_autoencoder_meta.joblib",
    )

    # ==================================================================
    # Save fault classifier
    # ==================================================================

    if fault_model is not None:

        joblib.dump(
            {
                "model": fault_model,
                "features": fault_feature_names,
                "labels": fault_labels_for_artifact,
                "target_type": (
                    "weak_group"
                    if fault_labels_for_artifact
                    == WEAK_GROUPS
                    else "fault_detail"
                ),
            },
            MODEL_DIR
            / "fault_unsupervised_model.joblib",
        )

    # ==================================================================
    # Save severity models
    # ==================================================================

    joblib.dump(
        {
            "model": sev_cls_model,
            "features": severity_feature_names,
            "labels": [
                "NORMAL",
                "WATCHLIST",
                "CRITICAL",
            ],
            "target_type": (
                "ieee_c57_104_2019_dga_status"
            ),
            "note": (
                "Surrogate model only. "
                "Rule-based IEEE status remains authoritative."
            ),
        },
        MODEL_DIR
        / "severity_classifier.joblib",
    )

    joblib.dump(
        {
            "model": sev_reg_model,
            "features": severity_feature_names,
            "target_type": (
                "ieee_c57_104_2019_application_score"
            ),
            "note": (
                "Surrogate model only. "
                "The rule-based severity score remains authoritative."
            ),
        },
        MODEL_DIR
        / "severity_regressor.joblib",
    )

    # ==================================================================
    # Save temporal model
    # ==================================================================

    torch.save(
        temp_model.state_dict(),
        MODEL_DIR
        / "temporal_model_state_dict.pt",
    )

    joblib.dump(
        temp_scaler,
        MODEL_DIR
        / "temporal_scaler.joblib",
    )

    joblib.dump(
        {
            "feature_names": feature_cols_with_embeddings,
            "seq_len": SEQ_LEN,
            "target": "severity_score",
            "device": str(DEVICE),
        },
        MODEL_DIR
        / "temporal_model_meta.joblib",
    )

    # ==================================================================
    # Save training metadata
    # ==================================================================

    metadata = {
        "standard": "IEEE C57.104-2019",
        "fluid_type": "MINERAL_OIL",
        "random_state": parsed.seed,
        "device": str(DEVICE),
        "static_feature_count": len(
            feature_cols_with_embeddings
        ),
        "base_feature_count": len(
            feature_cols
        ),
        "embedding_feature_count": len(
            emb_cols
        ),
        "sequence_length": SEQ_LEN,
        "weak_supervision": bool(
            parsed.weak_supervision
        ),
        "weak_confidence_threshold": (
            WEAK_CONFIDENCE_THRESHOLD
        ),
        "train_transformers": int(
            len(
                set(
                    groups[
                        train_idx
                    ].tolist()
                )
            )
        ),
        "validation_transformers": int(
            len(
                set(
                    groups[
                        val_idx
                    ].tolist()
                )
            )
        ),
        "note": (
            "Severity Status is rule-based according to "
            "IEEE C57.104-2019 norms; ML severity models are "
            "surrogate estimators and must not override the rule-based status."
        ),
    }

    joblib.dump(
        metadata,
        MODEL_DIR
        / "training_metadata.joblib",
    )

    logger.info(
        "Training complete. Models saved to %s",
        MODEL_DIR,
    )


if __name__ == "__main__":
    main()