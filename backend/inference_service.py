# inference_service.py
from __future__ import annotations
import logging, shutil, tempfile, threading, time
from pathlib import Path
from typing import Dict, List, Optional
import joblib, lightgbm as lgb, numpy as np, pandas as pd
from clean_dataset import clean_dataset
from consensus import apply_consensus, combine_consensus_and_student
from feature_engineering import build_training_features_from_clean, get_model_feature_columns
from logging_config import init_logging
from ranking import build_transformer_ranking
from severity import apply_severity
from weak_supervision import SNORKEL_AVAILABLE, WEAK_GROUPS, weak_supervision_pipeline
from config import DATASET_DIR, MODEL_DIR, REPORT_DIR, config as cfg

init_logging()
logger = logging.getLogger(__name__)

MODEL_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)
CLEAN_OUTPUT_DIR = DATASET_DIR / "processed"
STUDENT_MODEL_PATH = MODEL_DIR / "fault_classifier.joblib"
STUDENT_MODEL_ALT_PATH = MODEL_DIR / "fault_unsupervised_model.joblib"
ANOMALY_MODEL_PATH = MODEL_DIR / "anomaly_ensemble.joblib"
UNLABELED_PATH = Path(DATASET_DIR) / "processed" / "dga_unlabeled.parquet"
CONSENSUS_CONFIDENCE_THRESHOLD = 60.0
WEAK_CONFIDENCE_THRESHOLD = 0.70


def _build_X(df: pd.DataFrame, feature_cols: List[str]) -> pd.DataFrame:
    X = (df.reindex(columns=feature_cols, fill_value=0.0)
         .apply(pd.to_numeric, errors="coerce")
         .replace([np.inf, -np.inf], np.nan)
         .fillna(0.0))
    return X.astype(np.float32)


def _train_student_model_from_weak_labels(df_weak: pd.DataFrame):
    accepted = df_weak["weak_fault_group"] != "ABSTAIN"
    confidence = pd.to_numeric(df_weak["weak_fault_confidence"], errors="coerce").fillna(0.0)
    accepted &= confidence >= WEAK_CONFIDENCE_THRESHOLD
    df_filtered = df_weak[accepted].copy().reset_index(drop=True)
    if df_filtered.empty:
        raise ValueError("No confident weak labels available.")
    group_to_idx = {group: idx for idx, group in enumerate(WEAK_GROUPS)}
    y = df_filtered["weak_fault_group"].map(group_to_idx).astype(int).to_numpy()
    weights = (pd.to_numeric(df_filtered["weak_fault_confidence"], errors="coerce")
               .fillna(0.0).clip(0.0, 1.0).to_numpy())
    feature_cols = get_model_feature_columns(df_filtered)
    X = _build_X(df_filtered, feature_cols)
    groups = df_filtered["transformer_id"].to_numpy()
    from sklearn.model_selection import GroupShuffleSplit
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=42)
    train_idx, val_idx = next(splitter.split(X, y, groups=groups))
    dtrain = lgb.Dataset(X.iloc[train_idx], label=y[train_idx], weight=weights[train_idx])
    dval = lgb.Dataset(X.iloc[val_idx], label=y[val_idx], weight=weights[val_idx], reference=dtrain)
    params = {
        "objective": "multiclass", "num_class": len(WEAK_GROUPS), "metric": "multi_logloss",
        "learning_rate": 0.03, "num_leaves": 31, "min_data_in_leaf": 20,
        "feature_fraction": 0.85, "bagging_fraction": 0.85, "bagging_freq": 5,
        "verbosity": -1, "seed": 42,
    }
    model = lgb.train(params, dtrain, num_boost_round=600, valid_sets=[dval], valid_names=["validation"],
                      callbacks=[lgb.early_stopping(40, first_metric_only=True, verbose=False), lgb.log_evaluation(False)])
    artifact = {"model": model, "features": feature_cols, "labels": WEAK_GROUPS, "target_type": "weak_group"}
    joblib.dump(artifact, STUDENT_MODEL_PATH)
    return model, feature_cols


def _load_student_model():
    for path in [STUDENT_MODEL_PATH, STUDENT_MODEL_ALT_PATH]:
        if not path.exists():
            continue
        artifact = joblib.load(path)
        if not isinstance(artifact, dict):
            continue
        model = artifact.get("model")
        features = artifact.get("features")
        labels = artifact.get("labels", WEAK_GROUPS)
        if model is None or not features:
            continue
        logger.info("Loaded student model from %s", path)
        return model, list(features), list(labels)
    return None, None, None


def _apply_student_model(df: pd.DataFrame, model, feature_cols: List[str], labels: List[str]) -> pd.DataFrame:
    out = df.copy()
    X = _build_X(out, feature_cols)
    prediction = model.predict(X)
    if prediction.ndim == 2:
        indices = prediction.argmax(axis=1)
        confidence = prediction.max(axis=1)
    else:
        indices = prediction.astype(int)
        confidence = np.full(len(out), np.nan)
    indices = np.clip(indices, 0, len(labels) - 1)
    out["student_fault_label"] = [labels[idx] for idx in indices]
    out["student_fault_confidence"] = confidence
    return out


def _load_or_fit_anomaly(df: pd.DataFrame):
    from anomaly import UnsupervisedEnsemble
    gas_cols = ["h2", "ch4", "c2h6", "c2h4", "c2h2", "co", "co2"]
    X = (df.reindex(columns=gas_cols, fill_value=0.0)
         .apply(pd.to_numeric, errors="coerce")
         .fillna(0.0).to_numpy(dtype=np.float64))
    if ANOMALY_MODEL_PATH.exists():
        try:
            ensemble = joblib.load(ANOMALY_MODEL_PATH)
            logger.info("Loaded anomaly model from %s", ANOMALY_MODEL_PATH)
            scores = ensemble.predict(X)
            return ensemble, scores
        except Exception:
            logger.exception("Stored anomaly model could not be used; fitting a new one.")
    ensemble = UnsupervisedEnsemble()
    ensemble.fit(X, feature_names=gas_cols)
    scores = ensemble.predict(X)
    try:
        joblib.dump(ensemble, ANOMALY_MODEL_PATH)
    except Exception:
        logger.exception("Could not save anomaly ensemble.")
    return ensemble, scores


ROW_EXPORT_FIELDS = [
    "transformer_id", "sample_day", "loc", "name", "ser", "codetx", "mfg",
    "h2", "ch4", "c2h6", "c2h4", "c2h2", "co", "co2",
    "tdcg_raw", "tdcg_recalc", "tdcg", "tdcg_complete",
    "o2", "n2", "o2_n2_ratio", "transformer_age_years", "water", "temp",
    "ieee_dga_status", "ieee_dga_status_label", "ieee_dga_status_reason",
    "ieee_confirmation_required", "ieee_extreme_dga",
    "severity_score", "severity_label_text", "severity_gas_score", "severity_trend_score", "severity_anomaly_score",
    "consensus_fault", "consensus_fault_traditional", "mixed_components", "diagnostic_confidence", "diagnostic_coverage",
    "keygas_fault", "iec_fault", "rogers_fault", "doernenburg_fault", "duval_triangle_fault", "duval_pentagon_p1_fault",
    "duval_pentagon_p2_fault", "fault_p1", "fault_p2", "t_x", "t_y", "p_x", "p_y",
    "iec_r1_c2h2_c2h4", "iec_r2_ch4_h2", "iec_r3_c2h4_c2h6",
    "r1_c2h2_c2h4", "r2_ch4_h2", "r3_c2h4_c2h6",
    "dr_r1_ch4_h2", "dr_r2_c2h2_c2h4", "dr_r3_c2h2_ch4", "dr_r4_c2h6_c2h2",
    "h2_delta1", "ch4_delta1", "c2h2_delta1", "c2h4_delta1", "c2h6_delta1", "co_delta1", "co2_delta1",
    "h2_rate_per_year", "ch4_rate_per_year", "c2h2_rate_per_year", "c2h4_rate_per_year", "tdcg_rate_per_year",
    "rate_points", "rate_span_months", "student_fault_label", "student_fault_confidence",
    "anomaly_percentile", "fleet_priority_score", "fleet_priority_percent", "recommended_action",
]


def _trim_row(row_dict: Dict) -> Dict:
    return {key: row_dict[key] for key in ROW_EXPORT_FIELDS if key in row_dict}


def build_student_traditional_comparison(df: pd.DataFrame) -> pd.DataFrame:
    required = {"transformer_id", "sample_day", "student_fault_label", "consensus_fault_traditional"}
    if not required.issubset(df.columns):
        return pd.DataFrame()
    rows = []
    for transformer_id, group in df.groupby("transformer_id"):
        group = group.sort_values("sample_day")
        agreement = (group["student_fault_label"].astype(str) == group["consensus_fault_traditional"].astype(str))
        latest = group.iloc[-1]
        rows.append({
            "transformer_id": transformer_id,
            "n_samples": int(len(group)),
            "agreement_rate": float(agreement.mean()),
            "disagree_count": int((~agreement).sum()),
            "latest_student_fault": str(latest.get("student_fault_label", "ABSTAIN")),
            "latest_traditional_fault": str(latest.get("consensus_fault_traditional", "ABSTAIN")),
            "latest_severity_score": float(latest.get("severity_score", 0.0)),
            "latest_severity": str(latest.get("severity_label_text", "NORMAL")),
            "latest_sample_day": str(latest.get("sample_day", "")),
        })
    comparison = pd.DataFrame(rows)
    if comparison.empty:
        return comparison
    return comparison.sort_values(["agreement_rate", "latest_severity_score"], ascending=[True, False]).reset_index(drop=True)


def create_payload(df: pd.DataFrame, ranking_df: pd.DataFrame, comparison_df: Optional[pd.DataFrame] = None):
    predictions = []
    rows = []
    ordered = df.sort_values(["transformer_id", "sample_day"], ascending=[True, False])
    for idx, row in ordered.iterrows():
        severity_label = str(row.get("severity_label_text", "NORMAL"))
        severity_score = float(row.get("severity_score", 0.0))
        predictions.append({
            "row_index": int(idx),
            "transformer_id": row["transformer_id"],
            "pred_ensemble": severity_score,
            "severity": cfg.SEVERITY_TO_UI.get(severity_label, severity_label),
            "fault_type": row.get("consensus_fault", "ABSTAIN"),
            "reason": row.get("ieee_dga_status_reason", ""),
            "top_features": [],
        })
        rows.append(_trim_row(row.to_dict()))
    transformer_summary = []
    for _, rank_row in ranking_df.iterrows():
        severity_label = str(rank_row.get("severity_label_text", rank_row.get("ieee_dga_status_label", "NORMAL")))
        transformer_summary.append({
            "rank": int(rank_row["rank"]),
            "transformer_id": rank_row["transformer_id"],
            "latest_sample_day": str(rank_row.get("sample_day", "")),
            "latest_score": float(rank_row.get("severity_score", 0.0)),
            "severity": cfg.SEVERITY_TO_UI.get(severity_label, severity_label),
            "fault_type": rank_row.get("consensus_fault", "ABSTAIN"),
            "priority_score": float(rank_row.get("fleet_priority_percent", rank_row.get("final_score", 0.0))),
            "priority_label": cfg.SEVERITY_TO_UI.get(severity_label, severity_label),
            "recommended_action": rank_row.get("recommended_action", ""),
            "reason": rank_row.get("ieee_dga_status_reason", ""),
            "features": {},
            "loc": (rank_row.get("loc", "") if pd.notna(rank_row.get("loc", np.nan)) else ""),
            "name": (rank_row.get("name", "") if pd.notna(rank_row.get("name", np.nan)) else ""),
            "ranking_breakdown": {
                "fleet_priority_score": float(rank_row.get("fleet_priority_percent", 0.0)),
                "severity_score": float(rank_row.get("severity_score", 0.0)),
                "anomaly_percentile": float(rank_row.get("anomaly_percentile", 0.0)) if pd.notna(rank_row.get("anomaly_percentile", np.nan)) else None,
                "trend_slope": (float(rank_row["fleet_severity_trend_slope"]) if pd.notna(rank_row.get("fleet_severity_trend_slope", np.nan)) else None),
            },
        })
    timeseries = {}
    for transformer_id, group in ordered.groupby("transformer_id"):
        group = group.sort_values("sample_day")
        series = []
        for _, row in group.iterrows():
            series.append({
                "Sample Day": str(row["sample_day"]),
                "H2": float(row.get("h2", 0.0)),
                "C2H2": float(row.get("c2h2", 0.0)),
                "TDCG": float(row.get("tdcg", 0.0)),
                "pred_ensemble": float(row.get("severity_score", 0.0)),
                "fault_type": row.get("consensus_fault", "ABSTAIN"),
                "severity": row.get("severity_label_text", "NORMAL"),
            })
        timeseries[str(transformer_id)] = series
    dataset_summary = {"total_transformers": int(df["transformer_id"].nunique()), "total_rows": int(len(df))}
    return {
        "predictions": predictions,
        "rows": rows,
        "preview_rows": rows[:20],
        "transformer_summary": transformer_summary,
        "transformer_timeseries": timeseries,
        "dataset_summary": dataset_summary,
        "student_traditional_comparison": [] if comparison_df is None else comparison_df.to_dict(orient="records"),
        "chat_context_payload": {"transformer_summary": transformer_summary, "dataset_summary": dataset_summary},
    }


def _run_experiments_background():
    try:
        from experiment import run_full_experiments
        run_full_experiments()
    except Exception:
        logger.exception("Background experiments failed.")


def process_dataframe(uploaded_df: pd.DataFrame):
    start_time = time.time()
    tmp_dir = Path(tempfile.mkdtemp())
    CLEAN_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    try:
        excel_path = tmp_dir / "input.xlsx"
        uploaded_df.to_excel(excel_path, index=False, engine="openpyxl")
        logger.info("Step 1/15: Cleaning uploaded dataset...")
        df_clean, _ = clean_dataset(input_file=excel_path, output_dir=CLEAN_OUTPUT_DIR)
        logger.info("Step 2/15: Building features...")
        df_features = build_training_features_from_clean(df_clean)
        logger.info("Step 3/15: Applying traditional DGA consensus...")
        df_labeled = apply_consensus(df_features)
        logger.info("Step 4/15: Running weak supervision...")
        df_weak, label_model, groups = weak_supervision_pipeline(
            df_labeled, use_snorkel=SNORKEL_AVAILABLE, confidence_threshold=WEAK_CONFIDENCE_THRESHOLD)
        logger.info("Step 5/15: Loading student model...")
        student_model, student_features, student_labels = _load_student_model()
        if student_model is None:
            logger.warning("No saved student model. Training bootstrap student model from current weak labels.")
            student_model, student_features = _train_student_model_from_weak_labels(df_weak)
            student_labels = WEAK_GROUPS
        logger.info("Step 6/15: Applying student model...")
        df_labeled = _apply_student_model(df_labeled, student_model, student_features, student_labels)
        logger.info("Step 7/15: Preserving traditional fault labels...")
        df_labeled["consensus_fault_traditional"] = df_labeled["consensus_fault"]
        logger.info("Step 8/15: Loading/fitting anomaly model...")
        anomaly_model, anomaly_scores = _load_or_fit_anomaly(df_labeled)
        df_labeled["anomaly_percentile"] = anomaly_scores
        logger.info("Step 9/15: Combining consensus and student outputs...")
        df_labeled = combine_consensus_and_student(
            df_labeled, student_fault_col="student_fault_label",
            student_conf_col="student_fault_confidence",
            consensus_conf_threshold=CONSENSUS_CONFIDENCE_THRESHOLD)
        logger.info("Step 10/15: Computing IEEE severity/status...")
        df_labeled = apply_severity(df_labeled)
        logger.info("Step 11/15: Building transformer ranking...")
        ranking_df = build_transformer_ranking(df_labeled)
        logger.info("Step 12/15: Creating student-vs-traditional comparison...")
        comparison_df = build_student_traditional_comparison(df_labeled)
        if not comparison_df.empty:
            comparison_path = REPORT_DIR / "student_vs_traditional_by_transformer.csv"
            comparison_df.to_csv(comparison_path, index=False)
            logger.info("Comparison report saved to %s", comparison_path)
        logger.info("Step 13/15: Saving processed dataset...")
        UNLABELED_PATH.parent.mkdir(parents=True, exist_ok=True)
        df_labeled.to_parquet(UNLABELED_PATH, index=False)
        logger.info("Processed dataset saved to %s", UNLABELED_PATH)
        logger.info("Step 14/15: Creating payload...")
        payload = create_payload(df_labeled, ranking_df, comparison_df)
        from data_store import save_payload_to_db
        save_payload_to_db(payload)
        logger.info("Payload saved to SQLite.")
        logger.info("Step 15/15: Starting background experiments...")
        threading.Thread(target=_run_experiments_background, daemon=True).start()
        logger.info("DGA processing complete in %.1fs.", time.time() - start_time)
        return payload
    except Exception:
        logger.exception("Fatal error in process_dataframe.")
        raise
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)