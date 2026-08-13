# inference_service.py
import tempfile, shutil, json, time
from pathlib import Path
import pandas as pd
import numpy as np
import logging
import joblib
import lightgbm as lgb
from sklearn.model_selection import GroupShuffleSplit
import threading

from clean_dataset import clean_dataset
from feature_engineering import (
    add_nb_event_features, preprocess_types, sort_and_deduplicate, filter_rows_for_model,
    add_missingness_flags, impute_optional_context_by_transformer,
    add_tdcg, add_rating_features, add_metadata_features,
    add_ratio_features, add_duval_input_features,
    add_calendar_and_sequence_features, add_lag_delta_rate_features,
    add_rolling_features, add_ewm_features, add_cross_gas_trend_features,
    add_quality_flags
)
from consensus import apply_consensus, combine_consensus_and_student
from severity import apply_severity
from ranking import build_transformer_ranking
from weak_supervision import (
    weak_supervision_pipeline, WEAK_GROUPS, create_student_training_targets,
    ABSTAIN, build_label_matrix, fit_label_model, attach_probabilistic_labels,
    SNORKEL_AVAILABLE
)
from experiment import run_full_experiments

# Set detailed logging – use DEBUG for maximum verbosity
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s [%(levelname)s] %(name)s: %(message)s', datefmt='%H:%M:%S')
logger = logging.getLogger(__name__)

from config import config as cfg
from config import DATASET_DIR, DATABASE_DIR, MODEL_DIR, REPORT_DIR

CORE_GASES = ["h2", "ch4", "c2h6", "c2h4", "c2h2", "co", "co2"]
OPTIONAL_NUMERIC = ["o2", "n2", "water", "temp"]

ID_LIKE_COLS = [
    "transformer_id", "sample_day", "loc", "name", "ser", "codetx", "mfg",
    "sample_year", "sample_month", "sample_quarter", "sample_dayofyear",
    "sample_weekday", "record_idx", "tested_day", "tdcg_source",
    "severity_label", "severity_gas_score", "severity_trend_score",
    "severity_fault_score", "severity_aging_score", "severity_score",
    "consensus_fault", "mixed_components", "diagnostic_confidence", "diagnostic_votes",
    "student_fault_label", "student_fault_confidence",
    "consensus_fault_traditional",
]

# Scale factor for severity_score (which is 0-3) to 0-100 for frontend compatibility
SEVERITY_SCALE = 100.0 / 3.0

# ---------- Helpers ----------
def _build_X(df, feature_cols):
    logger.debug(f"Building feature matrix with {len(feature_cols)} columns")
    return (
        df.reindex(columns=feature_cols, fill_value=0)
        .apply(pd.to_numeric, errors="coerce")
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0)
    )

def _train_student_model_from_weak_labels(df_weak, use_snorkel):
    df_filtered = df_weak[df_weak["weak_fault_group"] != "ABSTAIN"].copy()
    n_abstain = len(df_weak) - len(df_filtered)
    if n_abstain > 0:
        logger.info(f"Filtered out {n_abstain} ABSTAIN samples for student training")
    if len(df_filtered) == 0:
        raise ValueError("No weak label samples available (all ABSTAIN).")

    y, weights = create_student_training_targets(df_filtered)
    groups_weak = df_filtered["transformer_id"].values

    feature_cols = [
        c for c in df_filtered.columns
        if c not in ID_LIKE_COLS
        and not c.startswith("weak_prob_")
        and not c.startswith("target_")
        and df_filtered[c].dtype in ("float64", "int64", "int32", "float32")
    ]
    logger.info(f"Student feature count: {len(feature_cols)}")
    X = _build_X(df_filtered, feature_cols)

    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    try:
        train_idx, val_idx = next(gss.split(X, y, groups=groups_weak))
    except ValueError:
        from sklearn.model_selection import train_test_split
        logger.warning("Not enough groups for GroupShuffleSplit, falling back to random split")
        train_idx, val_idx = train_test_split(
            np.arange(len(X)), test_size=0.2, random_state=42
        )

    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y[train_idx], y[val_idx]
    w_train, w_val = weights[train_idx], weights[val_idx]

    dtrain = lgb.Dataset(X_train, label=y_train, weight=w_train)
    dval = lgb.Dataset(X_val, label=y_val, weight=w_val, reference=dtrain)

    params = {
        "objective": "multiclass",
        "num_class": len(WEAK_GROUPS),
        "metric": "multi_logloss",
        "learning_rate": 0.05,
        "num_leaves": 31,
        "min_data_in_leaf": 10,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 5,
        "verbosity": 0,
        "seed": 42,
    }
    logger.info("Starting LightGBM training for student model...")
    model = lgb.train(
        params,
        dtrain,
        num_boost_round=200,
        valid_sets=[dtrain, dval],
        valid_names=["train", "val"],
        callbacks=[lgb.early_stopping(30), lgb.log_evaluation(10)],
    )
    return model, feature_cols

def _apply_student_model(df, model, feature_cols):
    X = _build_X(df, feature_cols)
    raw_pred = model.predict(X)
    if raw_pred.ndim == 2:
        fault_idx = raw_pred.argmax(axis=1)
        fault_conf = raw_pred.max(axis=1)
    else:
        fault_idx = raw_pred.astype(int)
        fault_conf = np.full(len(fault_idx), np.nan)

    df["student_fault_label"] = [WEAK_GROUPS[i] for i in fault_idx]
    df["student_fault_confidence"] = fault_conf
    return df

# ---------- Feature Engineering ----------
def build_features_from_clean(df: pd.DataFrame) -> pd.DataFrame:
    logger.info("=== Starting Feature Engineering ===")
    start = time.time()
    df = preprocess_types(df)
    df = sort_and_deduplicate(df)
    df = filter_rows_for_model(df, max_missing_core=3)
    df = add_nb_event_features(df)
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

    elapsed = time.time() - start
    logger.info(f"Feature engineering completed in {elapsed:.1f}s: {df.shape[0]} rows × {df.shape[1]} columns")
    return df

# ---------- Payload Helpers ----------
ROW_EXPORT_FIELDS = [
    "transformer_id", "sample_day", "loc", "name", "ser", "codetx", "mfg",
    "h2", "ch4", "c2h6", "c2h4", "c2h2", "co", "co2", "tdcg", "o2", "n2", "water", "temp",
    "severity_score", "severity_label", "consensus_fault", "mixed_components",
    "diagnostic_confidence", "diagnostic_votes",
    "keygas_fault", "iec_fault", "rogers_fault", "doernenburg_fault",
    "duval_triangle_fault", "fault_p1", "fault_p2", "duval_pentagon_fault",
    "t_x", "t_y", "p_x", "p_y",
    "iec_r1_c2h2_c2h4", "iec_r2_ch4_h2", "iec_r3_c2h4_c2h6",
    "r1_ch4_h2", "r2_c2h2_c2h4", "r3_c2h4_c2h6",
    "dr_r1_ch4_h2", "dr_r2_c2h2_c2h4", "dr_r3_c2h2_ch4", "dr_r4_c2h6_c2h2",
    "ratio_co2_co",
    "h2_rate_per_day", "c2h2_rate_per_day", "tdcg_rate_per_day",
    "severity_gas_score", "severity_trend_score", "severity_anomaly_score",
    "student_fault_label", "student_fault_confidence",
    "consensus_fault_traditional",
]

def _trim_row(row_dict):
    return {k: row_dict[k] for k in ROW_EXPORT_FIELDS if k in row_dict}

def build_student_traditional_comparison(df: pd.DataFrame) -> pd.DataFrame:
    required = {"transformer_id", "sample_day", "severity_score", "consensus_fault"}
    if not required.issubset(df.columns):
        return pd.DataFrame()
    if "student_fault_label" not in df.columns or "consensus_fault_traditional" not in df.columns:
        return pd.DataFrame()

    rows = []
    for tid, grp in df.groupby("transformer_id"):
        g = grp.sort_values("sample_day")
        n_samples = int(len(g))
        if n_samples == 0:
            continue
        agree_mask = (g["student_fault_label"].astype(str) == g["consensus_fault_traditional"].astype(str))
        agreement_rate = float(agree_mask.mean()) if n_samples > 0 else 0.0
        disagree_count = int((~agree_mask).sum())
        latest = g.iloc[-1]
        rows.append({
            "transformer_id": tid,
            "n_samples": n_samples,
            "agreement_rate": agreement_rate,
            "disagree_count": disagree_count,
            "latest_student_fault": str(latest.get("student_fault_label", "ABSTAIN")),
            "latest_traditional_fault": str(latest.get("consensus_fault_traditional", "ABSTAIN")),
            "latest_severity_score": float(latest.get("severity_score", 0.0)),
            "latest_severity_label": str(latest.get("severity_label", "ABSTAIN")),
            "latest_sample_day": str(latest.get("sample_day")),
        })

    comp_df = pd.DataFrame(rows)
    if comp_df.empty:
        logger.info("No student/traditional comparison data available")
        return comp_df
    comp_df = comp_df.sort_values(
        by=["agreement_rate", "latest_severity_score"],
        ascending=[True, False],
    ).reset_index(drop=True)
    logger.info(f"Student/traditional comparison built: {len(comp_df)} transformers")
    return comp_df

def create_payload(df, ranking_df, comparison_df: pd.DataFrame | None = None):
    logger.info("Creating API payload...")
    predictions = []
    rows = []
    df = df.sort_values(["transformer_id", "sample_day"], ascending=[True, False])
    for idx, row in df.iterrows():
        ui_severity = cfg.SEVERITY_TO_UI.get(row["severity_label"], row["severity_label"])
        severity_scaled = float(row["severity_score"]) * SEVERITY_SCALE
        pred = {
            "row_index": idx,
            "transformer_id": row["transformer_id"],
            "pred_ensemble": severity_scaled,
            "severity": ui_severity,
            "fault_type": row.get("consensus_fault", "ABSTAIN"),
            "reason": f"Severity score = {severity_scaled:.2f}",
            "top_features": []
        }
        predictions.append(pred)
        row_dict = _trim_row(row.to_dict())
        if "severity_score" in row_dict:
            row_dict["severity_score"] = float(row_dict["severity_score"]) * SEVERITY_SCALE
        rows.append(row_dict)

    transformer_summary = []
    for _, rrow in ranking_df.iterrows():
        ui_severity = cfg.SEVERITY_TO_UI.get(rrow["severity_label"], rrow["severity_label"])
        severity_scaled = float(rrow["severity_score"]) * SEVERITY_SCALE
        ts = {
            "rank": int(rrow["rank"]),
            "transformer_id": rrow["transformer_id"],
            "latest_sample_day": str(rrow["sample_day"]) if pd.notna(rrow["sample_day"]) else "",
            "latest_score": severity_scaled,
            "severity": ui_severity,
            "fault_type": rrow.get("consensus_fault", "ABSTAIN"),
            "priority_score": float(rrow["final_score"]),
            "priority_label": ui_severity,
            "recommended_action": rrow.get("recommended_action", ""),
            "reason": "",
            "features": {},
            "loc": rrow.get("loc") if pd.notna(rrow.get("loc")) else "",
            "name": rrow.get("name") if pd.notna(rrow.get("name")) else "",
            "ranking_breakdown": {
                "final_score": float(rrow["final_score"]),
                "severity_score": severity_scaled,
                "trend_slope": float(rrow["trend_slope"]) if pd.notna(rrow.get("trend_slope")) else None,
            },
        }
        transformer_summary.append(ts)

    timeseries = {}
    for tid, grp in df.groupby("transformer_id"):
        grp = grp.sort_values("sample_day")
        series = []
        for _, trow in grp.iterrows():
            severity_scaled = float(trow["severity_score"]) * SEVERITY_SCALE
            series.append({
                "Sample Day": str(trow["sample_day"]),
                "H2": float(trow.get("h2", 0)),
                "C2H2": float(trow.get("c2h2", 0)),
                "TCG": float(trow.get("tdcg", 0)),
                "pred_ensemble": severity_scaled,
                "fault_type": trow.get("consensus_fault", "ABSTAIN"),
                "severity": trow["severity_label"],
            })
        timeseries[str(tid)] = series

    dataset_summary = {
        "total_transformers": df["transformer_id"].nunique(),
        "total_rows": len(df),
    }

    payload = {
        "predictions": predictions,
        "rows": rows,
        "preview_rows": rows[:20],
        "transformer_summary": transformer_summary,
        "transformer_timeseries": timeseries,
        "dataset_summary": dataset_summary,
        "student_traditional_comparison": [] if comparison_df is None else comparison_df.to_dict(orient="records"),
        "chat_context_payload": {
            "transformer_summary": transformer_summary,
            "dataset_summary": dataset_summary
        }
    }
    logger.info("Payload creation completed.")
    return payload

# ---------- Auto Experiment Runner ----------
def _run_experiments_background():
    logger.info("Background experiments thread started.")
    try:
        run_full_experiments()
        logger.info("All experiments and report data generated successfully.")
    except Exception as e:
        logger.exception("Background experiment pipeline failed")

# ---------- Main Processing ----------
def process_dataframe(uploaded_df):
    total_start = time.time()
    tmp_dir = tempfile.mkdtemp()
    try:
        # 1. Clean & accumulate
        excel_path = Path(tmp_dir) / "input.xlsx"
        uploaded_df.to_excel(excel_path, index=False, engine='openpyxl')
        logger.info("Step 1/12: Cleaning dataset...")
        t0 = time.time()
        df_clean, _ = clean_dataset(input_file=excel_path, output_dir=Path(tmp_dir))
        logger.info(f"Cleaning completed in {time.time()-t0:.1f}s, raw rows: {len(df_clean)}")

        from dataset_accumulator import merge_with_accumulated
        t0 = time.time()
        df_clean = merge_with_accumulated(df_clean)
        logger.info(f"Merge with accumulated history done in {time.time()-t0:.1f}s, total rows: {len(df_clean)}")

        # 2. Feature engineering
        logger.info("Step 2/12: Feature engineering...")
        t0 = time.time()
        df_features = build_features_from_clean(df_clean)
        logger.info(f"Feature engineering took {time.time()-t0:.1f}s")

        # 3. Traditional consensus
        logger.info("Step 3/12: Running traditional DGA consensus...")
        t0 = time.time()
        df_labeled = apply_consensus(df_features)
        logger.info(f"Consensus completed in {time.time()-t0:.1f}s")

        # 4. Weak supervision + student model training
        logger.info("Step 4/12: Weak supervision (Snorkel)...")
        use_snorkel = SNORKEL_AVAILABLE
        t0 = time.time()
        df_weak, label_model, groups = weak_supervision_pipeline(df_labeled, use_snorkel=use_snorkel)
        logger.info(f"Weak supervision done in {time.time()-t0:.1f}s")
        logger.info("Step 5/12: Training student model from weak labels...")
        t0 = time.time()
        student_model, feature_cols = _train_student_model_from_weak_labels(df_weak, use_snorkel)
        logger.info(f"Student model trained in {time.time()-t0:.1f}s")

        # 5. Apply student model
        logger.info("Step 6/12: Applying student model predictions...")
        t0 = time.time()
        df_labeled = _apply_student_model(df_labeled, student_model, feature_cols)
        joblib.dump({
            "model": student_model,
            "features": feature_cols,
            "labels": WEAK_GROUPS,
            "target_type": "weak_group",
        }, MODEL_DIR / "fault_classifier.joblib")
        logger.info("Student model saved.")

        # 6. Anomaly ensemble
        logger.info("Step 7/12: Fitting unsupervised anomaly ensemble...")
        from anomaly import UnsupervisedEnsemble
        gas_cols = ['h2','ch4','c2h6','c2h4','c2h2','co','co2']
        X_anomaly = df_labeled[gas_cols].fillna(0).values
        ensemble = UnsupervisedEnsemble()
        ensemble.fit(X_anomaly)
        df_labeled['anomaly_percentile'] = ensemble.predict(X_anomaly)
        logger.info("Anomaly percentile scores added.")

        # 7. Combine consensus with student (instead of hard swap)
        if "student_fault_label" in df_labeled.columns:
            df_labeled["consensus_fault_traditional"] = df_labeled["consensus_fault"]
            df_labeled = combine_consensus_and_student(
                df_labeled,
                student_fault_col="student_fault_label",
                student_conf_col="student_fault_confidence",
                consensus_conf_threshold=60.0,
            )
            logger.info("Combined consensus fault with student model output.")

        # 8. Severity & Ranking
        logger.info("Step 8/12: Calculating severity scores...")
        t0 = time.time()
        df_labeled = apply_severity(df_labeled)
        logger.info(f"Severity calculation took {time.time()-t0:.1f}s")
        logger.info("Step 9/12: Building transformer ranking...")
        t0 = time.time()
        ranking_df = build_transformer_ranking(df_labeled)
        logger.info(f"Ranking built in {time.time()-t0:.1f}s, top 5:")
        for _, row in ranking_df.head(5).iterrows():
            logger.debug(f"  Rank {row['rank']}: {row['transformer_id']} score={row['final_score']:.2f}")

        # 9. Student vs Traditional comparison
        logger.info("Step 10/12: Creating student/traditional comparison...")
        comparison_df = build_student_traditional_comparison(df_labeled)
        if not comparison_df.empty:
            comparison_path = REPORT_DIR / "student_vs_traditional_by_transformer.csv"
            comparison_df.to_csv(comparison_path, index=False)
            logger.info(f"Comparison saved to {comparison_path}")

        # 10. Save unlabeled dataset for experiments
        logger.info("Step 11/12: Saving processed dataset for experiments...")
        unlabeled_path = Path(DATASET_DIR) / "processed" / "dga_unlabeled.parquet"
        unlabeled_path.parent.mkdir(parents=True, exist_ok=True)
        df_labeled.to_parquet(unlabeled_path)
        logger.info(f"Saved dga_unlabeled.parquet ({len(df_labeled)} rows)")

        # 11. Create payload
        logger.info("Step 12/12: Creating response payload...")
        payload = create_payload(df_labeled, ranking_df, comparison_df=comparison_df)
        from data_store import save_payload_to_db
        save_payload_to_db(payload)
        logger.info("Payload saved to database.")

        # 12. AUTO-RUN EXPERIMENTS IN BACKGROUND
        threading.Thread(target=_run_experiments_background, daemon=True).start()
        logger.info("Background experiments triggered.")

        total_elapsed = time.time() - total_start
        logger.info(f"Total processing time (main pipeline): {total_elapsed:.1f}s")
        return payload

    except Exception as e:
        logger.exception("Fatal error in process_dataframe")
        raise e
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)