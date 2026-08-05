# inference_service.py
import tempfile, shutil, json, time
from pathlib import Path
import pandas as pd
import numpy as np
import logging
import joblib
import lightgbm as lgb
from sklearn.model_selection import GroupShuffleSplit

from clean_dataset import clean_dataset
from feature_engineering import (
    preprocess_types, sort_and_deduplicate, filter_rows_for_model,
    add_missingness_flags, impute_optional_context_by_transformer,
    add_tdcg, add_rating_features, add_metadata_features,
    add_ratio_features, add_duval_input_features,
    add_calendar_and_sequence_features, add_lag_delta_rate_features,
    add_rolling_features, add_ewm_features, add_cross_gas_trend_features,
    add_quality_flags
)
from consensus import apply_consensus
from severity import apply_severity
from ranking import build_transformer_ranking
from weak_supervision import (
    weak_supervision_pipeline, WEAK_GROUPS, create_student_training_targets,
    ABSTAIN, build_label_matrix, fit_label_model, attach_probabilistic_labels,
    SNORKEL_AVAILABLE
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from config import config as cfg
from config import FAULT_LABELS, SEVERITY_LABELS, BACKEND_DATA_DIR, BACKEND_ROOT

MODEL_DIR = Path(__file__).resolve().parent / "models"
REPORT_DIR = Path(__file__).resolve().parent.parent / "reports"
REPORT_DIR.mkdir(parents=True, exist_ok=True)

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


def _build_X(df, feature_cols):
    return (
        df.reindex(columns=feature_cols, fill_value=0)
        .apply(pd.to_numeric, errors="coerce")
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0)
    )


def _train_student_model_from_weak_labels(df_weak, use_snorkel):
    """Huấn luyện LightGBM classifier từ weak labels và trả về model + danh sách features."""
    # Chỉ lấy các dòng không bị ABSTAIN
    df_filtered = df_weak[df_weak["weak_fault_group"] != "ABSTAIN"].copy()
    if len(df_filtered) == 0:
        raise ValueError("Không có mẫu nào có weak label khác ABSTAIN.")

    y, weights = create_student_training_targets(df_filtered)
    groups_weak = df_filtered["transformer_id"].values

    # Chọn feature columns (loại bỏ các cột không cần thiết)
    feature_cols = [
        c for c in df_filtered.columns
        if c not in ID_LIKE_COLS
        and not c.startswith("weak_prob_")
        and not c.startswith("target_")
        and df_filtered[c].dtype in ("float64", "int64", "int32", "float32")
    ]
    X = _build_X(df_filtered, feature_cols)

    # Chia train/val để early stopping (dùng GroupShuffleSplit)
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    try:
        train_idx, val_idx = next(gss.split(X, y, groups=groups_weak))
    except ValueError:
        # Nếu không đủ nhóm, dùng split ngẫu nhiên
        from sklearn.model_selection import train_test_split
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
        "verbosity": -1,
        "seed": 42,
    }
    callbacks = [lgb.early_stopping(30), lgb.log_evaluation(10)]
    model = lgb.train(
        params,
        dtrain,
        num_boost_round=200,
        valid_sets=[dtrain, dval],
        valid_names=["train", "val"],
        callbacks=callbacks,
    )
    return model, feature_cols


def _apply_student_model(df, model, feature_cols):
    """Dự đoán fault label cho toàn bộ DataFrame sử dụng model đã huấn luyện."""
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


def build_features_from_clean(df: pd.DataFrame) -> pd.DataFrame:
    logger.info("=== Bắt đầu feature engineering ===")
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

    # Debug: số lượng cột và vài dòng đầu
    logger.info(f"Feature engineering hoàn tất: {df.shape} - {len(df.columns)} cột")
    # Detailed debug summary — use logger.debug so it's captured in logs instead of stdout
    try:
        first_row = df.iloc[0]
        debug_lines = ["=== FEATURE ENGINEERING SUMMARY ===", f"Shape: {df.shape}", "First row keys (các cột chứa 'ratio' hoặc 'rate'):"]
        for k in first_row.index:
            if 'ratio' in k or 'rate' in k:
                debug_lines.append(f"  {k}: {first_row[k]}")
        debug_lines.append("===================================\n")
        for ln in debug_lines:
            logger.debug(ln)
    except Exception:
        logger.exception("Failed to produce feature engineering debug summary")
    return df

# Feature engineering produces ~368 columns (rolling means, EWM, lag deltas,
# quality flags, etc.) used internally by consensus/severity scoring, but the
# dashboard only ever reads this subset. Sending every column made the
# /predict response ~46MB for a 4.5k-row dataset — most of it unused bytes
# the browser still has to transfer and JSON.parse. Trimming to just what
# frontend/src/types/dga.ts's DgaRow reads cuts that dramatically without
# losing anything the UI shows.
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
    "severity_gas_score", "severity_trend_score", "severity_aging_score", "severity_fault_score",
    "student_fault_label", "student_fault_confidence",
    "student_severity_label", "student_severity_confidence", "student_severity_score",
    "consensus_fault_traditional",
]


def _trim_row(row_dict):
    return {k: row_dict[k] for k in ROW_EXPORT_FIELDS if k in row_dict}


def build_student_traditional_comparison(df: pd.DataFrame) -> pd.DataFrame:
    """Build per-transformer comparison between student and traditional diagnostics."""
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
        rows.append(
            {
                "transformer_id": tid,
                "n_samples": n_samples,
                "agreement_rate": agreement_rate,
                "disagree_count": disagree_count,
                "latest_student_fault": str(latest.get("student_fault_label", "ABSTAIN")),
                "latest_traditional_fault": str(latest.get("consensus_fault_traditional", "ABSTAIN")),
                "latest_severity_score": float(latest.get("severity_score", 0.0)),
                "latest_severity_label": str(latest.get("severity_label", "ABSTAIN")),
                "latest_sample_day": str(latest.get("sample_day")),
            }
        )

    comp_df = pd.DataFrame(rows)
    if comp_df.empty:
        return comp_df
    comp_df = comp_df.sort_values(
        by=["agreement_rate", "latest_severity_score"],
        ascending=[True, False],
    ).reset_index(drop=True)
    return comp_df


def create_payload(df, ranking_df, comparison_df: pd.DataFrame | None = None):
    logger.info("Tạo payload...")
    predictions = []
    rows = []
    df = df.sort_values(["transformer_id", "sample_day"], ascending=[True, False])
    for idx, row in df.iterrows():
        ui_severity = cfg.SEVERITY_TO_UI.get(row["severity_label"], row["severity_label"])
        pred = {
            "row_index": idx,
            "transformer_id": row["transformer_id"],
            "pred_ensemble": float(row["severity_score"] / 20.0),
            "severity": ui_severity,
            "fault_type": row.get("consensus_fault", "ABSTAIN"),
            "reason": f"Severity score = {row['severity_score']:.2f}",
            "top_features": []
        }
        predictions.append(pred)
        rows.append(_trim_row(row.to_dict()))

    rw = cfg.RANKING_WEIGHTS
    transformer_summary = []
    for _, rrow in ranking_df.iterrows():
        ui_severity = cfg.SEVERITY_TO_UI.get(rrow["severity_label"], rrow["severity_label"])
        historical_severity = float(rrow.get("historical_severity", 0) or 0)
        n_critical_past = int(rrow.get("n_critical_past", 0) or 0)
        days_since_last_critical = rrow.get("days_since_last_critical")
        fault_persistence = float(rrow.get("fault_persistence", 0) or 0)
        diagnostic_confidence = float(rrow.get("diagnostic_confidence", 0) or 0)
        trend_bonus_raw = {"worsening": 2.0, "stable": 0.0, "improving": -1.0}.get(rrow["trend"], 0.0)
        crit_bonus_raw = min(n_critical_past, 3)
        conf_penalty = 1.0 if diagnostic_confidence < 50 else 0.0
        ts = {
            "rank": int(rrow["rank"]),
            "transformer_id": rrow["transformer_id"],
            "latest_sample_day": str(rrow["sample_day"]) if pd.notna(rrow["sample_day"]) else "",
            "latest_score": float(rrow["severity_score"]),
            "severity": ui_severity,
            "fault_type": rrow["consensus_fault"],
            "trend": rrow["trend"],
            "priority_score": float(rrow["final_score"]),
            "priority_label": ui_severity,
            "recommended_action": rrow["recommended_action"],
            "reason": "",
            "features": {},
            "loc": rrow.get("loc") if pd.notna(rrow.get("loc")) else "",
            "name": rrow.get("name") if pd.notna(rrow.get("name")) else "",
            # Fleet-ranking breakdown — distinct from the per-record severity
            # score breakdown: this explains WHY this transformer ranks where
            # it does across the whole fleet (config.py RANKING_WEIGHTS).
            "ranking_breakdown": {
                "weights": rw,
                "current_severity": float(rrow["severity_score"]),
                "current_contribution": rw["current"] * float(rrow["severity_score"]),
                "historical_severity": historical_severity,
                "historical_contribution": rw["history"] * historical_severity,
                "trend_bonus": trend_bonus_raw,
                "trend_contribution": rw["trend"] * trend_bonus_raw,
                "critical_history_count": n_critical_past,
                "critical_history_contribution": rw["critical_history"] * crit_bonus_raw,
                "diagnostic_confidence": diagnostic_confidence,
                "confidence_contribution": rw["confidence"] * (1.0 - conf_penalty),
                "persistence_bonus_factor": cfg.PERSISTENCE_BONUS_FACTOR,
                "fault_persistence": fault_persistence,
                "days_since_last_critical": (
                    None if days_since_last_critical is None or pd.isna(days_since_last_critical)
                    else float(days_since_last_critical)
                ),
            },
        }
        transformer_summary.append(ts)

    timeseries = {}
    for tid, grp in df.groupby("transformer_id"):
        grp = grp.sort_values("sample_day")
        series = []
        for _, trow in grp.iterrows():
            series.append({
                "Sample Day": str(trow["sample_day"]),
                "H2": float(trow.get("h2", 0)),
                "C2H2": float(trow.get("c2h2", 0)),
                "TCG": float(trow.get("tdcg", 0)),
                "pred_ensemble": float(trow["severity_score"] / 20.0),
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
    logger.info("Payload hoàn tất.")
    return payload

def process_dataframe(uploaded_df):
    tmp_dir = tempfile.mkdtemp()
    try:
        # 1. Clean & accumulate
        excel_path = Path(tmp_dir) / "input.xlsx"
        uploaded_df.to_excel(excel_path, index=False, engine='openpyxl')
        logger.info("Bắt đầu clean dataset...")
        df_clean, _ = clean_dataset(input_file=excel_path, output_dir=Path(tmp_dir))
        logger.info(f"Clean xong: {df_clean.shape}")

        from dataset_accumulator import merge_with_accumulated
        df_clean = merge_with_accumulated(df_clean)
        logger.info(f"Sau khi gộp: {df_clean.shape}")

        # 2. Feature engineering
        df_features = build_features_from_clean(df_clean)

        # 3. Consensus truyền thống
        logger.info("Chạy consensus DGA...")
        df_labeled = apply_consensus(df_features)

        # 4. Weak supervision + huấn luyện student model tự động
        logger.info("Chạy weak supervision (Snorkel) để tạo probabilistic labels...")
        use_snorkel = SNORKEL_AVAILABLE  # dùng Snorkel nếu có
        df_weak, label_model, groups = weak_supervision_pipeline(
            df_labeled, use_snorkel=use_snorkel
        )
        logger.info("Weak supervision hoàn tất. Đang huấn luyện student model...")
        start_train = time.time()
        student_model, feature_cols = _train_student_model_from_weak_labels(df_weak, use_snorkel)
        logger.info(f"Huấn luyện student model xong trong {time.time() - start_train:.1f}s")

        # 5. Áp dụng student model lên toàn bộ dataset
        df_labeled = _apply_student_model(df_labeled, student_model, feature_cols)

        # Lưu student model để dùng cho các lần sau (có thể ghi đè)
        joblib.dump({
            "model": student_model,
            "features": feature_cols,
            "labels": WEAK_GROUPS,
            "target_type": "weak_group",
        }, MODEL_DIR / "fault_classifier.joblib")
        logger.info("Đã lưu student model mới vào fault_classifier.joblib")

        # 6. Dùng student fault label cho severity scoring
        if "student_fault_label" in df_labeled.columns:
            df_labeled["consensus_fault_traditional"] = df_labeled["consensus_fault"]
            df_labeled["consensus_fault"] = df_labeled["student_fault_label"]
            logger.info("Sử dụng student fault labels cho severity scoring.")

        # 7. Severity & Ranking
        logger.info("Chạy severity scoring...")
        df_labeled = apply_severity(df_labeled)

        logger.info("Tạo ranking...")
        ranking_df = build_transformer_ranking(df_labeled)

        # 8. Báo cáo so sánh (student vs traditional)
        comparison_df = build_student_traditional_comparison(df_labeled)
        if not comparison_df.empty:
            comparison_path = REPORT_DIR / "student_vs_traditional_by_transformer.csv"
            comparison_df.to_csv(comparison_path, index=False)

        # 9. Tạo payload và lưu SQLite
        payload = create_payload(df_labeled, ranking_df, comparison_df=comparison_df)
        from data_store import save_payload_to_db
        save_payload_to_db(payload)

        return payload

    except Exception as e:
        logger.exception("Lỗi trong process_dataframe")
        raise e
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)