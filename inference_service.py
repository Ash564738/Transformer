# inference_service.py
import tempfile, shutil, json
from pathlib import Path
import pandas as pd
import numpy as np

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
from dga.consensus import apply_consensus
from dga.severity import apply_severity
from dga.ranking import build_transformer_ranking

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from config import SEVERITY_TO_UI

CORE_GASES = ["h2", "ch4", "c2h6", "c2h4", "c2h2", "co", "co2"]
OPTIONAL_NUMERIC = ["o2", "n2", "water", "temp"]

def build_features_from_clean(df: pd.DataFrame) -> pd.DataFrame:
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

def create_payload(df, ranking_df):
    predictions = []
    rows = []
    for idx, row in df.iterrows():
        ui_severity = SEVERITY_TO_UI.get(row["severity_label"], row["severity_label"])
        pred = {
            "row_index": idx,
            "transformer_id": row["transformer_id"],
            "pred_ensemble": float(row["severity_score"] / 20.0),
            "severity": ui_severity,          # sử dụng nhãn UI
            "fault_type": row.get("consensus_fault", "UNCERTAIN"),
            "reason": f"Severity score = {row['severity_score']:.2f}",
            "top_features": []
        }
        predictions.append(pred)
        rows.append(row.to_dict())

    transformer_summary = []
    for _, rrow in ranking_df.iterrows():
        ui_severity = SEVERITY_TO_UI.get(rrow["severity_label"], rrow["severity_label"])
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
            "features": {}
        }
        transformer_summary.append(ts)
    # ... phần còn lại giữ nguyên

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
                "fault_type": trow.get("consensus_fault", "UNCERTAIN"),
                "severity": trow["severity_label"],
            })
        timeseries[str(tid)] = series

    dataset_summary = {
        "total_transformers": df["transformer_id"].nunique(),
        "total_rows": len(df),
    }

    return {
        "predictions": predictions,
        "rows": rows,
        "preview_rows": rows[:20],
        "transformer_summary": transformer_summary,
        "transformer_timeseries": timeseries,
        "dataset_summary": dataset_summary,
        "chat_context_payload": {
            "transformer_summary": transformer_summary,
            "dataset_summary": dataset_summary
        }
    }

def process_dataframe(uploaded_df):
    tmp_dir = tempfile.mkdtemp()
    try:
        excel_path = Path(tmp_dir) / "input.xlsx"
        uploaded_df.to_excel(excel_path, index=False, engine='openpyxl')
        logger.info("Bắt đầu clean dataset...")
        df_clean, _ = clean_dataset(input_file=excel_path, output_dir=Path(tmp_dir))
        logger.info(f"Clean xong: {df_clean.shape}")
        
        logger.info("Bắt đầu feature engineering...")
        df_features = build_features_from_clean(df_clean)
        logger.info(f"Features: {df_features.shape}, số cột: {len(df_features.columns)}")
        
        logger.info("Chạy consensus DGA...")
        df_labeled = apply_consensus(df_features)
        logger.info("Chạy severity scoring...")
        df_labeled = apply_severity(df_labeled)
        logger.info("Tạo ranking...")
        ranking_df = build_transformer_ranking(df_labeled)
        
        payload = create_payload(df_labeled, ranking_df)
        logger.info("Payload đã sẵn sàng.")
        return payload
    except Exception as e:
        logger.exception("Lỗi trong process_dataframe")
        raise e
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)