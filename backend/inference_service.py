# inference_service.py
import tempfile, shutil, json
from pathlib import Path
import pandas as pd
import numpy as np
import logging

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
from evaluation import evaluate_agreement_with_weak_labels, evaluate_diagnostic_performance

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from config import config as cfg

MODEL_DIR = Path(__file__).resolve().parent / "models"

CORE_GASES = ["h2", "ch4", "c2h6", "c2h4", "c2h2", "co", "co2"]
OPTIONAL_NUMERIC = ["o2", "n2", "water", "temp"]

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
    print(f"\n=== FEATURE ENGINEERING SUMMARY ===")
    print(f"Shape: {df.shape}")
    print("First row keys (các cột chứa 'ratio' hoặc 'rate'):")
    first_row = df.iloc[0]
    for k in first_row.index:
        if 'ratio' in k or 'rate' in k:
            print(f"  {k}: {first_row[k]}")
    print("===================================\n")
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
    "duval_triangle_fault", "fault_p1", "duval_pentagon_fault",
    "iec_r1_c2h2_c2h4", "iec_r2_ch4_h2", "iec_r3_c2h4_c2h6",
    "r1_ch4_h2", "r2_c2h2_c2h4", "r3_c2h4_c2h6",
    "dr_r1_ch4_h2", "dr_r2_c2h2_c2h4", "dr_r3_c2h2_ch4", "dr_r4_c2h6_c2h2",
    "ratio_co2_co",
    "h2_rate_per_day", "c2h2_rate_per_day", "tdcg_rate_per_day",
    "severity_gas_score", "severity_trend_score", "severity_aging_score", "severity_fault_score",
]


def _trim_row(row_dict):
    return {k: row_dict[k] for k in ROW_EXPORT_FIELDS if k in row_dict}


def create_payload(df, ranking_df):
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
            "fault_type": row.get("consensus_fault", "UNCERTAIN"),
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
                "fault_type": trow.get("consensus_fault", "UNCERTAIN"),
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
        excel_path = Path(tmp_dir) / "input.xlsx"
        uploaded_df.to_excel(excel_path, index=False, engine='openpyxl')
        logger.info("Bắt đầu clean dataset...")
        df_clean, _ = clean_dataset(input_file=excel_path, output_dir=Path(tmp_dir))
        logger.info(f"Clean xong: {df_clean.shape}")

        logger.info("Bắt đầu feature engineering...")
        df_features = build_features_from_clean(df_clean)

        logger.info("Chạy consensus DGA...")
        df_labeled = apply_consensus(df_features)

        logger.info("Chạy severity scoring...")
        df_labeled = apply_severity(df_labeled)

        logger.info("Tạo ranking...")
        ranking_df = build_transformer_ranking(df_labeled)

        payload = create_payload(df_labeled, ranking_df)
        return payload
    except Exception as e:
        logger.exception("Lỗi trong process_dataframe")
        raise e
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)