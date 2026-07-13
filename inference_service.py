# inference_service.py
"""
Pipeline tích hợp toàn bộ DGA: Clean → Features → Rule-based Labels → Severity → Ranking
Trả về payload cho dashboard Streamlit và Flask API.
"""
import tempfile
import shutil
from pathlib import Path
import numpy as np
import pandas as pd
import json

# Import các module xử lý
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
from generate_labels import (
    classify_fault_consensus, compute_gas_level_score, compute_trend_score,
    compute_aging_score, compute_fault_score, severity_class_from_score
)
from ranking import compute_transformer_ranking

# ============================================================
# Config
# ============================================================
CORE_GASES = ["h2", "ch4", "c2h6", "c2h4", "c2h2", "co", "co2"]
OPTIONAL_NUMERIC = ["o2", "n2", "water", "temp"]
ROLL_WINDOWS = [3, 5]
EWMA_SPANS = [3, 5]
LAG_STEPS = [1, 2, 3]

# ============================================================
# Feature engineering pipeline (giống hệt build_features.py)
# ============================================================
def build_features_from_clean(df: pd.DataFrame) -> pd.DataFrame:
    """Tạo toàn bộ feature từ DataFrame đã clean (sau clean_dataset)."""
    # Đảm bảo kiểu dữ liệu
    df = preprocess_types(df)
    df = sort_and_deduplicate(df)
    df = filter_rows_for_model(df, max_missing_core=3)

    # Missing flags
    df = add_missingness_flags(df, OPTIONAL_NUMERIC + ["year_energized", "tdcg_raw"])

    # Impute optional context (KHÔNG fill temp/water)
    df = impute_optional_context_by_transformer(df)

    # TDCG
    df = add_tdcg(df)

    # Rating features (trích xuất số từ chuỗi mva/kv)
    df = add_rating_features(df)

    # Metadata (transformer_age)
    df = add_metadata_features(df)

    # Ratios
    df = add_ratio_features(df)

    # Duval inputs
    df = add_duval_input_features(df)

    # Calendar & sequence
    df = add_calendar_and_sequence_features(df)

    # Temporal value columns
    temporal_value_cols = [c for c in CORE_GASES + ["tdcg"] if c in df.columns]
    for c in ["water", "temp"]:
        if c in df.columns:
            temporal_value_cols.append(c)

    # Lag/delta/rate
    df = add_lag_delta_rate_features(df, temporal_value_cols)

    # Rolling features
    df = add_rolling_features(df, temporal_value_cols)

    # EWMA features
    df = add_ewm_features(df, temporal_value_cols)

    # Cross-gas trends
    df = add_cross_gas_trend_features(df)

    # Quality flags
    df = add_quality_flags(df)

    return df


# ============================================================
# Rule-based labeling & severity (sử dụng logic từ generate_labels.py)
# ============================================================
def apply_rule_based_diagnostics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Áp dụng chuẩn đoán DGA bằng luật (IEC, Rogers, Duval, Key Gas)
    và tính điểm severity cho từng mẫu.
    Trả về DataFrame với các cột: fault_type_label, severity_score, severity_label,...
    """
    df = df.copy()

    # Fault labels
    final_fault_labels = []
    fault_rules = []
    for _, row in df.iterrows():
        label, detail = classify_fault_consensus(row)
        final_fault_labels.append(label)
        fault_rules.append(detail.get("aggregation", {}).get("strategy", ""))

    df["fault_type_label"] = final_fault_labels
    df["fault_rule"] = fault_rules

    # Severity sub-scores
    gas_scores = []
    trend_scores = []
    aging_scores = []
    fault_scores = []
    for _, row in df.iterrows():
        gas_s, _ = compute_gas_level_score(row)
        trend_s, _ = compute_trend_score(row)
        aging_s, _ = compute_aging_score(row)
        fault_s = compute_fault_score(row["fault_type_label"])

        gas_scores.append(gas_s)
        trend_scores.append(trend_s)
        aging_scores.append(aging_s)
        fault_scores.append(fault_s)

    df["severity_gas_score"] = gas_scores
    df["severity_trend_score"] = trend_scores
    df["severity_fault_score"] = fault_scores
    df["severity_aging_score"] = aging_scores

    # Composite severity score
    df["severity_score"] = (
        1.0 * df["severity_gas_score"]
        + 1.2 * df["severity_trend_score"]
        + 1.0 * df["severity_fault_score"]
        + 0.8 * df["severity_aging_score"]
    )
    df["severity_label"] = df["severity_score"].apply(severity_class_from_score)

    return df


# ============================================================
# Transformer ranking (dựa trên severity tổng hợp)
# ============================================================
def build_transformer_summary(df: pd.DataFrame) -> pd.DataFrame:
    """
    Tạo bảng xếp hạng transformer:
    - Lấy mẫu mới nhất của mỗi máy.
    - Tính điểm tổng hợp: 70% điểm hiện tại + 30% trung bình quá khứ.
    """
    latest_idx = df.groupby("transformer_id")["sample_day"].idxmax()
    latest_df = df.loc[latest_idx].copy()

    past_avg = df.groupby("transformer_id")["severity_score"].mean().reset_index()
    past_avg.rename(columns={"severity_score": "avg_past_severity"}, inplace=True)

    ranking = latest_df[["transformer_id", "loc", "name", "severity_score", "severity_label",
                         "fault_type_label", "sample_day"]].merge(
        past_avg, on="transformer_id", how="left"
    )
    ranking["final_score"] = 0.7 * ranking["severity_score"] + 0.3 * ranking["avg_past_severity"].fillna(ranking["severity_score"])
    ranking = ranking.sort_values("final_score", ascending=False).reset_index(drop=True)
    ranking["rank"] = range(1, len(ranking) + 1)

    # Thêm trend, recommended_action (đơn giản)
    def _trend_label(score):
        if score > 12: return "worsening"
        if score > 6: return "moderate"
        return "stable"

    ranking["trend"] = ranking["severity_score"].apply(_trend_label)

    def _recommended_action(row):
        if row["final_score"] > 10:
            return f"Inspect urgently, likely {row['fault_type_label']}"
        elif row["final_score"] > 5:
            return "Increase monitoring frequency"
        else:
            return "Routine monitoring"

    ranking["recommended_action"] = ranking.apply(_recommended_action, axis=1)

    return ranking


# ============================================================
# Tạo payload cho dashboard
# ============================================================
def create_payload(df: pd.DataFrame, ranking_df: pd.DataFrame) -> dict:
    """
    Chuyển đổi DataFrame đã xử lý và ranking thành payload đúng cấu trúc
    mà dashboard (components/results.py) mong đợi.
    """
    # Predictions (mỗi dòng là 1 record)
    predictions = []
    rows = []
    for idx, row in df.iterrows():
        pred = {
            "row_index": idx,
            "transformer_id": row["transformer_id"],
            "pred_ensemble": float(row["severity_score"] / 20.0),  # scale về 0-1 (max ~20)
            "severity": row["severity_label"],
            "fault_type": row["fault_type_label"],
            "reason": f"Severity score = {row['severity_score']:.2f}",
            "top_features": []  # có thể thêm nếu cần
        }
        predictions.append(pred)
        rows.append(row.to_dict())

    # Transformer summary (ranking)
    transformer_summary = []
    for _, rrow in ranking_df.iterrows():
        ts = {
            "rank": int(rrow["rank"]),
            "transformer_id": rrow["transformer_id"],
            "latest_sample_day": str(rrow["sample_day"]) if pd.notna(rrow["sample_day"]) else "",
            "latest_score": float(rrow["severity_score"]),
            "severity": rrow["severity_label"],
            "fault_type": rrow["fault_type_label"],
            "trend": rrow["trend"],
            "priority_score": float(rrow["final_score"]),
            "priority_label": rrow["severity_label"],  # tạm
            "recommended_action": rrow["recommended_action"],
            "reason": "",
            "features": {}  # có thể bổ sung gas trends
        }
        transformer_summary.append(ts)

    # Timeseries (cho biểu đồ trend)
    timeseries = {}
    for tid, grp in df.groupby("transformer_id"):
        grp = grp.sort_values("sample_day")
        series = []
        for _, trow in grp.iterrows():
            series.append({
                "Sample Day": str(trow["sample_day"]) if pd.notna(trow["sample_day"]) else None,
                "H2": float(trow.get("h2", 0)),
                "CH4": float(trow.get("ch4", 0)),
                "C2H2": float(trow.get("c2h2", 0)),
                "C2H4": float(trow.get("c2h4", 0)),
                "CO": float(trow.get("co", 0)),
                "CO2": float(trow.get("co2", 0)),
                "TCG": float(trow.get("tdcg", 0)),
                "pred_ensemble": float(trow["severity_score"] / 20.0),
                "fault_type": trow["fault_type_label"],
                "severity": trow["severity_label"],
            })
        timeseries[str(tid)] = series

    dataset_summary = {
        "total_transformers": df["transformer_id"].nunique(),
        "total_rows": len(df),
        "date_range": {
            "start": str(df["sample_day"].min()) if len(df) else None,
            "end": str(df["sample_day"].max()) if len(df) else None,
        }
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
    return payload


# ============================================================
# Hàm xử lý chính (được gọi từ main.py và app.py)
# ============================================================
def process_dataframe(uploaded_df: pd.DataFrame) -> dict:
    """
    Nhận DataFrame thô từ file upload (đã được parse bởi Streamlit/Flask).
    Thực hiện toàn bộ pipeline DGA và trả về payload cho dashboard.
    """
    # 1. Lưu DataFrame vào file Excel tạm để dùng với clean_dataset
    tmp_dir = tempfile.mkdtemp()
    try:
        excel_path = Path(tmp_dir) / "input.xlsx"
        uploaded_df.to_excel(excel_path, index=False, engine='openpyxl')

        # 2. Data cleaning
        df_clean, clean_summary = clean_dataset(input_file=excel_path, output_dir=Path(tmp_dir))
        # clean_dataset trả về df và summary, ta dùng df_clean

        # 3. Feature engineering
        df_features = build_features_from_clean(df_clean)

        # 4. Rule-based diagnostics (fault type + severity score)
        df_labeled = apply_rule_based_diagnostics(df_features)

        # 5. Transformer ranking
        ranking_df = build_transformer_summary(df_labeled)

        # 6. Tạo payload
        payload = create_payload(df_labeled, ranking_df)
        return payload
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)

    return {}