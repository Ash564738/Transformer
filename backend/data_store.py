# data_store.py
from __future__ import annotations
import json, logging, math, sqlite3
from pathlib import Path
import numpy as np, pandas as pd
from config import DATABASE_DIR
logger = logging.getLogger(__name__)
DB_PATH = DATABASE_DIR / "dga.db"

def _connect():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True); return sqlite3.connect(DB_PATH)

def _to_sql_value(value):
    if value is None: return None
    if isinstance(value, (dict, list, tuple)): return json.dumps(value, ensure_ascii=False)
    if isinstance(value, (bool, np.bool_)): return bool(value)
    if isinstance(value, (int, np.integer)): return int(value)
    if isinstance(value, (float, np.floating)):
        value = float(value)
        if math.isnan(value) or math.isinf(value): return None
        return value
    if isinstance(value, str): return value
    try:
        if pd.isna(value): return None
    except (TypeError, ValueError): pass
    return str(value)

def _resolve_status(row: dict) -> str:
    try: status = int(row.get("ieee_status", row.get("ieee_dga_status", 0)))
    except (TypeError, ValueError): status = 0
    extreme = bool(row.get("ieee_extreme_dga", False))
    if status == 1: return "Normal"
    if status == 2: return "Watch"
    if status == 3 and extreme: return "Critical"
    if status == 3: return "High"
    return "Insufficient data"

SAMPLE_COLUMNS = [
    "transformer_id", "sample_day", "loc", "name", "ser", "codetx", "mfg",
    "h2", "ch4", "c2h6", "c2h4", "c2h2", "co", "co2",
    "tdcg_raw", "tdcg_recalc", "tdcg",
    "o2", "n2", "o2_n2_ratio", "transformer_age_years", "water", "temp",
    "ieee_dga_status", "ieee_status", "ieee_dga_status_label", "ieee_dga_status_reason",
    "ieee_confirmation_required", "ieee_extreme_dga",
    "severity_score", "severity_label", "severity_label_text",
    "consensus_fault", "consensus_fault_traditional", "final_fault", "final_fault_group",
    "diagnostic_conflict", "mixed_components",
    "diagnostic_confidence", "diagnostic_coverage",
    "keygas_fault", "iec_fault", "rogers_fault", "doernenburg_fault",
    "duval_triangle_fault", "duval_pentagon_p1_fault", "duval_pentagon_p2_fault",
    "fault_p1", "fault_p2",
    "student_fault_label", "student_fault_confidence", "anomaly_percentile",
    "h2_delta1", "ch4_delta1", "c2h2_delta1", "c2h4_delta1", "c2h6_delta1",
    "co_delta1", "co2_delta1",
    "h2_rate_per_year", "ch4_rate_per_year", "c2h2_rate_per_year",
    "c2h4_rate_per_year", "tdcg_rate_per_year",
    "rate_points", "rate_span_months", "recommended_action",
]
TRANSFORMER_COLUMNS = [
    "transformer_id", "rank", "loc", "name", "latest_sample_day", "latest_score",
    "ieee_status", "status", "severity", "fault_type", "priority_score", "recommended_action",
    "ieee_dga_status", "ieee_dga_status_label", "diagnostic_confidence",
    "anomaly_percentile", "trend_slope",
]
SAMPLE_TEXT_COLS = {
    "transformer_id", "sample_day", "loc", "name", "ser", "codetx", "mfg",
    "ieee_dga_status_label", "ieee_dga_status_reason", "severity_label",
    "severity_label_text", "consensus_fault", "consensus_fault_traditional",
    "final_fault", "final_fault_group", "mixed_components", "diagnostic_coverage",
    "keygas_fault", "iec_fault", "rogers_fault", "doernenburg_fault",
    "duval_triangle_fault", "duval_pentagon_p1_fault", "duval_pentagon_p2_fault",
    "fault_p1", "fault_p2", "student_fault_label", "recommended_action",
}
TRANSFORMER_TEXT_COLS = {
    "transformer_id", "loc", "name", "latest_sample_day", "status", "severity",
    "fault_type", "recommended_action", "ieee_dga_status_label",
}

def _column_sql(column: str, text_columns: set) -> str:
    sql_type = "TEXT" if column in text_columns else "REAL"
    return f'"{column}" {sql_type}'

def save_payload_to_db(payload: dict) -> None:
    logger.info("Saving current DGA payload to SQLite.")
    conn = _connect()
    try:
        conn.execute("DROP TABLE IF EXISTS samples")
        conn.execute("DROP TABLE IF EXISTS transformers")
        sample_schema = ", ".join(_column_sql(c, SAMPLE_TEXT_COLS) for c in SAMPLE_COLUMNS)
        transformer_schema = ", ".join(_column_sql(c, TRANSFORMER_TEXT_COLS) for c in TRANSFORMER_COLUMNS)
        conn.execute(f"CREATE TABLE samples ({sample_schema})")
        conn.execute(f"CREATE TABLE transformers ({transformer_schema})")
        sample_rows = []
        for row in payload.get("rows", []):
            sample_rows.append(tuple(_resolve_status(row) if column == "status" else _to_sql_value(row.get(column)) for column in SAMPLE_COLUMNS))
        if sample_rows:
            placeholders = ", ".join(["?"] * len(SAMPLE_COLUMNS))
            conn.executemany(f"INSERT INTO samples VALUES ({placeholders})", sample_rows)
        transformer_rows = []
        for item in payload.get("transformer_summary", []):
            breakdown = item.get("ranking_breakdown") or {}
            record = {
                "transformer_id": item.get("transformer_id"), "rank": item.get("rank"),
                "loc": item.get("loc"), "name": item.get("name"),
                "latest_sample_day": item.get("latest_sample_day"), "latest_score": item.get("latest_score"),
                "ieee_status": item.get("ieee_status", breakdown.get("current_status")),
                "status": item.get("status") or _resolve_status(item),
                "severity": item.get("severity"), "fault_type": item.get("fault_type"),
                "priority_score": item.get("priority_score"), "recommended_action": item.get("recommended_action"),
                "ieee_dga_status": item.get("ieee_status", breakdown.get("current_status")),
                "ieee_dga_status_label": item.get("ieee_dga_status_label", breakdown.get("current_status_label")),
                "diagnostic_confidence": item.get("diagnostic_confidence", breakdown.get("diagnostic_confidence")),
                "anomaly_percentile": item.get("anomaly_percentile", breakdown.get("anomaly_percentile")),
                "trend_slope": item.get("trend_slope", breakdown.get("trend_slope")),
            }
            transformer_rows.append(tuple(_to_sql_value(record.get(c)) for c in TRANSFORMER_COLUMNS))
        if transformer_rows:
            placeholders = ", ".join(["?"] * len(TRANSFORMER_COLUMNS))
            conn.executemany(f"INSERT INTO transformers VALUES ({placeholders})", transformer_rows)
        conn.commit()
        logger.info("Inserted %d samples and %d transformers.", len(sample_rows), len(transformer_rows))
    except Exception:
        conn.rollback()
        logger.exception("Failed to save payload.")
        raise
    finally:
        conn.close()

def has_data() -> bool:
    if not DB_PATH.exists(): return False
    conn = _connect()
    try:
        exists = conn.execute("SELECT COUNT(*) FROM sqlite_master WHERE type='table' AND name='transformers'").fetchone()[0]
        if not exists: return False
        count = conn.execute("SELECT COUNT(*) FROM transformers").fetchone()[0]
        return count > 0
    finally:
        conn.close()

def reset_db() -> None:
    if not DB_PATH.exists(): return
    conn = _connect()
    try:
        conn.execute("DROP TABLE IF EXISTS samples")
        conn.execute("DROP TABLE IF EXISTS transformers")
        conn.commit()
    finally:
        conn.close()