# data_store.py
"""Persists the current /predict result into SQLite so the Text2SQL chatbot
(text2sql_chat.py) has something durable to query. Every /predict call DROPs
and rebuilds both tables from the payload it was just given — but that
payload itself is now built from the *accumulated* dataset (see
dataset_accumulator.py, which merges each new upload into everything
uploaded before, deduped by transformer_id+sample_day), not just the latest
file. So this snapshot already reflects the full history; there's nothing
incremental to do here. reset_db() clears it when the user explicitly wants
to start over (paired with dataset_accumulator.reset_accumulated_dataset()).
"""
import json
import math
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd

DB_PATH = Path(__file__).resolve().parent / "data" / "dga.db"

# Mirrors backend/config.py SEVERITY_CLASS_BOUNDARIES / frontend/src/lib/severity.ts
# classifyScore — the same 4-tier Normal/Watch/High/Critical the dashboard
# shows, computed the same way, so chatbot answers agree with what's on screen.
_STATUS_BOUNDARIES = [4, 8, 13]


def _status_from_score(score) -> str:
    try:
        s = float(score)
    except (TypeError, ValueError):
        return "Unknown"
    if s < _STATUS_BOUNDARIES[0]:
        return "Normal"
    if s < _STATUS_BOUNDARIES[1]:
        return "Watch"
    if s < _STATUS_BOUNDARIES[2]:
        return "High"
    return "Critical"


SAMPLE_COLUMNS = [
    "transformer_id", "sample_day", "loc", "name", "ser", "codetx", "mfg",
    "h2", "ch4", "c2h6", "c2h4", "c2h2", "co", "co2", "tdcg", "o2", "n2", "water", "temp",
    "severity_score", "severity_label", "status", "consensus_fault", "mixed_components",
    "diagnostic_confidence", "diagnostic_votes",
    "keygas_fault", "iec_fault", "rogers_fault", "doernenburg_fault",
    "duval_triangle_fault", "fault_p1", "duval_pentagon_fault",
    "iec_r1_c2h2_c2h4", "iec_r2_ch4_h2", "iec_r3_c2h4_c2h6",
    "r1_ch4_h2", "r2_c2h2_c2h4", "r3_c2h4_c2h6",
    "dr_r1_ch4_h2", "dr_r2_c2h2_c2h4", "dr_r3_c2h2_ch4", "dr_r4_c2h6_c2h2",
    "ratio_co2_co", "h2_rate_per_day", "c2h2_rate_per_day", "tdcg_rate_per_day",
    "severity_gas_score", "severity_trend_score", "severity_aging_score", "severity_fault_score",
]

TRANSFORMER_COLUMNS = [
    "transformer_id", "rank", "loc", "name", "latest_sample_day", "latest_score", "status",
    "severity", "fault_type", "trend", "priority_score", "recommended_action",
    "current_severity", "historical_severity", "trend_bonus", "critical_history_count",
    "diagnostic_confidence", "fault_persistence", "days_since_last_critical",
]


def _connect():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    return sqlite3.connect(DB_PATH)


def _to_sql_value(value):
    """Coerces a value from the in-memory payload dict (still full of pandas
    Timestamps, numpy scalar types, NaN/NaT, etc. — this runs before jsonify
    would otherwise normalize any of that) into something sqlite3 can bind:
    None, bool, int, float, or str."""
    if value is None:
        return None
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False)
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        v = float(value)
        return None if (math.isnan(v) or math.isinf(v)) else v
    if isinstance(value, str):
        return value
    try:
        if pd.isna(value):  # NaT and other pandas "missing" sentinels
            return None
    except (TypeError, ValueError):
        pass
    # pandas Timestamp, datetime.date, or anything else sqlite3 can't bind directly
    return str(value)


def save_payload_to_db(payload: dict) -> None:
    conn = _connect()
    try:
        conn.execute("DROP TABLE IF EXISTS samples")
        conn.execute("DROP TABLE IF EXISTS transformers")
        conn.execute(
            f"CREATE TABLE samples ({', '.join(c + (' REAL' if c not in _SAMPLE_TEXT_COLS else ' TEXT') for c in SAMPLE_COLUMNS)})"
        )
        conn.execute(
            f"CREATE TABLE transformers ({', '.join(c + (' REAL' if c not in _TRANSFORMER_TEXT_COLS else ' TEXT') for c in TRANSFORMER_COLUMNS)})"
        )

        sample_rows = []
        for row in payload.get("rows", []):
            status = _status_from_score(row.get("severity_score"))
            sample_rows.append(
                tuple(status if c == "status" else _to_sql_value(row.get(c)) for c in SAMPLE_COLUMNS)
            )
        if sample_rows:
            placeholders = ", ".join(["?"] * len(SAMPLE_COLUMNS))
            conn.executemany(f"INSERT INTO samples VALUES ({placeholders})", sample_rows)

        transformer_rows = []
        for t in payload.get("transformer_summary", []):
            rb = t.get("ranking_breakdown") or {}
            record = {
                "transformer_id": t.get("transformer_id"),
                "rank": t.get("rank"),
                "loc": t.get("loc"),
                "name": t.get("name"),
                "latest_sample_day": t.get("latest_sample_day"),
                "latest_score": t.get("latest_score"),
                "status": _status_from_score(t.get("latest_score")),
                "severity": t.get("severity"),
                "fault_type": t.get("fault_type"),
                "trend": t.get("trend"),
                "priority_score": t.get("priority_score"),
                "recommended_action": t.get("recommended_action"),
                "current_severity": rb.get("current_severity"),
                "historical_severity": rb.get("historical_severity"),
                "trend_bonus": rb.get("trend_bonus"),
                "critical_history_count": rb.get("critical_history_count"),
                "diagnostic_confidence": rb.get("diagnostic_confidence"),
                "fault_persistence": rb.get("fault_persistence"),
                "days_since_last_critical": rb.get("days_since_last_critical"),
            }
            transformer_rows.append(tuple(_to_sql_value(record[c]) for c in TRANSFORMER_COLUMNS))
        if transformer_rows:
            placeholders = ", ".join(["?"] * len(TRANSFORMER_COLUMNS))
            conn.executemany(f"INSERT INTO transformers VALUES ({placeholders})", transformer_rows)

        conn.commit()
    finally:
        conn.close()


_SAMPLE_TEXT_COLS = {
    "transformer_id", "sample_day", "loc", "name", "ser", "codetx", "mfg",
    "severity_label", "status", "consensus_fault", "mixed_components",
    "diagnostic_votes", "keygas_fault", "iec_fault", "rogers_fault", "doernenburg_fault",
    "duval_triangle_fault", "fault_p1", "duval_pentagon_fault",
}
_TRANSFORMER_TEXT_COLS = {
    "transformer_id", "loc", "name", "latest_sample_day", "status", "severity",
    "fault_type", "trend", "recommended_action",
}


def has_data() -> bool:
    if not DB_PATH.exists():
        return False
    conn = _connect()
    try:
        row = conn.execute(
            "SELECT COUNT(*) FROM sqlite_master WHERE type='table' AND name='transformers'"
        ).fetchone()
        if not row or row[0] == 0:
            return False
        return conn.execute("SELECT COUNT(*) FROM transformers").fetchone()[0] > 0
    finally:
        conn.close()


def reset_db() -> None:
    """Drops both tables — paired with dataset_accumulator.reset_accumulated_dataset()
    when the user explicitly clears the loaded dataset."""
    if not DB_PATH.exists():
        return
    conn = _connect()
    try:
        conn.execute("DROP TABLE IF EXISTS samples")
        conn.execute("DROP TABLE IF EXISTS transformers")
        conn.commit()
    finally:
        conn.close()
