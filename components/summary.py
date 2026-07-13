# components/summary.py

import logging
from typing import Any

import pandas as pd
import streamlit as st

logger = logging.getLogger(__name__)


def _debug_enabled() -> bool:
    return bool(st.session_state.get("debug_mode", False))


def _log_debug(event: str, **kwargs: Any) -> None:
    if _debug_enabled():
        logger.info("[summary] %s | %s", event, kwargs if kwargs else "{}")


def _safe_numeric_mean(df: pd.DataFrame, col: str) -> float:
    if df.empty or col not in df.columns:
        return 0.0
    series = pd.to_numeric(df[col], errors="coerce")
    return float(series.mean()) if not series.dropna().empty else 0.0


def render_summary(predictions):
    df = pd.DataFrame(predictions)

    _log_debug(
        "render_summary:start",
        prediction_count=len(df),
        columns=list(df.columns),
    )

    total = len(df)
    severe = int((df["severity"] == "Severe").sum()) if "severity" in df.columns else 0
    moderate = int((df["severity"] == "Moderate").sum()) if "severity" in df.columns else 0
    low = int((df["severity"] == "Low").sum()) if "severity" in df.columns else 0
    avg_score = _safe_numeric_mean(df, "pred_ensemble")

    _log_debug(
        "summary_metrics:computed",
        total=total,
        severe=severe,
        moderate=moderate,
        low=low,
        avg_score=avg_score,
    )

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total samples", total)
    c2.metric("Severe", severe)
    c3.metric("Moderate", moderate)
    c4.metric("Average score", f"{avg_score:.4f}")

    if total > 0:
        severe_pct = severe / total * 100
        moderate_pct = moderate / total * 100
        low_pct = low / total * 100

        st.caption(
            f"Severity mix — Severe: {severe_pct:.1f}% | Moderate: {moderate_pct:.1f}% | Low: {low_pct:.1f}%"
        )
        _log_debug(
            "summary_metrics:distribution",
            severe_pct=round(severe_pct, 2),
            moderate_pct=round(moderate_pct, 2),
            low_pct=round(low_pct, 2),
        )
    else:
        st.caption("No prediction rows available.")
        _log_debug("summary_metrics:empty_dataset")