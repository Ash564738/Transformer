# components/results.py
import logging
from typing import Any

import altair as alt
import pandas as pd
import streamlit as st

logger = logging.getLogger(__name__)


def _debug_enabled() -> bool:
    return bool(st.session_state.get("debug_mode", False))


def _log_debug(event: str, **kwargs: Any) -> None:
    if _debug_enabled():
        logger.info("[results] %s | %s", event, kwargs if kwargs else "{}")


def _render_card(title, value, subtitle, accent=""):
    accent_class = {
        "red": "dg-card-red",
        "amber": "dg-card-amber",
        "green": "dg-card-green",
        "blue": "dg-card-blue",
    }.get(accent, "dg-card-blue")

    st.markdown(
        f"""
        <div class="dg-card {accent_class}">
            <div class="dg-card-accent"></div>
            <div class="dg-card-title">{title}</div>
            <div class="dg-card-value">{value}</div>
            <div class="dg-card-subtitle">{subtitle}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _status_accent(severity):
    if severity == "Severe":
        return "red"
    if severity == "Moderate":
        return "amber"
    return "green"


def _render_chart(chart):
    try:
        st.altair_chart(chart, width="stretch")
    except TypeError:
        st.altair_chart(chart, use_container_width=True)


def _render_dataframe(df, height=None):
    _log_debug(
        "render_dataframe",
        rows=len(df) if hasattr(df, "__len__") else None,
        columns=list(df.columns) if hasattr(df, "columns") else None,
        height=height,
    )
    try:
        kwargs = {"width": "stretch"}
        if height is not None:
            kwargs["height"] = height
        st.dataframe(df, **kwargs)
    except TypeError:
        kwargs = {"use_container_width": True}
        if height is not None:
            kwargs["height"] = height
        st.dataframe(df, **kwargs)


def _filter_payload(payload, selected_transformer):
    predictions = payload.get("predictions", [])
    rows = payload.get("rows", [])
    preview_rows = payload.get("preview_rows", [])
    transformer_summary = payload.get("transformer_summary", [])

    if selected_transformer:
        selected = str(selected_transformer)
        predictions = [p for p in predictions if str(p.get("transformer_id", "")) == selected]
        rows = [r for r in rows if str(r.get("transformer_id", "")) == selected]
        preview_rows = [r for r in preview_rows if str(r.get("transformer_id", "")) == selected]
        transformer_summary = [r for r in transformer_summary if str(r.get("transformer_id", "")) == selected]

    _log_debug(
        "filter_payload",
        selected_transformer=selected_transformer,
        predictions=len(predictions),
        rows=len(rows),
        preview_rows=len(preview_rows),
        transformer_summary=len(transformer_summary),
    )
    return predictions, rows, preview_rows, transformer_summary


def _build_trend_frame(payload, selected_transformer):
    records = []
    for transformer_id, series in payload.get("transformer_timeseries", {}).items():
        if selected_transformer and str(transformer_id) != str(selected_transformer):
            continue
        for row in series:
            records.append({"transformer_id": transformer_id, **row})

    trend_df = pd.DataFrame(records)
    if trend_df.empty or not {"Sample Day", "pred_ensemble"}.issubset(trend_df.columns):
        _log_debug(
            "build_trend_frame:empty_or_invalid",
            row_count=len(trend_df),
            columns=list(trend_df.columns) if not trend_df.empty else [],
        )
        return pd.DataFrame()

    trend_df["Sample Day"] = pd.to_datetime(trend_df["Sample Day"], errors="coerce")
    trend_df["pred_ensemble"] = pd.to_numeric(trend_df["pred_ensemble"], errors="coerce")

    for col in ["H2", "C2H2", "CO", "TCG"]:
        if col in trend_df.columns:
            trend_df[col] = pd.to_numeric(trend_df[col], errors="coerce")

    trend_df = trend_df.dropna(subset=["Sample Day", "pred_ensemble"]).sort_values("Sample Day")
    _log_debug(
        "build_trend_frame:done",
        rows=len(trend_df),
        transformers=trend_df["transformer_id"].nunique() if not trend_df.empty else 0,
    )
    return trend_df


def _build_score_interpretation(row):
    parts = []
    c2h2 = row.get("C2H2", 0) or 0
    c2h4 = row.get("C2H4", 0) or 0
    tcg = row.get("TCG", 0) or 0
    roc_c2h2 = row.get("roc_C2H2", 0) or 0
    roc_tcg = row.get("roc_TCG", 0) or 0
    c2h2_c2h4 = row.get("C2H2_C2H4", 0) or 0
    co2_co = row.get("CO2_CO", 0) or 0
    co = row.get("CO", 0) or 0

    if c2h2 > 10:
        parts.append(f"C₂H₂ elevated ({c2h2:.1f} ppm)")
    if c2h2_c2h4 > 0.1:
        parts.append(f"C₂H₂/C₂H₄ ratio high ({c2h2_c2h4:.2f})")
    if roc_c2h2 > 0.5:
        parts.append(f"C₂H₂ rising fast ({roc_c2h2:.2f} ppm/day)")
    if roc_tcg > 5:
        parts.append(f"TCG increasing rapidly ({roc_tcg:.1f} ppm/day)")
    if tcg > 720:
        parts.append(f"TCG very high ({tcg:.0f} ppm)")
    if co > 300 and co2_co < 7:
        parts.append(f"CO elevated ({co:.0f} ppm) with low CO₂/CO ({co2_co:.1f})")

    if not parts:
        if c2h2_c2h4 > 0:
            parts.append(f"C₂H₂/C₂H₄ = {c2h2_c2h4:.2f}")
        if roc_tcg > 0:
            parts.append(f"TCG trend +{roc_tcg:.1f}/day")
        if not parts:
            parts.append("combined gas profile and trend")

    explanation = "Score driven by: " + ", ".join(parts) + "."
    _log_debug("score_interpretation", explanation=explanation)
    return explanation


def render_results(payload, selected_transformer=None):
    _log_debug(
        "render_results:start",
        payload_keys=list(payload.keys()) if isinstance(payload, dict) else None,
        selected_transformer=selected_transformer,
    )

    predictions, rows, preview_rows, transformer_summary = _filter_payload(payload, selected_transformer)

    if not predictions:
        _log_debug("render_results:no_predictions")
        st.warning("No results are available for the selected transformer.")
        return

    from components.summary import render_summary
    render_summary(predictions)

    pred_df = pd.DataFrame(predictions)
    summary_df = pd.DataFrame(transformer_summary)
    dataset_summary = payload.get("dataset_summary", {})

    _log_debug(
        "render_results:dataframes_ready",
        pred_rows=len(pred_df),
        pred_cols=list(pred_df.columns),
        summary_rows=len(summary_df),
        summary_cols=list(summary_df.columns) if not summary_df.empty else [],
        preview_rows=len(preview_rows),
    )

    st.subheader("Dashboard Overview")
    col1, col2, col3 = st.columns(3)

    with col1:
        total_transformers = (
            dataset_summary.get("total_transformers")
            if dataset_summary.get("total_transformers") is not None
            else (summary_df["transformer_id"].nunique() if not summary_df.empty else 0)
        )
        _render_card(
            "Transformers",
            str(total_transformers),
            "Transformer assets in the current dataset",
            "green",
        )

    with col2:
        highest = pred_df.loc[pred_df["pred_ensemble"].idxmax()]
        _log_debug(
            "dashboard_overview:highest",
            transformer_id=highest.get("transformer_id"),
            severity=highest.get("severity"),
            score=highest.get("pred_ensemble"),
        )
        _render_card(
            "Highest severity",
            str(highest["severity"]),
            f"{highest['transformer_id']}",
            _status_accent(highest["severity"]),
        )

    with col3:
        avg_score = pd.to_numeric(pred_df["pred_ensemble"], errors="coerce").mean()
        _render_card(
            "Average degradation score",
            f"{avg_score * 100:.1f}",
            "Mean sample-level ensemble score",
            "amber",
        )

    st.subheader("Key Gas Indicators")
    sel_idx = st.selectbox(
        "Select sample row",
        options=list(range(len(predictions))),
        format_func=lambda i: f"Row {i + 1} - {predictions[i].get('transformer_id', '')}",
        key="results_sample_selector",
    )
    selected_pred = predictions[sel_idx]
    raw_row = rows[sel_idx] if sel_idx < len(rows) else {}
    # Trong render_results, sau khi lấy selected_pred và raw_row
    gas_dict = raw_row if isinstance(raw_row, dict) else {}

    # Trong render_results, sau khi có gas_dict
    co2_co = gas_dict.get("ratio_co2_co")
    if co2_co is None:
        # fallback nếu không có ratio
        co_val = gas_dict.get("co")
        co2_val = gas_dict.get("co2")
        if co_val and co2_val and co_val > 0:
            co2_co = co2_val / co_val

    roc_h2 = gas_dict.get("h2_rate_per_day")
    roc_c2h2 = gas_dict.get("c2h2_rate_per_day")
    roc_tcg = gas_dict.get("tdcg_rate_per_day")

    # Định dạng hiển thị
    def fmt_rate(val, unit="ppm/d"):
        return f"{val:.2f} {unit}" if val is not None else "—"

    _log_debug(
        "sample_selected",
        index=sel_idx,
        transformer_id=selected_pred.get("transformer_id"),
        raw_row_keys=list(gas_dict.keys()) if gas_dict else [],
    )

    kpi_col1, kpi_col2, kpi_col3, kpi_col4 = st.columns(4)
    with kpi_col1:
        _render_card("CO₂/CO Ratio", f"{co2_co:.2f}" if co2_co is not None else "—", "Lower suggests cellulose stress", "blue")
    with kpi_col2:
        _render_card("roc H₂", fmt_rate(roc_h2), "Hydrogen rate-of-change", "blue")
    with kpi_col3:
        _render_card("roc C₂H₂", fmt_rate(roc_c2h2), "Acetylene rate-of-change", "blue")
    with kpi_col4:
        _render_card("roc TCG", fmt_rate(roc_tcg), "Total combustible gas trend", "blue")

    interpretation = _build_score_interpretation(gas_dict) if gas_dict else ""
    if interpretation:
        st.markdown(
            f"""
            <div class="insight-box">
                <div class="insight-box-title">Why this score?</div>
                <div class="insight-box-body">{interpretation}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    if gas_dict:
        votes = gas_dict.get("diagnostic_votes")
        if votes and isinstance(votes, dict):
            with st.expander("🔬 Diagnostic breakdown (why MIXED?)"):
                st.json(votes)
                # Liệt kê rõ ràng hơn
                st.markdown("**Individual methods:**")
                for method, fault in votes.items():
                    st.write(f"- {method}: {fault}")
    if not summary_df.empty:
        st.subheader("Transformer Ranking")
        ranking_cols = [
            "rank",
            "transformer_id",
            "latest_sample_day",
            "latest_score",
            "severity",
            "fault_type",
            "trend",
            "priority_score",
            "priority_label",
            "recommended_action",
        ]
        ranking_cols = [col for col in ranking_cols if col in summary_df.columns]
        _log_debug("ranking_table", columns=ranking_cols, rows=len(summary_df))
        _render_dataframe(summary_df[ranking_cols], height=360)

    st.subheader("Transformer Status Cards")
    top_rows = pred_df.head(3)
    cards_cols = st.columns(max(1, min(3, len(top_rows))))
    for i, (_, row) in enumerate(top_rows.iterrows()):
        with cards_cols[i % len(cards_cols)]:
            _render_card(
                str(row["transformer_id"]),
                f"{row['pred_ensemble'] * 100:.1f}",
                f"{row['severity']} — {row['fault_type']}",
                _status_accent(row["severity"]),
            )

    st.subheader("Prediction Results")
    prediction_cols = [
        "row_index",
        "transformer_id",
        "pred_lgb",
        "pred_seq",
        "pred_ensemble",
        "severity",
        "fault_type",
    ]
    prediction_cols = [c for c in prediction_cols if c in pred_df.columns]
    _render_dataframe(pred_df[prediction_cols], height=360)

    st.subheader("Severity Distribution")
    if "severity" in pred_df.columns:
        severity_counts = pred_df["severity"].value_counts().reset_index()
        severity_counts.columns = ["severity", "count"]
        _log_debug("severity_distribution", data=severity_counts.to_dict(orient="records"))
        severity_chart = alt.Chart(severity_counts).mark_bar(cornerRadiusTopLeft=6, cornerRadiusTopRight=6).encode(
            x=alt.X("severity:O", title="Severity"),
            y=alt.Y("count:Q", title="Count"),
            color=alt.Color("severity:N", legend=None),
            tooltip=["severity:N", "count:Q"],
        ).properties(height=220)
        _render_chart(severity_chart)

    st.subheader("Time Trend")
    trend_df = _build_trend_frame(payload, selected_transformer)
    if trend_df.empty:
        st.info("No valid time-series score data is available for the current selection.")
    else:
        tooltip_cols = ["Sample Day:T", "transformer_id:N", "pred_ensemble:Q"]
        for c in ["H2", "C2H2", "CO", "TCG"]:
            if c in trend_df.columns:
                tooltip_cols.append(f"{c}:Q")

        trend_chart = alt.Chart(trend_df).mark_line(point=True).encode(
            x=alt.X("Sample Day:T", title="Sample Day"),
            y=alt.Y("pred_ensemble:Q", title="Ensemble Score"),
            color="transformer_id:N",
            tooltip=tooltip_cols,
        ).properties(height=260)
        _render_chart(trend_chart)

    st.subheader("Data Preview")
    _render_dataframe(pd.DataFrame(preview_rows), height=320)

    st.subheader("Model Explanation")
    selected = st.selectbox(
        "Choose transformer row",
        options=list(range(len(predictions))),
        format_func=lambda i: f"Row {i + 1} - {predictions[i].get('transformer_id', '')}",
        key="results_explanation_selector",
    )
    selected_row = predictions[selected]

    _log_debug(
        "model_explanation:selected_row",
        index=selected,
        transformer_id=selected_row.get("transformer_id"),
        has_top_features=bool(selected_row.get("top_features")),
    )

    st.write(selected_row.get("reason", "No explanation was returned."))

    feature_df = pd.DataFrame(selected_row.get("top_features", []))
    if not feature_df.empty and {"feature", "importance"}.issubset(feature_df.columns):
        feature_df = feature_df[["feature", "importance"]].copy()
        feature_df["importance"] = pd.to_numeric(feature_df["importance"], errors="coerce").fillna(0.0)
        feature_df["importance_abs"] = feature_df["importance"].abs()
        feature_df = feature_df.sort_values("importance_abs", ascending=False).head(8)

        _log_debug(
            "top_features",
            features=feature_df[["feature", "importance"]].to_dict(orient="records"),
        )

        feature_chart = alt.Chart(feature_df).mark_bar(cornerRadiusTopRight=6, cornerRadiusBottomRight=6).encode(
            x=alt.X("importance_abs:Q", title="Absolute importance"),
            y=alt.Y("feature:N", sort="-x", title=None),
            color=alt.Color("importance:Q", legend=None),
            tooltip=["feature:N", "importance:Q"],
        ).properties(height=240)
        _render_chart(feature_chart)
        _render_dataframe(feature_df[["feature", "importance"]])

    raw_row = rows[selected] if selected < len(rows) else {}
    if raw_row:
        st.caption("Raw input row")
        st.json(raw_row)