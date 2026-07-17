# components/results.py
import logging
from typing import Any

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from dga import duval_triangle, duval_pentagon
import consensus

logger = logging.getLogger(__name__)

def _safe_fmt(val, fmt_spec=".3f", na_rep="—"):
    """Định dạng số an toàn: trả về na_rep nếu là None, NaN hoặc Inf."""
    if val is None:
        return na_rep
    if isinstance(val, (float, np.floating)):
        if np.isnan(val) or np.isinf(val):
            return na_rep
    try:
        return format(val, fmt_spec)
    except (ValueError, TypeError):
        return na_rep

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

    return predictions, rows, preview_rows, transformer_summary

def _build_trend_frame(payload, selected_transformer, transformer_ids=None):
    records = []
    for transformer_id, series in payload.get("transformer_timeseries", {}).items():
        if selected_transformer and str(transformer_id) != str(selected_transformer):
            continue
        if transformer_ids and str(transformer_id) not in transformer_ids:
            continue
        for row in series:
            records.append({"transformer_id": transformer_id, **row})

    trend_df = pd.DataFrame(records)
    if trend_df.empty or "Sample Day" not in trend_df.columns or "pred_ensemble" not in trend_df.columns:
        return pd.DataFrame()

    trend_df["Sample Day"] = pd.to_datetime(trend_df["Sample Day"], errors="coerce")
    trend_df["pred_ensemble"] = pd.to_numeric(trend_df["pred_ensemble"], errors="coerce")
    trend_df = trend_df.dropna(subset=["Sample Day", "pred_ensemble"]).sort_values("Sample Day")
    return trend_df

def _build_score_interpretation(row):
    parts = []
    c2h2 = row.get("c2h2", 0) or 0
    c2h4 = row.get("c2h4", 0) or 0
    tcg = row.get("tcg", 0) or 0
    roc_c2h2 = row.get("c2h2_rate_per_day", 0) or 0
    roc_tcg = row.get("tdcg_rate_per_day", 0) or 0
    c2h2_c2h4 = row.get("iec_r1_c2h2_c2h4", row.get("r2_c2h2_c2h4", 0)) or 0
    co2_co = row.get("ratio_co2_co", 0) or 0
    co = row.get("co", 0) or 0

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

    return "Score driven by: " + ", ".join(parts) + "."

def render_results(payload, selected_transformer=None):
    predictions, rows, preview_rows, transformer_summary = _filter_payload(payload, selected_transformer)

    if not predictions:
        st.warning("No results are available for the selected transformer.")
        return

    from components.summary import render_summary
    render_summary(predictions)

    pred_df = pd.DataFrame(predictions)
    summary_df = pd.DataFrame(transformer_summary)
    dataset_summary = payload.get("dataset_summary", {})

    # ===================== Dashboard Overview =====================
    st.subheader("Dashboard Overview")
    col1, col2, col3 = st.columns(3)

    with col1:
        total_transformers = (
            dataset_summary.get("total_transformers")
            or (summary_df["transformer_id"].nunique() if not summary_df.empty else 0)
        )
        _render_card("Transformers", str(total_transformers), "Transformer assets in dataset", "green")

    with col2:
        if not pred_df.empty and "pred_ensemble" in pred_df.columns:
            highest = pred_df.loc[pred_df["pred_ensemble"].idxmax()]
            _render_card("Highest severity", str(highest.get("severity", "N/A")),
                         f"{highest.get('transformer_id', '')}", _status_accent(highest.get("severity", "")))
        else:
            _render_card("Highest severity", "N/A", "", "blue")

    with col3:
        avg_score = pd.to_numeric(pred_df.get("pred_ensemble", pd.Series([0])), errors="coerce").mean()
        _render_card("Average degradation score", f"{avg_score * 100:.1f}", "Mean ensemble score", "amber")

    # ===================== Sample Inspector =====================
    st.subheader("Sample Inspector")
    sel_idx = st.selectbox(
        "Select sample row",
        options=list(range(len(predictions))),
        format_func=lambda i: f"Row {i + 1} - {predictions[i].get('transformer_id', '')}",
        key="results_sample_selector",
    )
    selected_pred = predictions[sel_idx]
    raw_row = rows[sel_idx] if sel_idx < len(rows) else {}
    gas_dict = raw_row if isinstance(raw_row, dict) else {}

    # -------- DEBUG CONSOLE --------
    print("=" * 60)
    print("DEBUG gas_dict keys and selected values:")
    for key in sorted(gas_dict.keys()):
        if any(k in key for k in ["ratio", "rate", "r1_", "r2_", "r3_", "iec_r", "dr_r"]):
            print(f"  {key} = {gas_dict[key]}")
    print("=" * 60)
    # ------------------------------

    def fmt_rate(val, unit="ppm/d"):
        formatted = _safe_fmt(val, fmt_spec=".2f")
        if formatted == "—":
            return "—"
        return f"{formatted} {unit}"

    co2_co = gas_dict.get("ratio_co2_co")
    if co2_co is None or (isinstance(co2_co, float) and np.isnan(co2_co)):
        co_val = gas_dict.get("co")
        co2_val = gas_dict.get("co2")
        if co_val and co2_val and co_val > 0:
            co2_co = co2_val / co_val

    roc_h2 = gas_dict.get("h2_rate_per_day")
    roc_c2h2 = gas_dict.get("c2h2_rate_per_day")
    roc_tcg = gas_dict.get("tdcg_rate_per_day")

    # ===================== Key Gas Indicators =====================
    st.subheader("Key Gas Indicators")

    # Hàng 1: Nồng độ các khí chính
    col1, col2, col3, col4, col5, col6 = st.columns(6)
    with col1:
        _render_card("H₂", f"{_safe_fmt(gas_dict.get('h2', 0), '.1f')} ppm", "Hydrogen", "blue")
    with col2:
        _render_card("CH₄", f"{_safe_fmt(gas_dict.get('ch4', 0), '.1f')} ppm", "Methane", "blue")
    with col3:
        _render_card("C₂H₆", f"{_safe_fmt(gas_dict.get('c2h6', 0), '.1f')} ppm", "Ethane", "blue")
    with col4:
        _render_card("C₂H₄", f"{_safe_fmt(gas_dict.get('c2h4', 0), '.1f')} ppm", "Ethylene", "blue")
    with col5:
        _render_card("C₂H₂", f"{_safe_fmt(gas_dict.get('c2h2', 0), '.1f')} ppm", "Acetylene", "blue")
    with col6:
        _render_card("CO", f"{_safe_fmt(gas_dict.get('co', 0), '.1f')} ppm", "Carbon Monoxide", "blue")

    # Hàng 2: TDCG, CO₂, CO₂/CO ratio, O₂
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        tdcg = gas_dict.get("tdcg", 0)
        if tdcg == 0:
            tdcg = sum(gas_dict.get(g, 0) for g in ["h2", "ch4", "c2h6", "c2h4", "c2h2", "co"])
        _render_card("TDCG", f"{_safe_fmt(tdcg, '.1f')} ppm", "Total Combustible Gas", "amber")
    with col2:
        _render_card("CO₂", f"{_safe_fmt(gas_dict.get('co2', 0), '.1f')} ppm", "Carbon Dioxide", "blue")
    with col3:
        _render_card("CO₂/CO Ratio", _safe_fmt(co2_co, ".2f"), "Cellulose stress indicator", "amber")
    with col4:
        _render_card("O₂", f"{_safe_fmt(gas_dict.get('o2', 0), '.1f')} ppm", "Oxygen", "blue")

    # Hàng 3: Tốc độ thay đổi
    st.caption("Rate of Change (ppm/day)")
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        _render_card("roc H₂", fmt_rate(roc_h2), "", "blue")
    with col2:
        _render_card("roc CH₄", fmt_rate(gas_dict.get("ch4_rate_per_day")), "", "blue")
    with col3:
        _render_card("roc C₂H₄", fmt_rate(gas_dict.get("c2h4_rate_per_day")), "", "blue")
    with col4:
        _render_card("roc C₂H₂", fmt_rate(roc_c2h2), "", "blue")
    with col5:
        _render_card("roc TCG", fmt_rate(roc_tcg), "", "blue")

    # Hàng 4: Các tỉ số chẩn đoán
    st.caption("Diagnostic Ratios")
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        ch4_h2 = gas_dict.get("iec_r2_ch4_h2", gas_dict.get("r1_ch4_h2"))
        print(f"DEBUG CH4/H2: iec_r2_ch4_h2={gas_dict.get('iec_r2_ch4_h2')}, r1_ch4_h2={gas_dict.get('r1_ch4_h2')}, final={ch4_h2}")
        _render_card("CH₄/H₂", _safe_fmt(ch4_h2, ".3f"), "Rogers/IEC R1", "blue")
    with col2:
        c2h2_c2h4 = gas_dict.get("iec_r1_c2h2_c2h4", gas_dict.get("r2_c2h2_c2h4"))
        print(f"DEBUG C2H2/C2H4: iec_r1_c2h2_c2h4={gas_dict.get('iec_r1_c2h2_c2h4')}, r2_c2h2_c2h4={gas_dict.get('r2_c2h2_c2h4')}, final={c2h2_c2h4}")
        _render_card("C₂H₂/C₂H₄", _safe_fmt(c2h2_c2h4, ".3f"), "Rogers/IEC R2", "blue")
    with col3:
        c2h4_c2h6 = gas_dict.get("iec_r3_c2h4_c2h6", gas_dict.get("r3_c2h4_c2h6"))
        print(f"DEBUG C2H4/C2H6: iec_r3_c2h4_c2h6={gas_dict.get('iec_r3_c2h4_c2h6')}, r3_c2h4_c2h6={gas_dict.get('r3_c2h4_c2h6')}, final={c2h4_c2h6}")
        _render_card("C₂H₄/C₂H₆", _safe_fmt(c2h4_c2h6, ".3f"), "Rogers/IEC R3", "blue")
    with col4:
        c2h2_ch4 = gas_dict.get("dr_r3_c2h2_ch4")
        print(f"DEBUG C2H2/CH4: dr_r3_c2h2_ch4={c2h2_ch4}")
        _render_card("C₂H₂/CH₄", _safe_fmt(c2h2_ch4, ".3f"), "Doernenburg R3", "blue")
    with col5:
        c2h6_c2h2 = gas_dict.get("dr_r4_c2h6_c2h2")
        print(f"DEBUG C2H6/C2H2: dr_r4_c2h6_c2h2={c2h6_c2h2}")
        _render_card("C₂H₆/C₂H₂", _safe_fmt(c2h6_c2h2, ".3f"), "Doernenburg R4", "blue")

    # Score interpretation
    interpretation = _build_score_interpretation(gas_dict) if gas_dict else ""
    if interpretation:
        st.markdown(
            f"""<div class="insight-box">
                <div class="insight-box-title">Why this score?</div>
                <div class="insight-box-body">{interpretation}</div>
            </div>""",
            unsafe_allow_html=True,
        )

    # ===================== Diagnostic Votes =====================
    if gas_dict and isinstance(gas_dict.get("diagnostic_votes"), dict):
        votes = gas_dict["diagnostic_votes"]
        consensus_fault = gas_dict.get("consensus_fault", "Uncertain")
        mixed_components = gas_dict.get("mixed_components", [])
        if consensus_fault == "MIXED":
            components_str = " + ".join(mixed_components)
            st.markdown(f"**Consensus fault:** Mixed fault ({components_str})")
        else:
            st.markdown(f"**Consensus fault:** {consensus_fault}")
        data = []
        for method, fault in votes.items():
            unified = consensus.unify_fault(fault)
            match = "✅" if unified == consensus_fault else "❌"
            data.append({
                "Method": method.replace("_fault", "").replace("_", " ").title(),
                "Original Fault": fault,
                "Unified Fault": unified,
                "Match Consensus": match
            })
        vote_df = pd.DataFrame(data)
        st.subheader("Diagnostic Votes vs. Consensus")
        st.caption(f"Consensus fault: **{consensus_fault}**")
        st.dataframe(vote_df, use_container_width=True)

    # ===================== Diagnostic Visualizations =====================
    if gas_dict:
        st.subheader("Diagnostic Visualizations")
        tab1, tab2, tab3, tab4 = st.tabs(["Duval Triangle", "Duval Pentagon", "Key Gas", "Ratio Methods"])

        with tab1:
            ch4 = gas_dict.get("ch4", 0)
            c2h4 = gas_dict.get("c2h4", 0)
            c2h2 = gas_dict.get("c2h2", 0)
            if ch4 + c2h4 + c2h2 > 0:
                tri_fault = gas_dict.get("duval_triangle_fault", None)
                fig_tri = duval_triangle.plot_duval_triangle(ch4, c2h4, c2h2, fault=tri_fault)
                if fig_tri:
                    st.pyplot(fig_tri)
                else:
                    st.warning("Insufficient data for Duval Triangle.")
            else:
                st.info("No hydrocarbon data for Duval Triangle.")

        with tab2:
            h2 = gas_dict.get("h2", 0)
            c2h6 = gas_dict.get("c2h6", 0)
            if h2 + ch4 + c2h6 + c2h4 + c2h2 > 0:
                p1_fault = gas_dict.get("fault_p1", None)
                p2_fault = gas_dict.get("duval_pentagon_fault", None)
                fig_pent = duval_pentagon.plot_pentagon_dual(h2, ch4, c2h6, c2h4, c2h2,
                                                             fault_p1=p1_fault, fault_p2=p2_fault)
                if fig_pent:
                    st.pyplot(fig_pent)
                else:
                    st.warning("Insufficient data for Duval Pentagon.")
            else:
                st.info("No data for Duval Pentagon.")

        with tab3:
            st.caption("Key Gas Method – Relative Gas Concentrations")
            gas_keys = ["h2", "ch4", "c2h6", "c2h4", "c2h2", "co"]
            gas_labels = ["H₂", "CH₄", "C₂H₆", "C₂H₄", "C₂H₂", "CO"]
            values = [gas_dict.get(k, 0) for k in gas_keys]
            total = sum(values)
            if total > 0:
                pcts = [v / total * 100 for v in values]
                df_gas = pd.DataFrame({"Gas": gas_labels, "Percentage": pcts})
                fig_kg, ax_kg = plt.subplots(figsize=(6, 4))
                bars = ax_kg.bar(df_gas["Gas"], df_gas["Percentage"],
                                 color=["#ff6b6b", "#4ecdc4", "#45b7d1", "#f9ca24", "#6c5ce7", "#a29bfe"])
                ax_kg.set_ylabel("Relative %")
                ax_kg.set_title(f"Key Gas Distribution – Fault: {gas_dict.get('keygas_fault', 'N/A')}")
                for bar, pct in zip(bars, pcts):
                    ax_kg.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                               f"{pct:.1f}%", ha='center', fontsize=9)
                st.pyplot(fig_kg)
            else:
                st.info("Insufficient gas data.")

        with tab4:
            st.caption("Ratio Methods – comparison table")
            # Sử dụng _safe_fmt đã định nghĩa ở ngoài
            def safe_fmt(val):
                return _safe_fmt(val, ".3f")
            
            ratio_rows = []
            ratio_rows.append({
                "Ratio": "CH₄/H₂",
                "Rogers": safe_fmt(gas_dict.get("r1_ch4_h2")),
                "IEC 60599": safe_fmt(gas_dict.get("iec_r2_ch4_h2")),
                "Doernenburg": safe_fmt(gas_dict.get("dr_r1_ch4_h2"))
            })
            ratio_rows.append({
                "Ratio": "C₂H₂/C₂H₄",
                "Rogers": safe_fmt(gas_dict.get("r2_c2h2_c2h4")),
                "IEC 60599": safe_fmt(gas_dict.get("iec_r1_c2h2_c2h4")),
                "Doernenburg": safe_fmt(gas_dict.get("dr_r2_c2h2_c2h4"))
            })
            ratio_rows.append({
                "Ratio": "C₂H₄/C₂H₆",
                "Rogers": safe_fmt(gas_dict.get("r3_c2h4_c2h6")),
                "IEC 60599": safe_fmt(gas_dict.get("iec_r3_c2h4_c2h6")),
                "Doernenburg": "—"
            })
            ratio_rows.append({
                "Ratio": "C₂H₂/CH₄",
                "Rogers": "—",
                "IEC 60599": "—",
                "Doernenburg": safe_fmt(gas_dict.get("dr_r3_c2h2_ch4"))
            })
            ratio_rows.append({
                "Ratio": "C₂H₆/C₂H₂",
                "Rogers": "—",
                "IEC 60599": "—",
                "Doernenburg": safe_fmt(gas_dict.get("dr_r4_c2h6_c2h2"))
            })
            ratio_rows.append({
                "Ratio": "**Fault**",
                "Rogers": gas_dict.get("rogers_fault", "N/A"),
                "IEC 60599": gas_dict.get("iec_fault", "N/A"),
                "Doernenburg": gas_dict.get("doernenburg_fault", "N/A")
            })
            
            df_ratios = pd.DataFrame(ratio_rows).set_index("Ratio")
            st.dataframe(df_ratios, use_container_width=True)

    # ===================== Transformer Ranking =====================
    if not summary_df.empty:
        st.subheader("Transformer Ranking")
        ranking_cols = ["rank", "transformer_id", "latest_sample_day", "latest_score",
                        "severity", "fault_type", "trend", "priority_score",
                        "priority_label", "recommended_action"]
        ranking_cols = [col for col in ranking_cols if col in summary_df.columns]
        _render_dataframe(summary_df[ranking_cols], height=360)

    # ===================== Status Cards =====================
    st.subheader("Transformer Status Cards")
    top_rows = pred_df.head(3)
    cards_cols = st.columns(max(1, min(3, len(top_rows))))
    for i, (_, row) in enumerate(top_rows.iterrows()):
        with cards_cols[i % len(cards_cols)]:
            _render_card(str(row["transformer_id"]), f"{row['pred_ensemble'] * 100:.1f}",
                         f"{row['severity']} — {row['fault_type']}",
                         _status_accent(row["severity"]))

    # ===================== Prediction Results =====================
    st.subheader("Prediction Results")
    prediction_cols = ["row_index", "transformer_id", "pred_lgb", "pred_seq",
                       "pred_ensemble", "severity", "fault_type"]
    prediction_cols = [c for c in prediction_cols if c in pred_df.columns]
    _render_dataframe(pred_df[prediction_cols], height=360)

    # ===================== Severity Distribution =====================
    st.subheader("Severity Distribution")
    if "severity" in pred_df.columns:
        severity_counts = pred_df["severity"].value_counts().reset_index()
        severity_counts.columns = ["severity", "count"]
        severity_chart = alt.Chart(severity_counts).mark_bar(cornerRadiusTopLeft=6, cornerRadiusTopRight=6).encode(
            x=alt.X("severity:O", title="Severity"),
            y=alt.Y("count:Q", title="Count"),
            color=alt.Color("severity:N", legend=None),
            tooltip=["severity:N", "count:Q"],
        ).properties(height=220)
        _render_chart(severity_chart)

    # ===================== Time Trend =====================
    st.subheader("Time Trend")
    if selected_transformer:
        top_ids = [str(selected_transformer)]
        st.caption(f"Showing trend for selected transformer: {selected_transformer}")
    else:
        pred_df["pred_ensemble"] = pd.to_numeric(pred_df["pred_ensemble"], errors="coerce")
        transformer_scores = pred_df.groupby("transformer_id")["pred_ensemble"].max().reset_index()
        transformer_scores = transformer_scores.sort_values("pred_ensemble", ascending=False)
        max_transformers = len(transformer_scores)
        default_top_n = min(5, max_transformers)
        top_n = st.slider("Number of top severity transformers to display",
                          min_value=1, max_value=max(1, max_transformers),
                          value=default_top_n, key="trend_top_n")
        top_ids = transformer_scores.head(top_n)["transformer_id"].astype(str).tolist()

    trend_df = _build_trend_frame(payload, selected_transformer, transformer_ids=top_ids)
    if trend_df.empty:
        st.info("No valid time-series score data is available for the current selection.")
    else:
        tooltip_cols = ["Sample Day:T", "transformer_id:N", "pred_ensemble:Q"]
        trend_chart = alt.Chart(trend_df).mark_line(point=True).encode(
            x=alt.X("Sample Day:T", title="Sample Day"),
            y=alt.Y("pred_ensemble:Q", title="Ensemble Score"),
            color="transformer_id:N",
            tooltip=tooltip_cols,
        ).properties(height=260)
        _render_chart(trend_chart)

    # ===================== Data Preview =====================
    st.subheader("Data Preview")
    _render_dataframe(pd.DataFrame(preview_rows), height=320)

    # ===================== Model Explanation =====================
    st.subheader("Model Explanation")
    selected = st.selectbox(
        "Choose transformer row",
        options=list(range(len(predictions))),
        format_func=lambda i: f"Row {i + 1} - {predictions[i].get('transformer_id', '')}",
        key="results_explanation_selector",
    )
    selected_row = predictions[selected]
    st.write(selected_row.get("reason", "No explanation was returned."))

    feature_df = pd.DataFrame(selected_row.get("top_features", []))
    if not feature_df.empty and "feature" in feature_df.columns and "importance" in feature_df.columns:
        feature_df = feature_df[["feature", "importance"]].copy()
        feature_df["importance"] = pd.to_numeric(feature_df["importance"], errors="coerce").fillna(0.0)
        feature_df["importance_abs"] = feature_df["importance"].abs()
        feature_df = feature_df.sort_values("importance_abs", ascending=False).head(8)
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