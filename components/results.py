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

# ------------------------------------------------------------
# Helper functions
# ------------------------------------------------------------
def _safe_fmt(val, fmt_spec=".3f", na_rep="—"):
    if val is None:
        return na_rep
    if isinstance(val, (float, np.floating)):
        if np.isnan(val) or np.isinf(val):
            return na_rep
    try:
        return format(val, fmt_spec)
    except (ValueError, TypeError):
        return na_rep

def _render_card(title, value, subtitle, accent="blue"):
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
        st.altair_chart(chart, use_container_width=True)
    except TypeError:
        st.altair_chart(chart)

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

def _build_trend_from_predictions(pred_df, top_ids):
    if pred_df.empty or "sample_day" not in pred_df.columns:
        return pd.DataFrame()
    trend_df = pred_df[pred_df["transformer_id"].astype(str).isin(top_ids)].copy()
    trend_df["Sample Day"] = pd.to_datetime(trend_df["sample_day"], errors="coerce")
    trend_df["pred_ensemble"] = pd.to_numeric(trend_df.get("pred_ensemble", pd.Series([0])), errors="coerce")
    return trend_df.dropna(subset=["Sample Day", "pred_ensemble"]).sort_values("Sample Day")

def _build_trend_frame(payload, top_ids):
    records = []
    ts_data = payload.get("transformer_timeseries", {})
    for tid, series in ts_data.items():
        if str(tid) not in top_ids:
            continue
        for row in series:
            records.append({"transformer_id": tid, **row})
    trend_df = pd.DataFrame(records)
    if not trend_df.empty and "Sample Day" in trend_df.columns and "pred_ensemble" in trend_df.columns:
        trend_df["Sample Day"] = pd.to_datetime(trend_df["Sample Day"], errors="coerce")
        trend_df["pred_ensemble"] = pd.to_numeric(trend_df["pred_ensemble"], errors="coerce")
        return trend_df.dropna(subset=["Sample Day", "pred_ensemble"]).sort_values("Sample Day")
    return pd.DataFrame()

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
        return "No strong diagnostic signal detected."
    return "Score driven by: " + ", ".join(parts) + "."

def _build_gas_summary_table(gas_dict):
    rows = []
    for gas, label in [("h2", "H₂"), ("ch4", "CH₄"), ("c2h2", "C₂H₂"), ("c2h4", "C₂H₄"), ("co", "CO"), ("co2", "CO₂")]:
        rows.append({"Parameter": label, "Value (ppm)": _safe_fmt(gas_dict.get(gas, 0), ".1f")})
    tdcg = gas_dict.get("tdcg", 0)
    if tdcg == 0:
        tdcg = sum(gas_dict.get(g, 0) for g in ["h2","ch4","c2h6","c2h4","c2h2","co"])
    rows.append({"Parameter": "TDCG", "Value (ppm)": _safe_fmt(tdcg, ".1f")})
    ratios = [
        ("CH₄/H₂", "iec_r2_ch4_h2", "r1_ch4_h2"),
        ("C₂H₂/C₂H₄", "iec_r1_c2h2_c2h4", "r2_c2h2_c2h4"),
        ("C₂H₄/C₂H₆", "iec_r3_c2h4_c2h6", "r3_c2h4_c2h6"),
        ("CO₂/CO", "ratio_co2_co", None),
    ]
    for name, key1, key2 in ratios:
        val = gas_dict.get(key1)
        if val is None and key2:
            val = gas_dict.get(key2)
        rows.append({"Parameter": name, "Value (ppm)": _safe_fmt(val, ".3f")})
    for rate, label in [("h2_rate_per_day", "H₂ rate"), ("c2h2_rate_per_day", "C₂H₂ rate"), ("tdcg_rate_per_day", "TCG rate")]:
        val = gas_dict.get(rate)
        rows.append({"Parameter": label, "Value (ppm)": _safe_fmt(val, ".2f") + " ppm/day" if val else "—"})
    return pd.DataFrame(rows)

# ------------------------------------------------------------
# Main rendering functions
# ------------------------------------------------------------
def render_results(payload, selected_transformer=None):
    predictions_all = payload.get("predictions", [])
    rows_all = payload.get("rows", [])
    preview_rows_all = payload.get("preview_rows", [])
    transformer_summary_all = payload.get("transformer_summary", [])

    if not predictions_all:
        st.warning("No results are available for the selected transformer.")
        return

    pred_df_all = pd.DataFrame(predictions_all)
    summary_df_all = pd.DataFrame(transformer_summary_all)

    # ---- Tính toán số transformer nghiêm trọng (không phải số dòng) ----
    if not summary_df_all.empty and "severity" in summary_df_all.columns:
        severe_count = (summary_df_all["severity"] == "Severe").sum()
    else:
        if "pred_ensemble" in pred_df_all.columns and "sample_day" in pred_df_all.columns:
            pred_df_all["pred_ensemble"] = pd.to_numeric(pred_df_all["pred_ensemble"], errors="coerce")
            latest_idx = pred_df_all.groupby("transformer_id")["sample_day"].idxmax()
            latest_scores = pred_df_all.loc[latest_idx, "pred_ensemble"]
            severe_count = (latest_scores > 0.7).sum()
        else:
            severe_count = 0

    # ---- Dashboard Overview ----
    _render_dashboard(payload, summary_df_all, pred_df_all, severe_count)

    # ---- Severity Ranking ----
    ranking_df = _prepare_ranking(summary_df_all, pred_df_all)
    if not ranking_df.empty:
        st.markdown("---")
        _render_severity_ranking(ranking_df)
        top_transformer = ranking_df.iloc[0]["transformer_id"]
        if not selected_transformer:
            selected_transformer = top_transformer
    else:
        top_transformer = None

    # ---- Analytics (distribution + trend) ----
    st.markdown("---")
    _render_analytics(pred_df_all, payload, ranking_df)

    # ---- Transformer Detail ----
    st.markdown("---")
    _render_transformer_detail(payload, predictions_all, rows_all, selected_transformer)

    # ---- Advanced (prediction results, raw data, explanation) ----
    st.markdown("---")
    _render_advanced(payload, predictions_all, rows_all, preview_rows_all)

def _render_dashboard(payload, summary_df, pred_df, severe_count):
    total_transformers = payload.get("dataset_summary", {}).get("total_transformers") or (
        summary_df["transformer_id"].nunique() if not summary_df.empty else 0
    )
    avg_score = pd.to_numeric(pred_df.get("pred_ensemble", pd.Series([0])), errors="coerce").mean()

    st.subheader("System Overview")
    col1, col2, col3 = st.columns(3)
    with col1:
        _render_card("Total Transformers", str(total_transformers), "Assets in dataset", "green")
    with col2:
        _render_card("Average Score", f"{avg_score:.2f}", "Degradation index", "amber")
    with col3:
        _render_card("Severe Units", str(severe_count), "Transformers with score > 0.7", "red")

def _prepare_ranking(summary_df, pred_df):
    """Tạo DataFrame ranking từ summary hoặc pred_df."""
    if not summary_df.empty:
        ranking_df = summary_df.copy()
        if "rank" not in ranking_df.columns:
            ranking_df = ranking_df.sort_values("latest_score", ascending=False)
            ranking_df["rank"] = range(1, len(ranking_df) + 1)
    else:
        if "pred_ensemble" in pred_df.columns:
            pred_df["pred_ensemble"] = pd.to_numeric(pred_df["pred_ensemble"], errors="coerce")
            ranking_df = pred_df.groupby("transformer_id").agg(
                latest_score=("pred_ensemble", "max"),
                latest_sample_day=("sample_day", "max")
            ).reset_index()
            ranking_df = ranking_df.sort_values("latest_score", ascending=False)
            ranking_df["rank"] = range(1, len(ranking_df) + 1)
            def _sev(score):
                if score >= 0.7: return "Severe"
                elif score >= 0.4: return "Moderate"
                else: return "Low"
            ranking_df["severity"] = ranking_df["latest_score"].apply(_sev)
        else:
            ranking_df = pd.DataFrame()
    return ranking_df

def _render_severity_ranking(ranking_df):
    st.subheader("Transformer Severity Ranking")
    ranking_cols = ["rank", "transformer_id", "latest_score", "severity",
                    "fault_type", "trend", "priority_score", "recommended_action"]
    ranking_cols = [c for c in ranking_cols if c in ranking_df.columns]

    def highlight_severity(val):
        color = 'red' if val == 'Severe' else 'orange' if val == 'Moderate' else 'green'
        return f'background-color: {color}; color: white'

    styled_df = ranking_df[ranking_cols].style.map(highlight_severity, subset=['severity'])
    st.dataframe(styled_df, width='stretch', height=400)

def _render_analytics(pred_df, payload, ranking_df):
    col_left, col_right = st.columns(2)
    with col_left:
        st.subheader("Severity Distribution")
        if "severity" in pred_df.columns:
            severity_counts = pred_df["severity"].value_counts().reset_index()
            severity_counts.columns = ["severity", "count"]
            chart = alt.Chart(severity_counts).mark_bar().encode(
                x=alt.X("severity:N", title=None, sort=["Low", "Moderate", "Severe"]),
                y="count:Q",
                color=alt.Color("severity:N", legend=None,
                                scale=alt.Scale(domain=["Low", "Moderate", "Severe"],
                                                range=["green", "orange", "red"])),
                tooltip=["severity", "count"]
            ).properties(height=220)
            _render_chart(chart)

    with col_right:
        st.subheader("Trend of Top 5 Severity")
        if not ranking_df.empty:
            top_ids = ranking_df.head(5)["transformer_id"].astype(str).tolist()
        else:
            top_ids = []
        trend_df = _build_trend_frame(payload, top_ids)
        if trend_df.empty:
            trend_df = _build_trend_from_predictions(pred_df, top_ids)
        if not trend_df.empty:
            chart = alt.Chart(trend_df).mark_line(point=True).encode(
                x=alt.X("Sample Day:T", title="Sample Day"),
                y=alt.Y("pred_ensemble:Q", title="Ensemble Score"),
                color="transformer_id:N",
                tooltip=["Sample Day", "transformer_id", "pred_ensemble"]
            ).properties(height=220)
            _render_chart(chart)
        else:
            st.info("No trend data available.")

def _render_transformer_detail(payload, predictions_all, rows_all, selected_transformer):
    st.subheader("Transformer Detail")
    all_ids = sorted(list(set(str(p.get("transformer_id", "")) for p in predictions_all)))
    if not all_ids:
        st.info("No transformer data.")
        return

    if selected_transformer and str(selected_transformer) in all_ids:
        default_idx = all_ids.index(str(selected_transformer))
    else:
        default_idx = 0
    sel_id = st.selectbox("Select transformer", options=all_ids, index=default_idx)

    detail_payload = {
        "predictions": predictions_all,
        "rows": rows_all,
        "preview_rows": [],
        "transformer_summary": []
    }
    detail_predictions, detail_rows, _, _ = _filter_payload(detail_payload, sel_id)

    if not detail_predictions or not detail_rows:
        st.warning("No data for selected transformer.")
        return

    # Lấy mẫu mới nhất
    pred = detail_predictions[0]
    raw = detail_rows[0]
    gas_dict = raw if isinstance(raw, dict) else {}

    severity_label = pred.get("severity", "")
    consensus_fault = pred.get("consensus_fault", gas_dict.get("consensus_fault", ""))
    score = pred.get("pred_ensemble", 0)

    # ---- Status cards ----
    col1, col2, col3 = st.columns(3)
    with col1:
        _render_card("Severity", severity_label, "", _status_accent(severity_label))
    with col2:
        _render_card("Score", f"{score:.2f}", "Ensemble prediction", "amber")
    with col3:
        _render_card("Consensus Fault", consensus_fault, "", "blue")

    # Key gas cards (chỉ các khí quan trọng)
    st.subheader("Key Gas Indicators")
    col1, col2, col3, col4, col5, col6 = st.columns(6)
    with col1:
        _render_card("H₂", f"{_safe_fmt(gas_dict.get('h2', 0), '.1f')} ppm", "Hydrogen", "blue")
    with col2:
        _render_card("CH₄", f"{_safe_fmt(gas_dict.get('ch4', 0), '.1f')} ppm", "Methane", "blue")
    with col3:
        _render_card("C₂H₂", f"{_safe_fmt(gas_dict.get('c2h2', 0), '.1f')} ppm", "Acetylene", "blue")
    with col4:
        _render_card("C₂H₄", f"{_safe_fmt(gas_dict.get('c2h4', 0), '.1f')} ppm", "Ethylene", "blue")
    with col5:
        _render_card("CO", f"{_safe_fmt(gas_dict.get('co', 0), '.1f')} ppm", "Carbon Monoxide", "blue")
    with col6:
        tdcg = gas_dict.get("tdcg", 0)
        if tdcg == 0:
            tdcg = sum(gas_dict.get(g, 0) for g in ["h2","ch4","c2h6","c2h4","c2h2","co"])
        _render_card("TDCG", f"{_safe_fmt(tdcg, '.1f')} ppm", "Total Combustible Gas", "amber")

    # Score interpretation
    interpretation = _build_score_interpretation(gas_dict) if gas_dict else ""
    if interpretation:
        st.info(interpretation)

    # Gas Summary (expander)
    with st.expander("Gas Summary", expanded=False):
        if gas_dict:
            gas_table = _build_gas_summary_table(gas_dict)
            st.dataframe(gas_table, width='stretch', hide_index=True)
        else:
            st.write("No gas data available.")

    # Diagnostic Votes (expander)
    with st.expander("Diagnostic Votes & Consensus", expanded=False):
        if gas_dict and isinstance(gas_dict.get("diagnostic_votes"), dict):
            votes = gas_dict["diagnostic_votes"]
            consensus_fault = gas_dict.get("consensus_fault", "Uncertain")
            mixed_components = gas_dict.get("mixed_components", [])
            if consensus_fault == "MIXED":
                st.markdown(f"**Consensus fault:** Mixed fault ({' + '.join(mixed_components)})")
            else:
                st.markdown(f"**Consensus fault:** {consensus_fault}")
            data = []
            for method, fault in votes.items():
                unified = consensus.unify_fault(fault)
                match = "Yes" if unified == consensus_fault else "No"
                data.append({
                    "Method": method.replace("_fault", "").replace("_", " ").title(),
                    "Original Fault": fault,
                    "Unified Fault": unified,
                    "Match Consensus": match
                })
            st.dataframe(pd.DataFrame(data), width='stretch')
        else:
            st.write("No vote information.")

    # Diagnostic Charts (expander)
    if gas_dict:
        with st.expander("Diagnostic Charts", expanded=False):
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
                    st.info("No hydrocarbon data for Duval Triangle.")
            with tab2:
                h2 = gas_dict.get("h2", 0)
                c2h6 = gas_dict.get("c2h6", 0)
                if sum([h2, ch4, c2h6, c2h4, c2h2]) > 0:
                    p1_fault = gas_dict.get("fault_p1", None)
                    p2_fault = gas_dict.get("duval_pentagon_fault", None)
                    fig_pent = duval_pentagon.plot_pentagon_dual(h2, ch4, c2h6, c2h4, c2h2,
                                                                 fault_p1=p1_fault, fault_p2=p2_fault)
                    if fig_pent:
                        st.pyplot(fig_pent)
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
                def safe_fmt(val):
                    return _safe_fmt(val, ".3f")
                ratio_rows = [
                    {"Ratio": "CH₄/H₂", "Rogers": safe_fmt(gas_dict.get("r1_ch4_h2")),
                     "IEC 60599": safe_fmt(gas_dict.get("iec_r2_ch4_h2")),
                     "Doernenburg": safe_fmt(gas_dict.get("dr_r1_ch4_h2"))},
                    {"Ratio": "C₂H₂/C₂H₄", "Rogers": safe_fmt(gas_dict.get("r2_c2h2_c2h4")),
                     "IEC 60599": safe_fmt(gas_dict.get("iec_r1_c2h2_c2h4")),
                     "Doernenburg": safe_fmt(gas_dict.get("dr_r2_c2h2_c2h4"))},
                    {"Ratio": "C₂H₄/C₂H₆", "Rogers": safe_fmt(gas_dict.get("r3_c2h4_c2h6")),
                     "IEC 60599": safe_fmt(gas_dict.get("iec_r3_c2h4_c2h6")),
                     "Doernenburg": "—"},
                    {"Ratio": "C₂H₂/CH₄", "Rogers": "—", "IEC 60599": "—",
                     "Doernenburg": safe_fmt(gas_dict.get("dr_r3_c2h2_ch4"))},
                    {"Ratio": "C₂H₆/C₂H₂", "Rogers": "—", "IEC 60599": "—",
                     "Doernenburg": safe_fmt(gas_dict.get("dr_r4_c2h6_c2h2"))},
                    {"Ratio": "**Fault**", "Rogers": gas_dict.get("rogers_fault", "N/A"),
                     "IEC 60599": gas_dict.get("iec_fault", "N/A"),
                     "Doernenburg": gas_dict.get("doernenburg_fault", "N/A")}
                ]
                st.dataframe(pd.DataFrame(ratio_rows).set_index("Ratio"), width='stretch')

def _render_advanced(payload, predictions_all, rows_all, preview_rows_all):
    with st.expander("Prediction Results", expanded=False):
        pred_df = pd.DataFrame(predictions_all)
        prediction_cols = ["row_index", "transformer_id", "pred_ensemble", "severity", "fault_type"]
        prediction_cols = [c for c in prediction_cols if c in pred_df.columns]
        st.dataframe(pred_df[prediction_cols], width='stretch', height=300)

    with st.expander("Raw Data Preview", expanded=False):
        st.dataframe(pd.DataFrame(preview_rows_all), width='stretch')

    with st.expander("Model Explanation", expanded=False):
        exp_idx = st.selectbox(
            "Choose transformer for explanation",
            options=list(range(len(predictions_all))),
            format_func=lambda i: f"{predictions_all[i].get('transformer_id', '')}",
            key="results_explanation_selector"
        )
        exp_row = predictions_all[exp_idx]
        st.write(exp_row.get("reason", "No explanation available."))
        features = exp_row.get("top_features", [])
        if features:
            feat_df = pd.DataFrame(features)
            if "feature" in feat_df.columns and "importance" in feat_df.columns:
                feat_df["importance"] = pd.to_numeric(feat_df["importance"], errors="coerce")
                feat_df = feat_df.sort_values("importance", ascending=False).head(10)
                chart = alt.Chart(feat_df).mark_bar().encode(
                    x=alt.X("importance:Q", title="Importance"),
                    y=alt.Y("feature:N", sort="-x", title=None),
                    color=alt.Color("importance:Q", legend=None),
                    tooltip=["feature", "importance"]
                ).properties(height=250)
                _render_chart(chart)
        raw_row_exp = rows_all[exp_idx] if exp_idx < len(rows_all) else {}
        if raw_row_exp:
            st.caption("Raw input row")
            st.json(raw_row_exp)