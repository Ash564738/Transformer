# chat.py
# Rule-based "DGA Assistant" answer generator used by POST /chat.
# Extracted from the original Streamlit components/chat.py (generate_chat_answer
# was already a pure function with no st.session_state dependency); the
# Streamlit-only rendering (_ask, render_chat, SUGGESTIONS) is now the
# frontend's concern (frontend/src/components/chat/floating-chat.tsx).
import logging

logger = logging.getLogger(__name__)


def _find_transformer(question: str, summaries: list) -> dict | None:
    q = question.lower()

    for item in summaries:
        tid = str(item.get("transformer_id", "")).lower()
        name = str(item.get("NAME", item.get("name", ""))).lower()
        if tid and tid in q:
            return item
        if name and name != "unknown" and name in q:
            return item

    if summaries and any(token in q for token in ["this transformer", "severe", "why"]):
        return summaries[0]

    return None


def _extract_gas_trend(summary_item, gas_name):
    feats = summary_item.get("features", {}) or {}
    rate = feats.get(f"{gas_name}_rate_per_day")
    roll_mean = feats.get(f"{gas_name}_roll3_mean")
    slope = feats.get(f"{gas_name}_roll3_slope")

    parts = []
    if rate is not None:
        parts.append(f"rate {rate:.2f} ppm/day")
    if roll_mean is not None:
        parts.append(f"3-sample avg {roll_mean:.1f} ppm")
    if slope is not None:
        direction = "up" if slope > 0 else "down" if slope < 0 else "flat"
        parts.append(f"slope {slope:.2f} ({direction})")

    return "; ".join(parts) if parts else "no trend data available"


def generate_chat_answer(question: str, context=None) -> str:
    q = question.lower().strip()
    context = context or {}

    transformer_summary = context.get("transformer_summary", [])
    dataset_summary = context.get("dataset_summary", {})
    top_ranked = transformer_summary[0] if transformer_summary else None
    matched_transformer = _find_transformer(question, transformer_summary)

    gas_keywords = {
        "h2": ["h2", "hydrogen", "h₂"],
        "ch4": ["ch4", "methane", "ch₄"],
        "c2h6": ["c2h6", "ethane", "c₂h₆"],
        "c2h4": ["c2h4", "ethylene", "c₂h₄"],
        "c2h2": ["c2h2", "acetylene", "c₂h₂"],
        "co": ["co", "carbon monoxide"],
        "co2": ["co2", "carbon dioxide", "co₂"],
        "tdcg": ["tdcg", "total combustible gas"],
    }

    for gas, aliases in gas_keywords.items():
        if any(alias in q for alias in aliases) and any(k in q for k in ["trend", "xu hướng", "gần đây", "recent", "change"]):
            if matched_transformer:
                info = _extract_gas_trend(matched_transformer, gas)
                return f"Trend for {gas.upper()} of transformer {matched_transformer.get('transformer_id', '')}: {info}."
            return f"Please specify a transformer to see {gas.upper()} trend (e.g., 'transformer T-01 H2 trend')."

    if ("compare" in q or "so sánh" in q) and len(transformer_summary) >= 2:
        ids = []
        for item in transformer_summary:
            tid = str(item.get("transformer_id", ""))
            if tid.lower() in q:
                ids.append(item)

        if len(ids) == 2:
            a, b = ids[0], ids[1]
            return (
                f"Comparison between {a['transformer_id']} and {b['transformer_id']}:\n"
                f"- Severity: {a['severity']} (score {a['latest_score']:.3f}) vs {b['severity']} (score {b['latest_score']:.3f})\n"
                f"- Fault type: {a['fault_type']} vs {b['fault_type']}\n"
                f"- Trend: {a['trend']} vs {b['trend']}\n"
                f"- Recommended action for {a['transformer_id']}: {a['recommended_action']}\n"
                f"- Recommended action for {b['transformer_id']}: {b['recommended_action']}"
            )
        if len(ids) == 0:
            return "Please specify two transformer IDs to compare, e.g., 'compare T-01 and T-02'."

    if any(k in q for k in ["co2/co", "co2 co", "co₂/co", "co2-co", "co2co"]):
        if any(w in q for w in ["low", "thấp", "thap", "decrease", "small", "meaning", "indicate"]):
            return (
                "A low CO₂/CO ratio (typically below 3–7) suggests that cellulose insulation (paper, pressboard) "
                "is being thermally stressed or degraded. CO is produced by cellulose overheating, while CO₂ is also "
                "generated. When the ratio drops, it often indicates a hot spot in the paper insulation. "
                "Combined with other gases, it can strengthen the case for cellulose involvement."
            )
        return (
            "CO₂/CO is a diagnostic ratio for cellulose degradation. A high ratio may indicate normal ageing, "
            "while a low ratio often suggests hot-spot overheating of paper insulation."
        )

    if any(k in q for k in ["fault type", "fault_type", "t3", "d2", "t1", "t2", "pd", "d1"]):
        if matched_transformer:
            ft = matched_transformer.get("fault_type", "General degradation")
            ft_lower = ft.lower()

            if "t3" in ft_lower or "thermal high" in ft_lower:
                return (
                    f"This transformer is classified as {ft}. "
                    "High-temperature thermal faults (T3) are associated with very high C₂H₄, often with C₂H₂, "
                    "and a high C₂H₄/C₂H₆ ratio, pointing to severe localized overheating."
                )
            if "d2" in ft_lower or "discharge" in ft_lower:
                return (
                    f"This transformer is classified as {ft}. "
                    "Discharge/arcing (D1/D2) is associated with significant acetylene (C₂H₂) generation, "
                    "a high C₂H₂/C₂H₄ ratio, and discharge-like gas signatures."
                )
            if "pd" in ft_lower:
                return (
                    f"This transformer is classified as {ft}. "
                    "Partial discharge typically produces hydrogen (H₂) with very low acetylene and characteristic ratio patterns."
                )
            if "cellulose" in ft_lower:
                return (
                    f"This transformer is classified as {ft}. "
                    "Cellulose overheating tends to yield elevated CO and a depressed CO₂/CO ratio."
                )
            return (
                f"Fault type '{ft}' is determined from rule-based gas thresholds, ratios, and model-derived indicators. "
                "Ask about a specific transformer for a more targeted explanation."
            )

        return (
            "I can explain fault types such as PD, D1, D2, T1, T2, T3, and cellulose degradation. "
            "Please choose a transformer or ask about a specific fault class."
        )

    if any(k in q for k in ["most critical", "highest risk", "top risk", "most dangerous", "worst transformer"]):
        if top_ranked:
            return (
                f"The highest-risk transformer is {top_ranked.get('transformer_id', 'Unknown')} "
                f"with severity {top_ranked.get('severity', 'Unknown')}, "
                f"fault type {top_ranked.get('fault_type', 'Unknown')}, "
                f"and latest score {top_ranked.get('latest_score', 'N/A'):.3f}."
            )
        return "No current ranking payload is available. Upload data and run prediction first."

    if any(k in q for k in ["how many transformers", "number of transformers", "transformer count"]):
        total = dataset_summary.get("total_transformers")
        if total is not None:
            return f"The current dataset contains {total} transformers."
        if transformer_summary:
            return f"The current dataset contains {len(transformer_summary)} transformers."
        return "No dataset summary is available for this question."

    if any(k in q for k in ["ensemble score", "how is the score calculated", "pred_ensemble"]):
        return (
            "The system combines a LightGBM model and a temporal LSTM-based model. "
            "The final pred_ensemble score is the weighted degradation ranking score used for severity and prioritization."
        )

    if any(k in q for k in ["need attention", "which need attention", "watch list", "who needs attention"]):
        at_risk = [s for s in transformer_summary if s.get("severity") != "Low"][:5]
        if not at_risk:
            return "No transformers currently need attention — all are within normal range."
        listed = ", ".join(f"{s['transformer_id']} ({s['severity']}, score {s['latest_score']:.2f})" for s in at_risk)
        return f"{len(at_risk)} transformer(s) need attention: {listed}."

    if any(k in q for k in ["dga", "gas", "c2h2", "h2", "ch4", "co", "co2"]) and not matched_transformer:
        return (
            "DGA uses dissolved gases to infer transformer condition. "
            "H₂ is commonly linked to partial discharge, C₂H₂ to discharge/arcing, "
            "C₂H₄ to higher-temperature thermal faults, and CO/CO₂ to cellulose insulation degradation."
        )

    if matched_transformer and any(k in q for k in ["why", "severe", "critical", "explain"]):
        reason = matched_transformer.get("reason")
        if reason:
            return (
                f"Transformer {matched_transformer.get('transformer_id', 'Unknown')} is labeled "
                f"{matched_transformer.get('severity', 'Unknown')} with fault type "
                f"{matched_transformer.get('fault_type', 'Unknown')}. {reason}"
            )
        return (
            f"Transformer {matched_transformer.get('transformer_id', 'Unknown')} is labeled "
            f"{matched_transformer.get('severity', 'Unknown')} with fault type "
            f"{matched_transformer.get('fault_type', 'Unknown')}. Latest risk score "
            f"{matched_transformer.get('latest_score', 0):.2f}, trend {matched_transformer.get('trend', 'stable')}. "
            f"{matched_transformer.get('recommended_action', '')}"
        )

    if "mixed" in q and matched_transformer:
        ft = matched_transformer.get("fault_type", "")
        if ft.lower() == "mixed":
            return (
                "MIXED means the diagnostic methods (Key Gas, IEC, Rogers, Duval Triangle, Duval Pentagon) "
                "disagree on the fault type. This often occurs when gas concentrations are borderline or incomplete. "
                "Ask for a specific transformer to see the detailed votes."
            )

    return (
        "I can answer questions about DGA gas interpretation, transformer degradation ranking, "
        "severity, fault type, feature importance, time trends, CO₂/CO meaning, and fault assignment rationale. "
        "You can also ask for gas trends or compare transformers."
    )
