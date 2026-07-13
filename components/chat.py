# components/chat.py
import logging

import streamlit as st

logger = logging.getLogger(__name__)

SUGGESTIONS = [
    "Which transformer is the most critical right now?",
    "Why is this transformer labeled Severe?",
    "What does low CO₂/CO indicate?",
    "Why is the fault type T3 or D2?",
    "How is the ensemble score calculated?",
    "What do roc_H₂ and roc_C₂H₂ tell us?",
    "Show H₂ trend for transformer T01",
    "Compare transformer A and B",
]


def _debug_enabled() -> bool:
    return bool(st.session_state.get("debug_mode", False))


def _log_debug(event: str, **kwargs) -> None:
    if _debug_enabled():
        logger.info("[chat] %s | %s", event, kwargs if kwargs else "{}")


def _build_context():
    payload = st.session_state.get("payload", {})
    if not isinstance(payload, dict):
        _log_debug("build_context:invalid_payload_type", payload_type=type(payload).__name__)
        return None

    context = payload.get("chat_context_payload") or {}
    selected_transformer = st.session_state.get("selected_transformer")

    if selected_transformer and context.get("transformer_summary"):
        selected = str(selected_transformer)
        selected_rows = [
            row for row in context["transformer_summary"]
            if str(row.get("transformer_id")) == selected
        ]
        other_rows = [
            row for row in context["transformer_summary"]
            if str(row.get("transformer_id")) != selected
        ]
        if selected_rows:
            context = {**context, "transformer_summary": selected_rows + other_rows}

    _log_debug(
        "build_context:done",
        selected_transformer=selected_transformer,
        summary_count=len(context.get("transformer_summary", [])),
        dataset_summary_keys=list((context.get("dataset_summary") or {}).keys()),
    )
    return context


def _find_transformer(question: str, summaries: list[dict]) -> dict | None:
    q = question.lower()

    for item in summaries:
        tid = str(item.get("transformer_id", "")).lower()
        name = str(item.get("NAME", "")).lower()
        if tid and tid in q:
            _log_debug("find_transformer:matched_by_id", transformer_id=item.get("transformer_id"))
            return item
        if name and name != "unknown" and name in q:
            _log_debug("find_transformer:matched_by_name", transformer_id=item.get("transformer_id"), name=name)
            return item

    if summaries and any(token in q for token in ["this transformer", "severe", "why"]):
        _log_debug("find_transformer:fallback_first_summary", transformer_id=summaries[0].get("transformer_id"))
        return summaries[0]

    _log_debug("find_transformer:no_match")
    return None


def _extract_gas_trend(summary_item, gas_name):
    feats = summary_item.get("features", {})
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

    result = "; ".join(parts) if parts else "no trend data available"
    _log_debug(
        "extract_gas_trend",
        transformer_id=summary_item.get("transformer_id"),
        gas_name=gas_name,
        result=result,
    )
    return result


def generate_chat_answer(question: str, context=None) -> str:
    q = question.lower().strip()
    context = context or {}

    transformer_summary = context.get("transformer_summary", [])
    dataset_summary = context.get("dataset_summary", {})
    top_ranked = transformer_summary[0] if transformer_summary else None
    matched_transformer = _find_transformer(question, transformer_summary)

    _log_debug(
        "generate_chat_answer:start",
        question=question,
        summary_count=len(transformer_summary),
        matched_transformer=matched_transformer.get("transformer_id") if matched_transformer else None,
    )

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
            _log_debug("branch:gas_trend", gas=gas)
            if matched_transformer:
                info = _extract_gas_trend(matched_transformer, gas)
                return f"Trend for {gas.upper()} of transformer {matched_transformer.get('transformer_id', '')}: {info}."
            return f"Please specify a transformer to see {gas.upper()} trend (e.g., 'transformer T-01 H2 trend')."

    if ("compare" in q or "so sánh" in q) and len(transformer_summary) >= 2:
        _log_debug("branch:compare")
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
        _log_debug("branch:co2_co_explanation")
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
        _log_debug("branch:fault_type")
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
        _log_debug("branch:most_critical")
        if top_ranked:
            return (
                f"The highest-risk transformer is {top_ranked.get('transformer_id', 'Unknown')} "
                f"with severity {top_ranked.get('severity', 'Unknown')}, "
                f"fault type {top_ranked.get('fault_type', 'Unknown')}, "
                f"and latest score {top_ranked.get('latest_score', 'N/A'):.3f}."
            )
        return "No current ranking payload is available. Upload data and run prediction first."

    if any(k in q for k in ["how many transformers", "number of transformers", "transformer count"]):
        _log_debug("branch:transformer_count")
        total = dataset_summary.get("total_transformers")
        if total is not None:
            return f"The current dataset contains {total} transformers."
        if transformer_summary:
            return f"The current dataset contains {len(transformer_summary)} transformers."
        return "No dataset summary is available for this question."

    if any(k in q for k in ["ensemble score", "how is the score calculated", "pred_ensemble"]):
        _log_debug("branch:ensemble_score")
        return (
            "The system combines a LightGBM model and a temporal LSTM-based model. "
            "The final pred_ensemble score is the weighted degradation ranking score used for severity and prioritization."
        )

    if any(k in q for k in ["dga", "gas", "c2h2", "h2", "ch4", "co", "co2"]) and not matched_transformer:
        _log_debug("branch:generic_dga")
        return (
            "DGA uses dissolved gases to infer transformer condition. "
            "H₂ is commonly linked to partial discharge, C₂H₂ to discharge/arcing, "
            "C₂H₄ to higher-temperature thermal faults, and CO/CO₂ to cellulose insulation degradation."
        )

    if matched_transformer and any(k in q for k in ["why", "severe", "critical", "explain"]):
        _log_debug("branch:matched_transformer_reason")
        reason = matched_transformer.get("reason")
        if reason:
            return (
                f"Transformer {matched_transformer.get('transformer_id', 'Unknown')} is labeled "
                f"{matched_transformer.get('severity', 'Unknown')} with fault type "
                f"{matched_transformer.get('fault_type', 'Unknown')}. {reason}"
            )

    _log_debug("branch:fallback")
    return (
        "I can answer questions about DGA gas interpretation, transformer degradation ranking, "
        "severity, fault type, feature importance, time trends, CO₂/CO meaning, and fault assignment rationale. "
        "You can also ask for gas trends or compare transformers."
    )


def _ask(question: str):
    if not question:
        _log_debug("ask:empty_question")
        return

    if "chat_messages" not in st.session_state:
        st.session_state.chat_messages = []

    _log_debug("ask:start", question=question, existing_messages=len(st.session_state.chat_messages))

    st.session_state.chat_messages.append(
        {
            "role": "user",
            "content": question,
        }
    )

    with st.spinner("Analyzing current DGA results..."):
        answer = generate_chat_answer(question, _build_context())

    st.session_state.chat_messages.append(
        {
            "role": "assistant",
            "content": answer,
        }
    )
    _log_debug("ask:done", answer_preview=answer[:180], total_messages=len(st.session_state.chat_messages))


def render_chat(question_ref=None, chat_trigger_ref=None):
    st.markdown("---")
    st.subheader("AI Chat Assistant")

    payload_ready = isinstance(st.session_state.get("payload"), dict)
    selected_transformer = st.session_state.get("selected_transformer") or "All transformers"
    status_text = (
        "Connected to the current prediction payload"
        if payload_ready
        else "Upload data and run prediction to enable result-aware answers"
    )

    _log_debug(
        "render_chat:start",
        payload_ready=payload_ready,
        selected_transformer=selected_transformer,
        chat_messages=len(st.session_state.get("chat_messages", [])),
    )

    st.markdown(
        f"""
        <div class="chat-shell">
            <div class="chat-header-row">
                <div>
                    <div class="chat-title">DGA analysis assistant</div>
                    <div class="chat-subtitle">{status_text}</div>
                </div>
                <div class="chat-context-pill">Context: {selected_transformer}</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if "chat_messages" not in st.session_state:
        st.session_state.chat_messages = [
            {
                "role": "assistant",
                "content": (
                    "I can explain DGA, transformer ranking, severity, fault hypotheses, "
                    "feature importance, dataset quality, CO₂/CO meaning, gas trends, and transformer comparisons."
                ),
            }
        ]
        _log_debug("render_chat:init_default_message")

    st.caption("Quick suggestions")
    suggestion_cols = st.columns(3)

    for idx, suggestion in enumerate(SUGGESTIONS):
        with suggestion_cols[idx % 3]:
            if st.button(suggestion, key=f"chat_suggestion_{idx}", use_container_width=True):
                _log_debug("suggestion_clicked", suggestion=suggestion)
                _ask(suggestion)
                st.rerun()

    for message in st.session_state.chat_messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    col1, _ = st.columns([1, 5])
    with col1:
        if st.button("Clear chat", use_container_width=True):
            _log_debug("clear_chat")
            st.session_state.chat_messages = []
            st.rerun()

    prompt = st.chat_input(
        "Ask about DGA, ranking, a transformer, CO₂/CO ratio, fault types, gas trends, or compare transformers..."
    )
    if prompt:
        _log_debug("chat_input_submitted", prompt=prompt)
        _ask(prompt)
        st.rerun()