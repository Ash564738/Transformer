# components/sidebar.py

import json
import logging
from typing import Any

import streamlit as st

logger = logging.getLogger(__name__)


def _debug_enabled() -> bool:
    return bool(st.session_state.get("debug_mode", False))


def _log_debug(event: str, **kwargs: Any) -> None:
    if _debug_enabled():
        logger.info("[sidebar] %s | %s", event, kwargs if kwargs else "{}")


def _safe_preview_json(text: str, max_len: int = 300) -> str:
    text = (text or "").strip()
    if len(text) <= max_len:
        return text
    return text[:max_len] + " ...[truncated]"


def _count_json_rows(text: str) -> int | None:
    try:
        parsed = json.loads(text or "[]")
        if isinstance(parsed, list):
            return len(parsed)
        return 1 if isinstance(parsed, dict) else None
    except Exception:
        return None


def _build_transformer_options(rows: list[dict]) -> list[str]:
    transformer_ids = sorted(
        {
            str(r.get("transformer_id", "")).strip()
            for r in rows
            if str(r.get("transformer_id", "")).strip()
        }
    )
    return ["All"] + transformer_ids if transformer_ids else ["All"]


def render_sidebar(
    uploaded_file_ref,
    json_text_ref,
    run_ref,
    question_ref,
    chat_trigger_ref,
):
    if "selected_transformer" not in st.session_state:
        st.session_state["selected_transformer"] = None
    if "debug_mode" not in st.session_state:
        st.session_state["debug_mode"] = False
    if "sidebar_input_mode" not in st.session_state:
        st.session_state["sidebar_input_mode"] = "File upload"

    payload = st.session_state.get("payload", {})
    rows = payload.get("rows", []) if isinstance(payload, dict) else []
    payload_ready = bool(rows)

    _log_debug(
        "render_sidebar:start",
        payload_ready=payload_ready,
        row_count=len(rows),
        existing_selected_transformer=st.session_state.get("selected_transformer"),
    )

    with st.sidebar:
        st.markdown("## Input Data")
        st.caption("Load transformer samples from file upload or inline JSON records.")

        with st.container(border=True):
            col_a, col_b = st.columns([1, 1])
            with col_a:
                st.session_state["debug_mode"] = st.toggle(
                    "Debug mode",
                    value=st.session_state["debug_mode"],
                    help="Enable verbose diagnostics in logs and the sidebar session panel.",
                    key="sidebar_debug_mode",
                )
            with col_b:
                source_mode = st.radio(
                    "Input source",
                    options=["File upload", "JSON text"],
                    horizontal=True,
                    key="sidebar_input_mode",
                    label_visibility="collapsed",
                )

        _log_debug("input_mode:selected", source_mode=source_mode)

        with st.container(border=True):
            st.markdown("### Dataset Source")

            if source_mode == "File upload":
                uploaded = st.file_uploader(
                    "Upload CSV or Excel",
                    type=["csv", "xlsx", "xls"],
                    help="Supported formats: .csv, .xlsx, .xls",
                    key="sidebar_file_uploader",
                )
                uploaded_file_ref["value"] = uploaded
                json_text_ref["value"] = st.session_state.get("sidebar_json_text", "[]")

                if uploaded is not None:
                    file_meta = {
                        "name": uploaded.name,
                        "type": uploaded.type,
                        "size_bytes": getattr(uploaded, "size", None),
                    }
                    _log_debug("file_uploaded", **file_meta)

                    st.success(f"Loaded file: `{uploaded.name}`")
                    c1, c2 = st.columns(2)
                    c1.metric("File size", f"{uploaded.size / 1024:.1f} KB" if uploaded.size else "—")
                    c2.metric("Format", uploaded.name.split(".")[-1].upper() if "." in uploaded.name else "Unknown")
                else:
                    st.info("No file uploaded yet.")
                    _log_debug("file_uploaded:none")

            else:
                json_text = st.text_area(
                    "Paste JSON rows",
                    value=st.session_state.get("sidebar_json_text", "[]"),
                    height=220,
                    placeholder='[{"transformer_id":"T01","H2":120,"CO":340}]',
                    key="sidebar_json_text",
                )
                json_text_ref["value"] = json_text
                uploaded_file_ref["value"] = None

                json_rows = _count_json_rows(json_text)
                _log_debug(
                    "json_input:updated",
                    estimated_rows=json_rows,
                    preview=_safe_preview_json(json_text),
                )

                if json_rows is None:
                    st.warning("JSON format looks invalid. Validation will run again during inference.")
                else:
                    st.success(f"Detected {json_rows} JSON row(s).")

        with st.container(border=True):
            st.markdown("### Inference")
            run_clicked = st.button(
                "Run prediction",
                type="primary",
                use_container_width=True,
                key="sidebar_run_prediction",
            )
            run_ref["value"] = run_clicked

            if run_clicked:
                _log_debug(
                    "run_prediction:clicked",
                    source_mode=source_mode,
                    has_file=uploaded_file_ref.get("value") is not None,
                    json_rows=_count_json_rows(json_text_ref.get("value", "[]")),
                )
                st.toast("Prediction request submitted.")

        st.markdown("## Transformer Filter")
        with st.container(border=True):
            if payload_ready:
                transformer_options = _build_transformer_options(rows)
                current_value = st.session_state.get("selected_transformer")
                current_label = current_value if current_value in transformer_options else "All"

                selected_transformer = st.selectbox(
                    "Select transformer",
                    options=transformer_options,
                    index=transformer_options.index(current_label),
                    key="sidebar_transformer_filter",
                    help="Filter dashboard tables, charts, and chat context to a single transformer.",
                )

                normalized_value = None if selected_transformer == "All" else selected_transformer
                st.session_state["selected_transformer"] = normalized_value

                _log_debug(
                    "transformer_filter:changed",
                    available_count=len(transformer_options) - 1,
                    selected_label=selected_transformer,
                    stored_value=normalized_value,
                )

                c1, c2 = st.columns(2)
                c1.metric("Rows in payload", len(rows))
                c2.metric("Transformers", max(0, len(transformer_options) - 1))

                if normalized_value:
                    st.caption(f"Active filter: `{normalized_value}`")
                else:
                    st.caption("Active filter: all transformers")
            else:
                st.session_state["selected_transformer"] = None
                st.info("Run prediction to enable transformer-level filtering.")
                _log_debug("transformer_filter:disabled_no_payload")

        with st.expander("Session diagnostics", expanded=False):
            payload_keys = list(payload.keys()) if isinstance(payload, dict) else []
            diagnostics = {
                "payload_ready": payload_ready,
                "payload_keys": payload_keys,
                "row_count": len(rows),
                "selected_transformer": st.session_state.get("selected_transformer"),
                "debug_mode": st.session_state.get("debug_mode"),
                "input_mode": st.session_state.get("sidebar_input_mode"),
            }
            st.write(diagnostics)
            _log_debug("session_diagnostics", **diagnostics)

        question_ref["value"] = ""
        chat_trigger_ref["value"] = False
        _log_debug("chat_state:reset", question="", chat_trigger=False)

        st.markdown("---")
        st.caption("Transformer Degradation Dashboard")