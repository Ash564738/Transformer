# main.py
import json
from pathlib import Path

import pandas as pd
import streamlit as st

from inference_service import process_dataframe
from components.chat import render_chat
from components.results import render_results
from components.sidebar import render_sidebar
from components.theme import apply_theme
apply_theme()


@st.cache_data(show_spinner=False)
def parse_uploaded_file(uploaded_file):
    suffix = Path(uploaded_file.name).suffix.lower()
    if suffix in {'.xlsx', '.xls'}:
        return pd.read_excel(uploaded_file, engine='openpyxl')
    if suffix == '.csv':
        return pd.read_csv(uploaded_file)
    raise ValueError('Unsupported file type. Please upload a CSV or XLSX file.')


def main():
    st.title('Transformer Degradation Ranking')
    st.caption('DGA-based transformer health scoring, ranking, and explanation dashboard')

    sidebar_state = {'value': None}
    json_text_state = {'value': '[]'}
    run_state = {'value': False}
    question_state = {'value': ''}
    chat_trigger_state = {'value': False}

    render_sidebar(
        sidebar_state,
        json_text_state,
        run_state,
        question_state,
        chat_trigger_state,
    )

    uploaded_file = sidebar_state['value']
    json_text = json_text_state['value']
    run = run_state['value']

    if run:
        try:
            if uploaded_file is not None:
                data_df = parse_uploaded_file(uploaded_file)
            else:
                payload = json.loads(json_text) if json_text.strip() else []
                if isinstance(payload, dict) and 'data' in payload:
                    payload = payload['data']
                data_df = pd.DataFrame(payload)

            # Gọi pipeline DGA
            st.session_state.payload = process_dataframe(data_df)

        except Exception as exc:
            st.error(str(exc))

    if 'payload' in st.session_state:
        selected_transformer = st.session_state.get('selected_transformer')
        render_results(st.session_state.payload, selected_transformer=selected_transformer)

    render_chat(question_state, chat_trigger_state)


if __name__ == '__main__':
    main()