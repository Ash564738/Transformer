# app.py
from pathlib import Path

import numpy as np
import pandas as pd
from flask import Flask, jsonify, render_template, request
from components.chat import generate_chat_answer
from inference_service import MODEL_DIR, process_dataframe

app = Flask(__name__)


def parse_file_input(file_storage):
    filename = getattr(file_storage, 'filename', '') or ''
    suffix = Path(filename).suffix.lower()
    try:
        if suffix in {'.xlsx', '.xls'}:
            df = pd.read_excel(file_storage, engine='openpyxl')
        elif suffix == '.csv' or not suffix:
            df = pd.read_csv(file_storage)
        else:
            raise ValueError('Only CSV, XLSX, and XLS files are supported.')
    except Exception as exc:
        raise ValueError(f'Unable to parse uploaded file: {exc}')
    if df.empty:
        raise ValueError('The uploaded file is empty or invalid.')
    return df  # Trả về DataFrame thay vì dict


def parse_request_data():
    if request.files and 'file' in request.files:
        return parse_file_input(request.files['file'])
    payload = request.get_json(silent=True)
    if payload is None:
        raise ValueError('Expected a JSON body, CSV file upload, or XLSX file upload.')
    if isinstance(payload, dict) and 'data' in payload:
        # Chuyển list of dict thành DataFrame
        return pd.DataFrame(payload['data'])
    return pd.DataFrame(payload)  # Giả sử payload là list of dict


@app.route('/', methods=['GET'])
def root():
    return render_template('index.html')


@app.route('/health', methods=['GET'])
def health():
    return jsonify(status='ok', model_dir=str(MODEL_DIR.resolve()))


@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = parse_request_data()   # trả về DataFrame
        if data.empty:
            return jsonify(error='No data provided.'), 400
        result = process_dataframe(data)
        return jsonify(result)
    except Exception as exc:
        return jsonify(error=str(exc)), 400


@app.route('/chat', methods=['POST'])
def chat():
    payload = request.get_json(force=True)
    question = payload.get('question', '').strip() if isinstance(payload, dict) else ''
    context = payload.get('context') if isinstance(payload, dict) else None
    if not question:
        return jsonify(error='Question is required.'), 400
    return jsonify(answer=generate_chat_answer(question, context))


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)