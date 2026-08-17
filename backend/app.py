# app.py
import io
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

from dotenv import load_dotenv
load_dotenv()

from logging_config import init_logging
init_logging()
import logging
logger = logging.getLogger(__name__)

import matplotlib
matplotlib.use("Agg")

import numpy as np
import pandas as pd
from flask import Flask, jsonify, request, Response

from inference_service import MODEL_DIR, process_dataframe
from text2sql_chat import answer_question
import auth

from flask import send_file


app = Flask(__name__)
auth.init_db()

# ────────────────── CORS headers ──────────────────
@app.after_request
def _add_cors_headers(response):
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type, Authorization"
    return response

# ────────────────── OPTIONS preflight (no auth) ──────────────────
@app.route('/predict', methods=['OPTIONS'])
@app.route('/chat', methods=['OPTIONS'])
@app.route('/auth/login', methods=['OPTIONS'])
@app.route('/auth/me', methods=['OPTIONS'])
@app.route('/auth/logout', methods=['OPTIONS'])
@app.route('/dataset/reset', methods=['OPTIONS'])
@app.route('/report/student-vs-traditional', methods=['OPTIONS'])
@app.route('/report/experiments', methods=['OPTIONS'])
def handle_options():
    return '', 204

# ────────────────── Helper functions ──────────────────
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
    return df

def _sanitize_for_json(value):
    if isinstance(value, float):
        return None if (np.isnan(value) or np.isinf(value)) else value
    if isinstance(value, dict):
        return {k: _sanitize_for_json(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_sanitize_for_json(v) for v in value]
    return value

def parse_request_data():
    if request.files and 'file' in request.files:
        return parse_file_input(request.files['file'])
    payload = request.get_json(silent=True)
    if payload is None:
        raise ValueError('Expected a JSON body, CSV file upload, or XLSX file upload.')
    if isinstance(payload, dict) and 'data' in payload:
        return pd.DataFrame(payload['data'])
    return pd.DataFrame(payload)

def _float_arg(name, default=0.0):
    try:
        return float(request.args.get(name, default))
    except (TypeError, ValueError):
        return default

# ────────────────── API endpoints ──────────────────
@app.route('/', methods=['GET'])
def root():
    return jsonify(service='Transformer Degradation Ranking API')

@app.route('/health', methods=['GET'])
def health():
    return jsonify(status='ok', model_dir=str(MODEL_DIR.resolve()))

@app.route('/auth/login', methods=['POST'])
def auth_login():
    payload = request.get_json(silent=True) or {}
    try:
        user, token = auth.login(payload.get('email'), payload.get('password'))
    except ValueError as exc:
        return jsonify(error=str(exc)), 401
    return jsonify(user=user, token=token)

@app.route('/auth/me', methods=['GET'])
@auth.require_auth
def auth_me():
    return jsonify(user=request.current_user)

@app.route('/auth/logout', methods=['POST'])
def auth_logout():
    header = request.headers.get('Authorization', '')
    token = header[len('Bearer '):].strip() if header.startswith('Bearer ') else None
    if token:
        auth.logout(token)
    return jsonify(ok=True)

@app.route('/predict', methods=['POST'])
@auth.require_auth
def predict():
    try:
        data = parse_request_data()
        if data.empty:
            return jsonify(error='No data provided.'), 400
        result = process_dataframe(data)
        return jsonify(_sanitize_for_json(result))
    except Exception as exc:
        return jsonify(error=str(exc)), 400

@app.route('/dataset/reset', methods=['POST'])
@auth.require_auth
def dataset_reset():
    from data_store import reset_db
    reset_db()
    return jsonify(ok=True)

@app.route('/report/student-vs-traditional', methods=['GET'])
@auth.require_auth
def student_vs_traditional_report():
    reports_dir = Path(__file__).resolve().parent.parent / 'reports'
    report_path = reports_dir / 'student_vs_traditional_by_transformer.csv'
    if not report_path.exists():
        return jsonify(error='Report not found.'), 404
    try:
        df = pd.read_csv(report_path)
        return jsonify(rows=df.to_dict(orient='records'), path=str(report_path))
    except Exception as exc:
        return jsonify(error=f'Failed to load report: {exc}'), 500

@app.route('/report/experiments', methods=['GET'])
@auth.require_auth
def report_experiments():
    import json
    from pathlib import Path
    backend_dir = Path(__file__).resolve().parent
    reports_dir = backend_dir / 'reports'
    data = {}

    # Existing experiment JSONs
    exp_path = reports_dir / 'experiment_results_full.json'
    sup_path = reports_dir / 'supervised_results.json'
    if exp_path.exists():
        with open(exp_path, 'r', encoding='utf-8') as f:
            data['anomaly'] = json.load(f)
    if sup_path.exists():
        with open(sup_path, 'r', encoding='utf-8') as f:
            data['supervised'] = json.load(f)

    # New section-specific JSONs for frontend charts
    for key, filename in [
        ('exploratory', 'section_3_1_data.json'),
        ('label_distribution', 'section_3_2_data.json'),
        ('risk_analysis', 'section_3_4_data.json')
    ]:
        path = reports_dir / filename
        if path.exists():
            with open(path, 'r', encoding='utf-8') as f:
                data[key] = json.load(f)

    logger.info("Experiment report data assembled")
    return jsonify(data)

@app.route('/chat', methods=['POST'])
@auth.require_auth
def chat():
    payload = request.get_json(force=True)
    question = payload.get('question', '').strip() if isinstance(payload, dict) else ''
    context = payload.get('context') if isinstance(payload, dict) else None
    history = payload.get('history') if isinstance(payload, dict) else None
    if not question:
        return jsonify(error='Question is required.'), 400
    return jsonify(answer=answer_question(question, context, history))

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)