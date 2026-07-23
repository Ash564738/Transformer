# app.py
import io
import sys
from pathlib import Path

# The pipeline logs and prints Vietnamese text; on Windows consoles (cp1252)
# that raises UnicodeEncodeError unless stdout/stderr are forced to UTF-8.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

import matplotlib
matplotlib.use("Agg")  # headless backend — must be set before any pyplot import

import numpy as np
import pandas as pd
from flask import Flask, jsonify, request, Response

from chat import generate_chat_answer
from inference_service import MODEL_DIR, process_dataframe
from dga import duval_triangle, duval_pentagon
import auth

app = Flask(__name__)
auth.init_db()


@app.after_request
def _add_cors_headers(response):
    # Long-running requests (large-dataset /predict calls can take ~1 minute)
    # were getting reset by the Next.js dev server's rewrite proxy. The
    # frontend now calls this Flask server directly from the browser instead
    # of going through that proxy, which needs CORS enabled here.
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
    # Authorization was missing here: every authenticated request (the
    # frontend sends `Authorization: Bearer <token>` on /auth/me, /predict,
    # /chat) is a non-simple CORS request, so the browser preflights it with
    # OPTIONS first and only sends the real request if the preflight response
    # says Authorization is allowed. Without it here, the browser silently
    # blocks the real request after every preflight — login/register would
    # store a token, then the immediate /auth/me check to confirm it would
    # get blocked, clearing the token and bouncing back to /login.
    response.headers["Access-Control-Allow-Headers"] = "Content-Type, Authorization"
    return response


@app.route('/predict', methods=['OPTIONS'])
@app.route('/chat', methods=['OPTIONS'])
@app.route('/auth/login', methods=['OPTIONS'])
@app.route('/auth/me', methods=['OPTIONS'])
@app.route('/auth/logout', methods=['OPTIONS'])
def _cors_preflight():
    return ('', 204)


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


def _sanitize_for_json(value):
    """Recursively replaces NaN/Infinity floats with None.

    The pipeline legitimately produces NaN for rate/lag/rolling features
    when a transformer doesn't have enough history yet (see feature_engineering.py).
    Python's json module serializes float('nan') as the bareword `NaN`, which
    is not valid JSON — browsers' JSON.parse() rejects it outright. Replacing
    it with `null` keeps the same "missing value" meaning while staying
    spec-compliant.
    """
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
        # Chuyển list of dict thành DataFrame
        return pd.DataFrame(payload['data'])
    return pd.DataFrame(payload)  # Giả sử payload là list of dict


def _fig_to_png_response(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=110, bbox_inches='tight')
    matplotlib.pyplot.close(fig)
    buf.seek(0)
    return Response(buf.getvalue(), mimetype='image/png')


def _float_arg(name, default=0.0):
    try:
        return float(request.args.get(name, default))
    except (TypeError, ValueError):
        return default


@app.route('/', methods=['GET'])
def root():
    return jsonify(
        service='Transformer Degradation Ranking API',
        endpoints=[
            '/health', '/predict', '/chat', '/chart/duval-triangle', '/chart/duval-pentagon',
            '/auth/login', '/auth/me', '/auth/logout',
        ],
    )


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
        data = parse_request_data()   # trả về DataFrame
        if data.empty:
            return jsonify(error='No data provided.'), 400
        result = process_dataframe(data)
        return jsonify(_sanitize_for_json(result))
    except Exception as exc:
        return jsonify(error=str(exc)), 400


@app.route('/chat', methods=['POST'])
@auth.require_auth
def chat():
    payload = request.get_json(force=True)
    question = payload.get('question', '').strip() if isinstance(payload, dict) else ''
    context = payload.get('context') if isinstance(payload, dict) else None
    if not question:
        return jsonify(error='Question is required.'), 400
    return jsonify(answer=generate_chat_answer(question, context))


@app.route('/chart/duval-triangle', methods=['GET'])
def chart_duval_triangle():
    """Renders the exact matplotlib figure from dga/duval_triangle.py — no
    reimplementation on the frontend, just the same plotting code reused
    as an image endpoint."""
    ch4 = _float_arg('ch4')
    c2h4 = _float_arg('c2h4')
    c2h2 = _float_arg('c2h2')
    fault = request.args.get('fault') or None
    fig = duval_triangle.plot_duval_triangle(ch4, c2h4, c2h2, fault=fault)
    if fig is None:
        return jsonify(error='Insufficient hydrocarbon data for Duval Triangle.'), 400
    return _fig_to_png_response(fig)


@app.route('/chart/duval-pentagon', methods=['GET'])
def chart_duval_pentagon():
    """Renders the exact matplotlib figure from dga/duval_pentagon.py (both
    Pentagon 1 and Pentagon 2 side by side)."""
    h2 = _float_arg('h2')
    ch4 = _float_arg('ch4')
    c2h6 = _float_arg('c2h6')
    c2h4 = _float_arg('c2h4')
    c2h2 = _float_arg('c2h2')
    fault_p1 = request.args.get('fault_p1') or None
    fault_p2 = request.args.get('fault_p2') or None
    fig = duval_pentagon.plot_pentagon_dual(h2, ch4, c2h6, c2h4, c2h2, fault_p1=fault_p1, fault_p2=fault_p2)
    if fig is None:
        return jsonify(error='Insufficient gas data for Duval Pentagon.'), 400
    return _fig_to_png_response(fig)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
