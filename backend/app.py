# backend/app.py

from __future__ import annotations

import hashlib
import logging
import os
import time
from pathlib import Path
from threading import Event, Lock

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from flask import Flask, jsonify, make_response, request

load_dotenv()
from logging_config import init_logging
init_logging()
import auth
from config import DATASET_DIR, MODEL_DIR, REPORT_DIR, config as cfg
from inference_service import process_dataframe
from text2sql_chat import answer_question

logger = logging.getLogger(__name__)

_PREDICTION_LOCK = Lock()
_PREDICTION_CACHE: dict[str, tuple[float, dict]] = {}
_INFLIGHT: dict[str, dict] = {}

_CACHE_TTL = max(30, int(os.getenv("DGA_PREDICTION_CACHE_TTL", "600")))
_MAX_UPLOAD_MB = max(1, int(os.getenv("DGA_MAX_UPLOAD_MB", "10")))

app = Flask(__name__)
app.json.ensure_ascii = False
app.config["MAX_CONTENT_LENGTH"] = _MAX_UPLOAD_MB * 1024 * 1024

# CORS is handled explicitly instead of relying on a static list of Vercel
# preview URLs. Vercel creates a new *.vercel.app hostname for many preview
# deployments, so an exact hostname whitelist breaks login preflight.
#
# Allowed:
#   - any https://*.vercel.app frontend
#   - configured exact origins from DGA_CORS_ORIGINS
#   - localhost/127.0.0.1 during development
_CORS_CONFIG = os.getenv("DGA_CORS_ORIGINS", "*").strip() or "*"
_CORS_EXACT = {
    origin.strip().rstrip("/")
    for origin in _CORS_CONFIG.split(",")
    if origin.strip() and origin.strip() != "*"
}
_CORS_ALLOW_ALL = "*" in _CORS_CONFIG

def _origin_allowed(origin: str | None) -> bool:
    if not origin:
        return True
    origin = origin.rstrip("/")
    if _CORS_ALLOW_ALL:
        return True
    if origin in _CORS_EXACT:
        return True
    if origin.startswith("https://") and origin.endswith(".vercel.app"):
        return True
    if origin.startswith("http://localhost:") or origin.startswith("http://127.0.0.1:"):
        return True
    return False

@app.before_request
def _handle_cors_preflight():
    if request.method != "OPTIONS":
        return None
    origin = request.headers.get("Origin")
    if not _origin_allowed(origin):
        return make_response(("", 204))
    response = make_response(("", 204))
    if origin:
        response.headers["Access-Control-Allow-Origin"] = origin
        response.headers["Vary"] = "Origin"
    else:
        response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Methods"] = "GET, POST, PUT, PATCH, DELETE, OPTIONS"
    response.headers["Access-Control-Allow-Headers"] = "Authorization, Content-Type, Accept, Origin, X-Requested-With"
    response.headers["Access-Control-Max-Age"] = "600"
    return response

@app.after_request
def _add_cors_headers(response):
    origin = request.headers.get("Origin")
    if not origin or not _origin_allowed(origin):
        return response

    response.headers["Access-Control-Allow-Origin"] = origin
    response.headers["Access-Control-Allow-Methods"] = "GET, POST, PUT, PATCH, DELETE, OPTIONS"
    response.headers["Access-Control-Allow-Headers"] = "Authorization, Content-Type, Accept, Origin, X-Requested-With"
    response.headers["Access-Control-Max-Age"] = "600"
    response.headers["Vary"] = "Origin"
    return response

auth.init_db()

@app.errorhandler(400)
def handle_400(error):
    return jsonify(error=str(getattr(error, "description", "Bad request.")), status=400), 400

@app.errorhandler(401)
def handle_401(error):
    return jsonify(error="Authentication required.", status=401), 401

@app.errorhandler(404)
def handle_404(error):
    return jsonify(error="API endpoint not found.", status=404), 404

@app.errorhandler(405)
def handle_405(error):
    return jsonify(error="HTTP method not allowed.", status=405), 405

@app.errorhandler(500)
def handle_500(error):
    logger.exception("Unhandled Flask 500 error")
    return jsonify(error="Internal server error.", status=500), 500

def parse_file_input(file_storage):
    filename = getattr(file_storage, "filename", "") or ""
    suffix = Path(filename).suffix.lower()
    try:
        if suffix in {".xlsx", ".xls"}:
            df = pd.read_excel(file_storage, engine="openpyxl")
        elif suffix in {".csv", ""}:
            df = pd.read_csv(file_storage)
        else:
            raise ValueError("Only CSV, XLSX, and XLS files are supported.")
    except Exception as exc:
        raise ValueError(f"Unable to parse uploaded file: {exc}") from exc
    if df.empty:
        raise ValueError("The uploaded file is empty or invalid.")
    return df

def _sanitize_for_json(value):
    if value is None or value is Ellipsis or value is pd.NA or value is pd.NaT:
        return None
    if isinstance(value, float):
        return value if np.isfinite(value) else None
    if isinstance(value, np.floating):
        x = float(value)
        return x if np.isfinite(x) else None
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, bool):
        return value
    if isinstance(value, pd.Timestamp):
        return None if pd.isna(value) else value.isoformat()
    if isinstance(value, (np.ndarray, pd.Series, pd.Index)):
        return [_sanitize_for_json(item) for item in value.tolist()]
    if isinstance(value, dict):
        return {str(k): _sanitize_for_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_sanitize_for_json(item) for item in value]
    try:
        missing = pd.isna(value)
        if isinstance(missing, (bool, np.bool_)) and missing:
            return None
    except Exception:
        pass
    return value

def parse_request_data():
    if request.files and "file" in request.files:
        return parse_file_input(request.files["file"])
    payload = request.get_json(silent=True)
    if payload is None:
        raise ValueError("Expected CSV/XLSX upload or JSON data.")
    if isinstance(payload, dict) and "data" in payload:
        return pd.DataFrame(payload["data"])
    return pd.DataFrame(payload)

@app.route("/", methods=["GET"])
def root():
    return jsonify(service="Transformer Degradation Ranking API", pipeline="production_inference", status="ok")

@app.route("/health", methods=["GET"])
def health():
    return jsonify(
        status="ok",
        service="Transformer Degradation Ranking API",
        pipeline="production_inference",
        model_dir=str(MODEL_DIR.resolve()),
        report_dir=str(REPORT_DIR.resolve()),
    )

@app.route("/auth/login", methods=["POST"])
def auth_login():
    payload = request.get_json(silent=True) or {}
    email = payload.get("email")
    password = payload.get("password")
    try:
        user, token = auth.login(email, password)
    except ValueError as exc:
        return jsonify(error=str(exc)), 401
    except RuntimeError as exc:
        logger.error("Authentication configuration error: %s", exc)
        return jsonify(error="Backend authentication is not configured correctly."), 503
    return jsonify(user=user, token=token)

@app.route("/auth/me", methods=["GET"])
@auth.require_auth
def auth_me():
    return jsonify(user=request.current_user)

@app.route("/auth/logout", methods=["POST"])
def auth_logout():
    header = request.headers.get("Authorization", "")
    token = header[len("Bearer "):].strip() if header.startswith("Bearer ") else None
    if token:
        auth.logout(token)
    return jsonify(ok=True)

def _prediction_cache_key(df: pd.DataFrame) -> str:
    try:
        data = df.to_csv(index=False).encode("utf-8", errors="replace")
    except Exception:
        data = repr(df).encode("utf-8", errors="replace")
    return hashlib.sha256(data).hexdigest()

def _prediction_cache_get(key: str):
    item = _PREDICTION_CACHE.get(key)
    if not item:
        return None
    created, payload = item
    if time.time() - created > _CACHE_TTL:
        _PREDICTION_CACHE.pop(key, None)
        return None
    return payload

def _prediction_cache_set(key: str, payload: dict) -> None:
    _PREDICTION_CACHE[key] = (time.time(), payload)
    while len(_PREDICTION_CACHE) > 2:
        oldest = min(_PREDICTION_CACHE.items(), key=lambda item: item[1][0])[0]
        if oldest == key:
            break
        _PREDICTION_CACHE.pop(oldest, None)

@app.route("/predict", methods=["POST"])
@auth.require_auth
def predict():
    try:
        input_df = parse_request_data()
    except ValueError as exc:
        return jsonify(error=str(exc)), 400
    if input_df.empty:
        return jsonify(error="Prediction input is empty."), 400

    key = _prediction_cache_key(input_df)
    cached = _prediction_cache_get(key)
    if cached is not None:
        cached = dict(cached)
        cached.setdefault("pipeline", {})["cache_hit"] = True
        return jsonify(_sanitize_for_json(cached)), 200

    # De-duplicate identical concurrent requests. The first request computes the
    # payload; later requests for the same exact input wait for that result
    # instead of returning a confusing 409 to the browser.
    with _PREDICTION_LOCK:
        inflight = _INFLIGHT.get(key)
        if inflight is None:
            inflight = {
                "event": Event(),
                "payload": None,
                "error": None,
            }
            _INFLIGHT[key] = inflight
            is_owner = True
        else:
            is_owner = False

    if not is_owner:
        wait_seconds = min(
            105,
            max(5, int(os.getenv("DGA_PREDICTION_WAIT_SECONDS", "105"))),
        )
        logger.info(
            "Duplicate prediction request detected; waiting up to %ss | key=%s",
            wait_seconds,
            key[:12],
        )
        finished = inflight["event"].wait(timeout=wait_seconds)
        if not finished:
            return jsonify(
                error="The current prediction is still running. Please wait a little longer.",
                retryable=True,
                status="running",
            ), 202

        if inflight["error"] is not None:
            error = inflight["error"]
            if isinstance(error, ValueError):
                return jsonify(error=str(error)), 400
            return jsonify(error=f"Prediction failed: {error}"), 500

        payload = inflight["payload"]
        if payload is None:
            return jsonify(
                error="Prediction finished without a result.",
                retryable=True,
            ), 503

        cached_payload = dict(payload)
        cached_payload.setdefault("pipeline", {})["cache_hit"] = True
        cached_payload["pipeline"]["deduplicated_request"] = True
        return jsonify(_sanitize_for_json(cached_payload)), 200

    # Only one request for a given input reaches the CPU-heavy pipeline.
    try:
        started = time.perf_counter()
        from inference_service import process_dataframe
        payload = process_dataframe(input_df)
        payload.setdefault("pipeline", {})["cache_hit"] = False
        payload["pipeline"]["deduplicated_request"] = False
        payload["pipeline"]["api_elapsed_seconds"] = round(
            time.perf_counter() - started, 4
        )
        _prediction_cache_set(key, payload)
        inflight["payload"] = payload
        return jsonify(_sanitize_for_json(payload)), 200
    except ValueError as exc:
        inflight["error"] = exc
        return jsonify(error=str(exc)), 400
    except Exception as exc:
        inflight["error"] = exc
        logger.exception("Prediction failed")
        return jsonify(error=f"Prediction failed: {exc}"), 500
    finally:
        inflight["event"].set()
        with _PREDICTION_LOCK:
            _INFLIGHT.pop(key, None)

@app.route("/predict/status/<job_id>", methods=["GET"])
def predict_status(job_id):
    return jsonify(error="Async jobs are disabled. POST /predict returns the completed payload directly."), 410


@app.route("/dataset/reset", methods=["POST"])
@auth.require_auth
def dataset_reset():
    from data_store import reset_db
    reset_db()
    return jsonify(ok=True)

@app.route("/report/student-vs-traditional", methods=["GET"])
@auth.require_auth
def student_vs_traditional_report():
    path = REPORT_DIR / "student_vs_traditional_by_transformer.csv"
    if not path.exists():
        return jsonify(error="Report not found."), 404
    try:
        df = pd.read_csv(path, encoding="utf-8-sig")
        df = df.replace([np.inf, -np.inf], np.nan).where(pd.notna(df), None)
        return jsonify(_sanitize_for_json({"rows": df.to_dict(orient="records"), "path": str(path)}))
    except Exception as exc:
        logger.exception("Failed to load student/traditional report")
        return jsonify(error=str(exc)), 500

@app.route("/report/experiments", methods=["GET"])
@auth.require_auth
def report_experiments():
    reports_dir = REPORT_DIR
    benchmark_dir = REPORT_DIR / "benchmark"
    processed_dir = DATASET_DIR / "processed"

    def read_csv(filename, benchmark=True):
        candidates = [benchmark_dir / filename] if benchmark else []
        candidates.append(reports_dir / filename)
        for path in candidates:
            if not path.exists():
                continue
            try:
                frame = pd.read_csv(path, encoding="utf-8-sig")
                frame = frame.replace([np.inf, -np.inf], np.nan).where(pd.notna(frame), None)
                return frame.to_dict(orient="records")
            except Exception:
                logger.exception("Failed to read report: %s", path)
                return []
        return []

    def read_json(path):
        if not path.exists():
            return {}
        try:
            import json
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            logger.exception("Failed to read metadata: %s", path)
            return {}

    def read_severity():
        path = processed_dir / "dga_unlabeled_processed.parquet"
        if not path.exists():
            return []
        try:
            frame = pd.read_parquet(path)
            fields = [
                "transformer_id", "sample_day", "ieee_dga_status", "ieee_dga_status_label",
                "ieee_dga_status_reason", "ieee_max_standardized_exceedance",
                "ieee_max_status3_standardized_exceedance", "ieee_table1_concentration_ratio_all",
                "ieee_table2_concentration_ratio_all", "ieee_table3_delta_ratio_all",
                "ieee_table4_rate_ratio_all", "ieee_continuous_evidence_ratio",
                "ieee_continuous_evidence_basis", "ieee_standard_trigger_count",
                "ieee_confirmation_required", "ieee_delta_available", "ieee_rate_available",
                "ieee_rate_span_months"
            ]
            fields = [c for c in fields if c in frame.columns]
            return (frame[fields].replace([np.inf, -np.inf], np.nan).tail(1000)
                    .where(lambda x: pd.notna(x), None).to_dict(orient="records"))
        except Exception:
            logger.exception("Failed to read severity records")
            return []

    weak_metadata = []
    for granularity in ("coarse", "fine"):
        metadata_path = processed_dir / f"dga_weak_label_metadata_{granularity}.json"
        metadata = read_json(metadata_path)
        if metadata:
            weak_metadata.append({
                "granularity": granularity,
                "backend": metadata.get("backend"),
                "n_rows": metadata.get("n_rows"),
                "n_lfs": metadata.get("n_lfs"),
                "rows_with_at_least_one_lf": metadata.get("rows_with_at_least_one_lf"),
                "abstain_rate": metadata.get("abstain_rate"),
                "mean_active_lf_count": metadata.get("mean_active_lf_count"),
                "uses_manual_lf_weights": False
            })

    traditional = read_csv("traditional_individual_benchmark.csv")
    combinations = read_csv("traditional_combinations_benchmark.csv")
    supervised = read_csv("supervised_fault_benchmark.csv")
    weak_transfer = read_csv("weak_transfer_fault_benchmark.csv")
    ranking = read_csv("transformer_ranking.csv", benchmark=False)
    inference_metadata = read_json(processed_dir / "dga_inference_metadata.json")
    training_metadata = read_json(MODEL_DIR / "training_metadata.json")

    summary = []
    def best(rows, split=None, granularity=None):
        selected = rows
        if split is not None:
            selected = [r for r in selected if r.get("split") == split]
        if granularity is not None:
            selected = [r for r in selected if r.get("granularity") == granularity]
        selected = [r for r in selected if r.get("macro_f1") is not None]
        if not selected:
            return None
        return max(selected, key=lambda r: float(r.get("macro_f1", 0) or 0))

    item = best(traditional, granularity="fine")
    if item:
        summary.append({"section": "Best Traditional Individual", "method": item.get("method"), "macro_f1": item.get("macro_f1")})
    item = best(combinations, split="locked_test", granularity="fine")
    if item:
        summary.append({"section": "Best Traditional Combination", "methods": item.get("methods"), "macro_f1": item.get("macro_f1")})
    item = best(supervised, split="locked_test", granularity="fine")
    if item:
        summary.append({"section": "Best Supervised Model", "model": item.get("model"), "feature_mode": item.get("feature_mode"), "macro_f1": item.get("macro_f1")})
    item = best(weak_transfer, split="locked_test", granularity="fine")
    if item:
        summary.append({"section": "Best Weak Transfer Model", "model": item.get("model"), "feature_mode": item.get("feature_mode"), "macro_f1": item.get("macro_f1")})
    if inference_metadata:
        summary.append({
            "section": "Latest Upload Pipeline",
            "rows": inference_metadata.get("n_rows"),
            "transformers": inference_metadata.get("n_transformers"),
            "processing_seconds": inference_metadata.get("processing_seconds")
        })

    response_payload = {
        "metadata": {
            "pipeline": inference_metadata,
            "training": training_metadata,
            "standard": cfg.STANDARD,
            "operational_data_is_unlabeled": True,
            "severity_is_weighted": False,
            "ranking_is_weighted": False
        },
        "executive_summary": summary,
        "traditional_methods": traditional,
        "traditional_per_class": [],
        "traditional_combinations": combinations,
        "method_coverage": read_csv("traditional_method_summary.csv"),
        "method_gas_range": read_csv("traditional_ppm_coverage.csv"),
        "supervised_ml": supervised,
        "weak_label_model": weak_metadata,
        "weak_ml_transfer": weak_transfer,
        "severity_records": read_severity(),
        "transformer_ranking": ranking,
        "ranking_stability": ranking
    }
    return jsonify(_sanitize_for_json(response_payload))

@app.route("/chat", methods=["POST"])
@auth.require_auth
def chat():
    payload = request.get_json(force=True)
    question = payload.get("question", "").strip() if isinstance(payload, dict) else ""
    context = payload.get("context") if isinstance(payload, dict) else None
    history = payload.get("history") if isinstance(payload, dict) else None
    if not question:
        return jsonify(error="Question is required."), 400
    try:
        answer = answer_question(question, context, history)
        return jsonify(answer=answer)
    except Exception as exc:
        logger.exception("Chat request failed")
        return jsonify(error=str(exc)), 500

if __name__ == "__main__":
    logger.info("Starting Transformer DGA API")
    logger.info("MODEL_DIR=%s", MODEL_DIR.resolve())
    logger.info("REPORT_DIR=%s", REPORT_DIR.resolve())
    port = int(os.environ.get("PORT", "5000"))
    app.run(host="0.0.0.0", port=port, debug=False)