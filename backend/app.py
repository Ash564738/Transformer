# backend/app.py

from __future__ import annotations

import hashlib
import json
import logging
import os
import tempfile
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
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
_PREDICTION_JOBS: dict[str, dict] = {}
_PREDICTION_FINGERPRINT_TO_JOB: dict[str, str] = {}
_CACHE_TTL = max(0, int(os.getenv("DGA_PREDICTION_CACHE_TTL", "0")))
_JOB_TTL_SECONDS = max(300, int(os.getenv("DGA_PREDICTION_JOB_TTL", "7200")))
_JOB_WORKERS = max(1, int(os.getenv("DGA_PREDICTION_WORKERS", "1")))
_PREDICTION_EXECUTOR = ThreadPoolExecutor(max_workers=_JOB_WORKERS, thread_name_prefix="dga-predict")

app = Flask(__name__)
app.json.ensure_ascii = False
# No application-level upload-size limit. The prediction contract is to process
# the complete user file. Render web services do support long-lived HTTP
# requests, but prediction itself is dispatched to a background executor so
# the upload request is not held open while pandas/NumPy performs inference.
app.config["MAX_CONTENT_LENGTH"] = None

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


def _cleanup_expired_prediction_jobs(now: float | None = None) -> None:
    now = time.time() if now is None else now
    expired = []
    with _PREDICTION_LOCK:
        for job_id, job in list(_PREDICTION_JOBS.items()):
            updated = float(job.get("updated_at", job.get("created_at", now)))
            if now - updated > _JOB_TTL_SECONDS:
                expired.append(job_id)
        for job_id in expired:
            job = _PREDICTION_JOBS.pop(job_id, None)
            if job:
                fingerprint = job.get("fingerprint")
                if fingerprint and _PREDICTION_FINGERPRINT_TO_JOB.get(fingerprint) == job_id:
                    _PREDICTION_FINGERPRINT_TO_JOB.pop(fingerprint, None)
            _remove_job_tempfile(job)


def _remove_job_tempfile(job: dict | None) -> None:
    if not job:
        return
    path = job.get("source_path")
    if path:
        try:
            Path(path).unlink(missing_ok=True)
        except Exception:
            logger.exception("Failed to remove temporary prediction input: %s", path)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(8 * 1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _parse_source_path(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    try:
        if suffix in {".xlsx", ".xls"}:
            return pd.read_excel(path, engine="openpyxl")
        if suffix in {".csv", ""}:
            return pd.read_csv(path)
        raise ValueError("Only CSV, XLSX, and XLS files are supported.")
    except Exception as exc:
        raise ValueError(f"Unable to parse uploaded file: {exc}") from exc


def _create_prediction_job_from_request() -> tuple[str, str]:
    _cleanup_expired_prediction_jobs()
    job_id = uuid.uuid4().hex
    created = time.time()

    if request.files and "file" in request.files:
        file_storage = request.files["file"]
        filename = getattr(file_storage, "filename", "") or "upload.csv"
        suffix = Path(filename).suffix.lower()
        if suffix not in {".csv", ".xlsx", ".xls"}:
            raise ValueError("Only CSV, XLSX, and XLS files are supported.")
        fd, raw_path = tempfile.mkstemp(prefix="dga-predict-", suffix=suffix)
        os.close(fd)
        path = Path(raw_path)
        try:
            file_storage.save(path)
            if path.stat().st_size == 0:
                raise ValueError("The uploaded file is empty or invalid.")
            fingerprint = f"file:{_sha256_file(path)}"
        except Exception:
            path.unlink(missing_ok=True)
            raise
        job = {
            "job_id": job_id,
            "fingerprint": fingerprint,
            "source_type": "file",
            "source_path": str(path),
            "status": "queued",
            "created_at": created,
            "updated_at": created,
            "payload": None,
            "error": None,
        }
    else:
        payload = request.get_json(silent=True)
        if payload is None:
            raise ValueError("Expected CSV/XLSX upload or JSON data.")
        try:
            raw_bytes = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        except Exception as exc:
            raise ValueError(f"Invalid JSON prediction payload: {exc}") from exc
        fingerprint = f"json:{_sha256_bytes(raw_bytes)}"
        if isinstance(payload, dict) and "data" in payload:
            frame_payload = payload["data"]
        else:
            frame_payload = payload
        job = {
            "job_id": job_id,
            "fingerprint": fingerprint,
            "source_type": "dataframe",
            "dataframe": pd.DataFrame(frame_payload),
            "status": "queued",
            "created_at": created,
            "updated_at": created,
            "payload": None,
            "error": None,
        }

    with _PREDICTION_LOCK:
        existing_job_id = _PREDICTION_FINGERPRINT_TO_JOB.get(fingerprint)
        if existing_job_id and existing_job_id in _PREDICTION_JOBS:
            existing = _PREDICTION_JOBS[existing_job_id]
            _remove_job_tempfile(job)
            return existing_job_id, fingerprint
        _PREDICTION_JOBS[job_id] = job
        _PREDICTION_FINGERPRINT_TO_JOB[fingerprint] = job_id
    return job_id, fingerprint


def _get_prediction_job(job_id: str) -> dict | None:
    _cleanup_expired_prediction_jobs()
    with _PREDICTION_LOCK:
        job = _PREDICTION_JOBS.get(job_id)
        if job is None:
            return None
        return dict(job)


def _set_prediction_job(job_id: str, **updates) -> None:
    with _PREDICTION_LOCK:
        job = _PREDICTION_JOBS.get(job_id)
        if job is None:
            return
        job.update(updates)
        job["updated_at"] = time.time()


def _run_prediction_job(job_id: str) -> None:
    job = _get_prediction_job(job_id)
    if job is None:
        return
    _set_prediction_job(job_id, status="running")
    source_path = job.get("source_path")
    try:
        if job.get("source_type") == "file":
            input_df = _parse_source_path(Path(source_path))
        else:
            input_df = job.get("dataframe")
        if input_df is None or input_df.empty:
            raise ValueError("Prediction input is empty.")

        started = time.perf_counter()
        payload = process_dataframe(input_df)
        payload.setdefault("pipeline", {})["api_async_job"] = True
        payload["pipeline"]["api_async_job_id"] = job_id
        payload["pipeline"]["api_async_elapsed_seconds"] = round(time.perf_counter() - started, 4)
        payload["pipeline"]["input_rows_are_unlimited"] = True
        payload["pipeline"]["samples_per_transformer_are_unlimited"] = True
        payload["pipeline"]["upload_size_limit"] = None

        if _CACHE_TTL > 0:
            _prediction_cache_set(job["fingerprint"], payload)

        _set_prediction_job(job_id, status="completed", payload=payload, error=None)
        logger.info("Prediction job completed | job=%s | rows=%d | transformers=%d", job_id, len(input_df), input_df.get("transformer_id", pd.Series(dtype=object)).nunique())
    except Exception as exc:
        logger.exception("Prediction job failed | job=%s", job_id)
        _set_prediction_job(job_id, status="failed", error=str(exc), payload=None)
    finally:
        _remove_job_tempfile(job)
        with _PREDICTION_LOCK:
            job_ref = _PREDICTION_JOBS.get(job_id)
            if job_ref is not None:
                job_ref.pop("source_path", None)
                job_ref.pop("dataframe", None)


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
    """Enqueue a complete uploaded dataset for full-file production inference."""
    try:
        job_id, fingerprint = _create_prediction_job_from_request()
    except ValueError as exc:
        return jsonify(error=str(exc)), 400
    except Exception as exc:
        logger.exception("Failed to enqueue prediction job")
        return jsonify(error=f"Unable to start prediction: {exc}"), 500

    job = _get_prediction_job(job_id)
    if job is None:
        return jsonify(error="Prediction job could not be created."), 500

    status = job.get("status")
    if status == "completed":
        payload = job.get("payload")
        return jsonify({
            "job_id": job_id,
            "prediction_job_id": job_id,
            "jobId": job_id,
            "status": "completed",
            "result": _sanitize_for_json(payload or {}),
        }), 200
    if status == "running":
        return jsonify({
            "job_id": job_id,
            "prediction_job_id": job_id,
            "jobId": job_id,
            "status": "running",
        }), 200

    cached = _prediction_cache_get(fingerprint) if _CACHE_TTL > 0 else None
    if cached is not None:
        _remove_job_tempfile(job)
        _set_prediction_job(job_id, status="completed", payload=cached, error=None)
        return jsonify({
            "job_id": job_id,
            "prediction_job_id": job_id,
            "jobId": job_id,
            "status": "completed",
            "result": _sanitize_for_json(cached),
        }), 200

    _PREDICTION_EXECUTOR.submit(_run_prediction_job, job_id)
    response = jsonify({
        "job_id": job_id,
        "prediction_job_id": job_id,
        "jobId": job_id,
        "status": "queued",
        "message": "Prediction job accepted. The complete uploaded dataset will be processed.",
        "rows_are_unlimited": True,
        "samples_per_transformer_are_unlimited": True,
        "upload_size_limit": None,
    })
    response.status_code = 202
    response.headers["X-Prediction-Job-Id"] = job_id
    return response


@app.route("/predict/status/<job_id>", methods=["GET"])
@auth.require_auth
def predict_status(job_id):
    job = _get_prediction_job(job_id)
    if job is None:
        return jsonify(error="Prediction job not found or expired."), 404

    status = job.get("status")
    if status in {"queued", "running"}:
        response = jsonify({
            "job_id": job_id,
            "prediction_job_id": job_id,
            "jobId": job_id,
            "status": status,
            "created_at": job.get("created_at"),
            "updated_at": job.get("updated_at"),
        })
        response.headers["X-Prediction-Job-Id"] = job_id
        return response, 200

    if status == "failed":
        response = jsonify({
            "job_id": job_id,
            "prediction_job_id": job_id,
            "jobId": job_id,
            "status": "failed",
            "error": job.get("error") or "Prediction failed.",
            "updated_at": job.get("updated_at"),
        })
        response.headers["X-Prediction-Job-Id"] = job_id
        return response, 200

    payload = job.get("payload")
    response = jsonify({
        "job_id": job_id,
        "prediction_job_id": job_id,
        "jobId": job_id,
        "status": "completed",
        "result": _sanitize_for_json(payload or {}),
        "updated_at": job.get("updated_at"),
    })
    response.headers["X-Prediction-Job-Id"] = job_id
    return response, 200


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

    # Experiment reports are immutable research artifacts. Do not serve an old
    # CSV/Excel set merely because those files remain on disk after the model
    # artifacts have been deleted. A new run writes this manifest.
    manifest_path = reports_dir / "experiment_run_manifest.json"
    experiment_manifest = {}
    if manifest_path.exists():
        try:
            experiment_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except Exception:
            logger.exception("Failed to read experiment manifest: %s", manifest_path)
            experiment_manifest = {}

    def _experiment_artifacts_valid():
        if not experiment_manifest:
            return False, "No current experiment run has been recorded."

        run_status = str(experiment_manifest.get("status", "")).upper()
        if run_status == "RUNNING":
            return False, "An experiment run is currently in progress."
        if run_status == "FAILED":
            return False, experiment_manifest.get("error") or "The current experiment run failed."
        if run_status != "COMPLETE":
            return False, "No completed experiment run is currently published."

        required = experiment_manifest.get("required_artifacts", [])
        missing = []
        backend_root = Path(__file__).resolve().parent
        for rel in required:
            path = Path(rel)
            if not path.is_absolute():
                path = backend_root / path
            if not path.exists():
                missing.append(str(path))
        if missing:
            return False, "Experiment results are stale because required artifacts are missing."
        return True, ""

    experiment_available, experiment_unavailable_reason = _experiment_artifacts_valid()
    if not experiment_available:
        return jsonify({
            "available": False,
            "reason": experiment_unavailable_reason,
            "run_id": experiment_manifest.get("run_id"),
            "status": experiment_manifest.get("status"),
            "failed_stage": experiment_manifest.get("failed_stage"),
            "error": experiment_manifest.get("error"),
            "stage_status": experiment_manifest.get("stage_status", {}),
            "metadata": {
                "operational_data_is_unlabeled": True,
                "standard": cfg.STANDARD,
                "severity_is_weighted": False,
                "ranking_is_weighted": False,
            },
            "executive_summary": [],
            "traditional_methods": [],
            "traditional_per_class": [],
            "traditional_combinations": [],
            "method_coverage": [],
            "method_gas_range": [],
            "supervised_ml": [],
            "weak_label_model": [],
            "weak_ml_transfer": [],
            "severity_records": [],
            "transformer_ranking": [],
            "ranking_stability": [],
            "cross_dataset_transfer": [],
            "rank_correlation_spearman": [],
        }), 200

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
            return (
                frame[fields]
                .replace([np.inf, -np.inf], np.nan)
                .where(lambda x: pd.notna(x), None)
                .to_dict(orient="records")
            )
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
    weak_label_transfer = read_csv("weak_label_model_transfer_fault_benchmark.csv")
    hybrid_transfer = read_csv("weak_traditional_hybrid_benchmark.csv")
    cross_dataset_transfer = read_csv("cross_dataset_transfer_grid.csv")
    rank_correlation_spearman = read_csv("rank_correlation_spearman.csv")
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
        "weak_label_model_transfer": weak_label_transfer,
        "weak_traditional_hybrid": hybrid_transfer,
        "cross_dataset_transfer": cross_dataset_transfer,
        "rank_correlation_spearman": rank_correlation_spearman,
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