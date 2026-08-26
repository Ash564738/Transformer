# backend/app.py

from __future__ import annotations

import logging
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from flask import Flask, jsonify, request
from flask_cors import CORS

load_dotenv()

from logging_config import init_logging

init_logging()

import auth

from config import (
    DATASET_DIR,
    MODEL_DIR,
    REPORT_DIR,
    config as cfg,
)

from inference_service import process_dataframe
from text2sql_chat import answer_question


logger = logging.getLogger(__name__)


# ============================================================================
# Flask application
# ============================================================================

app = Flask(__name__)

app.json.ensure_ascii = False


# ============================================================================
# CORS
# ============================================================================
#
# The frontend calls Render directly:
#
#   https://transformer-kgen.onrender.com
#
# We explicitly allow the Vercel deployments plus localhost development.
#
# flask-cors handles:
#
#   - normal responses
#   - OPTIONS preflight
#   - Authorization header
#   - POST multipart/form-data
#   - POST application/json
#
# This is more reliable than manually adding headers in after_request.
# ============================================================================

DEFAULT_ALLOWED_ORIGINS = [
    "https://transformer-ftgq1ic4z-ash564738s-projects.vercel.app",
    "https://transformer.vercel.app",
    "http://localhost:3000",
    "http://127.0.0.1:3000",
]

configured_origins = os.getenv(
    "CORS_ALLOWED_ORIGINS",
    "",
).strip()

if configured_origins:
    CORS_ORIGINS = [
        origin.strip().rstrip("/")
        for origin in configured_origins.split(",")
        if origin.strip()
    ]
else:
    CORS_ORIGINS = DEFAULT_ALLOWED_ORIGINS


CORS(
    app,
    resources={
        r"/*": {
            "origins": CORS_ORIGINS,
        }
    },
    methods=[
        "GET",
        "POST",
        "OPTIONS",
    ],
    allow_headers=[
        "Content-Type",
        "Authorization",
    ],
    expose_headers=[
        "Content-Type",
    ],
    supports_credentials=False,
    max_age=86400,
)


logger.info(
    "CORS configured | origins=%s",
    CORS_ORIGINS,
)


# ============================================================================
# Database initialization
# ============================================================================

auth.init_db()


# ============================================================================
# Generic OPTIONS handler
# ============================================================================
#
# Flask-CORS normally handles this automatically, but keeping explicit
# OPTIONS routes makes the behavior deterministic for the API endpoints.
# ============================================================================

@app.route("/predict", methods=["OPTIONS"])
@app.route("/chat", methods=["OPTIONS"])
@app.route("/auth/login", methods=["OPTIONS"])
@app.route("/auth/me", methods=["OPTIONS"])
@app.route("/auth/logout", methods=["OPTIONS"])
@app.route("/dataset/reset", methods=["OPTIONS"])
@app.route(
    "/report/student-vs-traditional",
    methods=["OPTIONS"],
)
@app.route(
    "/report/experiments",
    methods=["OPTIONS"],
)
@app.route(
    "/report/experiments/refresh",
    methods=["OPTIONS"],
)
def handle_options():
    return "", 204


# ============================================================================
# Error handlers
# ============================================================================
#
# These ensure Flask-generated errors are still JSON responses.
# flask-cors will add CORS headers to them.
# ============================================================================

@app.errorhandler(400)
def handle_400(error):
    return jsonify(
        error=str(
            getattr(
                error,
                "description",
                "Bad request.",
            )
        ),
        status=400,
    ), 400


@app.errorhandler(401)
def handle_401(error):
    return jsonify(
        error="Authentication required.",
        status=401,
    ), 401


@app.errorhandler(404)
def handle_404(error):
    return jsonify(
        error="API endpoint not found.",
        status=404,
    ), 404


@app.errorhandler(405)
def handle_405(error):
    return jsonify(
        error="HTTP method not allowed.",
        status=405,
    ), 405


@app.errorhandler(500)
def handle_500(error):
    logger.exception(
        "Unhandled Flask 500 error"
    )

    return jsonify(
        error="Internal server error.",
        status=500,
    ), 500


# ============================================================================
# File input
# ============================================================================

def parse_file_input(file_storage):
    filename = (
        getattr(
            file_storage,
            "filename",
            "",
        )
        or ""
    )

    suffix = Path(
        filename
    ).suffix.lower()

    try:
        if suffix in {
            ".xlsx",
            ".xls",
        }:
            df = pd.read_excel(
                file_storage,
                engine="openpyxl",
            )

        elif suffix in {
            ".csv",
            "",
        }:
            df = pd.read_csv(
                file_storage
            )

        else:
            raise ValueError(
                "Only CSV, XLSX, and XLS files are supported."
            )

    except Exception as exc:
        raise ValueError(
            f"Unable to parse uploaded file: {exc}"
        ) from exc

    if df.empty:
        raise ValueError(
            "The uploaded file is empty or invalid."
        )

    return df


# ============================================================================
# JSON sanitization
# ============================================================================

def _sanitize_for_json(value):
    if value is None:
        return None

    if value is Ellipsis:
        return None

    if value is pd.NA:
        return None

    if value is pd.NaT:
        return None

    if isinstance(value, float):
        return (
            value
            if np.isfinite(value)
            else None
        )

    if isinstance(
        value,
        np.floating,
    ):
        x = float(value)

        return (
            x
            if np.isfinite(x)
            else None
        )

    if isinstance(
        value,
        np.integer,
    ):
        return int(value)

    if isinstance(
        value,
        np.bool_,
    ):
        return bool(value)

    if isinstance(
        value,
        bool,
    ):
        return value

    if isinstance(
        value,
        pd.Timestamp,
    ):
        if pd.isna(value):
            return None

        return value.isoformat()

    if isinstance(
        value,
        np.ndarray,
    ):
        return [
            _sanitize_for_json(item)
            for item in value.tolist()
        ]

    if isinstance(
        value,
        pd.Series,
    ):
        return [
            _sanitize_for_json(item)
            for item in value.tolist()
        ]

    if isinstance(
        value,
        pd.Index,
    ):
        return [
            _sanitize_for_json(item)
            for item in value.tolist()
        ]

    if isinstance(
        value,
        dict,
    ):
        return {
            str(key): _sanitize_for_json(item)
            for key, item in value.items()
        }

    if isinstance(
        value,
        (
            list,
            tuple,
            set,
        ),
    ):
        return [
            _sanitize_for_json(item)
            for item in value
        ]

    try:
        missing = pd.isna(value)

        if isinstance(
            missing,
            (
                bool,
                np.bool_,
            ),
        ) and missing:
            return None

    except Exception:
        pass

    return value


# ============================================================================
# Request data parsing
# ============================================================================

def parse_request_data():
    if (
        request.files
        and "file" in request.files
    ):
        return parse_file_input(
            request.files["file"]
        )

    payload = request.get_json(
        silent=True
    )

    if payload is None:
        raise ValueError(
            "Expected CSV/XLSX upload or JSON data."
        )

    if (
        isinstance(
            payload,
            dict,
        )
        and "data" in payload
    ):
        return pd.DataFrame(
            payload["data"]
        )

    return pd.DataFrame(payload)


# ============================================================================
# Health/root endpoints
# ============================================================================

@app.route("/", methods=["GET"])
def root():
    return jsonify(
        service=(
            "Transformer Degradation Ranking API"
        ),
        pipeline=(
            "production_inference"
        ),
        status="ok",
    )


@app.route("/health", methods=["GET"])
def health():
    return jsonify(
        status="ok",
        service="Transformer Degradation Ranking API",
        pipeline="production_inference",
        model_dir=str(
            MODEL_DIR.resolve()
        ),
        report_dir=str(
            REPORT_DIR.resolve()
        ),
    )


# ============================================================================
# Authentication
# ============================================================================

@app.route(
    "/auth/login",
    methods=["POST"],
)
def auth_login():
    payload = (
        request.get_json(
            silent=True
        )
        or {}
    )

    email = payload.get(
        "email"
    )

    password = payload.get(
        "password"
    )

    try:
        user, token = auth.login(
            email,
            password,
        )

    except ValueError as exc:
        return jsonify(
            error=str(exc)
        ), 401

    return jsonify(
        user=user,
        token=token,
    )


@app.route(
    "/auth/me",
    methods=["GET"],
)
@auth.require_auth
def auth_me():
    return jsonify(
        user=request.current_user
    )


@app.route(
    "/auth/logout",
    methods=["POST"],
)
def auth_logout():
    header = request.headers.get(
        "Authorization",
        "",
    )

    token = (
        header[
            len("Bearer "):
        ].strip()
        if header.startswith(
            "Bearer "
        )
        else None
    )

    if token:
        auth.logout(token)

    return jsonify(
        ok=True
    )


# ============================================================================
# Production inference
# ============================================================================

@app.route(
    "/predict",
    methods=["POST"],
)
@auth.require_auth
def predict():
    started = pd.Timestamp.utcnow()

    try:
        logger.info(
            "UPLOAD REQUEST RECEIVED | origin=%s | content_type=%s",
            request.headers.get(
                "Origin"
            ),
            request.content_type,
        )

        data = parse_request_data()

        logger.info(
            "Uploaded dataset | rows=%d | columns=%d",
            len(data),
            len(data.columns),
        )

        if data.empty:
            return jsonify(
                error="No data provided."
            ), 400

        result = process_dataframe(
            data
        )

        clean_result = _sanitize_for_json(
            result
        )

        elapsed = (
            pd.Timestamp.utcnow()
            - started
        ).total_seconds()

        logger.info(
            "UPLOAD REQUEST COMPLETED | elapsed_seconds=%.2f",
            elapsed,
        )

        return jsonify(
            clean_result
        )

    except ValueError as exc:
        logger.exception(
            "Prediction validation error"
        )

        return jsonify(
            error=str(exc),
            pipeline_status="failed",
        ), 400

    except FileNotFoundError as exc:
        logger.exception(
            "Prediction model artifact missing"
        )

        return jsonify(
            error=str(exc),
            pipeline_status="model_missing",
        ), 503

    except Exception as exc:
        logger.exception(
            "Prediction request failed"
        )

        return jsonify(
            error=str(exc),
            pipeline_status="failed",
        ), 500


# ============================================================================
# Dataset reset
# ============================================================================

@app.route(
    "/dataset/reset",
    methods=["POST"],
)
@auth.require_auth
def dataset_reset():
    from data_store import reset_db

    reset_db()

    return jsonify(
        ok=True
    )


# ============================================================================
# Student vs traditional report
# ============================================================================

@app.route(
    "/report/student-vs-traditional",
    methods=["GET"],
)
@auth.require_auth
def student_vs_traditional_report():
    path = (
        REPORT_DIR
        / "student_vs_traditional_by_transformer.csv"
    )

    if not path.exists():
        return jsonify(
            error="Report not found."
        ), 404

    try:
        df = pd.read_csv(
            path,
            encoding="utf-8-sig",
        )

        df = df.replace(
            [
                np.inf,
                -np.inf,
            ],
            np.nan,
        )

        df = df.where(
            pd.notna(df),
            None,
        )

        response_payload = {
            "rows": df.to_dict(
                orient="records"
            ),
            "path": str(path),
        }

        return jsonify(
            _sanitize_for_json(
                response_payload
            )
        )

    except Exception as exc:
        logger.exception(
            "Failed to load student/traditional report"
        )

        return jsonify(
            error=str(exc)
        ), 500


# ============================================================================
# Research reports
# ============================================================================
#
# These endpoints ONLY READ precomputed reports.
# They never run training during a user's prediction request.
# ============================================================================

@app.route(
    "/report/experiments",
    methods=["GET"],
)
@auth.require_auth
def report_experiments():
    reports_dir = REPORT_DIR
    benchmark_dir = (
        REPORT_DIR / "benchmark"
    )
    processed_dir = (
        DATASET_DIR / "processed"
    )

    def read_csv(
        filename,
        benchmark=True,
    ):
        candidates = []

        if benchmark:
            candidates.append(
                benchmark_dir / filename
            )

        candidates.append(
            reports_dir / filename
        )

        for path in candidates:
            if not path.exists():
                continue

            try:
                frame = pd.read_csv(
                    path,
                    encoding="utf-8-sig",
                )

                frame = frame.replace(
                    [
                        np.inf,
                        -np.inf,
                    ],
                    np.nan,
                )

                frame = frame.where(
                    pd.notna(frame),
                    None,
                )

                return frame.to_dict(
                    orient="records"
                )

            except Exception:
                logger.exception(
                    "Failed to read report: %s",
                    path,
                )

                return []

        return []

    def read_json(path):
        if not path.exists():
            return {}

        try:
            import json

            return json.loads(
                path.read_text(
                    encoding="utf-8"
                )
            )

        except Exception:
            logger.exception(
                "Failed to read metadata: %s",
                path,
            )

            return {}

    def read_severity():
        path = (
            processed_dir
            / "dga_unlabeled_processed.parquet"
        )

        if not path.exists():
            return []

        try:
            frame = pd.read_parquet(
                path
            )

            fields = [
                "transformer_id",
                "sample_day",
                "ieee_dga_status",
                "ieee_dga_status_label",
                "ieee_dga_status_reason",
                "ieee_max_standardized_exceedance",
                "ieee_max_status3_standardized_exceedance",
                "ieee_table1_concentration_ratio_all",
                "ieee_table2_concentration_ratio_all",
                "ieee_table3_delta_ratio_all",
                "ieee_table4_rate_ratio_all",
                "ieee_continuous_evidence_ratio",
                "ieee_continuous_evidence_basis",
                "ieee_standard_trigger_count",
                "ieee_confirmation_required",
                "ieee_delta_available",
                "ieee_rate_available",
                "ieee_rate_span_months",
            ]

            fields = [
                column
                for column in fields
                if column in frame.columns
            ]

            return (
                frame[fields]
                .replace(
                    [
                        np.inf,
                        -np.inf,
                    ],
                    np.nan,
                )
                .tail(1000)
                .where(
                    lambda x: pd.notna(x),
                    None,
                )
                .to_dict(
                    orient="records"
                )
            )

        except Exception:
            logger.exception(
                "Failed to read severity records"
            )

            return []

    weak_metadata = []

    for granularity in (
        "coarse",
        "fine",
    ):
        metadata_path = (
            processed_dir
            / f"dga_weak_label_metadata_{granularity}.json"
        )

        metadata = read_json(
            metadata_path
        )

        if metadata:
            weak_metadata.append(
                {
                    "granularity": granularity,
                    "backend": metadata.get(
                        "backend"
                    ),
                    "n_rows": metadata.get(
                        "n_rows"
                    ),
                    "n_lfs": metadata.get(
                        "n_lfs"
                    ),
                    "rows_with_at_least_one_lf": metadata.get(
                        "rows_with_at_least_one_lf"
                    ),
                    "abstain_rate": metadata.get(
                        "abstain_rate"
                    ),
                    "mean_active_lf_count": metadata.get(
                        "mean_active_lf_count"
                    ),
                    "uses_manual_lf_weights": False,
                }
            )

    traditional = read_csv(
        "traditional_individual_benchmark.csv"
    )

    combinations = read_csv(
        "traditional_combinations_benchmark.csv"
    )

    supervised = read_csv(
        "supervised_fault_benchmark.csv"
    )

    weak_transfer = read_csv(
        "weak_transfer_fault_benchmark.csv"
    )

    ranking = read_csv(
        "transformer_ranking.csv",
        benchmark=False,
    )

    inference_metadata = read_json(
        processed_dir
        / "dga_inference_metadata.json"
    )

    training_metadata = read_json(
        MODEL_DIR
        / "training_metadata.json"
    )

    summary = []

    def best(
        rows,
        split=None,
        granularity=None,
    ):
        selected = rows

        if split is not None:
            selected = [
                row
                for row in selected
                if row.get(
                    "split"
                )
                == split
            ]

        if granularity is not None:
            selected = [
                row
                for row in selected
                if row.get(
                    "granularity"
                )
                == granularity
            ]

        selected = [
            row
            for row in selected
            if row.get(
                "macro_f1"
            )
            is not None
        ]

        if not selected:
            return None

        return max(
            selected,
            key=lambda row: float(
                row.get(
                    "macro_f1",
                    0,
                )
                or 0
            ),
        )

    item = best(
        traditional,
        granularity="fine",
    )

    if item:
        summary.append(
            {
                "section": (
                    "Best Traditional Individual"
                ),
                "method": item.get(
                    "method"
                ),
                "macro_f1": item.get(
                    "macro_f1"
                ),
            }
        )

    item = best(
        combinations,
        split="locked_test",
        granularity="fine",
    )

    if item:
        summary.append(
            {
                "section": (
                    "Best Traditional Combination"
                ),
                "methods": item.get(
                    "methods"
                ),
                "macro_f1": item.get(
                    "macro_f1"
                ),
            }
        )

    item = best(
        supervised,
        split="locked_test",
        granularity="fine",
    )

    if item:
        summary.append(
            {
                "section": (
                    "Best Supervised Model"
                ),
                "model": item.get(
                    "model"
                ),
                "feature_mode": item.get(
                    "feature_mode"
                ),
                "macro_f1": item.get(
                    "macro_f1"
                ),
            }
        )

    item = best(
        weak_transfer,
        split="locked_test",
        granularity="fine",
    )

    if item:
        summary.append(
            {
                "section": (
                    "Best Weak Transfer Model"
                ),
                "model": item.get(
                    "model"
                ),
                "feature_mode": item.get(
                    "feature_mode"
                ),
                "macro_f1": item.get(
                    "macro_f1"
                ),
            }
        )

    if inference_metadata:
        summary.append(
            {
                "section": (
                    "Latest Upload Pipeline"
                ),
                "rows": inference_metadata.get(
                    "n_rows"
                ),
                "transformers": inference_metadata.get(
                    "n_transformers"
                ),
                "processing_seconds": inference_metadata.get(
                    "processing_seconds"
                ),
            }
        )

    response_payload = {
        "metadata": {
            "pipeline": inference_metadata,
            "training": training_metadata,
            "standard": cfg.STANDARD,
            "operational_data_is_unlabeled": True,
            "severity_is_weighted": False,
            "ranking_is_weighted": False,
        },
        "executive_summary": summary,
        "traditional_methods": traditional,
        "traditional_per_class": [],
        "traditional_combinations": combinations,
        "method_coverage": read_csv(
            "traditional_method_summary.csv"
        ),
        "method_gas_range": read_csv(
            "traditional_ppm_coverage.csv"
        ),
        "supervised_ml": supervised,
        "weak_label_model": weak_metadata,
        "weak_ml_transfer": weak_transfer,
        "severity_records": read_severity(),
        "transformer_ranking": ranking,
        "ranking_stability": ranking,
    }

    return jsonify(
        _sanitize_for_json(
            response_payload
        )
    )


# ============================================================================
# Chat
# ============================================================================

@app.route(
    "/chat",
    methods=["POST"],
)
@auth.require_auth
def chat():
    payload = request.get_json(
        force=True
    )

    question = (
        payload.get(
            "question",
            "",
        ).strip()
        if isinstance(
            payload,
            dict,
        )
        else ""
    )

    context = (
        payload.get("context")
        if isinstance(
            payload,
            dict,
        )
        else None
    )

    history = (
        payload.get("history")
        if isinstance(
            payload,
            dict,
        )
        else None
    )

    if not question:
        return jsonify(
            error="Question is required."
        ), 400

    try:
        answer = answer_question(
            question,
            context,
            history,
        )

        return jsonify(
            answer=answer
        )

    except Exception as exc:
        logger.exception(
            "Chat request failed"
        )

        return jsonify(
            error=str(exc)
        ), 500


# ============================================================================
# Local execution
# ============================================================================

if __name__ == "__main__":
    logger.info(
        "Starting Transformer DGA API"
    )

    logger.info(
        "MODEL_DIR=%s",
        MODEL_DIR.resolve(),
    )

    logger.info(
        "REPORT_DIR=%s",
        REPORT_DIR.resolve(),
    )

    port = int(
        os.environ.get(
            "PORT",
            "5000",
        )
    )

    app.run(
        host="0.0.0.0",
        port=port,
        debug=False,
    )