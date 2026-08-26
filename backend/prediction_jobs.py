# backend/prediction_jobs.py
from __future__ import annotations

import logging
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from threading import Lock
from typing import Any

from inference_service import process_dataframe

logger = logging.getLogger(__name__)

_EXECUTOR = ThreadPoolExecutor(
    max_workers=1,
    thread_name_prefix="dga-predict",
)

_LOCK = Lock()
_JOBS: dict[str, dict[str, Any]] = {}

_JOB_TTL_SECONDS = 60 * 60
_MAX_JOBS = 8
_MAX_RUNNING_SECONDS = 30 * 60


def _cleanup_locked() -> None:
    now = time.time()

    for job_id in list(_JOBS):
        job = _JOBS[job_id]
        status = job.get("status")
        created_at = job.get("created_at") or now
        started_at = job.get("started_at") or created_at
        completed_at = job.get("completed_at")

        if now - (completed_at or created_at) > _JOB_TTL_SECONDS:
            _JOBS.pop(job_id, None)
            continue

        if status == "running" and now - started_at > _MAX_RUNNING_SECONDS:
            logger.error(
                "PREDICTION JOB STALE | job_id=%s | running_seconds=%.1f",
                job_id,
                now - started_at,
            )
            job["status"] = "failed"
            job["completed_at"] = now
            job["elapsed_seconds"] = round(now - created_at, 3)
            job["error"] = (
                "Prediction exceeded the maximum runtime of 30 minutes."
            )

    if len(_JOBS) <= _MAX_JOBS:
        return

    finished = sorted(
        (
            job for job in _JOBS.values()
            if job.get("status") in {"completed", "failed"}
        ),
        key=lambda job: job.get("completed_at") or job.get("created_at") or 0,
    )

    for job in finished[: max(0, len(_JOBS) - _MAX_JOBS)]:
        _JOBS.pop(job["job_id"], None)


def _run(job_id: str, dataframe) -> None:
    started_at = time.time()

    with _LOCK:
        job = _JOBS.get(job_id)
        if job is None:
            return
        job["status"] = "running"
        job["started_at"] = started_at

    logger.info(
        "PREDICTION JOB START | job_id=%s | rows=%d",
        job_id,
        len(dataframe),
    )

    try:
        result = process_dataframe(dataframe)
        elapsed = round(time.time() - started_at, 3)

        with _LOCK:
            job = _JOBS.get(job_id)
            if job is not None:
                job["status"] = "completed"
                job["completed_at"] = time.time()
                job["elapsed_seconds"] = elapsed
                job["result"] = result
                job["error"] = None

        logger.info(
            "PREDICTION JOB COMPLETED | job_id=%s | elapsed=%.3fs",
            job_id,
            elapsed,
        )

    except Exception as exc:
        elapsed = round(time.time() - started_at, 3)

        logger.exception(
            "PREDICTION JOB FAILED | job_id=%s",
            job_id,
        )

        with _LOCK:
            job = _JOBS.get(job_id)
            if job is not None:
                job["status"] = "failed"
                job["completed_at"] = time.time()
                job["elapsed_seconds"] = elapsed
                job["error"] = str(exc)


def submit_prediction(dataframe) -> str:
    with _LOCK:
        _cleanup_locked()

        for job in _JOBS.values():
            if job.get("status") != "running":
                continue

            started_at = (
                job.get("started_at")
                or job.get("created_at")
                or time.time()
            )

            running_seconds = time.time() - started_at

            if running_seconds <= _MAX_RUNNING_SECONDS:
                raise RuntimeError(
                    "A prediction is already running. "
                    "Please wait for it to finish."
                )

        job_id = uuid.uuid4().hex

        _JOBS[job_id] = {
            "job_id": job_id,
            "created_at": time.time(),
            "started_at": None,
            "completed_at": None,
            "status": "queued",
            "result": None,
            "error": None,
            "elapsed_seconds": None,
        }

        try:
            _EXECUTOR.submit(_run, job_id, dataframe)
        except Exception:
            _JOBS.pop(job_id, None)
            raise

    logger.info(
        "PREDICTION JOB QUEUED | job_id=%s | rows=%d",
        job_id,
        len(dataframe),
    )

    return job_id


def get_prediction_job(job_id: str) -> dict[str, Any] | None:
    with _LOCK:
        _cleanup_locked()

        job = _JOBS.get(job_id)
        if job is None:
            return None

        now = time.time()
        out = dict(job)
        created_at = job.get("created_at") or now
        started_at = job.get("started_at")

        out["elapsed_seconds"] = round(
            now - created_at,
            3,
        )

        out["running_seconds"] = (
            None
            if started_at is None
            else round(now - started_at, 3)
        )

        return out