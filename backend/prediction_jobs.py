# prediction_jobs.py
from __future__ import annotations

import logging
import multiprocessing as mp
import os
import time
import uuid
from threading import Lock
from typing import Any

from inference_service import process_dataframe

logger = logging.getLogger(__name__)

_LOCK = Lock()

_JOBS: dict[str, dict[str, Any]] = {}

_JOB_TTL_SECONDS = 60 * 60
_MAX_JOBS = 8


def _cleanup_locked() -> None:
    now = time.time()

    for job_id in list(_JOBS):
        job = _JOBS[job_id]

        reference_time = (
            job.get("completed_at")
            or job.get("started_at")
            or job["created_at"]
        )

        if now - reference_time > _JOB_TTL_SECONDS:
            _JOBS.pop(job_id, None)

    if len(_JOBS) <= _MAX_JOBS:
        return

    finished = sorted(
        (
            job
            for job in _JOBS.values()
            if job["status"] in {"completed", "failed"}
        ),
        key=lambda job: (
            job.get("completed_at")
            or job.get("created_at")
        ),
    )

    overflow = len(_JOBS) - _MAX_JOBS

    for job in finished[:overflow]:
        _JOBS.pop(job["job_id"], None)


def _child_runner(
    job_id: str,
    dataframe,
) -> None:
    """
    Runs outside the Gunicorn request worker.

    The child process writes only its final status/result through
    the shared multiprocessing manager dict.
    """
    started_at = time.time()

    try:
        logger.info(
            "PREDICTION CHILD START | job_id=%s | pid=%s | rows=%d",
            job_id,
            os.getpid(),
            len(dataframe),
        )

        result = process_dataframe(dataframe)

        elapsed = round(
            time.time() - started_at,
            3,
        )

        with _LOCK:
            job = _JOBS.get(job_id)

            if job is not None:
                job["status"] = "completed"
                job["completed_at"] = time.time()
                job["elapsed_seconds"] = elapsed
                job["result"] = result
                job["error"] = None

        logger.info(
            "PREDICTION CHILD COMPLETED | job_id=%s | elapsed=%.3fs",
            job_id,
            elapsed,
        )

    except Exception as exc:
        elapsed = round(
            time.time() - started_at,
            3,
        )

        logger.exception(
            "PREDICTION CHILD FAILED | job_id=%s",
            job_id,
        )

        with _LOCK:
            job = _JOBS.get(job_id)

            if job is not None:
                job["status"] = "failed"
                job["completed_at"] = time.time()
                job["elapsed_seconds"] = elapsed
                job["error"] = str(exc)


def _start_process(
    job_id: str,
    dataframe,
) -> mp.Process:
    process = mp.Process(
        target=_child_runner,
        args=(job_id, dataframe),
        daemon=True,
    )

    process.start()

    return process


def submit_prediction(dataframe) -> str:
    """
    Queue a prediction without occupying the HTTP request thread.

    Only one prediction is allowed at a time.
    """

    with _LOCK:
        _cleanup_locked()

        for existing in _JOBS.values():
            if existing["status"] in {
                "queued",
                "running",
            }:
                raise RuntimeError(
                    "A prediction is already running. "
                    "Please wait for it to finish before starting another."
                )

        job_id = uuid.uuid4().hex

        _JOBS[job_id] = {
            "job_id": job_id,
            "created_at": time.time(),
            "started_at": time.time(),
            "completed_at": None,
            "status": "running",
            "result": None,
            "error": None,
            "elapsed_seconds": None,
        }

        try:
            process = _start_process(
                job_id,
                dataframe,
            )

            _JOBS[job_id]["pid"] = process.pid

        except Exception:
            _JOBS.pop(job_id, None)
            raise

    logger.info(
        "PREDICTION JOB STARTED | job_id=%s | pid=%s | rows=%d",
        job_id,
        process.pid,
        len(dataframe),
    )

    return job_id


def get_prediction_job(
    job_id: str,
) -> dict[str, Any] | None:
    with _LOCK:
        _cleanup_locked()

        job = _JOBS.get(job_id)

        if job is None:
            return None

        out = dict(job)

        now = time.time()

        out["elapsed_seconds"] = round(
            now - job["created_at"],
            3,
        )

        if job.get("started_at") is None:
            out["running_seconds"] = None
        else:
            out["running_seconds"] = round(
                now - job["started_at"],
                3,
            )

        pid = job.get("pid")

        if (
            job["status"] == "running"
            and pid is not None
        ):
            try:
                os.kill(pid, 0)
            except ProcessLookupError:
                job["status"] = "failed"
                job["completed_at"] = now
                job["error"] = (
                    "Prediction worker process exited unexpectedly."
                )

                out = dict(job)
                out["elapsed_seconds"] = round(
                    now - job["created_at"],
                    3,
                )
                out["running_seconds"] = None

        return out