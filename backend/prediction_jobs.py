# backend/prediction_jobs.py

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

# Keep completed/failed jobs for one hour.
_JOB_TTL_SECONDS = 60 * 60

# Maximum number of jobs retained in memory.
_MAX_JOBS = 8

# A prediction should normally finish much earlier than this.
# A running job beyond this limit is considered stale.
_MAX_RUNNING_SECONDS = 30 * 60


def _cleanup_locked() -> None:
    """
    Remove expired jobs and detect stale/dead prediction processes.

    Must be called while _LOCK is held.
    """
    now = time.time()

    for job_id in list(_JOBS.keys()):
        job = _JOBS[job_id]

        status = job.get("status")

        created_at = job.get("created_at") or now
        started_at = job.get("started_at") or created_at
        completed_at = job.get("completed_at")

        # ---------------------------------------------------------
        # 1. Remove old completed/failed jobs.
        # ---------------------------------------------------------
        reference_time = (
            completed_at
            or created_at
        )

        if now - reference_time > _JOB_TTL_SECONDS:
            logger.info(
                "Removing expired prediction job | job_id=%s | status=%s",
                job_id,
                status,
            )

            _JOBS.pop(job_id, None)
            continue

        # ---------------------------------------------------------
        # 2. Detect stale running jobs.
        # ---------------------------------------------------------
        if status == "running":
            running_seconds = now - started_at

            if running_seconds > _MAX_RUNNING_SECONDS:
                logger.error(
                    "Prediction job marked stale | "
                    "job_id=%s | running_seconds=%.1f | pid=%s",
                    job_id,
                    running_seconds,
                    job.get("pid"),
                )

                job["status"] = "failed"
                job["completed_at"] = now
                job["elapsed_seconds"] = round(
                    now - created_at,
                    3,
                )
                job["error"] = (
                    "Prediction worker exceeded the maximum "
                    "runtime and was marked stale."
                )

                continue

            # -----------------------------------------------------
            # 3. Check child process.
            # -----------------------------------------------------
            pid = job.get("pid")

            if pid is not None:
                try:
                    os.kill(pid, 0)
                except ProcessLookupError:
                    logger.error(
                        "Prediction worker process disappeared | "
                        "job_id=%s | pid=%s",
                        job_id,
                        pid,
                    )

                    job["status"] = "failed"
                    job["completed_at"] = now
                    job["elapsed_seconds"] = round(
                        now - created_at,
                        3,
                    )
                    job["error"] = (
                        "Prediction worker process exited unexpectedly."
                    )

                except PermissionError:
                    # Process exists but permission prevents inspection.
                    pass

    # -------------------------------------------------------------
    # 4. Keep memory bounded.
    # -------------------------------------------------------------
    if len(_JOBS) <= _MAX_JOBS:
        return

    finished_jobs = sorted(
        (
            job
            for job in _JOBS.values()
            if job.get("status") in {
                "completed",
                "failed",
            }
        ),
        key=lambda job: (
            job.get("completed_at")
            or job.get("created_at")
            or 0
        ),
    )

    overflow = len(_JOBS) - _MAX_JOBS

    for job in finished_jobs[:overflow]:
        job_id = job.get("job_id")

        if job_id:
            _JOBS.pop(job_id, None)


def _child_runner(
    job_id: str,
    dataframe,
) -> None:
    """
    Run the ML pipeline in a separate process so the Gunicorn
    HTTP worker remains responsive to polling requests.
    """
    started_at = time.time()

    # Make sure scientific Python libraries do not create excessive
    # CPU threads inside the prediction worker.
    os.environ.setdefault(
        "OMP_NUM_THREADS",
        "1",
    )
    os.environ.setdefault(
        "OPENBLAS_NUM_THREADS",
        "1",
    )
    os.environ.setdefault(
        "MKL_NUM_THREADS",
        "1",
    )
    os.environ.setdefault(
        "NUMEXPR_NUM_THREADS",
        "1",
    )

    logger.info(
        "PREDICTION CHILD START | "
        "job_id=%s | pid=%s | rows=%d",
        job_id,
        os.getpid(),
        len(dataframe),
    )

    try:
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
            "PREDICTION CHILD COMPLETED | "
            "job_id=%s | elapsed=%.3fs",
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

    finally:
        logger.info(
            "PREDICTION CHILD EXIT | "
            "job_id=%s | pid=%s",
            job_id,
            os.getpid(),
        )


def _start_process(
    job_id: str,
    dataframe,
) -> mp.Process:
    """
    Start the prediction process.

    fork is preferred on Linux because Render runs Linux containers.
    """
    try:
        ctx = mp.get_context("fork")
    except ValueError:
        ctx = mp.get_context()

    process = ctx.Process(
        target=_child_runner,
        args=(
            job_id,
            dataframe,
        ),
        daemon=True,
    )

    process.start()

    return process


def submit_prediction(dataframe) -> str:
    """
    Start one prediction job.

    A second concurrent prediction is rejected to avoid exhausting
    Render CPU/RAM.
    """
    with _LOCK:
        # Clean stale/dead jobs first.
        _cleanup_locked()

        # ---------------------------------------------------------
        # Only block if a genuinely active job remains.
        # ---------------------------------------------------------
        for existing in _JOBS.values():
            if existing.get("status") != "running":
                continue

            existing_started = (
                existing.get("started_at")
                or existing.get("created_at")
                or time.time()
            )

            running_seconds = (
                time.time() - existing_started
            )

            if running_seconds <= _MAX_RUNNING_SECONDS:
                raise RuntimeError(
                    "A prediction is already running. "
                    "Please wait for it to finish."
                )

        # ---------------------------------------------------------
        # Create new job.
        # ---------------------------------------------------------
        job_id = uuid.uuid4().hex

        _JOBS[job_id] = {
            "job_id": job_id,
            "created_at": time.time(),
            "started_at": time.time(),
            "completed_at": None,
            "status": "running",
            "result": None,
            "error": None,
            "pid": None,
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
        "PREDICTION JOB STARTED | "
        "job_id=%s | pid=%s | rows=%d",
        job_id,
        process.pid,
        len(dataframe),
    )

    return job_id


def get_prediction_job(
    job_id: str,
) -> dict[str, Any] | None:
    """
    Return current prediction status.

    Also detects stale/dead workers so polling cannot leave the
    system permanently locked.
    """
    with _LOCK:
        _cleanup_locked()

        job = _JOBS.get(job_id)

        if job is None:
            return None

        now = time.time()

        created_at = (
            job.get("created_at")
            or now
        )

        started_at = job.get("started_at")

        output = dict(job)

        output["elapsed_seconds"] = round(
            now - created_at,
            3,
        )

        if started_at is None:
            output["running_seconds"] = None
        else:
            output["running_seconds"] = round(
                now - started_at,
                3,
            )

        return output


def cancel_prediction(
    job_id: str,
) -> bool:
    """
    Mark a running job as failed.

    This does not forcibly kill a child process. It prevents the
    job from blocking future predictions.
    """
    with _LOCK:
        job = _JOBS.get(job_id)

        if job is None:
            return False

        if job.get("status") != "running":
            return False

        now = time.time()

        job["status"] = "failed"
        job["completed_at"] = now
        job["elapsed_seconds"] = round(
            now - job.get("created_at", now),
            3,
        )
        job["error"] = "Prediction cancelled."

        logger.warning(
            "Prediction job cancelled | job_id=%s",
            job_id,
        )

        return True