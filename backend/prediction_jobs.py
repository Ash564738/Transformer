# backend/prediction_jobs.py
from __future__ import annotations

import json
import logging
import math
import sqlite3
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from threading import Lock
from typing import Any

from config import DATABASE_DIR
from inference_service import process_dataframe

logger = logging.getLogger(__name__)

# One inference task at a time per backend instance.
_EXECUTOR = ThreadPoolExecutor(
    max_workers=1,
    thread_name_prefix="dga-predict",
)

# Protects submission/cleanup operations inside the current process.
_LOCK = Lock()

_JOB_DB_PATH = DATABASE_DIR / "prediction_jobs.db"
_JOB_TTL_SECONDS = 24 * 60 * 60
_MAX_JOBS = 256
_MAX_RUNNING_SECONDS = 30 * 60


# Initialize the database schema at process startup. This guarantees that the
# status endpoint can open the same persistent store immediately after deploy.
def _initialize_job_store() -> None:
    try:
        with _connect() as conn:
            _ensure_schema(conn)
        logger.info(
            "Prediction job store initialized | db=%s",
            _JOB_DB_PATH,
        )
    except Exception:
        logger.exception(
            "Prediction job store initialization failed | db=%s",
            _JOB_DB_PATH,
        )


def _connect() -> sqlite3.Connection:
    """Open the shared prediction-job database."""
    _JOB_DB_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    conn = sqlite3.connect(
        _JOB_DB_PATH,
        timeout=30,
        check_same_thread=False,
    )

    conn.row_factory = sqlite3.Row

    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA busy_timeout=30000")

    return conn


def _ensure_schema(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS prediction_jobs (
            job_id TEXT PRIMARY KEY,
            created_at REAL NOT NULL,
            started_at REAL,
            completed_at REAL,
            status TEXT NOT NULL,
            result_json TEXT,
            error TEXT,
            elapsed_seconds REAL
        )
        """
    )

    conn.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_prediction_jobs_status
        ON prediction_jobs(status)
        """
    )

    conn.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_prediction_jobs_completed_at
        ON prediction_jobs(completed_at)
        """
    )

    conn.commit()


_initialize_job_store()


def _json_default(value: Any):
    """Serialize common NumPy/Pandas objects returned by inference."""
    try:
        import numpy as np
        import pandas as pd

        if value is pd.NA or value is pd.NaT:
            return None

        if isinstance(value, np.integer):
            return int(value)

        if isinstance(value, np.floating):
            value = float(value)
            return value if math.isfinite(value) else None

        if isinstance(value, np.bool_):
            return bool(value)

        if isinstance(value, np.ndarray):
            return value.tolist()

        if isinstance(value, pd.Timestamp):
            return value.isoformat()

        try:
            if pd.isna(value):
                return None
        except Exception:
            pass

    except ImportError:
        pass

    raise TypeError(
        f"Object of type {type(value).__name__} is not JSON serializable"
    )


def _serialize_result(result: Any) -> str:
    return json.dumps(
        result,
        ensure_ascii=False,
        allow_nan=False,
        default=_json_default,
        separators=(",", ":"),
    )


def _deserialize_result(result_json: str | None):
    if not result_json:
        return None

    return json.loads(result_json)


def _cleanup_locked(conn: sqlite3.Connection) -> None:
    """Clean up only terminal jobs and mark genuinely stale workers failed."""
    now = time.time()

    stale_cutoff = now - _MAX_RUNNING_SECONDS

    conn.execute(
        """
        UPDATE prediction_jobs
        SET status = 'failed',
            completed_at = ?,
            elapsed_seconds = ROUND(? - created_at, 3),
            error = ?
        WHERE status = 'running'
          AND COALESCE(started_at, created_at) < ?
        """,
        (
            now,
            now,
            "Prediction exceeded the maximum runtime of 30 minutes.",
            stale_cutoff,
        ),
    )

    expiry_cutoff = now - _JOB_TTL_SECONDS

    # IMPORTANT: never delete queued/running jobs. The browser may poll them
    # throughout the whole inference period.
    conn.execute(
        """
        DELETE FROM prediction_jobs
        WHERE status IN ('completed', 'failed')
          AND COALESCE(completed_at, created_at) < ?
        """,
        (expiry_cutoff,),
    )

    # Limit only terminal history. Active jobs are never evicted.
    rows = conn.execute(
        """
        SELECT job_id
        FROM prediction_jobs
        WHERE status IN ('completed', 'failed')
        ORDER BY COALESCE(completed_at, created_at) DESC
        LIMIT -1 OFFSET ?
        """,
        (_MAX_JOBS,),
    ).fetchall()

    if rows:
        conn.executemany(
            "DELETE FROM prediction_jobs WHERE job_id = ?",
            [
                (row["job_id"],)
                for row in rows
            ],
        )

    conn.commit()


def _get_job_locked(
    conn: sqlite3.Connection,
    job_id: str,
) -> dict[str, Any] | None:
    row = conn.execute(
        """
        SELECT
            job_id,
            created_at,
            started_at,
            completed_at,
            status,
            result_json,
            error,
            elapsed_seconds
        FROM prediction_jobs
        WHERE job_id = ?
        """,
        (job_id,),
    ).fetchone()

    if row is None:
        return None

    now = time.time()
    created_at = row["created_at"] or now
    started_at = row["started_at"]

    status = row["status"]

    if status == "completed":
        elapsed_seconds = row["elapsed_seconds"]
    else:
        elapsed_seconds = round(
            now - created_at,
            3,
        )

    running_seconds = (
        None
        if started_at is None
        else round(
            now - started_at,
            3,
        )
    )

    return {
        "job_id": row["job_id"],
        "created_at": created_at,
        "started_at": started_at,
        "completed_at": row["completed_at"],
        "status": status,
        "result": _deserialize_result(row["result_json"]),
        "error": row["error"],
        "elapsed_seconds": elapsed_seconds,
        "running_seconds": running_seconds,
    }


def _mark_running(job_id: str, started_at: float) -> bool:
    with _connect() as conn:
        _ensure_schema(conn)

        cursor = conn.execute(
            """
            UPDATE prediction_jobs
            SET status = 'running',
                started_at = ?
            WHERE job_id = ?
              AND status = 'queued'
            """,
            (
                started_at,
                job_id,
            ),
        )

        conn.commit()
        return cursor.rowcount == 1


def _mark_completed(
    job_id: str,
    started_at: float,
    result: Any,
) -> None:
    completed_at = time.time()
    elapsed = round(
        completed_at - started_at,
        3,
    )

    result_json = _serialize_result(
        result
    )

    with _connect() as conn:
        _ensure_schema(conn)

        conn.execute(
            """
            UPDATE prediction_jobs
            SET status = 'completed',
                completed_at = ?,
                elapsed_seconds = ?,
                result_json = ?,
                error = NULL
            WHERE job_id = ?
            """,
            (
                completed_at,
                elapsed,
                result_json,
                job_id,
            ),
        )

        conn.commit()

    logger.info(
        "PREDICTION JOB COMPLETED | job_id=%s | elapsed=%.3fs",
        job_id,
        elapsed,
    )


def _mark_failed(
    job_id: str,
    started_at: float,
    exc: Exception,
) -> None:
    completed_at = time.time()
    elapsed = round(
        completed_at - started_at,
        3,
    )
    error = str(exc) or exc.__class__.__name__

    with _connect() as conn:
        _ensure_schema(conn)

        conn.execute(
            """
            UPDATE prediction_jobs
            SET status = 'failed',
                completed_at = ?,
                elapsed_seconds = ?,
                error = ?,
                result_json = NULL
            WHERE job_id = ?
            """,
            (
                completed_at,
                elapsed,
                error,
                job_id,
            ),
        )

        conn.commit()

    logger.error(
        "PREDICTION JOB FAILED | job_id=%s | elapsed=%.3fs | error=%s",
        job_id,
        elapsed,
        error,
    )


def _run(job_id: str, dataframe) -> None:
    started_at = time.time()

    try:
        if not _mark_running(
            job_id,
            started_at,
        ):
            logger.warning(
                "PREDICTION JOB WORKER SKIPPED | job_id=%s | job missing or no longer queued",
                job_id,
            )
            return

        logger.info(
            "PREDICTION JOB START | job_id=%s | rows=%d",
            job_id,
            len(dataframe),
        )

        result = process_dataframe(
            dataframe
        )

        _mark_completed(
            job_id,
            started_at,
            result,
        )

    except Exception as exc:
        logger.exception(
            "PREDICTION JOB WORKER ERROR | job_id=%s",
            job_id,
        )

        try:
            _mark_failed(
                job_id,
                started_at,
                exc,
            )
        except Exception:
            logger.exception(
                "PREDICTION JOB ERROR STATE COULD NOT BE STORED | job_id=%s",
                job_id,
            )


def submit_prediction(dataframe) -> str:
    """Create a persistent job record and start inference in the worker."""
    with _LOCK:
        with _connect() as conn:
            _ensure_schema(conn)
            _cleanup_locked(conn)

            running_job = conn.execute(
                """
                SELECT job_id, started_at, created_at
                FROM prediction_jobs
                WHERE status IN ('queued', 'running')
                ORDER BY created_at ASC
                LIMIT 1
                """
            ).fetchone()

            if running_job is not None:
                started_at = (
                    running_job["started_at"]
                    or running_job["created_at"]
                    or time.time()
                )

                running_seconds = time.time() - started_at

                if running_seconds <= _MAX_RUNNING_SECONDS:
                    raise RuntimeError(
                        "A prediction is already running. "
                        "Please wait for it to finish."
                    )

                # The cleanup function normally catches this; this branch is
                # defensive in case the row changed between checks.
                conn.execute(
                    """
                    UPDATE prediction_jobs
                    SET status = 'failed',
                        completed_at = ?,
                        elapsed_seconds = ?,
                        error = ?
                    WHERE job_id = ?
                    """,
                    (
                        time.time(),
                        round(
                            running_seconds,
                            3,
                        ),
                        "Prediction worker became stale.",
                        running_job["job_id"],
                    ),
                )

            job_id = uuid.uuid4().hex
            created_at = time.time()

            conn.execute(
                """
                INSERT INTO prediction_jobs (
                    job_id,
                    created_at,
                    started_at,
                    completed_at,
                    status,
                    result_json,
                    error,
                    elapsed_seconds
                )
                VALUES (?, ?, NULL, NULL, 'queued', NULL, NULL, NULL)
                """,
                (
                    job_id,
                    created_at,
                ),
            )

            conn.commit()

        try:
            _EXECUTOR.submit(
                _run,
                job_id,
                dataframe,
            )

        except Exception:
            with _connect() as conn:
                _ensure_schema(conn)
                conn.execute(
                    """
                    UPDATE prediction_jobs
                    SET status = 'failed',
                        completed_at = ?,
                        elapsed_seconds = 0,
                        error = ?
                    WHERE job_id = ?
                    """,
                    (
                        time.time(),
                        "Unable to start prediction worker.",
                        job_id,
                    ),
                )
                conn.commit()
            raise

    logger.info(
        "PREDICTION JOB QUEUED | job_id=%s | rows=%d",
        job_id,
        len(dataframe),
    )

    return job_id


def get_prediction_job(
    job_id: str,
) -> dict[str, Any] | None:
    """Return a job without deleting or evicting it during polling."""
    with _connect() as conn:
        _ensure_schema(conn)
        job = _get_job_locked(conn, job_id)

        if job is None:
            logger.warning(
                "PREDICTION JOB NOT FOUND | job_id=%s | db=%s",
                job_id,
                _JOB_DB_PATH,
            )
        else:
            logger.info(
                "PREDICTION JOB STATUS | job_id=%s | status=%s",
                job_id,
                job.get("status"),
            )

        return job


def prediction_job_store_health() -> dict[str, object]:
    """Return minimal diagnostics proving that the persistent job store works."""
    with _connect() as conn:
        _ensure_schema(conn)
        row = conn.execute(
            "SELECT COUNT(*) AS count FROM prediction_jobs"
        ).fetchone()

    return {
        "ok": True,
        "job_count": int(row["count"] if row else 0),
    }


def delete_prediction_job(
    job_id: str,
) -> None:
    with _connect() as conn:
        _ensure_schema(conn)
        conn.execute(
            "DELETE FROM prediction_jobs WHERE job_id = ?",
            (job_id,),
        )
        conn.commit()