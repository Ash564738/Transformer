# prediction_jobs.py
from __future__ import annotations
import logging, time, uuid
from concurrent.futures import ThreadPoolExecutor
from threading import Lock
from typing import Any
from inference_service import process_dataframe
logger = logging.getLogger(__name__)
_EXECUTOR = ThreadPoolExecutor(max_workers=1, thread_name_prefix="dga-predict")
_LOCK = Lock()
_JOBS: dict[str, dict[str, Any]] = {}
_JOB_TTL_SECONDS = 30 * 60
_MAX_JOBS = 4

def _cleanup_locked() -> None:
    now = time.time()
    for jid in list(_JOBS):
        job = _JOBS[jid]
        if now - (job.get("completed_at") or job["created_at"]) > _JOB_TTL_SECONDS:
            _JOBS.pop(jid, None)
    if len(_JOBS) > _MAX_JOBS:
        finished = sorted((j for j in _JOBS.values() if j["status"] in {"completed", "failed"}), key=lambda j: j.get("completed_at") or j["created_at"])
        for job in finished[: max(0, len(_JOBS)-_MAX_JOBS)]:
            _JOBS.pop(job["job_id"], None)

def _run(job_id: str, dataframe) -> None:
    with _LOCK:
        job = _JOBS.get(job_id)
        if not job: return
        job["status"] = "running"; job["started_at"] = time.time()
    logger.info("PREDICTION JOB START | job_id=%s | rows=%d", job_id, len(dataframe))
    try:
        result = process_dataframe(dataframe)
        with _LOCK:
            job = _JOBS.get(job_id)
            if job:
                job["status"] = "completed"; job["completed_at"] = time.time(); job["result"] = result
        logger.info("PREDICTION JOB COMPLETED | job_id=%s", job_id)
    except Exception as exc:
        logger.exception("PREDICTION JOB FAILED | job_id=%s", job_id)
        with _LOCK:
            job = _JOBS.get(job_id)
            if job:
                job["status"] = "failed"; job["completed_at"] = time.time(); job["error"] = str(exc)

def submit_prediction(dataframe) -> str:
    job_id = uuid.uuid4().hex
    with _LOCK:
        _cleanup_locked()
        _JOBS[job_id] = {"job_id":job_id,"created_at":time.time(),"status":"queued","started_at":None,"completed_at":None,"result":None,"error":None}
    _EXECUTOR.submit(_run, job_id, dataframe)
    logger.info("PREDICTION JOB QUEUED | job_id=%s | rows=%d", job_id, len(dataframe))
    return job_id

def get_prediction_job(job_id: str) -> dict[str, Any] | None:
    with _LOCK:
        _cleanup_locked()
        job = _JOBS.get(job_id)
        if not job: return None
        now=time.time(); out=dict(job); out["elapsed_seconds"]=round(now-job["created_at"],3); out["running_seconds"]=None if job["started_at"] is None else round(now-job["started_at"],3)
        return out
