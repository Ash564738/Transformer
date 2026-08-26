# celery_worker.py
from celery import Celery
import os

celery = Celery(
    "dga",
    broker=os.getenv("REDIS_URL", "redis://localhost:6379/0"),
    backend=os.getenv("REDIS_URL", "redis://localhost:6379/0"),
)

celery.conf.update(
    task_track_started=True,
    task_time_limit=600,  # 10 phút tối đa
    task_soft_time_limit=540,
)