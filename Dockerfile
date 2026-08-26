# Dockerfile
FROM python:3.11-slim
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1
ENV MALLOC_ARENA_MAX=2
ENV PORT=10000
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends build-essential gcc g++ libgomp1 libglib2.0-0 libsm6 libxext6 libxrender1 && rm -rf /var/lib/apt/lists/*
COPY backend/requirements.txt ./requirements.txt
RUN pip install --upgrade pip && pip install -r requirements.txt && pip install gunicorn
COPY backend/ ./
EXPOSE 10000
CMD ["sh", "-c", "exec gunicorn app:app --bind 0.0.0.0:${PORT} --workers 1 --threads 4 --timeout 900 --graceful-timeout 900 --keep-alive 75 --worker-tmp-dir /dev/shm"]