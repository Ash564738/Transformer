# Dockerfile
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1

WORKDIR /app

# System libraries commonly required by scientific Python packages.
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
        gcc \
        g++ \
        libgomp1 \
        libglib2.0-0 \
        libsm6 \
        libxext6 \
        libxrender1 \
    && rm -rf /var/lib/apt/lists/*

# Install backend dependencies first for better Docker layer caching.
COPY backend/requirements.txt ./requirements.txt

RUN pip install --upgrade pip \
    && pip install -r requirements.txt \
    && pip install gunicorn

# Copy backend application.
COPY backend/ ./

# Render normally injects PORT=10000.
# This is only a fallback/default.
ENV PORT=10000

EXPOSE 10000

# IMPORTANT:
# The prediction pipeline is CPU/ML intensive and can take more than 120s.
# Gunicorn's default/previous 120s timeout was likely killing the worker
# before /predict could return the JSON response.
#
# 15 minutes is intentionally chosen here. Render supports substantially
# longer HTTP requests, so this does not conflict with Render's platform.
CMD ["sh", "-c", "exec gunicorn app:app --bind 0.0.0.0:${PORT} --workers 1 --threads 2 --timeout 900 --graceful-timeout 900 --keep-alive 75 --worker-tmp-dir /dev/shm"]