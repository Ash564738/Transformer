# Dockerfile
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    MALLOC_ARENA_MAX=2 \
    PORT=10000 \
    OMP_NUM_THREADS=1 \
    OPENBLAS_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 \
    NUMEXPR_NUM_THREADS=1 \
    VECLIB_MAXIMUM_THREADS=1 \
    DGA_MAX_ROWS=0 \
    DGA_PERSIST_OUTPUTS=0 \
    DGA_SAVE_DATABASE=0 \
    DGA_PRODUCTION_SELECTION_FALLBACK=traditional

WORKDIR /app

COPY backend/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY backend/ ./

EXPOSE 10000

CMD ["sh", "-c", "exec gunicorn app:app --bind 0.0.0.0:${PORT:-10000} --workers 1 --threads 2 --worker-class gthread --timeout 120 --graceful-timeout 20 --keep-alive 5 --worker-tmp-dir /dev/shm --access-logfile - --error-logfile - --log-level info"]