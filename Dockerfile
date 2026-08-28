# Dockerfile
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    MALLOC_ARENA_MAX=2 \
    PORT=10000 \
    MPLBACKEND=Agg \
    OMP_NUM_THREADS=1 \
    OPENBLAS_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 \
    NUMEXPR_NUM_THREADS=1

WORKDIR /app

# Runtime dependencies only.
# Training dependencies such as torch, CUDA, Snorkel, XGBoost,
# LightGBM and CatBoost are intentionally NOT installed on Render.
COPY backend/requirements.txt ./requirements.txt

RUN pip install --no-cache-dir -r requirements.txt

# This copies the already-generated production artifacts:
#   backend/models/*
#   backend/reports/*
#   backend/dataset/processed/*
#
# It does NOT execute the research pipeline.
COPY backend/ ./

# Fail the deployment when the local/offline research artifacts were
# not committed/pushed correctly.
#
# Render is only the inference server. It must never train the model.
RUN test -f /app/models/production_fault_selection.joblib \
    && test -f /app/models/training_metadata.json \
    && test -f /app/models/fault_classifiers_fine.joblib \
    && test -f /app/models/fault_classifiers_coarse.joblib \
    && test -f /app/reports/experiment_run_manifest.json \
    && test -f /app/reports/dga_research_report.xlsx \
    && test -f /app/reports/transformer_ranking.csv \
    && test -f /app/dataset/processed/dga_unlabeled_processed.parquet \
    && test -f /app/dataset/processed/transformer_ranking.parquet

EXPOSE 10000

CMD ["sh", "-c", "exec gunicorn app:app --bind 0.0.0.0:${PORT:-10000} --workers 1 --threads 2 --worker-class gthread --timeout 6000 --graceful-timeout 30 --keep-alive 5 --worker-tmp-dir /dev/shm --access-logfile - --error-logfile - --log-level info"]