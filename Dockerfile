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

COPY backend/requirements.txt ./requirements.txt

RUN pip install --no-cache-dir -r requirements.txt

COPY backend/ ./

RUN mkdir -p \
    /app/models \
    /app/reports \
    /app/reports/benchmark \
    /app/dataset/processed \
    /app/database

# Build-time research publication.
#
# This executes the complete offline research pipeline against the
# labeled datasets + unlabeled operational dataset bundled in the image.
#
# The resulting:
#   models/*
#   reports/*
#   dataset/processed/*
#   database/*
# are therefore part of the deployed Render image and do not depend
# on Render's ephemeral runtime filesystem.
RUN python run_full_experiment.py \
    --mode all \
    --seed 42 \
    --use-snorkel

# Hard verification during image build.
# Fail the Render deployment instead of publishing an image whose
# experiment page would report stale/missing artifacts.
RUN test -f /app/reports/dga_research_report.xlsx \
    && test -f /app/reports/experiment_run_manifest.json \
    && test -f /app/reports/benchmark/traditional_individual_benchmark.csv \
    && test -f /app/reports/benchmark/traditional_combinations_benchmark.csv \
    && test -f /app/reports/benchmark/traditional_ppm_coverage.csv \
    && test -f /app/reports/benchmark/traditional_fault_class_coverage.csv \
    && test -f /app/reports/benchmark/traditional_pairwise_agreement.csv \
    && test -f /app/reports/benchmark/traditional_method_summary.csv \
    && test -f /app/reports/benchmark/supervised_fault_benchmark.csv \
    && test -f /app/reports/benchmark/weak_transfer_fault_benchmark.csv \
    && test -f /app/reports/benchmark/weak_label_model_transfer_fault_benchmark.csv \
    && test -f /app/reports/benchmark/weak_traditional_hybrid_benchmark.csv \
    && test -f /app/reports/benchmark/domain_gap_absolute_vs_ratio.csv \
    && test -f /app/reports/benchmark/domain_gap_representation_summary.csv \
    && test -f /app/reports/benchmark/rank_correlation_spearman.csv \
    && test -f /app/reports/benchmark/rank_correlation_kendall.csv \
    && test -f /app/reports/benchmark/cross_dataset_transfer_grid.csv \
    && test -f /app/models/fault_classifiers_coarse.joblib \
    && test -f /app/models/fault_classifiers_fine.joblib \
    && test -f /app/models/training_metadata.json \
    && test -f /app/models/production_fault_selection.joblib \
    && test -f /app/reports/transformer_ranking.csv \
    && test -f /app/dataset/processed/dga_unlabeled_processed.parquet \
    && test -f /app/dataset/processed/transformer_ranking.parquet

EXPOSE 10000

CMD ["sh", "-c", "exec gunicorn app:app --bind 0.0.0.0:${PORT:-10000} --workers 1 --threads 2 --worker-class gthread --timeout 6000 --graceful-timeout 30 --keep-alive 5 --worker-tmp-dir /dev/shm --access-logfile - --error-logfile - --log-level info"]