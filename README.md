# Transformer Degradation Dashboard

This repository implements an auditable DGA research and monitoring pipeline for
unlabeled transformer time series. It combines traditional diagnostic rules with
Snorkel weak supervision, evaluates alternatives on labeled benchmarks, preserves
transformer history, and exposes an anonymized fleet ranking through Flask and Next.js.

## Data

Place the supplied files under the configured `backend/dataset/` locations:

| File | Records | Use |
|---|---:|---|
| `DGA dataset.csv` | 201 labeled | Benchmark (five gases, descriptive labels) |
| `IEC_TC10_121.csv` | 121 labeled | Independent benchmark (seven gases, standardized labels) |
| `DGA of Main Tank only KT 11022026_09062026.xlsx` | 4,561 unlabeled / 627 transformers | Operational weak-label training and inference |

The operational file is not treated as ground truth. A transformer may have one or
many records, and each record may abstain, conflict, or contain mixed-fault evidence.

## Architecture

- `backend/inference_service.py`: cleaning, features, diagnostics, severity, ranking,
  and dashboard payload creation.
- `backend/dga/`: IEC 60599, Rogers, Doernenburg, Duval Triangle, Duval Pentagon and
  Key Gas methods.
- `backend/weak_supervision.py`: labeling functions and Snorkel `LabelModel`.
- `backend/train_unsupervised_models.py`: weak-label students, balanced training,
  benchmark splits, transfer evaluation, and report generation.
- `backend/ranking.py`: one history-aware row per transformer.
- `backend/experiment.py`: report-ready Excel workbook and chart-specific tables.
- `frontend/`: Next.js dashboard.

## Local setup (Windows)

```powershell
cd backend
python -m venv ..\.venv
..\.venv\Scripts\activate
pip install -r requirements.txt
python app.py
```

In another terminal:

```powershell
cd frontend
npm install
npm run dev
```

Open `http://localhost:3000`. The API runs at `http://127.0.0.1:5000`.

## Research workflow

Run the repository's configured training/experiment entry point after placing the
datasets. The workflow:

1. Cleans and normalizes data while retaining raw values.
2. Runs every traditional method and measures coverage, abstention, conflicts and
   empirical ppm/class ranges.
3. Fits Snorkel weak labels, then trains group-balanced student classifiers.
4. Uses transformer/group-aware train, development and locked-test splits.
5. Reports balanced accuracy, macro/weighted F1, coverage, abstain-aware accuracy and
   cross-dataset transfer.
6. Applies IEEE C57.104-2019 Status 1/2/3 evidence per record and aggregates history.
7. Writes the `.xlsx` research report.

The cleaning stage always writes `backend/dataset/processed/dga_cleaned.xlsx` with
`Cleaned_Data`, `Missing_Summary`, and `Cleaning_Summary` sheets. The preparation
stage also writes `dga_unlabeled.xlsx`; these workbooks are the report-facing
artifacts, while Parquet files are used only for efficient internal loading.
Source snapshots use the `_raw` suffix. Context fields may be imputed within
transformer/manufacturer groups; gas measurements are never fabricated for
labeling or severity.

The ranking is an unweighted lexicographic evidence order. Current IEEE status is
primary; current concentration, delta/rate, trigger counts and historical maxima are
explicit subsequent keys. It is not a failure probability or an invented weighted
health score.

## Privacy and outputs

Runtime rows, ranking, time series, dashboard and chatbot use deterministic aliases
(`TR-XXXXXXXXXX`) instead of source transformer IDs. The Excel report contains a
separate `Transformer_ID_Map` sheet for controlled research reporting. Do not expose
that sheet or raw source files through a public API.

The single report-facing workbook is `backend/reports/dga_research_report.xlsx`.
It contains separate sheets for method comparison, combinations, ppm and class
coverage, model transfer, domain gap, rank correlation, validation, ranking,
ID mapping, and chart inputs. CSV files under `reports/benchmark/` are internal
intermediate tables used to assemble that workbook; they are not additional
paper/report formats. T1/T2 ambiguity is retained explicitly.
The anonymized-to-source transformer mapping is also exported explicitly as
`backend/reports/transformer_id_mapping.xlsx` (and is included as the
`Transformer_ID_Map` sheet in the ranking and research workbooks). Treat this
file as restricted because it can re-identify operational assets. T1/T2 ambiguity is retained explicitly.

Duval Pentagon 1 and 2 are both computed and evaluated. Pentagon 1 is the
default traditional pentagon output because the labeled-benchmark evaluation
currently supports it better; Pentagon 2 remains a comparison labeling function,
not a silently discarded alternative. Production selection must still be based
on development results and reported locked-test results.

## Chatbot

The assistant is read-only and scoped to the current SQLite snapshot. It can answer
DGA questions and query loaded results; it cannot mutate data. An OpenRouter key is
optional and must be stored only in the gitignored `backend/.env`.

## Scientific boundaries

Traditional methods are noisy labeling functions, not labels. No expert annotation is
assumed. Benchmark performance is the only accuracy evidence available here; unlabeled
operational predictions must be interpreted with coverage, conflict, history and the
documented domain gap.
