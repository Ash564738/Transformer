# Transformer Degradation Dashboard

A DGA (Dissolved Gas Analysis) transformer health monitoring system: a Python/Flask
backend runs the DGA diagnostic pipeline (data cleaning, feature engineering, six
traditional fault-diagnosis methods, severity scoring, and fleet-wide ranking), and a
Next.js dashboard visualizes the results — fleet overview, analytics, per-transformer
detail, and an AI chat assistant — behind a login.

## Project layout

```
backend/    Flask API + DGA diagnostic pipeline + offline model training
frontend/   Next.js dashboard (calls the Flask API directly from the browser)
```

- `backend/app.py` — the Flask server: `/auth/*`, `/predict`, `/chat`, `/dataset/reset`, `/chart/*`
- `backend/inference_service.py` — orchestrates clean → accumulate → feature-engineer → diagnose → score → rank
- `backend/dataset_accumulator.py` — merges each new upload into everything uploaded
  before (`backend/data/accumulated_clean.csv`), deduped by transformer + sample date
- `backend/dga/` — the six traditional diagnostic methods (Duval Triangle, Duval Pentagon,
  IEC 60599, Rogers Ratio, Doernenburg, Key Gas)
- `backend/consensus.py`, `backend/severity.py`, `backend/ranking.py` — combine the six
  methods' votes into a consensus fault, score severity, and rank the fleet
- `backend/train_models.py` — offline training for the LightGBM/XGBoost/PyTorch models
  (not required to run the live dashboard — the pipeline above is rule-based)
- `backend/auth.py` — local SQLite-backed single-user login (no self-service registration)
- `backend/data_store.py` — mirrors every `/predict` result into SQLite (`backend/data/dga.db`)
  so the chatbot has something to query
- `backend/text2sql_chat.py` — the DGA Assistant: turns a question into SQL against that
  SQLite snapshot, runs it read-only, and explains the result (falls back to a simple
  rule-based responder if no `OPENROUTER_API_KEY` is configured)

## Prerequisites

- Python 3.10+
- Node.js 20+
- `git`

## 1. Backend setup

From the repository root:

```powershell
cd backend
python -m venv ../.venv
../.venv/Scripts/activate       # Windows
# source ../.venv/bin/activate  # macOS/Linux
pip install -r requirements.txt
python app.py
```

The API starts at `http://127.0.0.1:5000`. Confirm it's up:

```powershell
curl http://127.0.0.1:5000/health
```

No trained model is required to use the dashboard — `/predict` runs the rule-based
diagnostic pipeline directly on whatever CSV/XLSX you upload. Training
(`python train_models.py`) is only needed if you're working on the separate
LightGBM/XGBoost/PyTorch models; it expects a labeled dataset under `dataset/`
(not included in this repo).

## 2. Frontend setup

In a second terminal:

```powershell
cd frontend
npm install
npm run dev
```

Open `http://localhost:3000`. The frontend calls the Flask API directly from the
browser (not through a Next.js proxy — large uploads can take close to a minute, and
a dev-server rewrite proxy was found to reset those connections). If your backend
isn't on `127.0.0.1:5000`, set:

```
NEXT_PUBLIC_BACKEND_URL=http://your-backend-host:5000
```

in `frontend/.env.local`.

## 3. Sign in

The dashboard is behind a login, but there's no self-service registration — it's a
single configured account. Set it once (from `backend/`, with the venv active):

```powershell
python seed_user.py you@example.com "a-real-password" "Your Name"
```

Then sign in at `http://localhost:3000/login`. The account lives in
`backend/data/users.db` (SQLite, gitignored) with a hashed password. Re-run
`seed_user.py` any time to change the email/password — it replaces whichever
account exists and signs out its active sessions.

## 4. DGA Assistant chatbot (optional, but recommended)

The floating chat assistant answers two kinds of questions: general DGA/transformer
domain knowledge (answered directly), and questions about the currently loaded
dataset (answered by writing and running SQL against the SQLite snapshot in
`backend/data/dga.db`). It refuses anything outside that scope, and answers in
whichever language you asked in (English or Vietnamese).

This needs a free [OpenRouter](https://openrouter.ai) API key (OpenRouter proxies to
several no-cost ":free" models — no payment method required):

1. Sign in at https://openrouter.ai (Google/GitHub login works), then create a key at
   https://openrouter.ai/keys
2. Create `backend/.env` (gitignored) with:
   ```
   OPENROUTER_API_KEY=sk-or-v1-...
   ```
3. Restart the backend.

Without a key, `/chat` still works — it silently falls back to a smaller rule-based
responder instead of erroring. The model used is set by `DGA_CHAT_MODEL` (default
`openai/gpt-oss-20b:free`); if OpenRouter retires that free slug, swap in another one
from https://openrouter.ai/models?max_price=0 without touching any code.

## Using the dashboard

Once signed in: open the **Data Source** panel (gear icon, top right) and drop in a
CSV/XLSX file — prediction runs automatically as soon as it's selected, no extra
button. (Pasting JSON rows instead still has a **Run prediction** button, since
there's no natural "upload" moment for typed text.) Processing a few-thousand-row
dataset (six diagnostic methods × severity scoring × fleet ranking)
can take up to a minute — the panel shows a running timer so it's clear it's still
working, not stuck.

Uploads **accumulate** rather than replace: each new file is merged with everything
uploaded before (deduped by transformer + sample date — the newest upload wins on an
exact-date conflict), then the *entire* combined dataset is re-cleaned, re-scored, and
re-ranked. Different files can have different columns/formats — each is normalized by
`clean_dataset.py` independently before merging. Click **Clear data** (trash icon) to
wipe the accumulated history and start fresh on the next upload.

From there:
- **Overview** (`/`) — fleet-wide status counts and the risk-ranked table
- **Analytics** (`/analytics`) — severity/fault-type distribution, top-5-risk trend
- **Fleet** (`/fleet`) — transformer directory grouped by station
- **Transformer detail** (`/transformer/:id`) — gas indicators, a severity trend chart
  (colored by status) and gas trend chart (6 gases, toggleable), the six-method
  diagnostic switcher, the fleet-ranking score breakdown, and a single searchable/
  sortable history table combining every sample record's gas readings with each
  method's individual vote for that record — plus field inspection confirmation
- **DGA Assistant** (floating button, resizable and remembers conversation context
  within the session) — asks the backend's `/chat` endpoint questions
  scoped to the currently loaded dataset

## Helper scripts

`run_all.ps1` / `run_all.bat` / `run_all.sh` install `backend/requirements.txt` and
run `train_models.py`; pass `-StartApi` (PowerShell) to also launch the Flask server
afterward. These are for the offline training workflow — for normal dashboard use,
just run the two `npm run dev` / `python app.py` commands above.

## Running the full B1..B6 pipeline manually (feature -> weak labels -> student -> severity -> ranking -> evaluation)

This project provides scripts that implement the pipeline described in the repository:

B1. Feature engineering (assume `backend/dataset/processed/dga_unlabeled.parquet` contains the time-series features per sample)
B2. Weak Supervision Labeling (traditional methods -> label model -> probabilistic labels)
B3. Train student classifier on probabilistic labels (LightGBM by default)
B4. Compute per-sample severity scores (domain-driven)
B5. Aggregate per-transformer ranking (latest sample + EWMA + bonuses/penalties)
B6. Produce evaluation reports (LF stats, ranking CSV)

Below are the exact manual commands to reproduce the pipeline on Windows (PowerShell examples). All outputs are written under `backend/dataset/processed/`, `backend/models/` and `reports/`.

Prerequisites (from repository root, Windows PowerShell):

```powershell
# create and activate venv (one-time)
python -m venv .venv
.\.venv\Scripts\Activate.ps1   # PowerShell
# (or) .\.venv\Scripts\activate.bat  # cmd.exe

# install required Python packages (in venv)
pip install -r backend/requirements.txt
# ensure parquet support
pip install pyarrow
```

Step 0 — prepare data

- Place your cleaned, feature-engineered per-sample time-series table at:
  `backend/dataset/processed/dga_unlabeled.parquet`
  Each row should be one sample with columns: transformer_id, sample_day, and the numeric features (CO, CO2, H2, C2H4, C2H6, tdcg, derived rates/trends). The repo contains a small synthetic example for testing.

Step 1 — (Optional, safe) Normalize legacy labels and generate initial reports

```powershell
# This script will create backups under backend/dataset/processed/backups_uncertain/
# and will replace literal 'UNCERTAIN' strings in Parquet string columns with 'ABSTAIN' if found.
python backend/maintenance_replace_and_report.py
```

- Output: `reports/lf_stats.csv` and `reports/transformer_ranking.csv` (initial)
- NOTE: backups are created before any destructive change. The script will report how many replacements were made.

Step 2 — Generate weak (pseudo) labels only (useful to inspect before training)

```powershell
# Generates backend/dataset/processed/dga_weak_labels.parquet and does NOT train student
python backend/train_models.py --weak-supervision --weak-only --use-snorkel
```

- Flags:
  - `--weak-supervision`: run the traditional diagnostics and label-model step, write `dga_weak_labels.parquet`.
  - `--weak-only`: stop after generating weak labels (do not train models).
  - `--use-snorkel`: attempt to use Snorkel's LabelModel; if Snorkel is not installed the code will fall back to the built-in EM-style estimator.

Step 3 — Generate weak labels and train the student + severity + temporal models (end-to-end)

```powershell
# This runs weak supervision then trains the student and severity models, then saves models under backend/models/
python backend/train_models.py --weak-supervision --use-snorkel
```

- Outputs:
  - `backend/dataset/processed/dga_weak_labels.parquet` (per-sample P(y) columns: `weak_prob_<GROUP>`, `weak_fault_group`, `weak_fault_confidence`)
  - `backend/models/fault_classifier.joblib`, `backend/models/severity_classifier.joblib`, `backend/models/severity_regressor.joblib`, and temporal model artifacts

Step 4 — Produce final reports (LF stats + transformer ranking)

```powershell
# Re-run the maintenance/report script to build LF statistics and the final fleet ranking CSV
python backend/maintenance_replace_and_report.py
```

- Outputs (examples):
  - `reports/lf_stats.csv` — per-LF abstain rates and vote counts
  - `reports/transformer_ranking.csv` — per-transformer final_score and rank (normalized 0–100)

Quick troubleshooting

- If you see `Snorkel is not installed; falling back to the built-in weak supervision estimator.`, either install Snorkel (`pip install snorkel`) or allow fallback (EM) to run.
- If Parquet read/write raises an error, ensure `pyarrow` is installed: `pip install pyarrow`.
- If your dataset is large and LightGBM runs out of memory, reduce rows or increase machine memory; you can also use `--weak-supervision --weak-only` to inspect labels before training.

Debug logs and where to look

- Backend structured logs are written to `backend/logs/pipeline.log`. This file contains timestamps, module names and level (INFO/DEBUG/ERROR) — useful when diagnosing why Snorkel import failed or why a subprocess returned a non-zero exit code.
- To run the pipeline manually and watch logs:

```powershell
# Activate venv (from repo root)
.\.venv\Scripts\Activate.ps1
# Run the full pipeline (writes weak labels, trains student, saves models and reports)
python backend/train_models.py --weak-supervision --use-snorkel
# Check the logs (tail)
Get-Content backend\logs\pipeline.log -Tail 200 -Wait
```

- The Flask API endpoint `/pipeline/run` also captures stdout/stderr from the training and maintenance subprocesses and writes them to the same log file. If you trigger the pipeline from the UI and it fails, inspect `backend/logs/pipeline.log` and the returned JSON from the endpoint (it includes stdout/stderr on error).

Automated run script

- There is `run_all.ps1` at the repository root (Windows) that installs backend requirements and runs `train_models.py`. Use it if you want a single command to install and run the training flow.

Where the important outputs land

- backend/dataset/processed/dga_weak_labels.parquet — weak supervision probabilistic labels
- backend/dataset/processed/backups_uncertain/ — backups of original Parquet files before UNCERTAIN→ABSTAIN replacement
- backend/models/ — trained LightGBM/XGBoost/PyTorch models
- reports/lf_stats.csv, reports/transformer_ranking.csv — LF statistics and fleet ranking

If you'd like, I'll commit this README update (so it's persisted). I can also add a short example PowerShell snippet to `run_all.ps1` to call the exact recommended sequence (maintenance -> weak-only -> train -> reports).
## Docker (backend only)

```bash
docker build -t dga-backend .
docker run -p 5000:5000 dga-backend
```

Builds and runs `backend/` behind Flask's dev server (see `Dockerfile`). The frontend
isn't containerized here — run it with `npm run dev` / `npm run build && npm start`.

## K?t lu?n ng?n

Chi?n lu?c d�ng Snorkel + student classifier l� h?p l� cho DGA kh�ng nh�n. �? c� k?t qu? tin c?y c?n:

1. Thi?t k? labeling functions (LFs) c?n tr?ng v� ghi nh?n ABSTAIN khi m?t LF kh�ng th? quy?t d?nh.
2. D�ng label model (Snorkel n?u c�) ho?c fallback generative model d? u?c lu?ng d? ch�nh x�c c?a t?ng LF v� h?p nh?t votes th�nh probabilistic labels.
3. Hu?n luy?n student model v?i soft labels v�/ho?c sample weights d?a tr�n confidence t? label model (v� d? LightGBM v?i sample weights, ho?c MLP v?i soft targets).
4. T�ch h?p temporal features + EWMA d? t?ng h?p per-transformer ranking (hi?n t?i tr?ng s? l?n nhung v?n gi? s? d?ng l?ch s?).
5. Th�m active learning / expert review cho c�c m?u c� d? kh�ng ch?c ch?n (low confidence) d? c?i thi?n LF v� label model theo v�ng l?p.
6. B? sung unsupervised anomaly detection v� clustering d? ph�t hi?n fault types chua c� trong c�c phuong ph�p truy?n th?ng.
