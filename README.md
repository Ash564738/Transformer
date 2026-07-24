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

## Docker (backend only)

```bash
docker build -t dga-backend .
docker run -p 5000:5000 dga-backend
```

Builds and runs `backend/` behind Flask's dev server (see `Dockerfile`). The frontend
isn't containerized here — run it with `npm run dev` / `npm run build && npm start`.
