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

- `backend/app.py` — the Flask server: `/auth/*`, `/predict`, `/chat`, `/chart/*`
- `backend/inference_service.py` — orchestrates clean → feature-engineer → diagnose → score → rank
- `backend/dga/` — the six traditional diagnostic methods (Duval Triangle, Duval Pentagon,
  IEC 60599, Rogers Ratio, Doernenburg, Key Gas)
- `backend/consensus.py`, `backend/severity.py`, `backend/ranking.py` — combine the six
  methods' votes into a consensus fault, score severity, and rank the fleet
- `backend/train_models.py` — offline training for the LightGBM/XGBoost/PyTorch models
  (not required to run the live dashboard — the pipeline above is rule-based)
- `backend/auth.py` — local SQLite-backed login/registration

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

## Using the dashboard

Once signed in: open the **Data Source** panel (gear icon, top right) and drop in a
CSV/XLSX file — prediction runs automatically as soon as it's selected, no extra
button. (Pasting JSON rows instead still has a **Run prediction** button, since
there's no natural "upload" moment for typed text.) Processing a few-thousand-row
dataset (six diagnostic methods × severity scoring × fleet ranking)
can take up to a minute — the panel shows a running timer so it's clear it's still
working, not stuck.

From there:
- **Overview** (`/`) — fleet-wide status counts and the risk-ranked table
- **Analytics** (`/analytics`) — severity/fault-type distribution, top-5-risk trend
- **Fleet** (`/fleet`) — transformer directory grouped by station
- **Transformer detail** (`/transformer/:id`) — gas indicators, severity trend over
  time, the six-method diagnostic switcher, a fault-type history table (every sample
  record, not just the latest), the fleet-ranking score breakdown, and field
  inspection confirmation
- **DGA Assistant** (floating button) — asks the backend's `/chat` endpoint questions
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
