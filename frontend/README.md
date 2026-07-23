# DGA Monitor — frontend

Next.js dashboard for the Transformer Degradation Dashboard (see the repo-root
[README](../README.md) for the full setup). This app renders fleet overview,
analytics, per-transformer detail, and the DGA Assistant chat — all backed by the
Flask API in `../backend/`.

## Run it

```bash
npm install
npm run dev
```

Open [http://localhost:3000](http://localhost:3000). You'll be redirected to
`/login`. There's no registration page — the single login is set from the backend
with `python seed_user.py <email> <password>` (see the root README).

The Flask backend must be running separately at `http://127.0.0.1:5000` (see the
root README's "Backend setup"). Requests go **directly from the browser to Flask**
(`src/lib/api.ts`), not through a Next.js rewrite — long-running `/predict` calls on
large datasets were getting reset by the dev-server proxy. Point at a different
backend with:

```
NEXT_PUBLIC_BACKEND_URL=http://your-backend-host:5000
```

in `.env.local`.

## Structure

- `src/app/` — routes: `/` (overview), `/analytics`, `/fleet`,
  `/transformer/[id]`, `/login`
- `src/components/layout/` — app shell, top nav, data-source upload panel
- `src/components/detail/` — transformer-detail widgets (gas/severity trend charts,
  diagnostic switcher, fault-type history table, ranking breakdown, confirmation dialog)
- `src/components/chat/` — the floating DGA Assistant
- `src/components/auth/` — login shell and the route guard
- `src/store/` — Zustand stores (`use-dashboard-store.ts` for the loaded dataset and
  chat, `use-auth-store.ts` for the signed-in user)
- `src/lib/api.ts` — the only place that calls the Flask backend
- `src/app/api/confirmations/route.ts` — the one local (non-Flask) API route, a
  file-backed store for field-inspection confirmations

## Build

```bash
npm run build
npm start
```
