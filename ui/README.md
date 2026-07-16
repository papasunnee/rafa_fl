# RAFA-FL demo UI

Three independent pieces, all under `ui/`. Nothing outside this folder is touched.

**Every Python command below (export script and backend) needs the project's
`venv` active** — they import `numpy`/`torch`/etc. from the main repo, which
your system Python won't have:

```powershell
# from repo root, PowerShell
.\venv\Scripts\Activate.ps1
```

## 1. Export results for the Dashboard

Run whenever `results/` changes (one-time/manual, not triggered from the UI):

```bash
# from repo root, venv active
python ui/scripts/export_results.py
```

Writes `ui/frontend/src/data/results_export.json`, reusing `analysis/report.py`'s
own summary logic so the numbers match `python -m analysis.report` exactly.

## 2. Backend (FastAPI)

Must run from the **repo root**, not from inside `ui/backend/`, so
`from agent.reactive_agent import ReactiveAgent` etc. resolve without path hacks:

```bash
# from repo root, with the project venv active
pip install -r ui/backend/requirements.txt
python -m uvicorn ui.backend.main:app --reload --port 8000
```

- `POST /decide` — Agent Playground. Calls the real `ReactiveAgent.decide()`.
- `POST /run`, `GET /run/{job_id}` — Runner (local dev only, see below).

## 3. Frontend (Vite + React + Tailwind + Recharts)

```bash
cd ui/frontend
npm install
npm run dev       # dev server, proxies /decide and /run to :8000
npm run build      # production build
```

Pages:
- **Dashboard** (`/`) — reads `src/data/results_export.json` directly, no backend needed.
  Shows per-condition summary stats and real per-round accuracy/transmission curves for
  whatever's already in that exported file.
- **Agent Playground** (`/playground`) — sliders/inputs feeding the live `/decide` endpoint.
  Doesn't need `results/` or a training run at all; it's the agent's one-shot decision rule
  in isolation, for building intuition about the 5 priority rules (P1-P5).
- **Runner** (`/runner`) — form that launches a real `experiments.run_experiment` subprocess.
  Local dev only, see gating below.

### How the three pages relate

```
Runner  --(new results/*.json)-->  export_results.py  --(results_export.json)-->  Dashboard
Playground -- independent, talks straight to the agent, doesn't touch results/ at all
```

The Dashboard is a *snapshot*, not live: results from a Runner run won't show up
there until you manually re-run step 1 (`export_results.py`) to refresh the export.

### Runner gating

The Runner triggers real, possibly multi-minute training subprocesses on
whatever machine hosts the backend. It's gated by `VITE_ENABLE_RUNNER`:

- `.env` → `true` (visible in `npm run dev`)
- `.env.production` → `false` (hidden in `npm run build`, Vite's default
  production mode)

This only hides the *frontend UI*. The backend itself (`ui/backend/main.py`)
should never be deployed publicly — its `/run` endpoint has no auth and will
happily execute training jobs for anyone who can reach it.
