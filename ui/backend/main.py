"""
main.py — FastAPI app entrypoint for the RAFA-FL UI backend (LOCAL DEV ONLY).

Must be run from the repo root so `from agent.reactive_agent import ReactiveAgent`
etc. resolve without path hacks, the same convention experiments/run_experiment.py
already relies on:

    python -m uvicorn ui.backend.main:app --reload --port 8000

The /run endpoint (routes/runner.py) executes real training subprocesses on
whatever machine hosts this backend -- do not deploy this backend publicly.
The frontend's Runner page is separately gated behind VITE_ENABLE_RUNNER, off
by default in production builds, but that only hides the UI; this backend
itself should never be exposed outside local development.
"""

import os
import sys

# Repo root must be importable so `agent`, `config`, etc. resolve regardless
# of the working directory uvicorn happens to be launched from.
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .routes import playground, runner

app = FastAPI(title="RAFA-FL UI backend (local dev only)")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(playground.router)
app.include_router(runner.router)


@app.get("/health")
def health():
    return {"status": "ok"}
