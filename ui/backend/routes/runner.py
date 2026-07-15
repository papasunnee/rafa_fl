"""
runner.py — POST /run, GET /run/{job_id} (LOCAL DEV ONLY -- see main.py/README).

Launches `python -m experiments.run_experiment` exactly as documented in that
module's own CLI, as a subprocess, and streams its stdout back via polling.
Jobs are tracked in memory only (lost on backend restart) -- acceptable for a
local-dev-only tool with no persistence requirement.
"""

import os
import subprocess
import sys
import threading
import uuid

from fastapi import APIRouter
from pydantic import BaseModel

router = APIRouter()

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))

_jobs = {}
_jobs_lock = threading.Lock()


class RunRequest(BaseModel):
    method: str
    ablation: str | None = None
    alpha: float
    num_clients: int
    seed: int
    drift: bool = False


def _stream_job(job_id, cmd):
    proc = subprocess.Popen(
        cmd, cwd=REPO_ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, bufsize=1,
    )
    with _jobs_lock:
        _jobs[job_id]["proc"] = proc
    for line in proc.stdout:
        with _jobs_lock:
            _jobs[job_id]["output"] += line
    proc.wait()
    with _jobs_lock:
        _jobs[job_id]["status"] = "done" if proc.returncode == 0 else "error"
        _jobs[job_id]["returncode"] = proc.returncode


@router.post("/run")
def start_run(req: RunRequest):
    job_id = uuid.uuid4().hex[:12]
    cmd = [sys.executable, "-m", "experiments.run_experiment",
          "--method", req.method, "--alpha", str(req.alpha),
          "--num_clients", str(req.num_clients), "--seed", str(req.seed),
          "--out", "results/"]
    if req.drift:
        cmd.append("--drift")
    if req.ablation:
        cmd += ["--ablation", req.ablation]

    with _jobs_lock:
        _jobs[job_id] = {"status": "running", "output": "", "proc": None, "returncode": None}

    thread = threading.Thread(target=_stream_job, args=(job_id, cmd), daemon=True)
    thread.start()
    return {"job_id": job_id}


@router.get("/run/{job_id}")
def get_run(job_id: str):
    with _jobs_lock:
        job = _jobs.get(job_id)
    if job is None:
        return {"status": "not_found", "output": ""}
    return {"status": job["status"], "output": job["output"], "returncode": job["returncode"]}
