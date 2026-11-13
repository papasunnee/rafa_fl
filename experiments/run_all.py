"""
run_all.py — Full experiment grid (Section 4.8).

Iterates over methods x alpha x N x seeds x {drift, no-drift}, plus the ablation
grid at the primary condition. Each job runs in its own subprocess (`python -m
experiments.run_experiment ...`) with a hard timeout, rather than calling
run() in-process:

  - Ray's Windows simulation backend has been observed to hang indefinitely
    (zero progress, no exception raised) after a transient "Failed to
    register worker to Raylet" error -- a plain try/except retry can't catch
    a hang, since nothing ever raises. A subprocess with a timeout can be
    force-killed and retried instead.
  - Running multiple such simulations concurrently (e.g. sharding by
    num_clients across parallel processes) was tried and found to be
    UNRELIABLE on this Ray/Windows setup: confirmed failures included fatal
    Raylet deaths and a native Windows access-violation crash inside Ray's
    actor-invocation code, not catchable at the Python level at all. Stick to
    one run_all.py process at a time.

Idempotent: a job is skipped if its output JSON already exists in --out, so
this is safe to interrupt and re-run after a crash/hang/manual stop.

Usage:
    python -m experiments.run_all --out results/
    python -m experiments.run_all --quick    # small grid for a smoke test
"""

import os, sys, argparse, itertools, subprocess
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import EXPERIMENT
from experiments.run_experiment import result_tag, ABLATIONS

BASELINES = ["fedavg", "fedprox", "scaffold", "fedcs", "oort"]
ALL_METHODS = ["rafa_fl"] + BASELINES
THETA_RHO_GRID = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50]  # sweep around the config default (0.20)
MAX_ATTEMPTS = 3      # retries before giving up on a job
JOB_TIMEOUT_SECONDS = 1800  # jobs take ~7.5 min with three shards sharing the CPU; 600 s
                            # was too close. A real hang is still killed after 30 min.
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _already_done(out_dir, method, alpha, num_clients, seed, drift, ablation=None, theta_rho=None):
    tag = result_tag(method, alpha, num_clients, seed, drift, ablation, theta_rho)
    return os.path.exists(os.path.join(out_dir, f"{tag}.json"))


def _kill_process_tree(pid):
    # Popen.kill() only signals the direct child; Ray spawns further child
    # processes (raylet, gcs_server, actor workers) that would otherwise be
    # orphaned on timeout. taskkill /T kills the whole tree.
    subprocess.run(["taskkill", "/F", "/T", "/PID", str(pid)],
                   capture_output=True)


def _run_job_subprocess(method, alpha, num_clients, seed, drift, out_dir, ablation=None, theta_rho=None):
    cmd = [sys.executable, "-m", "experiments.run_experiment",
          "--method", method, "--alpha", str(alpha), "--num_clients", str(num_clients),
          "--seed", str(seed), "--out", out_dir]
    if drift:
        cmd.append("--drift")
    if ablation:
        cmd += ["--ablation", ablation]
    if theta_rho is not None:
        cmd += ["--theta_rho", str(theta_rho)]

    proc = subprocess.Popen(cmd, cwd=PROJECT_ROOT,
                            creationflags=subprocess.CREATE_NEW_PROCESS_GROUP)
    try:
        return proc.wait(timeout=JOB_TIMEOUT_SECONDS) == 0
    except subprocess.TimeoutExpired:
        print(f"  TIMED OUT after {JOB_TIMEOUT_SECONDS}s -- killing process tree (pid={proc.pid})")
        _kill_process_tree(proc.pid)
        proc.wait()
        return False


def _run_with_retries(label, *args, **kwargs):
    for attempt in range(1, MAX_ATTEMPTS + 1):
        if _run_job_subprocess(*args, **kwargs):
            return True
        print(f"  FAILED (attempt {attempt}/{MAX_ATTEMPTS}): {label}")
    print(f"  GAVE UP after {MAX_ATTEMPTS} attempts: {label}")
    return False


def main_grid(out_dir, quick=False, only_n=None):
    seeds = EXPERIMENT.seeds[:1] if quick else EXPERIMENT.seeds
    alphas = [EXPERIMENT.primary_alpha] if quick else EXPERIMENT.alpha_grid
    ns = [EXPERIMENT.primary_n] if quick else EXPERIMENT.num_clients_grid
    if only_n is not None:
        ns = [n for n in ns if n == only_n]
    drift_opts = [True] if quick else [False, True]

    jobs = list(itertools.product(ALL_METHODS, alphas, ns, seeds, drift_opts))
    print(f"Main grid: {len(jobs)} runs")
    for i, (method, alpha, n, seed, drift) in enumerate(jobs, 1):
        label = f"{method} a={alpha} n={n} s={seed} drift={drift}"
        if _already_done(out_dir, method, alpha, n, seed, drift):
            print(f"[{i}/{len(jobs)}] SKIP (already done): {label}")
            continue
        print(f"[{i}/{len(jobs)}] {label}")
        _run_with_retries(label, method, alpha, n, seed, drift, out_dir)


def ablation_grid(out_dir, quick=False, only_n=None):
    """Ablation variants at the primary condition under drift (Section 3.13)."""
    a, n = EXPERIMENT.primary_alpha, EXPERIMENT.primary_n
    if only_n is not None and only_n != n:
        print(f"Ablation grid: 0 runs (primary_n={n} != only_n={only_n})")
        return
    seeds = EXPERIMENT.seeds[:1] if quick else EXPERIMENT.seeds
    jobs = list(itertools.product(ABLATIONS.keys(), seeds))
    print(f"Ablation grid: {len(jobs)} runs")
    for i, (ablation, seed) in enumerate(jobs, 1):
        label = f"{ablation} a={a} n={n} s={seed} drift=True"
        if _already_done(out_dir, "rafa_fl", a, n, seed, True, ablation):
            print(f"[{i}/{len(jobs)}] SKIP (already done): {label}")
            continue
        print(f"[{i}/{len(jobs)}] {label}")
        _run_with_retries(label, "rafa_fl", a, n, seed, True, out_dir, ablation=ablation)


def sensitivity_grid(out_dir, quick=False):
    """Sweep theta_rho at the primary condition under drift (Chapter 5 sensitivity figure)."""
    a, n = EXPERIMENT.primary_alpha, EXPERIMENT.primary_n
    seeds = EXPERIMENT.seeds[:1] if quick else EXPERIMENT.seeds
    jobs = list(itertools.product(THETA_RHO_GRID, seeds))
    print(f"Sensitivity grid: {len(jobs)} runs")
    for i, (theta_rho, seed) in enumerate(jobs, 1):
        label = f"rafa_fl a={a} n={n} s={seed} drift=True theta_rho={theta_rho}"
        if _already_done(out_dir, "rafa_fl", a, n, seed, True, theta_rho=theta_rho):
            print(f"[{i}/{len(jobs)}] SKIP (already done): {label}")
            continue
        print(f"[{i}/{len(jobs)}] {label}")
        _run_with_retries(label, "rafa_fl", a, n, seed, True, out_dir, theta_rho=theta_rho)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/")
    ap.add_argument("--quick", action="store_true", help="small smoke-test grid")
    ap.add_argument("--skip_main", action="store_true")
    ap.add_argument("--skip_ablation", action="store_true")
    ap.add_argument("--sensitivity", action="store_true",
                    help="run the theta_rho sensitivity grid instead of main+ablation")
    ap.add_argument("--only_n", type=int, default=None,
                    help="restrict to one num_clients value")
    args = ap.parse_args()
    if args.sensitivity:
        sensitivity_grid(args.out, args.quick)
    else:
        if not args.skip_main:
            main_grid(args.out, args.quick, args.only_n)
        if not args.skip_ablation:
            ablation_grid(args.out, args.quick, args.only_n)
    print("Grid complete.")
