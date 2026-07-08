"""
run_all.py — Full experiment grid (Section 4.8).

Iterates over methods x alpha x N x seeds x {drift, no-drift}, plus the ablation
grid at the primary condition. Runs sequentially to avoid Ray/outer-loop
parallelism conflicts in Flower simulation mode.

Usage:
    python -m experiments.run_all --out results/
    python -m experiments.run_all --quick    # small grid for a smoke test
"""

import os, sys, argparse, itertools, traceback
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import EXPERIMENT
from experiments.run_experiment import run, ABLATIONS

BASELINES = ["fedavg", "fedprox", "scaffold", "fedcs", "oort"]
ALL_METHODS = ["rafa_fl"] + BASELINES
MAX_ATTEMPTS = 3   # Flower/Ray simulation is occasionally flaky between back-to-back
                   # runs in one long-lived process; retry before giving up on a job.


def _run_with_retries(label, *args, **kwargs):
    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            run(*args, **kwargs)
            return True
        except Exception:
            print(f"  FAILED (attempt {attempt}/{MAX_ATTEMPTS}): {label}")
            traceback.print_exc()
    print(f"  GAVE UP after {MAX_ATTEMPTS} attempts: {label}")
    return False


def main_grid(out_dir, quick=False):
    seeds = EXPERIMENT.seeds[:1] if quick else EXPERIMENT.seeds
    alphas = [EXPERIMENT.primary_alpha] if quick else EXPERIMENT.alpha_grid
    ns = [EXPERIMENT.primary_n] if quick else EXPERIMENT.num_clients_grid
    drift_opts = [True] if quick else [False, True]

    jobs = list(itertools.product(ALL_METHODS, alphas, ns, seeds, drift_opts))
    print(f"Main grid: {len(jobs)} runs")
    for i, (method, alpha, n, seed, drift) in enumerate(jobs, 1):
        label = f"{method} a={alpha} n={n} s={seed} drift={drift}"
        print(f"[{i}/{len(jobs)}] {label}")
        _run_with_retries(label, method, alpha, n, seed, drift, out_dir)


def ablation_grid(out_dir, quick=False):
    """Ablation variants at the primary condition under drift (Section 3.13)."""
    seeds = EXPERIMENT.seeds[:1] if quick else EXPERIMENT.seeds
    a, n = EXPERIMENT.primary_alpha, EXPERIMENT.primary_n
    jobs = list(itertools.product(ABLATIONS.keys(), seeds))
    print(f"Ablation grid: {len(jobs)} runs")
    for i, (ablation, seed) in enumerate(jobs, 1):
        label = f"{ablation} a={a} n={n} s={seed} drift=True"
        print(f"[{i}/{len(jobs)}] {label}")
        _run_with_retries(label, "rafa_fl", a, n, seed, True, out_dir, ablation=ablation)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/")
    ap.add_argument("--quick", action="store_true", help="small smoke-test grid")
    ap.add_argument("--skip_main", action="store_true")
    ap.add_argument("--skip_ablation", action="store_true")
    args = ap.parse_args()
    if not args.skip_main:
        main_grid(args.out, args.quick)
    if not args.skip_ablation:
        ablation_grid(args.out, args.quick)
    print("Grid complete.")
