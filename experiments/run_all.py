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


def main_grid(out_dir, quick=False):
    seeds = EXPERIMENT.seeds[:1] if quick else EXPERIMENT.seeds
    alphas = [EXPERIMENT.primary_alpha] if quick else EXPERIMENT.alpha_grid
    ns = [EXPERIMENT.primary_n] if quick else EXPERIMENT.num_clients_grid
    drift_opts = [True] if quick else [False, True]

    jobs = list(itertools.product(ALL_METHODS, alphas, ns, seeds, drift_opts))
    print(f"Main grid: {len(jobs)} runs")
    for i, (method, alpha, n, seed, drift) in enumerate(jobs, 1):
        print(f"[{i}/{len(jobs)}] {method} a={alpha} n={n} s={seed} drift={drift}")
        try:
            run(method, alpha, n, seed, drift, out_dir)
        except Exception:
            print(f"  FAILED: {method} a={alpha} n={n} s={seed} drift={drift}")
            traceback.print_exc()


def ablation_grid(out_dir, quick=False):
    """Ablation variants at the primary condition under drift (Section 3.13)."""
    seeds = EXPERIMENT.seeds[:1] if quick else EXPERIMENT.seeds
    a, n = EXPERIMENT.primary_alpha, EXPERIMENT.primary_n
    jobs = list(itertools.product(ABLATIONS.keys(), seeds))
    print(f"Ablation grid: {len(jobs)} runs")
    for i, (ablation, seed) in enumerate(jobs, 1):
        print(f"[{i}/{len(jobs)}] {ablation} a={a} n={n} s={seed} drift=True")
        try:
            run("rafa_fl", a, n, seed, True, out_dir, ablation=ablation)
        except Exception:
            print(f"  FAILED ablation: {ablation} s={seed}")
            traceback.print_exc()


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
