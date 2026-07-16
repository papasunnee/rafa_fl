"""
export_results.py — Flatten results/*.json into one compact JSON for the dashboard.

Manual/one-time script: run it whenever results/ changes, commit or otherwise
ship the output file, and the frontend reads that static file directly (no
backend call). Reuses analysis.report/analysis.metrics for the per-condition
summary so the numbers match `python -m analysis.report` exactly -- this
script does not recompute or reinterpret anything.

Usage:
    python ui/scripts/export_results.py --results_dir results/ \
        --out ui/frontend/src/data/results_export.json
"""

import os, sys, json, argparse

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO_ROOT)

from analysis.report import load_results, grid_table


def export(results_dir, out_path):
    runs = load_results(results_dir)
    if not runs:
        raise SystemExit(f"No result JSON files found in {results_dir}")

    # Per-run records: only fields actually present in the source JSON (see
    # experiments/run_experiment.py's result dict) -- no per-client data
    # exists in these logs, so none is synthesised here.
    run_records = [{
        "method": r["method"],                        # "rafa_fl" | baseline | ablation variant name
        "alpha": r["alpha"],                           # Dirichlet non-IID heterogeneity (lower = more skewed)
        "num_clients": r["num_clients"],               # simulated client population size (N)
        "seed": r["seed"],                             # random seed for this run
        "drift": r["drift"],                           # whether concept drift was injected mid-run
        "param_count": r["param_count"],               # model parameter count (for bytes_communicated math)
        "accuracy_curve": r["accuracy_curve"],         # global test accuracy after each round
        "macro_f1_curve": r["macro_f1_curve"],         # macro-averaged F1 after each round
        "round_log": r["round_log"],                   # [{round, active, deferred, cumulative_transmissions}]
        "total_transmissions": r["total_transmissions"],  # total messages sent across the whole run
    } for r in runs]

    # Per-CONDITION summary (one row per method/alpha/num_clients/drift,
    # aggregated across its seeds): n_seeds, and [mean, 95% CI] pairs for
    # peak_accuracy, rounds_to_convergence, total_transmissions, and
    # bytes_communicated. See analysis/report.py's grid_table() -- this is
    # that function's own output, not a re-derivation of it.
    summary_rows = grid_table(runs)

    payload = {
        "generated_from": results_dir,
        "n_runs": len(runs),
        "runs": run_records,
        "summary": summary_rows,
    }

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(payload, f)
    print(f"Wrote {out_path} ({len(runs)} runs, {len(summary_rows)} summary rows, "
         f"{os.path.getsize(out_path) / 1024:.0f} KB)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--results_dir", default=os.path.join(REPO_ROOT, "results"))
    ap.add_argument("--out", default=os.path.join(REPO_ROOT, "ui", "frontend", "src", "data", "results_export.json"))
    args = ap.parse_args()
    export(args.results_dir, args.out)
