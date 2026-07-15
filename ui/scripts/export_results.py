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
        "method": r["method"],
        "alpha": r["alpha"],
        "num_clients": r["num_clients"],
        "seed": r["seed"],
        "drift": r["drift"],
        "param_count": r["param_count"],
        "accuracy_curve": r["accuracy_curve"],
        "macro_f1_curve": r["macro_f1_curve"],
        "round_log": r["round_log"],
        "total_transmissions": r["total_transmissions"],
    } for r in runs]

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
