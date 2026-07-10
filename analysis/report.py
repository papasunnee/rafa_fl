"""
report.py — Aggregate results/*.json into Chapter 5 tables (Sections 3.11, 3.12).

Loads every per-run JSON log, summarises each with analysis.metrics.summarise_run,
groups by (method, alpha, num_clients, drift) across seeds, and prints:
  - a full grid table (mean +/- 95% CI per metric, across available seeds)
  - a paired significance table at the primary condition (RAFA-FL vs each baseline)
  - an ablation comparison at the primary condition under drift

Safe to run against a partially-complete results/ directory (e.g. while
experiments.run_all is still writing files) — unreadable/incomplete files are
skipped, and comparisons requiring >=2 common seeds are reported as
"insufficient data" rather than failing.

Usage:
    python -m analysis.report --results_dir results/
    python -m analysis.report --results_dir results/ --metric total_transmissions
"""

import os, sys, json, glob, argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import EXPERIMENT
from analysis.metrics import summarise_run
from analysis.stats import compare_all, mean_ci

BASELINES = ["fedavg", "fedprox", "scaffold", "fedcs", "oort"]
ABLATIONS = ["rafa_no_drift", "rafa_no_resource", "rafa_fixed_epochs"]
GRID_METRICS = ["peak_accuracy", "rounds_to_convergence", "total_transmissions", "bytes_communicated"]


def load_results(results_dir):
    runs = []
    for path in sorted(glob.glob(os.path.join(results_dir, "*.json"))):
        try:
            with open(path) as f:
                d = json.load(f)
            d["_summary"] = summarise_run(
                d["accuracy_curve"], d["macro_f1_curve"],
                d.get("total_transmissions") or 0, d["param_count"],
            )
            runs.append(d)
        except (json.JSONDecodeError, KeyError, OSError):
            continue          # file still being written, or malformed
    return runs


def group_by(runs, keys):
    groups = {}
    for r in runs:
        key = tuple(r[k] for k in keys)
        groups.setdefault(key, []).append(r)
    return groups


def grid_table(runs):
    """One row per (method, alpha, num_clients, drift), mean +/- CI across seeds."""
    groups = group_by(runs, ["method", "alpha", "num_clients", "drift"])
    rows = []
    for key, group in sorted(groups.items(), key=lambda kv: (str(kv[0][0]), kv[0][1], kv[0][2], kv[0][3])):
        method, alpha, n, drift = key
        row = {"method": method, "alpha": alpha, "num_clients": n, "drift": drift, "n_seeds": len(group)}
        for m in GRID_METRICS:
            vals = [g["_summary"][m] for g in group if g["_summary"].get(m) is not None]
            row[m] = mean_ci(vals) if vals else (None, None)
        rows.append(row)
    return rows


def _paired_by_seed(groups, key, metric):
    if key not in groups:
        return None
    return {g["seed"]: g["_summary"][metric] for g in groups[key]}


def significance_table(runs, alpha, num_clients, drift, metric="peak_accuracy"):
    """Paired RAFA-FL vs each baseline at one (alpha, num_clients, drift) condition."""
    groups = group_by(runs, ["method", "alpha", "num_clients", "drift"])
    rafa_by_seed = _paired_by_seed(groups, ("rafa_fl", alpha, num_clients, drift), metric)
    if not rafa_by_seed:
        return None

    baselines_by_seed = {}
    for b in BASELINES:
        vals = _paired_by_seed(groups, (b, alpha, num_clients, drift), metric)
        if vals:
            baselines_by_seed[b] = vals
    if not baselines_by_seed:
        return None

    common = sorted(set(rafa_by_seed).intersection(*[set(v) for v in baselines_by_seed.values()]))
    if len(common) < 2:
        return None

    rafa_vals = [rafa_by_seed[s] for s in common]
    baseline_vals = {b: [v[s] for s in common] for b, v in baselines_by_seed.items()}
    # Fixed family size (EXPERIMENT.bonferroni_baselines), not len(baseline_vals) --
    # this table is safe to run against a partially-complete results/ directory,
    # and inferring the correction from whatever baselines happen to have
    # results yet would silently loosen it (interim peeking) until all 5 finish.
    result = compare_all(rafa_vals, baseline_vals, alpha=EXPERIMENT.alpha_significance,
                         num_comparisons=EXPERIMENT.bonferroni_baselines)
    result["n_common_seeds"] = len(common)
    if len(baseline_vals) < len(BASELINES):
        result["incomplete_baselines"] = sorted(set(BASELINES) - set(baseline_vals))
    return result


def ablation_table(runs, metric="peak_accuracy"):
    """Paired rafa_full vs each ablation variant, at the primary condition under drift."""
    groups = group_by(runs, ["method", "alpha", "num_clients", "drift"])
    key = ("rafa_full", EXPERIMENT.primary_alpha, EXPERIMENT.primary_n, True)
    full_by_seed = _paired_by_seed(groups, key, metric)
    if not full_by_seed:
        return None

    variants_by_seed = {}
    for a in ABLATIONS:
        vals = _paired_by_seed(groups, (a, EXPERIMENT.primary_alpha, EXPERIMENT.primary_n, True), metric)
        if vals:
            variants_by_seed[a] = vals
    if not variants_by_seed:
        return None

    common = sorted(set(full_by_seed).intersection(*[set(v) for v in variants_by_seed.values()]))
    if len(common) < 2:
        return None

    full_vals = [full_by_seed[s] for s in common]
    variant_vals = {a: [v[s] for s in common] for a, v in variants_by_seed.items()}
    # Fixed family size (len(ABLATIONS)), same reasoning as significance_table.
    result = compare_all(full_vals, variant_vals, alpha=EXPERIMENT.alpha_significance,
                         num_comparisons=len(ABLATIONS))
    result["n_common_seeds"] = len(common)
    if len(variant_vals) < len(ABLATIONS):
        result["incomplete_ablations"] = sorted(set(ABLATIONS) - set(variant_vals))
    return result


def full_grid_significance(runs, metric="peak_accuracy"):
    """
    significance_table() at every (alpha, num_clients, drift) condition in the
    grid, not just the primary one -- shows whether an advantage is universal
    or condition-specific. "Better" assumes higher is better for the metric;
    flip the sign for metrics where lower is better (e.g. total_transmissions).
    """
    rows = []
    for alpha in EXPERIMENT.alpha_grid:
        for n in EXPERIMENT.num_clients_grid:
            for drift in [False, True]:
                sig = significance_table(runs, alpha, n, drift, metric)
                rows.append({"alpha": alpha, "num_clients": n, "drift": drift, "sig": sig})
    return rows


def print_full_grid_significance(rows, metric, lower_is_better=False):
    print(f"\nSignificance across the full grid, metric={metric} "
          f"({'lower' if lower_is_better else 'higher'} is better for RAFA-FL)")
    header = f"{'alpha':6s} {'N':5s} {'drift':6s} {'rafa mean':16s} {'sig.better':12s} {'sig.worse':12s} {'n.s.':6s}"
    print(header)
    print("-" * len(header))
    for r in rows:
        sig = r["sig"]
        if sig is None:
            print(f"{r['alpha']:<6} {r['num_clients']:<5} {str(r['drift']):6s} "
                  f"{'insufficient data':16s}")
            continue
        better, worse, ns = [], [], []
        for name, c in sig["comparisons"].items():
            if not c["significant"]:
                ns.append(name)
            elif (c["cohens_d"] > 0) != lower_is_better:
                better.append(name)
            else:
                worse.append(name)
        print(f"{r['alpha']:<6} {r['num_clients']:<5} {str(r['drift']):6s} "
              f"{sig['rafa_mean']:<16.4f} {len(better):<12} {len(worse):<12} {len(ns):<6}"
              f"  better={better} worse={worse}")


def fmt(pair, decimals=4):
    mean, ci = pair
    if mean is None:
        return "n/a"
    return f"{mean:.{decimals}f}+/-{ci:.{decimals}f}"


def print_grid_table(rows):
    header = (f"{'method':18s} {'alpha':6s} {'N':5s} {'drift':6s} {'seeds':6s} "
              f"{'peak_acc':16s} {'rounds_conv':14s} {'transmissions':16s} {'bytes':16s}")
    print(header)
    print("-" * len(header))
    for r in rows:
        print(f"{str(r['method']):18s} {r['alpha']:<6} {r['num_clients']:<5} {str(r['drift']):6s} "
              f"{r['n_seeds']:<6} {fmt(r['peak_accuracy']):16s} {fmt(r['rounds_to_convergence'], 1):14s} "
              f"{fmt(r['total_transmissions'], 1):16s} {fmt(r['bytes_communicated'], 0):16s}")


def print_significance(sig, title):
    print(f"\n{title}")
    if sig is None:
        print("  insufficient data (need >=2 common seeds across all methods being compared)")
        return
    if sig.get("incomplete_baselines"):
        print(f"  NOTE: still missing results for {sig['incomplete_baselines']} -- "
              f"verdicts below use the full pre-registered Bonferroni family size "
              f"({sig['bonferroni_threshold']:.5f}) but only reflect the baselines present so far.")
    if sig.get("incomplete_ablations"):
        print(f"  NOTE: still missing results for {sig['incomplete_ablations']} -- "
              f"verdicts below use the full pre-registered Bonferroni family size "
              f"({sig['bonferroni_threshold']:.5f}) but only reflect the variants present so far.")
    print(f"  rafa mean={sig['rafa_mean']:.4f}+/-{sig['rafa_ci95']:.4f}  "
          f"n_seeds={sig['n_common_seeds']}  bonferroni_threshold={sig['bonferroni_threshold']:.5f}")
    for name, c in sig["comparisons"].items():
        flag = "*" if c["significant"] else " "
        print(f"  {flag} {name:16s} mean={c['baseline_mean']:.4f}+/-{c['baseline_ci95']:.4f}  "
              f"t={c['t']:.3f} p={c['p']:.4f} d={c['cohens_d']:.3f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results_dir", default="results/")
    ap.add_argument("--metric", default="peak_accuracy",
                    choices=["peak_accuracy", "final_accuracy", "final_macro_f1",
                             "rounds_to_convergence", "total_transmissions",
                             "bytes_communicated", "convergence_speed"])
    ap.add_argument("--out_json", default=None, help="optional path to dump the grid table as JSON")
    ap.add_argument("--full_grid", action="store_true",
                    help="test RAFA-FL vs baselines at every (alpha, N, drift) condition, not just the primary one")
    ap.add_argument("--lower_is_better", action="store_true",
                    help="pass when --metric is one where lower is better for RAFA-FL (e.g. total_transmissions)")
    args = ap.parse_args()

    runs = load_results(args.results_dir)
    if not runs:
        print(f"No result JSON files found in {args.results_dir}")
        return
    print(f"Loaded {len(runs)} run(s) from {args.results_dir}\n")

    rows = grid_table(runs)
    print_grid_table(rows)

    if args.full_grid:
        print_full_grid_significance(full_grid_significance(runs, args.metric),
                                     args.metric, args.lower_is_better)
        if args.out_json:
            with open(args.out_json, "w") as f:
                json.dump(rows, f, indent=2)
            print(f"\nWrote {args.out_json}")
        return

    print_significance(
        significance_table(runs, EXPERIMENT.primary_alpha, EXPERIMENT.primary_n, True, args.metric),
        f"Significance @ primary condition (alpha={EXPERIMENT.primary_alpha}, "
        f"N={EXPERIMENT.primary_n}, drift=True), metric={args.metric}")

    print_significance(
        significance_table(runs, EXPERIMENT.primary_alpha, EXPERIMENT.primary_n, False, args.metric),
        f"Significance @ primary condition (alpha={EXPERIMENT.primary_alpha}, "
        f"N={EXPERIMENT.primary_n}, drift=False), metric={args.metric}")

    print_significance(
        ablation_table(runs, args.metric),
        f"Ablation comparison (rafa_full vs variants, drift=True), metric={args.metric}")

    if args.out_json:
        with open(args.out_json, "w") as f:
            json.dump(rows, f, indent=2)
        print(f"\nWrote {args.out_json}")


if __name__ == "__main__":
    main()
