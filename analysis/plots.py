"""
plots.py — Generate Chapter 5 thesis figures from results/*.json.

Figures:
  5.1 Accuracy convergence curves over rounds -- RAFA-FL vs all baselines
  5.2 Communication cost vs alpha in {0.1, 0.5, 1.0} (N=50)
  5.3 Per-client deferral rate distribution (primary condition, RAFA-FL)
  5.4 Ablation study accuracy over rounds -- full agent vs ablated variants
  5.5 Scalability: transmissions vs N in {10, 50, 100}
  5.6 Sensitivity analysis: key metrics vs theta_rho
  5.7 Mean accuracy after round 50 by alpha, drift vs no drift (N=50)
  5.8 RAFA-FL decision-rule mix before/after drift, drifted vs unaffected clients

5.6 needs the theta_rho sweep from `python -m experiments.run_all --sensitivity
--out results/`; the sweep runs sit in results/ next to the main grid and are
recognised by the theta_rho override recorded in their provenance.
Both are skipped with a message (not an error) if their data isn't there yet,
so this is safe to run against a partially-complete results/ directory.

Usage:
    python -m analysis.plots            # defaults: results/ -> figures/
"""

import os, sys, glob, argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from config import EXPERIMENT
from analysis.report import load_results, group_by, BASELINES, ABLATIONS
from analysis.metrics import deferral_by_client_from_log, summarise_run

# Categorical palette, fixed hue order (dataviz skill reference palette).
# RAFA-FL/rafa_full always take slot 1 so the "star of the show" reads the
# same color across every figure.
COLORS = {
    "rafa_fl": "#2a78d6", "rafa_full": "#2a78d6",
    "fedavg": "#eb6834", "rafa_no_drift": "#eb6834",
    "fedprox": "#1baf7a", "rafa_no_resource": "#1baf7a",
    "scaffold": "#eda100", "rafa_fixed_epochs": "#eda100",
    "fedcs": "#e87ba4", "rafa_equal_weight": "#e87ba4",
    "oort": "#008300",
}
INK = "#0b0b0b"
MUTED = "#898781"
GRID = "#e1e0d9"

LABELS = {
    "rafa_fl": "RAFA-FL", "fedavg": "FedAvg", "fedprox": "FedProx",
    "scaffold": "SCAFFOLD", "fedcs": "FedCS", "oort": "Oort",
    "rafa_full": "Full agent", "rafa_no_drift": "No drift-awareness",
    "rafa_no_resource": "No resource-awareness", "rafa_fixed_epochs": "Fixed epochs",
    "rafa_equal_weight": "Equal weight ($\\lambda$=0.5)",
}


def _style_axes(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(MUTED)
    ax.spines["bottom"].set_color(MUTED)
    ax.tick_params(colors=MUTED)
    ax.yaxis.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.xaxis.label.set_color(INK)
    ax.yaxis.label.set_color(INK)
    ax.title.set_color(INK)


def _savefig(fig, out_path):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"Wrote {out_path}")


def fig_5_1_convergence(results_dir, out_path):
    runs = load_results(results_dir)
    a, n = EXPERIMENT.primary_alpha, EXPERIMENT.primary_n
    groups = group_by(runs, ["method", "alpha", "num_clients", "drift"])
    methods = ["rafa_fl"] + BASELINES

    fig, ax = plt.subplots(figsize=(7, 4.5))
    plotted = 0
    for m in methods:
        group = [g for g in groups.get((m, a, n, True), []) if g.get("accuracy_curve")]
        if not group:
            continue
        min_len = min(len(g["accuracy_curve"]) for g in group)
        curves = np.array([g["accuracy_curve"][:min_len] for g in group])
        mean = curves.mean(axis=0)
        rounds = np.arange(1, len(mean) + 1)
        ax.plot(rounds, mean, label=LABELS.get(m, m), color=COLORS.get(m, INK), linewidth=1.8)
        plotted += 1

    if plotted == 0:
        print(f"[5.1] SKIPPED -- no runs found at alpha={a}, N={n}, drift=True in {results_dir}")
        plt.close(fig)
        return

    ax.axvline(EXPERIMENT.drift_round, color=MUTED, linestyle=":", linewidth=1)
    ax.text(EXPERIMENT.drift_round + 1, ax.get_ylim()[0], "drift onset", color=MUTED,
            fontsize=8, va="bottom")
    central = os.path.join(results_dir, "centralised", "summary.json")
    if os.path.exists(central):
        import json
        with open(central) as f:
            c = json.load(f)
        ax.axhline(c["accuracy_mean"], color=INK, linestyle="--", linewidth=1,
                   label=f"Centralised reference ({c['accuracy_mean']:.3f})")

    ax.set_xlabel("Communication round")
    ax.set_ylabel("Test accuracy")
    ax.set_title(f"Accuracy convergence: RAFA-FL vs baselines ($\\alpha$={a}, N={n}, drift)")
    _style_axes(ax)
    ax.legend(frameon=False, fontsize=9)
    _savefig(fig, out_path)


def _grouped_bar(ax, x_labels, series, series_order, value_fn, err_fn=None):
    """series: dict[series_key] -> dict[x_label] -> group (list of runs)."""
    n_series = len(series_order)
    width = 0.8 / n_series
    x = np.arange(len(x_labels))
    for i, key in enumerate(series_order):
        vals, errs = [], []
        for xl in x_labels:
            group = series.get(key, {}).get(xl)
            if not group:
                vals.append(0.0)
                errs.append(0.0)
                continue
            vals.append(value_fn(group))
            errs.append(err_fn(group) if err_fn else 0.0)
        offset = (i - (n_series - 1) / 2) * width
        ax.bar(x + offset, vals, width, yerr=errs, capsize=2,
              label=LABELS.get(key, key), color=COLORS.get(key, INK))
    ax.set_xticks(x)
    ax.set_xticklabels([str(xl) for xl in x_labels])


def fig_5_2_comm_cost(results_dir, out_path):
    runs = load_results(results_dir)
    n = EXPERIMENT.primary_n
    groups = group_by(runs, ["method", "alpha", "num_clients", "drift"])
    methods = ["rafa_fl"] + BASELINES
    alphas = EXPERIMENT.alpha_grid

    series = {m: {a: groups.get((m, a, n, True)) for a in alphas} for m in methods}
    if not any(v for m in series.values() for v in m.values()):
        print(f"[5.2] SKIPPED -- no runs found at N={n}, drift=True in {results_dir}")
        return

    def mean_bytes(group):
        return float(np.mean([g["_summary"]["bytes_communicated"] for g in group]))

    def std_bytes(group):
        return float(np.std([g["_summary"]["bytes_communicated"] for g in group]))

    fig, ax = plt.subplots(figsize=(7, 4.5))
    _grouped_bar(ax, alphas, series, methods, mean_bytes, std_bytes)
    ax.set_xlabel("Dirichlet concentration $\\alpha$")
    ax.set_ylabel("Bytes communicated")
    ax.set_title(f"Communication cost vs $\\alpha$ (N={n}, drift)")
    _style_axes(ax)
    ax.legend(frameon=False, fontsize=8, ncol=2)
    _savefig(fig, out_path)


def fig_5_3_deferral(results_dir, out_path):
    runs = load_results(results_dir)
    a, n = EXPERIMENT.primary_alpha, EXPERIMENT.primary_n
    groups = group_by(runs, ["method", "alpha", "num_clients", "drift"])
    group = groups.get(("rafa_fl", a, n, True), [])

    rates = []
    for g in group:
        by_client = deferral_by_client_from_log(g.get("round_log", []))
        rates.extend(np.mean(v) for v in by_client.values() if v)

    if not rates:
        print(f"[5.3] SKIPPED -- no per-client deferral data in {results_dir} "
              f"for rafa_fl @ alpha={a}, N={n}, drift=True. Re-run "
              f"experiments.run_experiment for this condition (round_log now "
              f"records per-client deferral flags).")
        return

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.hist(rates, bins=20, range=(0, 1), color=COLORS["rafa_fl"], edgecolor="white", linewidth=0.5)
    ax.set_xlabel("Per-client deferral rate (fraction of rounds deferred)")
    ax.set_ylabel("Number of clients")
    ax.set_title(f"Per-client deferral rate distribution (RAFA-FL, $\\alpha$={a}, N={n})")
    _style_axes(ax)
    _savefig(fig, out_path)


def fig_5_4_ablation(results_dir, out_path):
    runs = load_results(results_dir)
    a, n = EXPERIMENT.primary_alpha, EXPERIMENT.primary_n
    groups = group_by(runs, ["method", "alpha", "num_clients", "drift"])
    variants = ["rafa_full"] + ABLATIONS

    fig, ax = plt.subplots(figsize=(7, 4.5))
    plotted = 0
    for v in variants:
        group = [g for g in groups.get((v, a, n, True), []) if g.get("accuracy_curve")]
        if not group:
            continue
        min_len = min(len(g["accuracy_curve"]) for g in group)
        curves = np.array([g["accuracy_curve"][:min_len] for g in group])
        mean = curves.mean(axis=0)
        rounds = np.arange(1, len(mean) + 1)
        ax.plot(rounds, mean, label=LABELS.get(v, v), color=COLORS.get(v, INK), linewidth=1.8)
        plotted += 1

    if plotted == 0:
        print(f"[5.4] SKIPPED -- no ablation runs found at alpha={a}, N={n}, drift=True in {results_dir}")
        plt.close(fig)
        return

    ax.set_xlabel("Communication round")
    ax.set_ylabel("Test accuracy")
    ax.set_title(f"Ablation study: full agent vs variants ($\\alpha$={a}, N={n}, drift)")
    _style_axes(ax)
    ax.legend(frameon=False, fontsize=9)
    _savefig(fig, out_path)


def fig_5_5_scalability(results_dir, out_path):
    runs = load_results(results_dir)
    a = EXPERIMENT.primary_alpha
    groups = group_by(runs, ["method", "alpha", "num_clients", "drift"])
    methods = ["rafa_fl"] + BASELINES
    ns = EXPERIMENT.num_clients_grid

    series = {m: {n: groups.get((m, a, n, True)) for n in ns} for m in methods}
    if not any(v for m in series.values() for v in m.values()):
        print(f"[5.5] SKIPPED -- no runs found at alpha={a}, drift=True in {results_dir}")
        return

    def mean_tx(group):
        return float(np.mean([g["_summary"]["total_transmissions"] for g in group]))

    def std_tx(group):
        return float(np.std([g["_summary"]["total_transmissions"] for g in group]))

    fig, ax = plt.subplots(figsize=(7, 4.5))
    _grouped_bar(ax, ns, series, methods, mean_tx, std_tx)
    ax.set_xlabel("Number of clients N")
    ax.set_ylabel("Total transmissions")
    ax.set_title(f"Scalability: transmissions vs N ($\\alpha$={a}, drift)")
    _style_axes(ax)
    ax.legend(frameon=False, fontsize=8, ncol=2)
    _savefig(fig, out_path)


def fig_5_6_sensitivity(sensitivity_dir, out_path):
    if not os.path.isdir(sensitivity_dir) or not glob.glob(os.path.join(sensitivity_dir, "*.json")):
        print(f"[5.6] SKIPPED -- no runs found in {sensitivity_dir}. Generate with "
              f"`python -m experiments.run_all --sensitivity --out {sensitivity_dir}`.")
        return

    runs = load_results(sensitivity_dir, kind="sweep")
    groups = group_by(runs, ["theta_rho"])
    thetas = sorted(key[0] for key in groups.keys())
    if not thetas:
        print(f"[5.6] SKIPPED -- no theta_rho grouping found in {sensitivity_dir}")
        return

    metrics = [
        ("peak_accuracy", "Peak accuracy"),
        ("total_transmissions", "Total transmissions"),
        ("rounds_to_convergence", "Rounds to convergence"),
    ]
    fig, axes = plt.subplots(1, len(metrics), figsize=(13, 4))
    for ax, (key, ylabel) in zip(axes, metrics):
        means, stds = [], []
        for theta in thetas:
            group = groups[(theta,)]
            vals = [g["_summary"][key] for g in group if g["_summary"].get(key) is not None]
            means.append(np.mean(vals) if vals else np.nan)
            stds.append(np.std(vals) if vals else 0.0)
        ax.errorbar(thetas, means, yerr=stds,
                   marker="o", markersize=5, color=COLORS["rafa_fl"], linewidth=1.8, capsize=3)
        ax.set_xlabel("$\\theta_\\rho$ (resource threshold)")
        ax.set_ylabel(ylabel)
        _style_axes(ax)
    fig.suptitle(f"Sensitivity analysis: key metrics vs $\\theta_\\rho$ "
                f"($\\alpha$={EXPERIMENT.primary_alpha}, N={EXPERIMENT.primary_n}, drift)", color=INK)
    _savefig(fig, out_path)


def fig_5_7_post_drift(results_dir, out_path):
    """Mean accuracy over rounds 51-100 by alpha (N=50), drift vs no drift side by side."""
    runs = load_results(results_dir)
    n = EXPERIMENT.primary_n
    groups = group_by(runs, ["method", "alpha", "num_clients", "drift"])
    methods = ["rafa_fl"] + BASELINES
    alphas = EXPERIMENT.alpha_grid
    key = "post_drift_mean_accuracy"

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)
    for ax, drift in zip(axes, [False, True]):
        series = {m: {a: groups.get((m, a, n, drift)) for a in alphas} for m in methods}
        _grouped_bar(ax, alphas, series, methods,
                     lambda g: float(np.mean([r["_summary"][key] for r in g])),
                     lambda g: float(np.std([r["_summary"][key] for r in g])))
        ax.set_xlabel("Dirichlet concentration $\\alpha$")
        ax.set_title("With drift (onset at round 50)" if drift else "No drift")
        _style_axes(ax)
    axes[0].set_ylabel("Mean test accuracy, rounds 51-100")
    axes[0].set_ylim(0.5, 1.0)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, fontsize=9, ncol=6, loc="lower center",
               bbox_to_anchor=(0.5, 0.0))
    fig.suptitle(f"Accuracy after round 50 (N={n}; bars = mean over 5 seeds, whiskers = std)", color=INK)
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"Wrote {out_path}")


def fig_5_8_rule_mix(results_dir, out_path):
    """Share of RAFA-FL decisions by rule, before/after drift, for drifted vs unaffected clients."""
    from data.dirichlet import make_drift_clients
    runs = load_results(results_dir)
    a, n = EXPERIMENT.primary_alpha, EXPERIMENT.primary_n
    group = group_by(runs, ["method", "alpha", "num_clients", "drift"]).get(("rafa_fl", a, n, True), [])
    if not group:
        print(f"[5.8] SKIPPED -- no rafa_fl runs at alpha={a}, N={n}, drift=True")
        return
    rules = ["P0_first", "P1_force", "P2_drift", "P3_quality", "P4_defer", "P5_default"]
    rule_colors = ["#898781", "#0b0b0b", "#eb6834", "#1baf7a", "#e87ba4", "#2a78d6"]
    cats = [("Before drift", "unaffected"), ("Before drift", "drifted"),
            ("After drift", "unaffected"), ("After drift", "drifted")]
    counts = {c: {r: 0 for r in rules} for c in cats}
    for run in group:
        dc = make_drift_clients(n, EXPERIMENT.drift_fraction, drift_seed=1000 + run["seed"])
        for e in run["round_log"]:
            # drift is applied from round drift_round (50) onwards
            phase = "Before drift" if e["round"] < EXPERIMENT.drift_round else "After drift"
            for c in e["clients"]:
                kind = "drifted" if c["pid"] in dc else "unaffected"
                if c["rule"] in counts[(phase, kind)]:
                    counts[(phase, kind)][c["rule"]] += 1

    fig, ax = plt.subplots(figsize=(8, 4.5))
    y = np.arange(len(cats))
    left = np.zeros(len(cats))
    for r, col in zip(rules, rule_colors):
        share = np.array([counts[c][r] / max(sum(counts[c].values()), 1) for c in cats])
        ax.barh(y, share, left=left, color=col, label=r, edgecolor="white", linewidth=0.5)
        left += share
    ax.set_yticks(y)
    ax.set_yticklabels([f"{p}, {k} clients\n(n={sum(counts[(p, k)].values())})" for p, k in cats])
    ax.invert_yaxis()
    ax.set_xlim(0, 1)
    ax.set_xlabel("Share of decisions")
    ax.set_title(f"RAFA-FL decision rules ($\\alpha$={a}, N={n}, drift, 5 seeds)")
    _style_axes(ax)
    ax.xaxis.grid(True, color=GRID, linewidth=0.8)
    ax.legend(frameon=False, fontsize=8, ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.15))
    _savefig(fig, out_path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results_dir", default="results/")
    ap.add_argument("--sensitivity_dir", default="results/",
                    help="folder holding the theta_rho sweep runs (identified by their override)")
    ap.add_argument("--out_dir", default="figures/")
    args = ap.parse_args()

    fig_5_1_convergence(args.results_dir, os.path.join(args.out_dir, "fig5_1_convergence.png"))
    fig_5_2_comm_cost(args.results_dir, os.path.join(args.out_dir, "fig5_2_comm_cost.png"))
    fig_5_3_deferral(args.results_dir, os.path.join(args.out_dir, "fig5_3_deferral.png"))
    fig_5_4_ablation(args.results_dir, os.path.join(args.out_dir, "fig5_4_ablation.png"))
    fig_5_5_scalability(args.results_dir, os.path.join(args.out_dir, "fig5_5_scalability.png"))
    fig_5_6_sensitivity(args.sensitivity_dir, os.path.join(args.out_dir, "fig5_6_sensitivity.png"))
    fig_5_7_post_drift(args.results_dir, os.path.join(args.out_dir, "fig5_7_post_drift_accuracy.png"))
    fig_5_8_rule_mix(args.results_dir, os.path.join(args.out_dir, "fig5_8_rule_mix.png"))


if __name__ == "__main__":
    main()
