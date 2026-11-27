"""
thesis_outputs.py — Every Chapter 5 table, generated from one results folder.

Writes, into --out (default tables/):
  table5_1_grid.csv            all methods x alpha x N x drift: accuracy metrics,
                               transmissions, deferral rate (mean and 95% CI)
  table5_2_primary_sig.csv     RAFA-FL vs each baseline at alpha=0.5, N=50
                               (drift and no drift), several metrics, paired
                               t-test + Bonferroni + Cohen's d
  table5_3_grid_sig.csv        same test at every condition (post-drift mean
                               accuracy and final-10 accuracy)
  table5_4_ablation.csv        full agent vs each ablation, several metrics
  table5_5_communication.csv   transmissions, bytes, reduction vs FedAvg
  table5_6_sensitivity.csv     theta_rho sweep
  table5_7_rule_mix.csv        RAFA-FL decisions before/after drift, for
                               drifted vs unaffected clients
  table5_8_reference.csv       every method at the primary condition next to
                               the centralised upper bound
  table5_9_validity.csv        run counts, validity and provenance
  table5_10_replicate_noise.csv identical-configuration replicates (main run,
                               rafa_full ablation, theta_rho=0.20 sweep run)
  table5_11_h2_convergence.csv H2 test at alpha=0.1, N=50: rounds to 95% of peak
                               (sustained) and rounds to 98% of peak, RAFA-FL vs
                               FedAvg and FedProx
  thesis_tables.md             all of the above, formatted for reading

Usage:
    python -m analysis.thesis_outputs            # defaults: results/ -> tables/
"""

import os, sys, csv, json, argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from collections import Counter, defaultdict

import numpy as np

from config import EXPERIMENT
from analysis.report import (load_results, group_by, significance_table,
                             ablation_table, BASELINES, ABLATIONS)
from analysis.stats import mean_ci
from data.dirichlet import make_drift_clients

METHODS = ["rafa_fl"] + BASELINES
LABEL = {"rafa_fl": "RAFA-FL", "fedavg": "FedAvg", "fedprox": "FedProx",
         "scaffold": "SCAFFOLD", "fedcs": "FedCS", "oort": "OORT",
         "rafa_full": "Full agent", "rafa_no_drift": "No drift-awareness",
         "rafa_no_resource": "No resource-awareness", "rafa_fixed_epochs": "Fixed epochs",
         "rafa_equal_weight": "Equal weight (lambda=0.5)", "centralised": "Centralised (reference)"}
ACC_METRICS = ["peak_accuracy", "final10_accuracy", "post_drift_mean_accuracy",
               "post_drift_min_accuracy", "post_drift_std_accuracy"]
A0, N0 = EXPERIMENT.primary_alpha, EXPERIMENT.primary_n


def _mc(vals):
    vals = [v for v in vals if v is not None]
    return mean_ci(vals) if vals else (None, None)


def _f(pair, d=4):
    m, h = pair
    return "n/a" if m is None else f"{m:.{d}f} ± {h:.{d}f}"


def _write(path, header, rows):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)


def _md_table(header, rows):
    out = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


# ---------------------------------------------------------------------------
def table_grid(runs):
    g = group_by(runs, ["method", "alpha", "num_clients", "drift"])
    header = ["method", "alpha", "N", "drift", "seeds"] + \
             [x for m in ACC_METRICS + ["total_transmissions", "deferral_rate_mean"] for x in (m, m + "_ci95")]
    rows, md = [], []
    for m in METHODS:
        for a in EXPERIMENT.alpha_grid:
            for n in EXPERIMENT.num_clients_grid:
                for d in [False, True]:
                    grp = g.get((m, a, n, d), [])
                    r = [m, a, n, d, len(grp)]
                    pairs = {}
                    for k in ACC_METRICS + ["total_transmissions", "deferral_rate_mean"]:
                        pairs[k] = _mc([x["_summary"].get(k) for x in grp])
                        r += list(pairs[k])
                    rows.append(r)
                    md.append([LABEL[m], a, n, "yes" if d else "no",
                               _f(pairs["peak_accuracy"]), _f(pairs["final10_accuracy"]),
                               _f(pairs["post_drift_mean_accuracy"]),
                               _f(pairs["total_transmissions"], 1)])
    return header, rows, _md_table(["Method", "α", "N", "Drift", "Peak acc", "Final-10 acc",
                                    "Mean acc rounds 51-100", "Transmissions"], md)


def _sig_rows(sig, cond, metric):
    rows = []
    if sig is None:
        return rows
    for b, c in sig["comparisons"].items():
        rows.append(cond + [metric, sig["rafa_mean"], sig["rafa_ci95"], b, c["baseline_mean"],
                            c["baseline_ci95"], c["t"], c["p"], c["cohens_d"],
                            sig["bonferroni_threshold"], c["significant"]])
    return rows


SIG_HEADER = ["alpha", "N", "drift", "metric", "rafa_mean", "rafa_ci95", "baseline", "baseline_mean",
              "baseline_ci95", "t", "p", "cohens_d", "bonferroni_threshold", "significant"]


def table_primary_sig(runs):
    rows, md = [], []
    for d in [True, False]:
        for metric in ["post_drift_mean_accuracy", "final10_accuracy", "peak_accuracy",
                       "total_transmissions"]:
            sig = significance_table(runs, A0, N0, d, metric)
            rows += _sig_rows(sig, [A0, N0, d], metric)
            if sig is None:
                continue
            for b, c in sig["comparisons"].items():
                verdict = "n.s."
                if c["significant"]:
                    higher = c["cohens_d"] > 0
                    if metric == "total_transmissions":
                        verdict = "RAFA-FL fewer" if not higher else "RAFA-FL more"
                    else:
                        verdict = "RAFA-FL better" if higher else "RAFA-FL worse"
                md.append(["yes" if d else "no", metric, f"{sig['rafa_mean']:.4f}", LABEL[b],
                           f"{c['baseline_mean']:.4f}", f"{c['p']:.4f}", f"{c['cohens_d']:.2f}", verdict])
    return SIG_HEADER, rows, _md_table(
        ["Drift", "Metric", "RAFA-FL", "Baseline", "Baseline mean", "p", "Cohen's d",
         "Verdict (Bonferroni α=0.01)"], md)


def table_grid_sig(runs):
    rows, md = [], []
    for metric in ["post_drift_mean_accuracy", "final10_accuracy"]:
        for a in EXPERIMENT.alpha_grid:
            for n in EXPERIMENT.num_clients_grid:
                for d in [False, True]:
                    sig = significance_table(runs, a, n, d, metric)
                    rows += _sig_rows(sig, [a, n, d], metric)
                    if sig is None:
                        continue
                    better = [LABEL[b] for b, c in sig["comparisons"].items()
                              if c["significant"] and c["cohens_d"] > 0]
                    worse = [LABEL[b] for b, c in sig["comparisons"].items()
                             if c["significant"] and c["cohens_d"] < 0]
                    md.append([metric, a, n, "yes" if d else "no", f"{sig['rafa_mean']:.4f}",
                               ", ".join(better) or "-", ", ".join(worse) or "-"])
    return SIG_HEADER, rows, _md_table(
        ["Metric", "α", "N", "Drift", "RAFA-FL mean", "RAFA-FL sig. better than",
         "RAFA-FL sig. worse than"], md)


def table_ablation(runs):
    g = group_by(runs, ["method", "alpha", "num_clients", "drift"])
    metrics = ["post_drift_mean_accuracy", "final10_accuracy", "peak_accuracy", "total_transmissions"]
    tests = {m: ablation_table(runs, m) for m in metrics}
    header = ["variant"] + [x for m in metrics for x in (m, m + "_ci95", m + "_p_vs_full", m + "_sig")]
    rows, md = [], []
    for v in ["rafa_full"] + ABLATIONS:
        grp = g.get((v, A0, N0, True), [])
        r, mdr = [v], [LABEL[v]]
        for m in metrics:
            pair = _mc([x["_summary"].get(m) for x in grp])
            c = (tests[m] or {}).get("comparisons", {}).get(v)
            p = c["p"] if c else None
            s = c["significant"] if c else None
            r += [pair[0], pair[1], p, s]
            cell = _f(pair, 1 if m == "total_transmissions" else 4)
            if p is not None:
                cell += f" (p={p:.4f}{'*' if s else ''})"
            mdr.append(cell)
        rows.append(r)
        md.append(mdr)
    return header, rows, _md_table(["Variant", "Mean acc rounds 51-100", "Final-10 acc", "Peak acc",
                                    "Transmissions"], md)


def table_comm(runs):
    g = group_by(runs, ["method", "alpha", "num_clients", "drift"])
    header = ["method", "alpha", "N", "drift", "transmissions", "transmissions_ci95",
              "bytes", "reduction_vs_fedavg_pct", "deferral_rate_mean"]
    rows, md = [], []
    for a in EXPERIMENT.alpha_grid:
        for n in EXPERIMENT.num_clients_grid:
            for d in [False, True]:
                base = np.mean([x["_summary"]["total_transmissions"] for x in g[("fedavg", a, n, d)]])
                grp = g[("rafa_fl", a, n, d)]
                tx = _mc([x["_summary"]["total_transmissions"] for x in grp])
                by = np.mean([x["_summary"]["bytes_communicated"] for x in grp])
                red = 100 * (1 - tx[0] / base)
                dr = np.mean([x["_summary"].get("deferral_rate_mean", 0) for x in grp])
                rows.append(["rafa_fl", a, n, d, tx[0], tx[1], by, red, dr])
                md.append([a, n, "yes" if d else "no", f"{base:.0f}", _f(tx, 1),
                           f"{red:.1f}%", f"{by / 1e6:.1f}", f"{dr:.3f}"])
    return header, rows, _md_table(["α", "N", "Drift", "FedAvg tx", "RAFA-FL tx",
                                    "Reduction", "RAFA-FL MB", "Mean per-client deferral rate"], md)


def table_sensitivity(sweep):
    g = group_by(sweep, ["theta_rho"])
    metrics = ["post_drift_mean_accuracy", "final10_accuracy", "peak_accuracy",
               "total_transmissions", "deferral_rate_mean"]
    header = ["theta_rho", "seeds"] + [x for m in metrics for x in (m, m + "_ci95")]
    rows, md = [], []
    for (t,), grp in sorted(g.items()):
        r, mdr = [t, len(grp)], [t, len(grp)]
        for m in metrics:
            pair = _mc([x["_summary"].get(m) for x in grp])
            r += list(pair)
            mdr.append(_f(pair, 1 if m == "total_transmissions" else 4))
        rows.append(r)
        md.append(mdr)
    return header, rows, _md_table(["θ_ρ", "Seeds", "Mean acc rounds 51-100", "Final-10 acc",
                                    "Peak acc", "Transmissions", "Deferral rate"], md)


def table_rule_mix(runs):
    """RAFA-FL decisions at the primary drift condition, split by phase and client type."""
    g = group_by(runs, ["method", "alpha", "num_clients", "drift"])
    counts, epochs = Counter(), defaultdict(list)
    for r in g[("rafa_fl", A0, N0, True)]:
        dc = make_drift_clients(N0, EXPERIMENT.drift_fraction, drift_seed=1000 + r["seed"])
        for e in r["round_log"]:
            phase = "pre-drift (1-49)" if e["round"] < EXPERIMENT.drift_round else "post-drift (50-100)"
            for c in e["clients"]:
                kind = "drifted client" if c["pid"] in dc else "unaffected client"
                counts[(phase, kind, c["rule"])] += 1
                if not c["deferred"]:
                    epochs[(phase, kind)].append(c["epochs"])
    rules = ["P0_first", "P1_force", "P2_drift", "P3_quality", "P4_defer", "P5_default"]
    header = ["phase", "client_type", "decisions"] + rules + ["mean_epochs_when_training"]
    rows, md = [], []
    for phase in ["pre-drift (1-49)", "post-drift (50-100)"]:
        for kind in ["unaffected client", "drifted client"]:
            tot = sum(counts[(phase, kind, r)] for r in rules)
            ep = np.mean(epochs[(phase, kind)]) if epochs[(phase, kind)] else float("nan")
            rows.append([phase, kind, tot] + [counts[(phase, kind, r)] for r in rules] + [ep])
            md.append([phase, kind, tot] +
                      [f"{counts[(phase, kind, r)]} ({100 * counts[(phase, kind, r)] / max(tot, 1):.0f}%)"
                       for r in rules] + [f"{ep:.2f}"])
    return header, rows, _md_table(["Phase", "Clients", "Decisions"] + rules + ["Mean epochs"], md)


def table_reference(runs, results_dir):
    g = group_by(runs, ["method", "alpha", "num_clients", "drift"])
    central = os.path.join(results_dir, "centralised", "summary.json")
    c = json.load(open(central)) if os.path.exists(central) else None
    header = ["method", "condition", "metric", "mean", "ci95", "pct_of_centralised"]
    rows, md = [], []
    if c:
        rows.append(["centralised", "pooled data", "test_accuracy", c["accuracy_mean"],
                     c["accuracy_std"], 100.0])
        md.append([LABEL["centralised"], f"{c['accuracy_mean']:.4f} ± {c['accuracy_std']:.4f} (sd)",
                   "-", "-", "100%"])
    for m in METHODS:
        nd = _mc([x["_summary"]["final10_accuracy"] for x in g[(m, A0, N0, False)]])
        dr = _mc([x["_summary"]["post_drift_mean_accuracy"] for x in g[(m, A0, N0, True)]])
        pct = 100 * nd[0] / c["accuracy_mean"] if c else None
        rows.append([m, f"a={A0} N={N0} no drift", "final10_accuracy", nd[0], nd[1], pct])
        rows.append([m, f"a={A0} N={N0} drift", "post_drift_mean_accuracy", dr[0], dr[1],
                     100 * dr[0] / c["accuracy_mean"] if c else None])
        md.append([LABEL[m], "-", _f(nd), _f(dr), f"{pct:.1f}%" if pct else "-"])
    return header, rows, _md_table(["Method", "Centralised acc", "Final-10 acc (no drift)",
                                    "Mean acc rounds 51-100 (drift)",
                                    "No-drift final-10 as % of centralised"], md)


def table_validity(all_runs):
    commits, dirty, invalid, by_kind = Counter(), 0, 0, Counter()
    for r in all_runs:
        p = r.get("provenance") or {}
        commits[(p.get("git_commit") or "?")[:7]] += 1
        dirty += bool(p.get("git_dirty"))
        invalid += r.get("valid") is False
        if "theta_rho" in p.get("overrides", {}):
            by_kind["theta_rho sweep"] += 1
        elif r["method"].startswith("rafa_") and r["method"] != "rafa_fl":
            by_kind["ablation"] += 1
        else:
            by_kind["main grid"] += 1
    rows = [[k, v] for k, v in by_kind.items()] + [
        ["total", len(all_runs)], ["invalid (excluded)", invalid],
        ["dirty working tree", dirty], ["source commits", dict(commits)],
        ["dataset sha256 values", len({(r.get("provenance") or {}).get("dataset_sha256") for r in all_runs})]]
    return ["item", "value"], rows, _md_table(["Item", "Value"], rows)


def table_replicates(results_dir):
    """Identical configurations run three times: gives the run-to-run noise floor."""
    rows, md = [], []
    for s in EXPERIMENT.seeds:
        files = {"main": f"rafa_fl_a{A0}_n{N0}_s{s}_d1.json",
                 "rafa_full": f"rafa_full_a{A0}_n{N0}_s{s}_d1.json",
                 "sweep_0.2": f"rafa_fl_a{A0}_n{N0}_s{s}_d1_rho0.2.json"}
        vals = {}
        for k, fn in files.items():
            path = os.path.join(results_dir, fn)
            if not os.path.exists(path):
                break
            acc = np.array(json.load(open(path))["accuracy_curve"])
            vals[k] = (acc.max(), acc[50:].mean())
        if len(vals) < 3:
            continue
        peaks = [v[0] for v in vals.values()]
        posts = [v[1] for v in vals.values()]
        rows.append([s] + [round(x, 4) for x in peaks] + [round(max(peaks) - min(peaks), 4)]
                    + [round(x, 4) for x in posts] + [round(max(posts) - min(posts), 4)])
        md.append(rows[-1])
    header = ["seed", "peak_main", "peak_rafa_full", "peak_sweep0.2", "peak_range",
              "post_main", "post_rafa_full", "post_sweep0.2", "post_range"]
    return header, rows, _md_table(header, md)


def table_h2(runs, alpha=0.1, n=None):
    """
    H2 test: rounds to convergence for RAFA-FL vs FedAvg and FedProx at
    alpha=0.1, N=50, with and without drift. Two operationalisations: the
    sustained 95%-of-peak criterion and the first round reaching 98% of peak.
    Plain paired t-test per comparison (as reported in the thesis table).
    """
    from scipy import stats
    n = n or N0
    g = group_by(runs, ["method", "alpha", "num_clients", "drift"])
    measures = [("rounds_to_convergence", "Rounds to 95% (sustained)"),
                ("rounds_to_98pct_peak", "Rounds to 98% of peak")]
    header = ["measure", "drift", "rafa_mean", "baseline", "baseline_mean", "t", "p"]
    rows, md = [], []
    for key, label in measures:
        for d in [False, True]:
            rafa = {r["seed"]: r["_summary"][key] for r in g.get(("rafa_fl", alpha, n, d), [])}
            for b in ["fedavg", "fedprox"]:
                base = {r["seed"]: r["_summary"][key] for r in g.get((b, alpha, n, d), [])}
                seeds = sorted(set(rafa) & set(base))
                x = np.array([rafa[s] for s in seeds], dtype=float)
                y = np.array([base[s] for s in seeds], dtype=float)
                if len(seeds) < 2 or np.all(x - y == 0):
                    t, p = float("nan"), float("nan")
                else:
                    t, p = stats.ttest_rel(x, y)
                rows.append([key, d, x.mean(), b, y.mean(), t, p])
                md.append([label, "drift" if d else "no drift", f"{x.mean():.1f}", LABEL[b],
                           f"{y.mean():.1f}", f"{t:.2f}", f"{p:.3f}"])
    return header, rows, _md_table(["Measure", "Condition", "RAFA-FL", "Baseline", "Baseline mean",
                                    "t", "p"], md)


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results_dir", default="results/")
    ap.add_argument("--out", default="tables/")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    runs = load_results(args.results_dir, kind="main")
    sweep = load_results(args.results_dir, kind="sweep")
    all_runs = load_results(args.results_dir, kind="all")

    tables = [
        ("table5_1_grid", "Table 5.1 — Full grid (mean ± 95% CI over 5 seeds)", table_grid(runs)),
        ("table5_2_primary_sig", f"Table 5.2 — RAFA-FL vs baselines at α={A0}, N={N0} "
                                 f"(paired t-test, Bonferroni α=0.05/5)", table_primary_sig(runs)),
        ("table5_3_grid_sig", "Table 5.3 — Significance at every condition", table_grid_sig(runs)),
        ("table5_4_ablation", f"Table 5.4 — Ablation at α={A0}, N={N0}, drift "
                              f"(p vs full agent, Bonferroni α=0.05/4; * = significant)",
         table_ablation(runs)),
        ("table5_5_communication", "Table 5.5 — Communication (RAFA-FL vs FedAvg)", table_comm(runs)),
        ("table5_6_sensitivity", f"Table 5.6 — θ_ρ sensitivity (α={A0}, N={N0}, drift)",
         table_sensitivity(sweep)),
        ("table5_7_rule_mix", f"Table 5.7 — RAFA-FL decisions by phase and client type "
                              f"(α={A0}, N={N0}, drift, 5 seeds pooled)", table_rule_mix(runs)),
        ("table5_8_reference", f"Table 5.8 — Methods vs centralised reference (α={A0}, N={N0})",
         table_reference(runs, args.results_dir)),
        ("table5_9_validity", "Table 5.9 — Run validity and provenance", table_validity(all_runs)),
        ("table5_10_replicate_noise", "Table 5.10 — Identical-configuration replicates "
                                      "(run-to-run noise floor)", table_replicates(args.results_dir)),
        ("table5_11_h2_convergence", f"Table 5.11 — H2: rounds to convergence at α=0.1, N={N0}, "
                                     f"RAFA-FL vs FedAvg and FedProx (paired t-test)", table_h2(runs)),
    ]
    md = ["# Chapter 5 tables", "",
          f"Generated by `python -m analysis.thesis_outputs` from `{args.results_dir}` "
          f"({len(runs)} main/ablation runs, {len(sweep)} sweep runs).", "",
          "Metric definitions: *peak acc* = best single round; *final-10 acc* = mean of rounds "
          "91-100; *mean acc rounds 51-100* = mean over the rounds after drift onset (round 50; "
          "in no-drift runs the same rounds, for comparison).", ""]
    for name, title, (header, rows, mdtab) in tables:
        _write(os.path.join(args.out, name + ".csv"), header, rows)
        md += [f"## {title}", "", mdtab, ""]
        print(f"wrote {name}.csv ({len(rows)} rows)")
    with open(os.path.join(args.out, "thesis_tables.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(md))
    print(f"wrote {os.path.join(args.out, 'thesis_tables.md')}")


if __name__ == "__main__":
    main()
