"""
pilot.py — Threshold-initialisation pilot (Sections 3.9 and 4.8.1).

Runs a REAL federated pilot (actual local training, same client code, same
observation process as the main experiments) at alpha=0.5, N=50, no drift,
seed 42, with the agent in observe-only mode: every selected client trains
E_std epochs and communicates (i.e. behaves as FedAvg), while recording the
same signals the agent would act on:

    delta = L_post(previous participation) - L_post(this participation)
            (post-training local validation loss, Algorithm 1 lines 24-25)
    d     = KL(label histogram of a drift_batch sample || reference), Section 3.6.1
    rho   ~ Beta(2, 2)

Observe-only mode is used so the observations do not depend on the very
thresholds being calibrated. Thresholds are then:

    theta_delta = 25th percentile of observed delta
    theta_d     = 90th percentile of observed no-drift d
    theta_rho, s_max: analytic (taken from config.py, not estimated here)

Because delta needs two participations by the same client and only 10% of
clients are selected per round, the pilot runs --rounds rounds (default 30)
rather than the 10 stated in Section 4.8.1; the report records how many
delta observations that produced, and the script refuses to update config.py
if there are fewer than --min_delta_obs.

Outputs (in --out, default experiments/pilot_output/):
    pilot_run.json      the raw run (round log, accuracy, provenance)
    pilot_report.json   observations, thresholds, epoch distribution,
                        estimated rule mix / deferral rate, provenance
With --write_config, theta_delta and theta_d in config.py are replaced and
annotated with the report's SHA-256 so the frozen values are traceable.

Usage:
    python -m experiments.pilot                    # run + report only
    python -m experiments.pilot --write_config     # also freeze into config.py
    python -m experiments.pilot --from_run experiments/pilot_output/pilot_run.json
"""

import os, sys, re, json, hashlib, argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from collections import Counter, defaultdict
from dataclasses import replace, asdict

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_OUT = os.path.join(REPO, "experiments", "pilot_output")


def run_pilot(rounds, alpha, num_clients, seed, out_dir):
    from experiments.run_experiment import run
    res = run("rafa_fl", alpha, num_clients, seed, drift=False, out_dir=out_dir,
              observe_only=True, num_rounds=rounds, tag_suffix="_pilot")
    path = os.path.join(out_dir, "pilot_run.json")
    with open(path, "w") as f:
        json.dump(res, f, indent=2)
    return res


def extract_observations(run_result):
    """Pull (delta, d, rho) observations out of the round log."""
    post_losses = defaultdict(list)          # cid -> [L_post per participation, in order]
    drifts, rhos = [], []
    for r in run_result["round_log"]:
        for c in r["clients"]:
            if c.get("drift") is not None:
                drifts.append(float(c["drift"]))
            if c.get("rho") is not None:
                rhos.append(float(c["rho"]))
            if c.get("val_loss_post") is not None:
                # group by data-partition index (stable across runs); older
                # logs without "pid" fall back to the per-run Flower node id
                key = str(c.get("pid", c["cid"]))
                post_losses[key].append(float(c["val_loss_post"]))
    deltas = [prev - cur for seq in post_losses.values()
              for prev, cur in zip(seq[:-1], seq[1:])]
    # Sorted so the report and the resampling in analyse() do not depend on
    # the order in which Ray workers returned results within a round.
    return {
        "delta": sorted(deltas), "drift": sorted(drifts), "rho": sorted(rhos),
        "participations_per_client": dict(sorted((k, len(v)) for k, v in post_losses.items())),
    }


def _stats(x):
    x = np.asarray(x, dtype=float)
    if len(x) == 0:
        return {"n": 0}
    q = np.percentile(x, [5, 10, 25, 50, 75, 90, 95])
    return {"n": int(len(x)), "min": float(x.min()), "max": float(x.max()),
            "mean": float(x.mean()), "std": float(x.std()),
            **{f"p{p}": float(v) for p, v in zip([5, 10, 25, 50, 75, 90, 95], q)}}


def analyse(obs, agent_cfg):
    from agent.reactive_agent import ReactiveAgent
    theta_delta = float(np.percentile(obs["delta"], 25))
    theta_d = float(np.percentile(obs["drift"], 90))
    cfg = replace(agent_cfg, theta_delta=theta_delta, theta_d=theta_d)

    # Estimate the decision mix the calibrated policy would produce on these
    # observations (sigma=0, so P1 never fires; no-drift regime). The three
    # signals are paired by resampling, since delta exists only for repeat
    # participations; this is an estimate, not a replay.
    rng = np.random.RandomState(0)
    n = 5000
    ds = rng.choice(obs["delta"], n)
    dd = rng.choice(obs["drift"], n)
    rr = rng.choice(obs["rho"], n)
    rules, epochs = Counter(), Counter()
    for dl, dr, rh in zip(ds, dd, rr):
        a = ReactiveAgent(cfg, 6, rng=np.random.RandomState(0))
        act = a.decide(float(dl), float(dr), float(rh))
        rules[act.rule] += 1
        if act.participate:
            epochs[int(act.epochs)] += 1
    ep = np.repeat(list(epochs.keys()), list(epochs.values()))
    return {
        "theta_delta_p25": theta_delta,
        "theta_d_p90": theta_d,
        "theta_rho_analytic": agent_cfg.theta_rho,
        "s_max_analytic": agent_cfg.s_max,
        "empirical_P_rho_below_theta_rho": float(np.mean(np.asarray(obs["rho"]) < agent_cfg.theta_rho)),
        "estimated_rule_mix": {k: v / n for k, v in sorted(rules.items())},
        "estimated_deferral_rate": rules.get("P4_defer", 0) / n,
        "epoch_distribution": {int(k): int(v) for k, v in sorted(epochs.items())},
        "epoch_mean": float(ep.mean()) if len(ep) else None,
        "epoch_std": float(ep.std()) if len(ep) else None,
        "epoch_function_collapsed": bool(len(ep) and ep.std() < 0.1),
    }


def write_config(theta_delta, theta_d, report_sha):
    path = os.path.join(REPO, "config.py")
    with open(path) as f:
        src = f.read()
    tag = f"pilot report sha256 {report_sha[:12]}"
    src, n1 = re.subn(r"(?m)^(\s*theta_delta: float = )[-0-9.eE]+.*$",
                      rf"\g<1>{theta_delta:.6f}  # 25th pctile of pilot delta ({tag})", src)
    src, n2 = re.subn(r"(?m)^(\s*theta_d: float = )[-0-9.eE]+.*$",
                      rf"\g<1>{theta_d:.6f}      # 90th pctile of pilot no-drift d ({tag})", src)
    if n1 != 1 or n2 != 1:
        raise RuntimeError("could not locate theta_delta / theta_d lines in config.py")
    with open(path, "w") as f:
        f.write(src)
    print(f"config.py updated ({tag})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rounds", type=int, default=30)
    ap.add_argument("--alpha", type=float, default=0.5)
    ap.add_argument("--num_clients", type=int, default=50)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument("--from_run", default=None, help="analyse an existing pilot_run.json")
    ap.add_argument("--min_delta_obs", type=int, default=50)
    ap.add_argument("--write_config", action="store_true")
    args = ap.parse_args()

    from config import AGENT
    os.makedirs(args.out, exist_ok=True)
    if args.from_run:
        with open(args.from_run) as f:
            run_result = json.load(f)
    else:
        run_result = run_pilot(args.rounds, args.alpha, args.num_clients, args.seed, args.out)

    problems = []
    if not run_result.get("valid", False):
        problems.append(f"pilot run invalid (incomplete rounds {run_result.get('incomplete_rounds')})")
    obs = extract_observations(run_result)
    if len(obs["delta"]) < args.min_delta_obs:
        problems.append(f"only {len(obs['delta'])} delta observations (< {args.min_delta_obs}); "
                        f"increase --rounds")
    if len(obs["drift"]) == 0 or np.allclose(obs["drift"], 0):
        problems.append("all drift observations are 0 -- theta_d would be degenerate "
                        "(is AgentConfig.drift_batch None?)")

    analysis = analyse(obs, AGENT) if (obs["delta"] and obs["drift"]) else {}
    if analysis.get("epoch_function_collapsed"):
        problems.append("epoch function collapsed to a near-constant value")

    report = {
        "settings": {"rounds": run_result.get("num_rounds"), "alpha": run_result["alpha"],
                     "num_clients": run_result["num_clients"], "seed": run_result["seed"],
                     "drift": run_result["drift"], "observe_only": run_result.get("observe_only")},
        "agent_config_at_pilot": asdict(AGENT),
        "summary": {"delta": _stats(obs["delta"]), "drift": _stats(obs["drift"]),
                    "rho": _stats(obs["rho"])},
        "analysis": analysis,
        "problems": problems,
        "observations": obs,
        "pilot_run_provenance": run_result.get("provenance"),
    }
    blob = json.dumps(report, indent=2, sort_keys=True)
    sha = hashlib.sha256(blob.encode()).hexdigest()
    report_path = os.path.join(args.out, "pilot_report.json")
    with open(report_path, "w") as f:
        f.write(blob)

    print(json.dumps({"summary": report["summary"], "analysis": analysis,
                      "problems": problems}, indent=2))
    print(f"\nreport: {report_path}  (sha256 {sha[:12]})")

    if args.write_config:
        if problems:
            print("NOT writing config.py because of the problems listed above.")
            sys.exit(1)
        write_config(analysis["theta_delta_p25"], analysis["theta_d_p90"], sha)


if __name__ == "__main__":
    main()
