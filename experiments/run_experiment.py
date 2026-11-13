"""
run_experiment.py — Single experiment runner (Section 4.8).

Runs one condition: one method, one alpha, one N, one seed. Writes a structured
JSON log with per-round accuracy, macro-F1, active/deferred counts, cumulative
transmissions, and per-client agent metadata.

Usage:
    python -m experiments.run_experiment --method rafa_fl --alpha 0.5 \
        --num_clients 50 --seed 42 --drift --out results/
"""

import os, sys, json, shutil, pickle, argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dataclasses import replace

import numpy as np
import torch
import flwr as fl
from sklearn.metrics import f1_score

from config import AGENT, MODEL, EXPERIMENT
from data.har_loader import load_har, stratified_val_split
from data.dirichlet import dirichlet_partition, make_drift_clients
from models.mlp import HAR_MLP, get_parameters, num_parameters
from fl.rafa_client import RAFAFLClient
from fl.baselines import build_strategy

METHODS = ["rafa_fl", "fedavg", "fedprox", "scaffold", "fedcs", "oort"]

# Ablation variants map to RAFA-FL with switches toggled. "lam" is not a
# RAFAFLClient constructor arg -- it's an AgentConfig field, popped out and
# applied via dataclasses.replace() in build_client_fn() instead of being
# spread into the client constructor with the other switches.
ABLATIONS = {
    "rafa_full":        dict(drift_enabled=True,  resource_enabled=True,  adaptive_epochs=True),
    "rafa_no_drift":    dict(drift_enabled=False, resource_enabled=True,  adaptive_epochs=True),
    "rafa_no_resource": dict(drift_enabled=True,  resource_enabled=False, adaptive_epochs=True),
    "rafa_fixed_epochs":dict(drift_enabled=True,  resource_enabled=True,  adaptive_epochs=False),
    "rafa_equal_weight":dict(drift_enabled=True,  resource_enabled=True,  adaptive_epochs=True, lam=0.5),
}


def result_tag(method, alpha, num_clients, seed, drift, ablation=None, theta_rho=None):
    tag = f"{ablation or method}_a{alpha}_n{num_clients}_s{seed}_d{int(drift)}"
    if theta_rho is not None:
        tag += f"_rho{theta_rho}"
    return tag


def _pkg_version(name):
    from importlib.metadata import version, PackageNotFoundError
    try:
        return version(name)
    except PackageNotFoundError:
        return None


def provenance(X_train, y_train, theta_rho=None, ablation=None):
    """
    Everything needed to tell whether two result files are comparable:
    source revision (and whether the tree was dirty), dependency versions,
    a fingerprint of the training data, and the full effective configuration.
    """
    import hashlib, platform, subprocess
    from config import full_config_dict
    repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    def _git(*args):
        try:
            return subprocess.run(["git", *args], cwd=repo, capture_output=True,
                                  text=True, timeout=10).stdout.strip()
        except (OSError, subprocess.SubprocessError):
            return None

    _git("update-index", "-q", "--refresh")   # clear stale stat info before status
    h = hashlib.sha256()
    h.update(np.ascontiguousarray(X_train).tobytes())
    h.update(np.ascontiguousarray(y_train).tobytes())
    cfg = full_config_dict()
    overrides = dict(ABLATIONS.get(ablation, {}))
    if theta_rho is not None:
        overrides["theta_rho"] = theta_rho
    return {
        "git_commit": _git("rev-parse", "HEAD"),
        "git_dirty": bool(_git("status", "--porcelain", "--untracked-files=no")),
        "python": platform.python_version(),
        "versions": {"torch": torch.__version__, "flwr": _pkg_version("flwr"),
                     "numpy": np.__version__, "ray": _pkg_version("ray")},
        "dataset_sha256": h.hexdigest(),
        "config": cfg,
        "overrides": overrides,
    }


def set_all_seeds(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    import random; random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def build_client_fn(method, X_train, y_train, client_idx, X_test, y_test,
                    seed, drift_clients, drift_state, device, state_dir, ablation=None,
                    theta_rho=None, observe_only=False):
    use_agent = (method == "rafa_fl") or (ablation is not None)
    switches = dict(ABLATIONS.get(ablation, dict(drift_enabled=True, resource_enabled=True,
                                                 adaptive_epochs=True)))
    # lam and theta_rho aren't RAFAFLClient constructor args -- they live on
    # AgentConfig, so they're applied via a per-run replace() instead of being
    # spread in below. AGENT is a single shared module-level instance imported
    # by every concurrently-running condition; replace() returns a fresh copy
    # rather than mutating it in place.
    lam_override = switches.pop("lam", None)
    agent_overrides = {}
    if lam_override is not None:
        agent_overrides["lam"] = lam_override
    if theta_rho is not None:
        agent_overrides["theta_rho"] = theta_rho
    agent_cfg = replace(AGENT, **agent_overrides) if agent_overrides else AGENT

    def client_fn(cid):
        k = int(cid)
        rng = np.random.RandomState(seed + k)
        idx = client_idx[k]
        tr_idx, val_idx = stratified_val_split(idx, y_train,
                                               EXPERIMENT.val_fraction, rng)
        y_tr = y_train[tr_idx].copy()
        Xtr, Xval = X_train[tr_idx], X_train[val_idx]
        y_val = y_train[val_idx].copy()

        client = RAFAFLClient(
            cid=k, X_tr=Xtr, y_tr=y_tr, X_val=Xval, y_val=y_val,
            X_test=X_test, y_test=y_test,
            model_cfg=MODEL, agent_cfg=agent_cfg, rng=rng, device=device,
            use_agent=use_agent, method=method,
            drift_clients=drift_clients, drift_active=drift_state,
            state_dir=state_dir, resource_seed=seed + k, exp_seed=seed, observe_only=observe_only, **switches,
        )
        # attach epoch default for non-agent baselines
        client.model_cfg.e_std = AGENT.e_std
        return client.to_client()

    return client_fn


def run(method, alpha, num_clients, seed, drift, out_dir, ablation=None, theta_rho=None,
        observe_only=False, num_rounds=None, tag_suffix=""):
    num_rounds = num_rounds or EXPERIMENT.num_rounds
    set_all_seeds(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    import ray
    if ray.is_initialized():
        ray.shutdown()

    # Flower's default ray_init_args omit `address`, so ray.init() auto-detects
    # and attaches to any already-running local Ray cluster instead of starting
    # its own -- fatal when multiple run_all.py --only_n shards run concurrently
    # as separate OS processes (confirmed: "Failed to register worker to
    # Raylet"). address="local" forces a fresh cluster per process, and a
    # PID-scoped _temp_dir keeps their on-disk state from colliding too.
    ray_init_args = {
        "ignore_reinit_error": True,
        "include_dashboard": False,
        "address": "local",
        "_temp_dir": os.path.join(os.environ.get("TEMP", "C:\\ray_tmp"), f"ray_{os.getpid()}"),
    }

    X_train, y_train, X_test, y_test = load_har(EXPERIMENT.dataset_root)
    part_rng = np.random.RandomState(seed)
    client_idx = dirichlet_partition(y_train, num_clients, MODEL.num_classes,
                                     alpha, part_rng, EXPERIMENT.min_client_samples)
    drift_clients = make_drift_clients(num_clients, EXPERIMENT.drift_fraction,
                                       drift_seed=1000 + seed) if drift else set()

    param_count = num_parameters(HAR_MLP(MODEL))
    param_shapes = [p.shape for p in get_parameters(HAR_MLP(MODEL))]
    # Explicit, seed-determined initial global model (same for every method
    # at a given seed).
    torch.manual_seed(seed)
    init_params = fl.common.ndarrays_to_parameters(get_parameters(HAR_MLP(MODEL)))

    tag = result_tag(method, alpha, num_clients, seed, drift, ablation, theta_rho) + tag_suffix
    # Per-client state (reactive agent sigma/previous_loss, SCAFFOLD control
    # variates) round-trips through disk since Flower's simulation engine
    # reconstructs the client object -- possibly in a different worker
    # process -- every round. Scoped to this run's tag and wiped afterwards.
    state_dir = os.path.join(out_dir, ".state", tag)
    if os.path.isdir(state_dir):
        shutil.rmtree(state_dir)

    # Centralised evaluation: server-side eval of the aggregated model on the
    # shared global test set, once per round (Flower's evaluate_fn hook).
    history = {"accuracy": [], "macro_f1": [], "rounds": []}
    eval_model = HAR_MLP(MODEL).to(device)
    X_test_t = torch.tensor(X_test).to(device)
    y_test_t = torch.tensor(y_test, dtype=torch.long).to(device)

    def centralised_evaluate_fn(server_round, parameters, config):
        from models.mlp import set_parameters
        import torch.nn as nn
        set_parameters(eval_model, parameters)
        eval_model.eval()
        with torch.no_grad():
            out = eval_model(X_test_t)
            loss = nn.CrossEntropyLoss()(out, y_test_t).item()
            pred = out.argmax(1).cpu().numpy()
        acc = float((pred == y_test).mean())
        f1 = f1_score(y_test, pred, average="macro", zero_division=0)
        # Flower's Server.fit() evaluates once at server_round=0 (the untrained
        # global model) before the round loop even starts, then again after
        # each of rounds 1..num_rounds. Excluding round 0 keeps accuracy_curve
        # at exactly num_rounds entries, index i == round i+1.
        if server_round > 0:
            history["accuracy"].append(acc)
            history["macro_f1"].append(float(f1))
        return loss, {"accuracy": acc, "macro_f1": float(f1)}

    strategy_name = "rafa_fl" if (method == "rafa_fl" or ablation) else method
    strategy = build_strategy(
        strategy_name,
        fraction_fit=EXPERIMENT.fraction_fit, num_clients=num_clients,
        evaluate_fn=centralised_evaluate_fn,
        param_shapes=param_shapes if strategy_name == "scaffold" else None,
        selection_seed=seed, oort_exploration_floor=EXPERIMENT.oort_exploration_floor,
        initial_parameters=init_params,
    )

    def on_fit_config(server_round):
        cfg = {"server_round": server_round}
        if strategy_name == "scaffold":
            cfg["global_control"] = pickle.dumps(strategy.global_control)
        return cfg

    strategy.on_fit_config_fn = on_fit_config

    client_fn = build_client_fn(method, X_train, y_train, client_idx,
                                X_test, y_test, seed, drift_clients,
                                drift_state=drift, device=device,
                                state_dir=state_dir, ablation=ablation,
                                theta_rho=theta_rho, observe_only=observe_only)

    try:
        fl.simulation.start_simulation(
            client_fn=client_fn,
            num_clients=num_clients,
            config=fl.server.ServerConfig(num_rounds=num_rounds),
            strategy=strategy,
            client_resources={"num_cpus": 1},
            ray_init_args=ray_init_args,
        )
    finally:
        shutil.rmtree(state_dir, ignore_errors=True)
        if ray.is_initialized():
            ray.shutdown()

    result = {
        "method": ablation or method,
        "alpha": alpha, "num_clients": num_clients, "seed": seed, "drift": drift,
        "theta_rho": theta_rho if theta_rho is not None else AGENT.theta_rho,
        "param_count": param_count,
        "num_rounds": num_rounds, "observe_only": observe_only,
        "accuracy_curve": history["accuracy"],
        "macro_f1_curve": history["macro_f1"],
        "round_log": getattr(strategy, "round_log", []),
        "total_transmissions": getattr(strategy, "total_transmissions", None),
    }
    # Validity: a run is invalid if any round lost a selected client to a
    # failure, or if the accuracy curve is short. Invalid runs are still
    # written (for diagnosis) but analysis/report.py excludes them.
    incomplete = list(getattr(strategy, "incomplete_rounds", []))
    result["incomplete_rounds"] = incomplete
    result["failure_messages"] = list(getattr(strategy, "failure_messages", []))
    result["valid"] = (not incomplete) and len(history["accuracy"]) == num_rounds
    result["provenance"] = provenance(X_train, y_train, theta_rho=theta_rho,
                                      ablation=ablation)
    if not result["valid"]:
        print(f"WARNING: run {tag} is INVALID: {len(incomplete)} incomplete rounds "
              f"{incomplete[:10]}; accuracy points={len(history['accuracy'])}")

    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"{tag}.json")
    with open(path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"Wrote {path}")
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", choices=METHODS, default="rafa_fl")
    ap.add_argument("--ablation", choices=list(ABLATIONS.keys()), default=None)
    ap.add_argument("--alpha", type=float, default=0.5)
    ap.add_argument("--num_clients", type=int, default=50)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--drift", action="store_true")
    ap.add_argument("--out", default="results/")
    ap.add_argument("--theta_rho", type=float, default=None,
                    help="override AgentConfig.theta_rho for a sensitivity-analysis run")
    args = ap.parse_args()
    run(args.method, args.alpha, args.num_clients, args.seed, args.drift,
        args.out, args.ablation, args.theta_rho)


if __name__ == "__main__":
    main()
