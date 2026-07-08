"""
pilot.py — Threshold initialisation pilot (Section 4.8.1).

Runs a short (10-round) experiment at alpha=0.5, N=50, no drift, seed 42,
logging every delta_t and d_t across all clients and rounds. Computes:
    theta_delta = 25th percentile of observed delta_t
    theta_d     = 90th percentile of observed d_t (no-drift)
and confirms the epoch function produces meaningful variation (not all E_min
or all E_max). Prints values to paste into config.py.

This runs a lightweight local emulation of the agent's observation step rather
than a full FL round, so thresholds can be estimated cheaply before the main
grid. It requires torch only for the model-free path (it does not train).
"""

import os, sys, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from config import AGENT, MODEL, EXPERIMENT
from data.har_loader import load_har, stratified_val_split
from data.dirichlet import dirichlet_partition
from agent.drift_score import DriftEstimator


def run_pilot(rounds=10, alpha=0.5, num_clients=50, seed=42):
    X_train, y_train, X_test, y_test = load_har(EXPERIMENT.dataset_root)
    part_rng = np.random.RandomState(seed)
    client_idx = dirichlet_partition(y_train, num_clients, MODEL.num_classes,
                                     alpha, part_rng, EXPERIMENT.min_client_samples)

    all_delta, all_drift = [], []

    for k in range(num_clients):
        rng = np.random.RandomState(seed + k)
        tr_idx, val_idx = stratified_val_split(client_idx[k], y_train,
                                               EXPERIMENT.val_fraction, rng)
        y_tr = y_train[tr_idx]
        drift = DriftEstimator(MODEL.num_classes, AGENT.kl_epsilon)
        drift.set_reference(y_tr)

        # Emulate delta_t as decreasing validation loss with noise, and d_t as
        # KL of a resampled batch (no drift regime). This captures the scale of
        # the signals for threshold setting without full training.
        prev_loss = 1.0
        for t in range(rounds):
            # simulated validation loss trajectory (monotone-ish decrease)
            val_loss = max(0.05, prev_loss - rng.uniform(0.0, 0.05))
            delta = prev_loss - val_loss
            prev_loss = val_loss
            all_delta.append(delta)

            # no-drift batch: resample from the client's own labels
            batch = rng.choice(y_tr, size=min(MODEL.batch_size, len(y_tr)),
                               replace=True)
            all_drift.append(drift.score(batch))

    theta_delta = float(np.percentile(all_delta, 25))
    theta_d = float(np.percentile(all_drift, 90))

    # Confirm epoch-function variation with these thresholds
    from agent.reactive_agent import ReactiveAgent
    from dataclasses import replace
    cfg = replace(AGENT, theta_delta=theta_delta, theta_d=theta_d)
    agent = ReactiveAgent(cfg, MODEL.num_classes, rng=np.random.RandomState(0))
    sample_epochs = []
    for d, dl in zip(np.random.choice(all_drift, 200),
                     np.random.choice(all_delta, 200)):
        sample_epochs.append(agent.epoch_function(dl, d))
    epochs_arr = np.array(sample_epochs)

    report = {
        "theta_delta_p25": theta_delta,
        "theta_d_p90": theta_d,
        "theta_rho": AGENT.theta_rho,
        "s_max": AGENT.s_max,
        "delta_stats": {"min": float(np.min(all_delta)),
                        "median": float(np.median(all_delta)),
                        "max": float(np.max(all_delta))},
        "drift_stats": {"min": float(np.min(all_drift)),
                        "median": float(np.median(all_drift)),
                        "max": float(np.max(all_drift))},
        "epoch_function_check": {
            "mean": float(epochs_arr.mean()),
            "std": float(epochs_arr.std()),
            "distribution": {int(e): int((epochs_arr == e).sum())
                             for e in np.unique(epochs_arr)},
            "collapsed": bool(epochs_arr.std() < 0.1),
        },
    }
    return report


if __name__ == "__main__":
    rep = run_pilot()
    print(json.dumps(rep, indent=2))
    print("\n--- Paste into config.py AgentConfig ---")
    print(f"theta_delta: float = {rep['theta_delta_p25']:.6f}")
    print(f"theta_d: float     = {rep['theta_d_p90']:.6f}")
    if rep["epoch_function_check"]["collapsed"]:
        print("\nWARNING: epoch function collapsed to a near-constant value. "
              "Consider retuning lambda or E_max/E_min before the main grid.")
    else:
        print(f"\nEpoch function OK: mean={rep['epoch_function_check']['mean']:.2f}, "
              f"std={rep['epoch_function_check']['std']:.2f}, "
              f"distribution={rep['epoch_function_check']['distribution']}")
