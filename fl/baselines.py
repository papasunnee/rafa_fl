"""
baselines.py — Baseline FL methods (Section 4.7).

FedAvg, FedProx, SCAFFOLD, FedCS, OORT. Each shares the MLP, HAR partitioning,
and seeds with RAFA-FL. Simulation adaptations for FedCS (resource-proxy
ranking) and OORT (statistical-utility only) are documented in Sections 4.7.4
and 4.7.5 and implemented here.

This module provides:
  - build_strategy(name, ...) -> a Flower strategy for the named baseline
  - client-side hooks (proximal term, control variates, utility reporting)
    are applied via the client factory in run_experiment.py.
"""

import numpy as np


SELECTION_MODES = {"fedcs": "fedcs", "oort": "oort"}


def build_strategy(name, fraction_fit, num_clients, min_active_fraction=0.1,
                   evaluate_metrics_aggregation_fn=None, evaluate_fn=None,
                   param_shapes=None, selection_seed=0, oort_exploration_floor=0.1):
    """
    Return a configured Flower strategy for a baseline, or the RAFA-FL strategy.
    Client-side behaviour (proximal term, control variates) is selected in the
    client factory keyed on the same name.
    """
    from fl.rafa_strategy import RAFAFLStrategy

    common = dict(
        fraction_fit=fraction_fit,
        fraction_evaluate=0.0,                      # centralised eval via evaluate_fn instead
        min_fit_clients=max(1, int(fraction_fit * num_clients)),
        min_available_clients=num_clients,
        evaluate_metrics_aggregation_fn=evaluate_metrics_aggregation_fn,
        evaluate_fn=evaluate_fn,
    )

    # RAFAFLStrategy filters clients whose fit metrics report "deferred": True;
    # baseline clients never set that key, so it behaves as plain FedAvg for
    # them while still recording total_transmissions/round_log for the
    # communication-cost metrics all methods are compared on.
    return RAFAFLStrategy(min_active_fraction=min_active_fraction,
                          scaffold=(name == "scaffold"), param_shapes=param_shapes,
                          selection_mode=SELECTION_MODES.get(name, "random"),
                          selection_seed=selection_seed,
                          oort_exploration_floor=oort_exploration_floor,
                          **common)


# ---- Client-side proximal term (FedProx, Section 4.7.2) -------------------
def proximal_penalty(model, global_params, mu):
    """Compute (mu/2) * ||w - w_global||^2 over model parameters."""
    import torch
    penalty = 0.0
    for p, gp in zip(model.parameters(), global_params):
        penalty = penalty + torch.sum((p - torch.tensor(gp, device=p.device)) ** 2)
    return 0.5 * mu * penalty


# ---- OORT statistical utility (Section 4.7.5) -----------------------------
def oort_statistical_utility(losses, num_samples):
    """
    U_k = |B_k| * sqrt( sum_j loss_j^2 / |B_k| ).
    System-speed component excluded (documented simulation simplification).
    """
    losses = np.asarray(losses, dtype=np.float64)
    if len(losses) == 0:
        return 0.0
    return float(num_samples * np.sqrt(np.mean(losses ** 2)))


# ---- FedCS / OORT selection helpers ---------------------------------------
def select_topk_by_score(scores, k, rng, exploration_floor=0.0):
    """
    Select top-k client indices by score. If exploration_floor > 0 (OORT), each
    client retains a minimum selection probability to prevent starvation.
    """
    scores = np.asarray(scores, dtype=np.float64)
    n = len(scores)
    k = min(k, n)
    if exploration_floor > 0:
        n_explore = int(round(exploration_floor * k))
        n_exploit = k - n_explore
        exploit = list(np.argsort(scores)[::-1][:n_exploit])
        remaining = [i for i in range(n) if i not in exploit]
        explore = list(rng.choice(remaining, size=min(n_explore, len(remaining)),
                                  replace=False)) if remaining else []
        return exploit + explore
    return list(np.argsort(scores)[::-1][:k])
