"""
rafa_strategy.py — RAFAFLStrategy(FedAvg) (Section 4.6).

Overrides aggregate_fit() to filter deferred clients before the weighted
average, carries forward the previous global model when too few clients are
active, and instruments total transmissions for the communication-cost metric.

Also maintains SCAFFOLD's global control variate (Section 4.7.3) when
scaffold=True: each active client returns its local control-variate delta
(pickled, under the "scaffold_delta_c" fit metric); this strategy accumulates
them into self.global_control, which the client factory in run_experiment.py
reads via a closure and forwards to clients through on_fit_config_fn.

Overrides configure_fit() for the FedCS/OORT client-selection simplifications
(Sections 4.7.4-4.7.5): selection_mode="fedcs" ranks all available clients by
a per-round resource-proxy sample (in place of a wall-clock deadline);
selection_mode="oort" ranks by each client's statistical utility from its
last reported training loss, with an exploration floor so low-utility clients
are still occasionally sampled. Both fall back to plain FedAvg-style random
sampling ("random", the default) for every other method.
"""

import pickle

import numpy as np
import flwr as fl
from flwr.common import FitIns

from fl.baselines import select_topk_by_score, oort_statistical_utility


class RAFAFLStrategy(fl.server.strategy.FedAvg):
    def __init__(self, *args, min_active_fraction=0.1, scaffold=False,
                param_shapes=None, selection_mode="random", selection_seed=0,
                oort_exploration_floor=0.1, **kwargs):
        super().__init__(*args, **kwargs)
        self.min_active_fraction = min_active_fraction
        self.previous_parameters = None
        self.total_transmissions = 0
        self.round_log = []          # per-round (round, active, deferred)

        self.scaffold = scaffold
        self.global_control = ([np.zeros(s, dtype=np.float32) for s in param_shapes]
                               if scaffold else None)

        self.selection_mode = selection_mode
        self.selection_seed = selection_seed
        self.oort_exploration_floor = oort_exploration_floor
        self.client_utility = {}     # oort only: cid -> last statistical utility

    def configure_fit(self, server_round, parameters, client_manager):
        config = {}
        if self.on_fit_config_fn is not None:
            config = self.on_fit_config_fn(server_round)
        fit_ins = FitIns(parameters, config)

        sample_size, min_num_clients = self.num_fit_clients(client_manager.num_available())

        if self.selection_mode == "random":
            clients = client_manager.sample(num_clients=sample_size,
                                            min_num_clients=min_num_clients)
            return [(client, fit_ins) for client in clients]

        # FedCS / OORT: rank every currently-available client, then take the top-k.
        available = client_manager.sample(num_clients=client_manager.num_available(),
                                          min_num_clients=min_num_clients)
        rng = np.random.RandomState(self.selection_seed + server_round)

        if self.selection_mode == "fedcs":
            # Resource-proxy ranking in place of a wall-clock deadline (Section 4.7.4).
            scores = rng.beta(2, 2, size=len(available))
            idx = select_topk_by_score(scores, sample_size, rng)
        elif self.selection_mode == "oort":
            # Statistical-utility ranking, excluding the system-speed component
            # (Section 4.7.5). Never-yet-sampled clients default to max utility
            # so every client gets tried at least once.
            scores = np.array([self.client_utility.get(c.cid, np.inf) for c in available])
            idx = select_topk_by_score(scores, sample_size, rng,
                                       exploration_floor=self.oort_exploration_floor)
        else:
            raise ValueError(f"Unknown selection_mode: {self.selection_mode}")

        clients = [available[i] for i in idx]
        return [(client, fit_ins) for client in clients]

    def aggregate_fit(self, server_round, results, failures):
        active = [(cp, fr) for cp, fr in results
                  if not fr.metrics.get("deferred", False)]
        deferred = [(cp, fr) for cp, fr in results
                    if fr.metrics.get("deferred", False)]

        self.total_transmissions += len(active)
        self.round_log.append({
            "round": server_round,
            "active": len(active),
            "deferred": len(deferred),
            "cumulative_transmissions": self.total_transmissions,
        })

        if self.scaffold and active:
            num_clients = self.min_available_clients
            for _, fr in active:
                delta_c = pickle.loads(fr.metrics["scaffold_delta_c"])
                self.global_control = [c + d / num_clients
                                       for c, d in zip(self.global_control, delta_c)]

        if self.selection_mode == "oort":
            for cp, fr in active:
                if "train_loss" in fr.metrics:
                    self.client_utility[cp.cid] = oort_statistical_utility(
                        [fr.metrics["train_loss"]], fr.num_examples)

        min_required = max(1, int(self.min_active_fraction * len(results)))
        if len(active) < min_required and self.previous_parameters is not None:
            # Carry forward previous global model; round effectively skipped
            return self.previous_parameters, {}

        aggregated, metrics = super().aggregate_fit(server_round, active, failures)
        if aggregated is not None:
            self.previous_parameters = aggregated
        return aggregated, metrics
