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
"""

import pickle

import numpy as np
import flwr as fl


class RAFAFLStrategy(fl.server.strategy.FedAvg):
    def __init__(self, *args, min_active_fraction=0.1, scaffold=False,
                param_shapes=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.min_active_fraction = min_active_fraction
        self.previous_parameters = None
        self.total_transmissions = 0
        self.round_log = []          # per-round (round, active, deferred)

        self.scaffold = scaffold
        self.global_control = ([np.zeros(s, dtype=np.float32) for s in param_shapes]
                               if scaffold else None)

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

        min_required = max(1, int(self.min_active_fraction * len(results)))
        if len(active) < min_required and self.previous_parameters is not None:
            # Carry forward previous global model; round effectively skipped
            return self.previous_parameters, {}

        aggregated, metrics = super().aggregate_fit(server_round, active, failures)
        if aggregated is not None:
            self.previous_parameters = aggregated
        return aggregated, metrics
