"""
reactive_agent.py — The RAFA-FL reactive agent (GFSM decision policy).

Implements the joint action triple a_t = (p_t, e_t, c_t) produced by the
priority-ordered policy of Table 3.1, with the continuous epoch function of
Section 3.6.3. Pure NumPy — no torch — so the decision logic is unit-testable
in isolation from the FL machinery.

The agent maintains persistent state between rounds (skip counter sigma and the
previous validation loss), and exposes the four-stage decision cycle through
observe() and decide().
"""

from collections import namedtuple
import numpy as np

from .drift_score import DriftEstimator

# Joint action triple: participation, epochs, communication
Action = namedtuple("Action", ["participate", "epochs", "communicate", "rule"])

# GFSM behavioural states
STATE_ACTIVE = "Active"
STATE_ACCUMULATING = "Accumulating"
STATE_FORCED = "Forced"


class ReactiveAgent:
    def __init__(self, agent_config, num_classes, rng=None,
                 drift_enabled=True, resource_enabled=True,
                 adaptive_epochs=True):
        self.cfg = agent_config
        self.num_classes = num_classes
        self.rng = rng if rng is not None else np.random.RandomState()

        # Ablation switches (Section 3.13)
        self.drift_enabled = drift_enabled
        self.resource_enabled = resource_enabled
        self.adaptive_epochs = adaptive_epochs

        # Persistent state
        self.sigma = 0
        self.previous_loss = 1.0            # round-0 sentinel -> delta_0 > theta_delta
        self.drift = DriftEstimator(num_classes, epsilon=self.cfg.kl_epsilon)
        self.state = STATE_ACTIVE

    # ---- Stage 1: Observe -------------------------------------------------
    def set_drift_reference(self, labels):
        self.drift.set_reference(labels)

    def sample_resource(self):
        if not self.resource_enabled:
            return 1.0                      # resource never constrains when ablated
        return float(self.rng.beta(2, 2))

    def compute_drift(self, labels):
        if not self.drift_enabled:
            return 0.0                      # P2 disabled when drift ablated
        return self.drift.score(labels)

    # ---- Epoch function (Section 3.6.3) ----------------------------------
    def epoch_function(self, delta_t, drift_t):
        if not self.adaptive_epochs:
            return self.cfg.e_std           # fixed-epochs ablation
        lam = self.cfg.lam
        g = lam * (drift_t / max(self.cfg.theta_d, 1e-12)) \
            + (1 - lam) * (delta_t / max(self.cfg.theta_delta, 1e-12))
        g = float(np.clip(g, 0.0, 1.0))
        return int(self.cfg.e_min + int((self.cfg.e_max - self.cfg.e_min) * g))

    # ---- Stages 2-3: Evaluate guards + select joint action ---------------
    def decide(self, delta_t, drift_t, rho_t):
        """Return the joint action triple from the priority-ordered policy."""
        e_fn = self.epoch_function(delta_t, drift_t)

        # P1 — bounded participation guarantee
        if self.sigma >= self.cfg.s_max:
            self.sigma = 0
            self.state = STATE_ACTIVE
            return Action(1, e_fn, 1, "P1_force")

        # P2 — drift-triggered
        if self.drift_enabled and drift_t >= self.cfg.theta_d:
            self.sigma = 0
            self.state = STATE_ACTIVE
            return Action(1, e_fn, 1, "P2_drift")

        # P3 — quality-triggered
        if delta_t >= self.cfg.theta_delta:
            self.sigma = 0
            self.state = STATE_ACTIVE
            return Action(1, e_fn, 1, "P3_quality")

        # P4 — resource-aware deferral (conjunction)
        if (rho_t < self.cfg.theta_rho) and (delta_t < self.cfg.theta_delta):
            self.sigma += 1
            self.state = STATE_FORCED if self.sigma >= self.cfg.s_max else STATE_ACCUMULATING
            return Action(0, 0, 0, "P4_defer")

        # P5 — default participation
        self.sigma = 0
        self.state = STATE_ACTIVE
        return Action(1, self.cfg.e_std, 1, "P5_default")

    # ---- Stage 4 (loss bookkeeping) --------------------------------------
    def update_loss(self, new_val_loss):
        """Update the previous-loss reference after a communicating round."""
        self.previous_loss = new_val_loss

    def compute_delta(self, current_val_loss):
        return self.previous_loss - current_val_loss
