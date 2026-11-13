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
        # Loss-improvement bookkeeping (Algorithm 1, lines 24-25). Both values
        # are POST-training local validation losses:
        #   previous_loss = L at the end of this client's last communicating round
        #   last_delta    = L(prev communicating round) - L(this one)
        # None means "not yet observed": the client has never trained (first
        # selection, handled by rule P0) or has trained only once (no delta yet).
        # This replaces the old previous_loss=1.0 sentinel, which did not
        # guarantee first-round communication -- initial weighted cross-entropy
        # over 6 classes is ~ln 6 = 1.79 > 1.0, giving a negative delta_0.
        self.previous_loss = None
        self.last_delta = None
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
        """
        Return the joint action triple from the priority-ordered policy.

        delta_t=None means no loss improvement has been observed yet (the
        client has trained fewer than two times). Such a client always
        communicates: rule P0_first fires after P1/P2 and before P3-P5, with
        E_std epochs.
        """
        first = delta_t is None
        e_fn = self.epoch_function(0.0 if first else delta_t, drift_t)

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

        # P0 — no delta observed yet: participate so that one can be measured
        if first:
            self.sigma = 0
            self.state = STATE_ACTIVE
            return Action(1, self.cfg.e_std, 1, "P0_first")

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
    def update_loss(self, post_train_val_loss):
        """
        Record the post-training validation loss of a communicating round
        (Algorithm 1 line 24) and update delta = L(prev) - L(now) (line 25).
        Never called on deferred rounds, so delta reflects local learning
        progress between successive participations.
        """
        if self.previous_loss is not None:
            self.last_delta = self.previous_loss - post_train_val_loss
        self.previous_loss = post_train_val_loss

    def compute_delta(self):
        """delta for the current decision: last observed improvement, or None."""
        return self.last_delta
