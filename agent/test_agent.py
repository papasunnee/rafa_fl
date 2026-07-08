"""
test_agent.py — Validation checks for the reactive agent and drift score.

Corresponds to the implementation-validation checks of Section 4.10:
  1. GFSM state-machine validation (bounded participation, transitions)
  2. Drift-score validation against scipy reference
These run without torch, so they verify the correctness-critical logic in
isolation from the FL machinery.
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from config import AgentConfig
from agent.reactive_agent import ReactiveAgent
from agent.drift_score import kl_divergence, label_histogram


def test_bounded_participation():
    """Proposition 1: sigma never exceeds S_max; force fires at the boundary."""
    cfg = AgentConfig()
    agent = ReactiveAgent(cfg, num_classes=6, rng=np.random.RandomState(0))
    agent.set_drift_reference(np.zeros(50, dtype=int))

    forced_rounds = []
    max_sigma = 0
    # Feed inputs that always trigger the defer branch (low resource, low gain)
    for t in range(30):
        delta = -1.0                     # below theta_delta -> no P3
        drift = 0.0                      # below theta_d -> no P2
        rho = 0.0                        # below theta_rho -> P4 defer condition
        act = agent.decide(delta, drift, rho)
        max_sigma = max(max_sigma, agent.sigma)
        if act.rule == "P1_force":
            forced_rounds.append(t)

    assert max_sigma <= cfg.s_max, f"sigma exceeded S_max: {max_sigma}"
    # force must fire periodically, at most every S_max+1 rounds
    gaps = np.diff([0] + forced_rounds)
    assert all(g <= cfg.s_max + 1 for g in gaps), f"participation gap too large: {gaps}"
    print(f"[PASS] bounded participation: max sigma={max_sigma}, "
          f"force fired at rounds {forced_rounds}")


def test_deadlock_freedom():
    """Proposition 2: a communicating action always occurs within S_max rounds."""
    cfg = AgentConfig()
    agent = ReactiveAgent(cfg, num_classes=6, rng=np.random.RandomState(1))
    agent.set_drift_reference(np.zeros(50, dtype=int))

    last_comm = -1
    max_gap = 0
    for t in range(40):
        act = agent.decide(delta_t=-1.0, drift_t=0.0, rho_t=0.0)  # worst case: always defer-inclined
        if act.communicate == 1:
            max_gap = max(max_gap, t - last_comm)
            last_comm = t
    assert max_gap <= cfg.s_max + 1, f"communication gap {max_gap} exceeds S_max+1"
    print(f"[PASS] deadlock freedom: max communication gap={max_gap} (<= {cfg.s_max + 1})")


def test_determinism():
    """Lemma 1: identical inputs and state produce identical actions."""
    cfg = AgentConfig()
    a1 = ReactiveAgent(cfg, 6, rng=np.random.RandomState(2))
    a2 = ReactiveAgent(cfg, 6, rng=np.random.RandomState(2))
    a1.set_drift_reference(np.zeros(50, dtype=int))
    a2.set_drift_reference(np.zeros(50, dtype=int))
    for _ in range(20):
        act1 = a1.decide(0.05, 0.2, 0.5)
        act2 = a2.decide(0.05, 0.2, 0.5)
        assert act1 == act2, f"nondeterministic: {act1} != {act2}"
    print("[PASS] determinism: identical inputs yield identical actions")


def test_drift_triggers_communication():
    """A drift spike (P2) forces communication regardless of low loss/resource."""
    cfg = AgentConfig()
    agent = ReactiveAgent(cfg, 6, rng=np.random.RandomState(3))
    agent.set_drift_reference(np.zeros(50, dtype=int))
    # High drift, low gain, low resource -> P2 should still communicate
    act = agent.decide(delta_t=-1.0, drift_t=cfg.theta_d + 0.5, rho_t=0.0)
    assert act.communicate == 1 and act.rule == "P2_drift", f"drift did not trigger: {act}"
    print(f"[PASS] drift triggers communication: {act.rule}, epochs={act.epochs}")


def test_epoch_function_variation():
    """Epoch function should produce a range of values, not collapse to a constant."""
    cfg = AgentConfig()
    agent = ReactiveAgent(cfg, 6, rng=np.random.RandomState(4))
    lo = agent.epoch_function(delta_t=0.0, drift_t=0.0)
    hi = agent.epoch_function(delta_t=cfg.theta_delta, drift_t=cfg.theta_d)
    drift_only = agent.epoch_function(delta_t=0.0, drift_t=cfg.theta_d)
    assert lo == cfg.e_min, f"expected e_min at zero signal, got {lo}"
    assert hi == cfg.e_max, f"expected e_max at threshold signals, got {hi}"
    print(f"[PASS] epoch function varies: low={lo}, drift_only={drift_only}, high={hi}")


def test_kl_against_scipy():
    """Drift-score KL agrees with scipy.stats.entropy on synthetic pairs."""
    try:
        from scipy.stats import entropy
    except ImportError:
        print("[SKIP] scipy not available; KL reference check skipped")
        return
    rng = np.random.RandomState(5)
    max_err = 0.0
    for _ in range(5):
        p = rng.dirichlet(np.ones(6))
        q = rng.dirichlet(np.ones(6))
        ours = kl_divergence(p, q, epsilon=1e-12)
        ref = float(entropy(p, q))
        max_err = max(max_err, abs(ours - ref))
    assert max_err < 1e-6, f"KL mismatch vs scipy: {max_err}"
    print(f"[PASS] KL matches scipy: max error={max_err:.2e}")


if __name__ == "__main__":
    print("Running RAFA-FL agent validation (Section 4.10)\n" + "-" * 50)
    test_bounded_participation()
    test_deadlock_freedom()
    test_determinism()
    test_drift_triggers_communication()
    test_epoch_function_variation()
    test_kl_against_scipy()
    print("-" * 50 + "\nAll agent validation checks passed.")
