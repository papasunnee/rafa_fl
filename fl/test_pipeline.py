"""
test_pipeline.py — Integration checks for the FL pipeline (beyond agent logic).

Covers what agent/test_agent.py does not: partition integrity, OORT
exploration and utility, the post-training delta bookkeeping, client state
persistence across Flower's per-round client reconstruction, baseline
training, per-round determinism, and server-side failure accounting.

Run:  python fl/test_pipeline.py      (or: python -m pytest fl/test_pipeline.py)
Needs torch + flwr but NOT the HAR dataset (synthetic data throughout).
"""

import os, sys, tempfile
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from types import SimpleNamespace

import numpy as np

from config import AgentConfig, ModelConfig
from data.dirichlet import dirichlet_partition, validate_partition
from fl.baselines import (select_topk_by_score, oort_statistical_utility,
                          oort_utility_from_stats)
from agent.reactive_agent import ReactiveAgent


# ---- partitioning ----------------------------------------------------------
def test_partition_integer_and_complete_under_extreme_skew():
    """alpha tiny, N large, few samples: empty initial clients must not yield float indices."""
    labels = np.repeat(np.arange(6), 100)               # 600 samples
    for seed in range(20):
        parts = dirichlet_partition(labels, num_clients=100, num_classes=6,
                                    alpha=0.05, rng=np.random.RandomState(seed),
                                    min_samples=3)
        for p in parts:
            assert p.dtype.kind in "iu", p.dtype
            _ = labels[p]                               # used to raise IndexError
        validate_partition(parts, len(labels), 3)
    print("[PASS] partition: integer dtype, disjoint, complete, min size (20 skewed seeds)")


def test_validate_partition_rejects_bad_partitions():
    good = [np.array([0, 1], dtype=np.int64), np.array([2, 3], dtype=np.int64)]
    validate_partition(good, 4, 2)
    bad_cases = {
        "float": [np.array([0., 1.]), np.array([2, 3], dtype=np.int64)],
        "overlap": [np.array([0, 1]), np.array([1, 3])],
        "missing": [np.array([0, 1]), np.array([2])],
    }
    for name, parts in bad_cases.items():
        try:
            validate_partition(parts, 4, 1)
        except ValueError:
            continue
        raise AssertionError(f"validate_partition accepted {name} partition")
    print("[PASS] validate_partition rejects float / overlapping / incomplete partitions")


# ---- OORT ------------------------------------------------------------------
def test_oort_exploration_nonzero_at_small_k():
    """Old code: round(0.1*k) == 0 for k<=5. Expected exploration must be ~floor at every k."""
    n, floor = 50, 0.1
    scores = np.arange(n, dtype=float)                  # top-k is deterministic
    for k in (1, 5, 10):
        rng = np.random.RandomState(0)
        top = set(np.argsort(scores)[::-1][:k])
        explored, trials = 0, 4000
        for _ in range(trials):
            sel = select_topk_by_score(scores, k, rng, exploration_floor=floor)
            assert len(sel) == k and len(set(sel)) == k
            explored += sum(1 for i in sel if i not in top)
        frac = explored / (trials * k)
        # explored slots can coincidentally land in `top` only via exploit, never explore
        assert 0.07 < frac < 0.13, f"k={k}: exploration fraction {frac:.3f}"
    print("[PASS] OORT exploration ~10% of slots at k=1,5,10")


def test_oort_utility_from_stats_matches_per_sample_formula():
    losses = np.random.RandomState(0).exponential(size=200)
    ref = oort_statistical_utility(losses, len(losses))
    ours = oort_utility_from_stats(len(losses), float(np.mean(losses ** 2)))
    assert abs(ref - ours) < 1e-9 * ref
    print("[PASS] OORT utility from reported mean-squared loss == per-sample formula")


# ---- agent delta bookkeeping ----------------------------------------------
def test_delta_is_post_training_and_first_selection_explicit():
    cfg = AgentConfig()
    a = ReactiveAgent(cfg, 6, rng=np.random.RandomState(0))
    a.set_drift_reference(np.zeros(10, dtype=int))
    # first selection: no delta, must communicate regardless of resource
    assert a.compute_delta() is None
    assert a.decide(a.compute_delta(), 0.0, rho_t=0.0).rule == "P0_first"
    a.update_loss(2.0)                                  # post-training loss, round 1
    assert a.compute_delta() is None                    # one observation is not a delta
    assert a.decide(a.compute_delta(), 0.0, rho_t=0.0).rule == "P0_first"
    a.update_loss(1.5)                                  # round 2
    assert abs(a.compute_delta() - 0.5) < 1e-12
    a.update_loss(1.5)                                  # no improvement
    assert a.compute_delta() == 0.0
    assert a.decide(a.compute_delta(), 0.0, rho_t=0.0).rule == "P4_defer"
    print("[PASS] delta = L_post(prev) - L_post(now); first selections fire P0_first")


# ---- client-level (torch + flwr) ------------------------------------------
def _make_client(state_dir, cid=3, exp_seed=42, method="rafa_fl", use_agent=True):
    from fl.rafa_client import RAFAFLClient
    rs = np.random.RandomState(cid)
    X = rs.randn(80, 561).astype(np.float32)
    y = rs.randint(0, 6, size=80)
    model_cfg = ModelConfig()
    model_cfg.e_std = 1
    return RAFAFLClient(cid=cid, X_tr=X[:60], y_tr=y[:60], X_val=X[60:], y_val=y[60:],
                        X_test=X[:20], y_test=y[:20], model_cfg=model_cfg,
                        agent_cfg=AgentConfig(e_std=1, e_max=2),
                        rng=np.random.RandomState(cid), state_dir=state_dir,
                        use_agent=use_agent, method=method,
                        resource_seed=exp_seed + cid, exp_seed=exp_seed)


def _init_params():
    import torch
    from models.mlp import HAR_MLP, get_parameters
    torch.manual_seed(0)
    return get_parameters(HAR_MLP(ModelConfig()))


def test_client_state_persists_across_reconstruction():
    params = _init_params()
    with tempfile.TemporaryDirectory() as d:
        rules, deltas = [], []
        for rnd in (1, 2, 3):
            c = _make_client(d)                         # fresh object every round, like Flower
            _, n, m = c.fit(params, {"server_round": rnd})
            rules.append(m["rule"]); deltas.append(m["delta"])
            assert "val_loss_post" in m
        assert rules[:2] == ["P0_first", "P0_first"], rules
        assert np.isnan(deltas[0]) and np.isnan(deltas[1]) and np.isfinite(deltas[2]), deltas
    print(f"[PASS] client state survives reconstruction; rules={rules}")


def test_fit_is_deterministic_per_seed_client_round():
    params = _init_params()
    outs = []
    for _ in range(2):
        with tempfile.TemporaryDirectory() as d:
            p, _, _ = _make_client(d).fit(params, {"server_round": 7})
            outs.append(p)
    assert all(np.array_equal(a, b) for a, b in zip(*outs)), "same seed/cid/round gave different weights"
    with tempfile.TemporaryDirectory() as d:
        p3, _, _ = _make_client(d, exp_seed=43).fit(params, {"server_round": 7})
    assert not all(np.array_equal(a, b) for a, b in zip(outs[0], p3)), "seed has no effect"
    print("[PASS] fit() reproducible for fixed (seed, cid, round); differs across seeds")


def test_baselines_train_and_report_oort_stats():
    params = _init_params()
    for method in ("fedavg", "fedprox", "scaffold", "oort"):
        with tempfile.TemporaryDirectory() as d:
            c = _make_client(d, method=method, use_agent=False)
            cfg = {"server_round": 1}
            p, n, m = c.fit(params, cfg)
            assert n == 60 and not m["deferred"]
            assert m["train_loss_sq_mean"] is not None and m["train_loss_sq_mean"] > 0
            assert any(not np.array_equal(a, b) for a, b in zip(p, params)), f"{method} did not train"
    print("[PASS] FedAvg/FedProx/SCAFFOLD/OORT clients train and report loss stats")


def test_observe_only_and_batch_drift():
    """Pilot mode always communicates with E_std epochs; batch drift is non-degenerate."""
    params = _init_params()
    drifts = []
    with tempfile.TemporaryDirectory() as d:
        for rnd in range(1, 6):
            c = _make_client(d)
            c.observe_only = True
            _, n, m = c.fit(params, {"server_round": rnd})
            assert m["rule"] == "OBSERVE" and not m["deferred"] and m["epochs"] == 1
            drifts.append(m["drift"])
    assert any(x > 0 for x in drifts), f"no-drift batch d should be noisy, got {drifts}"
    print(f"[PASS] observe-only mode; no-drift batch d in [{min(drifts):.3f}, {max(drifts):.3f}]")


# ---- server failure accounting --------------------------------------------
def test_strategy_records_failed_and_missing_clients():
    from flwr.common import FitRes, Status, Code, ndarrays_to_parameters
    from fl.rafa_strategy import RAFAFLStrategy

    params = _init_params()
    strat = RAFAFLStrategy(fraction_fit=0.1, min_fit_clients=1, min_available_clients=10,
                           fraction_evaluate=0.0)
    proxies = [SimpleNamespace(cid=str(i)) for i in range(3)]
    strat._record_selection(1, proxies)

    def res(deferred):
        return FitRes(status=Status(code=Code.OK, message=""),
                      parameters=ndarrays_to_parameters(params),
                      num_examples=0 if deferred else 10,
                      metrics={"deferred": deferred})

    # client 0 trains, client 1 defers, client 2 never answers (failure)
    results = [(proxies[0], res(False)), (proxies[1], res(True))]
    strat.aggregate_fit(1, results, [RuntimeError("worker died")])
    log = strat.round_log[-1]
    assert (log["selected"], log["active"], log["deferred"], log["failed"]) == (3, 1, 1, 1), log
    assert log["missing_cids"] == ["2"]
    assert strat.incomplete_rounds == [1] and strat.total_transmissions == 1
    assert strat.failure_messages and "worker died" in strat.failure_messages[0]

    strat._record_selection(2, proxies[:2])
    strat.aggregate_fit(2, results, [])
    assert strat.round_log[-1]["failed"] == 0 and strat.incomplete_rounds == [1]
    print("[PASS] strategy separates selected / active / deferred / failed and flags incomplete rounds")


if __name__ == "__main__":
    print("RAFA-FL pipeline checks\n" + "-" * 50)
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
    print("-" * 50 + "\nAll pipeline checks passed.")
