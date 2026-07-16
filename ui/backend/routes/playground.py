"""
playground.py — POST /decide.

Instantiates a fresh agent.reactive_agent.ReactiveAgent per request and calls
its real decide() method -- the priority-ordered rule logic (P1-P5) itself is
not reimplemented here in any form. A fresh instance means sigma starts at 0
each call, so this endpoint's output is deterministic and reproducible: it
matches calling ReactiveAgent(...).decide(delta_t, drift_t, rho_t) directly in
a Python shell for the same inputs.
"""

import numpy as np
from fastapi import APIRouter
from pydantic import BaseModel

from agent.reactive_agent import ReactiveAgent

router = APIRouter()


class AgentConfigIn(BaseModel):
    """Mirrors config.py's AgentConfig -- see there for the authoritative defaults/meaning."""
    theta_delta: float = 0.013148  # min loss improvement that counts as "still learning" (rule P3)
    theta_d: float = 0.137177      # KL-divergence above which drift is considered detected (rule P2)
    theta_rho: float = 0.20        # resource sample below which a client is resource-constrained (rule P4)
    s_max: int = 5                 # max consecutive skipped rounds before a client is force-included (rule P1)
    e_min: int = 1                 # fewest local epochs the adaptive epoch function can choose
    e_max: int = 5                 # most local epochs the adaptive epoch function can choose
    e_std: int = 3                 # fixed epoch count for non-agent baselines (unused by decide() itself)
    lam: float = 0.6               # weight on drift vs. quality in the epoch function (0=quality, 1=drift)
    kl_epsilon: float = 1e-8       # Laplace smoothing epsilon for the drift estimator's KL divergence


class DecideRequest(BaseModel):
    delta_t: float   # observed loss improvement this round (see AgentConfig.theta_delta)
    drift_t: float   # observed label-distribution drift this round (see AgentConfig.theta_d)
    rho_t: float     # observed resource availability this round, in [0, 1] (see AgentConfig.theta_rho)
    config: AgentConfigIn = AgentConfigIn()


@router.post("/decide")
def decide(req: DecideRequest):
    agent = ReactiveAgent(req.config, num_classes=6, rng=np.random.RandomState(0))
    action = agent.decide(req.delta_t, req.drift_t, req.rho_t)
    return {
        "participate": bool(action.participate),
        "epochs": action.epochs,
        "communicate": bool(action.communicate),
        "rule": action.rule,
    }
