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
    theta_delta: float = 0.013148
    theta_d: float = 0.137177
    theta_rho: float = 0.20
    s_max: int = 5
    e_min: int = 1
    e_max: int = 5
    e_std: int = 3
    lam: float = 0.6
    kl_epsilon: float = 1e-8


class DecideRequest(BaseModel):
    delta_t: float
    drift_t: float
    rho_t: float
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
