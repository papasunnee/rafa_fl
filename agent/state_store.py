"""
state_store.py — Cross-process per-client state persistence for Flower simulation.

Flower's Virtual Client Engine reconstructs the client object fresh every
round, often in a different worker process, so anything that must persist
across rounds for the same client id -- the reactive agent's skip counter
and previous validation loss (Section 3.6), and SCAFFOLD's local control
variate (Section 4.7.3) -- is round-tripped through small per-cid files
under a per-run state directory instead of kept as an instance attribute.
"""

import os
import json
import numpy as np


def _agent_path(state_dir, cid):
    return os.path.join(state_dir, f"agent_{cid}.json")


def load_agent_state(state_dir, cid):
    path = _agent_path(state_dir, cid)
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def save_agent_state(state_dir, cid, state):
    os.makedirs(state_dir, exist_ok=True)
    with open(_agent_path(state_dir, cid), "w") as f:
        json.dump(state, f)


def _control_path(state_dir, cid):
    return os.path.join(state_dir, f"control_{cid}.npz")


def load_control_variate(state_dir, cid, shapes):
    """Return the client's persisted local control variate, or zeros on round 1."""
    path = _control_path(state_dir, cid)
    if not os.path.exists(path):
        return [np.zeros(s, dtype=np.float32) for s in shapes]
    with np.load(path) as data:
        return [data[f"arr_{i}"] for i in range(len(shapes))]


def save_control_variate(state_dir, cid, arrays):
    os.makedirs(state_dir, exist_ok=True)
    np.savez(_control_path(state_dir, cid), *arrays)
