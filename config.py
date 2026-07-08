"""
config.py — Centralised hyperparameter configuration for RAFA-FL.

All thresholds, model, and experiment parameters live here so that the same
values are shared across clients and are version-controlled alongside the code.
Threshold values theta_delta and theta_d are placeholders here; they are set by
the pilot experiment (experiments/pilot.py) and written back into this file.
"""

from dataclasses import dataclass, field, asdict
from typing import List


@dataclass
class AgentConfig:
    # Threshold parameters (theta_delta and theta_d set by pilot; others analytic)
    theta_delta: float = 0.013148 # loss-improvement threshold (25th pctile of pilot delta)
    theta_d: float = 0.137177     # drift threshold (90th pctile of pilot no-drift d)
    theta_rho: float = 0.20       # resource threshold (20th pctile of Beta(2,2))
    s_max: int = 5                # bounded-participation skip limit

    # Continuous epoch function
    e_min: int = 1
    e_max: int = 5
    e_std: int = 3
    lam: float = 0.6              # drift weight in epoch function g(delta, d)

    # Numerical
    kl_epsilon: float = 1e-8      # Laplace smoothing for KL divergence


@dataclass
class ModelConfig:
    input_dim: int = 561
    hidden1: int = 128
    hidden2: int = 64
    num_classes: int = 6
    dropout: float = 0.3
    lr: float = 0.01
    momentum: float = 0.9
    batch_size: int = 32


@dataclass
class ExperimentConfig:
    dataset_root: str = "./data/UCI HAR Dataset"
    num_rounds: int = 100
    fraction_fit: float = 0.10          # 10% clients selected per round
    val_fraction: float = 0.20          # local validation split
    drift_round: int = 50
    drift_fraction: float = 0.30        # 30% of clients receive drift
    min_client_samples: int = 10
    seeds: List[int] = field(default_factory=lambda: [42, 123, 456, 789, 1024])
    num_clients_grid: List[int] = field(default_factory=lambda: [10, 50, 100])
    alpha_grid: List[float] = field(default_factory=lambda: [0.1, 0.5, 1.0])
    primary_n: int = 50
    primary_alpha: float = 0.5

    # Baseline hyperparameters (Section 4.7)
    fedprox_mu: float = 0.01            # proximal term weight
    oort_exploration_floor: float = 0.1 # fraction of OORT selections reserved for exploration

    # Statistical
    bonferroni_baselines: int = 5
    alpha_significance: float = 0.05


AGENT = AgentConfig()
MODEL = ModelConfig()
EXPERIMENT = ExperimentConfig()


def full_config_dict():
    return {"agent": asdict(AGENT), "model": asdict(MODEL), "experiment": asdict(EXPERIMENT)}


if __name__ == "__main__":
    import json
    print(json.dumps(full_config_dict(), indent=2))
