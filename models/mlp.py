"""
mlp.py — Local MLP model for HAR classification (Section 3.7).

Architecture: 561 -> 128 -> 64 -> 6, ReLU, dropout 0.3, raw logits out.
Includes parameter get/set utilities for Flower's NumPy serialisation.
"""

from collections import OrderedDict
import numpy as np

import torch
import torch.nn as nn


class HAR_MLP(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(cfg.input_dim, cfg.hidden1),
            nn.ReLU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.hidden1, cfg.hidden2),
            nn.ReLU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.hidden2, cfg.num_classes),   # raw logits (CrossEntropyLoss applies log-softmax)
        )

    def forward(self, x):
        return self.net(x)


def get_parameters(model):
    """Return model parameters as a list of NumPy arrays (Flower convention)."""
    return [p.detach().cpu().numpy() for p in model.parameters()]


def set_parameters(model, parameters):
    """Load a list of NumPy arrays into the model."""
    params_dict = zip(model.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
    model.load_state_dict(state_dict, strict=True)


def num_parameters(model):
    return sum(p.numel() for p in model.parameters())
