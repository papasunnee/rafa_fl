"""
rafa_client.py — RAFAFLClient(NumPyClient) (Section 4.5).

The fit() method is the primary integration site of the RAFA-FL contribution:
it computes the agent state, obtains the joint action triple, and executes
train / defer / communicate accordingly. evaluate() uses the global test set.

Flower's simulation engine reconstructs the client object fresh every round
(and may run it in a different worker process each time), so per-client state
that must persist across rounds -- the reactive agent's skip counter and
previous validation loss (Section 3.6), and SCAFFOLD's local control variate
(Section 4.7.3) -- is round-tripped through agent/state_store.py rather than
kept as an instance attribute. The current round number is likewise only
available via the Flower `config` argument to fit(), not at client
construction time, so concept-drift label permutation (Section 3.4) is also
decided inside fit().
"""

import pickle

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

import flwr as fl

from config import EXPERIMENT
from models.mlp import HAR_MLP, get_parameters, set_parameters
from agent.reactive_agent import ReactiveAgent
from agent.state_store import (load_agent_state, save_agent_state,
                               load_control_variate, save_control_variate)
from data.dirichlet import apply_label_permutation
from fl.baselines import proximal_penalty
from sklearn.metrics import f1_score


def _loader(X, y, batch_size, shuffle):
    ds = TensorDataset(torch.tensor(X), torch.tensor(y, dtype=torch.long))
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)


def _class_weights(y, num_classes):
    counts = np.bincount(y, minlength=num_classes).astype(np.float64) + 1.0
    inv = 1.0 / counts
    w = inv / inv.sum() * num_classes
    return torch.tensor(w, dtype=torch.float32)


class RAFAFLClient(fl.client.NumPyClient):
    def __init__(self, cid, X_tr, y_tr, X_val, y_val, X_test, y_test,
                 model_cfg, agent_cfg, rng, device="cpu",
                 drift_enabled=True, resource_enabled=True, adaptive_epochs=True,
                 use_agent=True, method="rafa_fl",
                 drift_clients=frozenset(), drift_active=False,
                 state_dir=None, resource_seed=0):
        self.cid = cid
        self.device = device
        self.model_cfg = model_cfg
        self.use_agent = use_agent
        self.method = method
        self.state_dir = state_dir
        # Flower reconstructs this client fresh every round from an RNG
        # re-seeded identically each time (needed so the train/val partition
        # stays stable across rounds) -- so agent.rng would otherwise draw the
        # exact same "random" resource sample every round. Reseed it from this
        # per-round-varying seed right before sampling instead (see fit()).
        self.resource_seed = resource_seed

        # Concept-drift configuration (Section 3.4): whether this client is one
        # of the drift-affected clients, and whether drift is active this run
        # at all. The round check (>= EXPERIMENT.drift_round) happens in fit(),
        # once the current round number is known.
        self.drift_active = drift_active and (cid in drift_clients)

        self.model = HAR_MLP(model_cfg).to(device)
        self.X_tr, self.y_tr = X_tr, y_tr
        self.X_val, self.y_val = X_val, y_val
        self.test_loader = _loader(X_test, y_test, model_cfg.batch_size, False)

        self.agent = ReactiveAgent(
            agent_cfg, model_cfg.num_classes, rng=rng,
            drift_enabled=drift_enabled, resource_enabled=resource_enabled,
            adaptive_epochs=adaptive_epochs,
        )
        self.agent.set_drift_reference(y_tr)      # P_hat_0 from pre-drift labels

        if self.use_agent and self.state_dir:
            state = load_agent_state(self.state_dir, self.cid)
            if state is not None:
                self.agent.sigma = state["sigma"]
                self.agent.previous_loss = state["previous_loss"]

    # ---- helpers ------------------------------------------------------
    def get_parameters(self, config):
        return get_parameters(self.model)

    def _active_labels(self, server_round):
        """Apply the cyclic label permutation once this client's drift round has passed."""
        if self.drift_active and server_round >= EXPERIMENT.drift_round:
            num_classes = self.model_cfg.num_classes
            return (apply_label_permutation(self.y_tr, num_classes),
                    apply_label_permutation(self.y_val, num_classes))
        return self.y_tr, self.y_val

    def _evaluate_local(self, val_loader, class_weights):
        self.model.eval()
        crit = nn.CrossEntropyLoss(weight=class_weights)
        total, loss_sum = 0, 0.0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(self.device), yb.to(self.device)
                out = self.model(xb)
                loss_sum += crit(out, yb).item() * len(yb)
                total += len(yb)
        return loss_sum / max(total, 1)

    def _train(self, epochs, train_loader, class_weights,
              proximal_global=None, proximal_mu=0.0,
              scaffold_control=None, scaffold_local=None):
        self.model.train()
        crit = nn.CrossEntropyLoss(weight=class_weights)
        opt = torch.optim.SGD(self.model.parameters(),
                              lr=self.model_cfg.lr, momentum=self.model_cfg.momentum)
        steps = 0
        for _ in range(epochs):
            for xb, yb in train_loader:
                xb, yb = xb.to(self.device), yb.to(self.device)
                opt.zero_grad()
                loss = crit(self.model(xb), yb)
                if proximal_mu > 0 and proximal_global is not None:
                    loss = loss + proximal_penalty(self.model, proximal_global, proximal_mu)
                loss.backward()
                if scaffold_control is not None:
                    with torch.no_grad():
                        for p, c, ci in zip(self.model.parameters(), scaffold_control, scaffold_local):
                            p.grad += torch.tensor(c - ci, device=p.device, dtype=p.grad.dtype)
                opt.step()
                steps += 1
        return steps

    # ---- Flower fit (Section 4.5.2) ---------------------------------------
    def fit(self, parameters, config):
        set_parameters(self.model, parameters)
        server_round = int(config.get("server_round", 1))
        y_tr_active, y_val_active = self._active_labels(server_round)
        train_loader = _loader(self.X_tr, y_tr_active, self.model_cfg.batch_size, True)
        val_loader = _loader(self.X_val, y_val_active, self.model_cfg.batch_size, False)
        class_weights = _class_weights(y_tr_active, self.model_cfg.num_classes).to(self.device)

        # Non-agent baselines: standard fixed-epoch training, always communicate.
        if not self.use_agent:
            return self._fit_baseline(parameters, config, train_loader, class_weights)

        # Agent path
        val_loss = self._evaluate_local(val_loader, class_weights)
        delta_t = self.agent.compute_delta(val_loss)
        drift_t = self.agent.compute_drift(y_tr_active)
        self.agent.rng = np.random.RandomState(self.resource_seed * 1_000_003 + server_round)
        rho_t = self.agent.sample_resource()

        action = self.agent.decide(delta_t, drift_t, rho_t)

        if action.participate == 0:                     # defer triple (0,0,0)
            self._save_agent_state()
            return get_parameters(self.model), 0, {
                "deferred": True, "cid": self.cid, "sigma": self.agent.sigma,
                "drift": drift_t, "epochs": 0, "rule": action.rule,
            }

        self._train(action.epochs, train_loader, class_weights)
        self.agent.update_loss(val_loss)
        self._save_agent_state()
        return get_parameters(self.model), len(train_loader.dataset), {
            "deferred": False, "cid": self.cid, "sigma": 0, "drift": drift_t,
            "epochs": action.epochs, "rule": action.rule,
        }

    def _fit_baseline(self, parameters, config, train_loader, class_weights):
        epochs = self.model_cfg_epochs()

        if self.method == "fedprox":
            self._train(epochs, train_loader, class_weights,
                       proximal_global=parameters, proximal_mu=EXPERIMENT.fedprox_mu)
            return get_parameters(self.model), len(train_loader.dataset), \
                {"deferred": False, "cid": self.cid, "epochs": epochs}

        if self.method == "scaffold":
            shapes = [p.shape for p in parameters]
            c_global = (pickle.loads(config["global_control"])
                       if "global_control" in config else [np.zeros(s, dtype=np.float32) for s in shapes])
            c_local = load_control_variate(self.state_dir, self.cid, shapes)
            steps = self._train(epochs, train_loader, class_weights,
                               scaffold_control=c_global, scaffold_local=c_local)
            w_local = get_parameters(self.model)
            denom = max(steps, 1) * self.model_cfg.lr
            c_local_new = [ci - c + (wg - wl) / denom
                          for ci, c, wg, wl in zip(c_local, c_global, parameters, w_local)]
            delta_c = [new - old for new, old in zip(c_local_new, c_local)]
            save_control_variate(self.state_dir, self.cid, c_local_new)
            return w_local, len(train_loader.dataset), {
                "deferred": False, "cid": self.cid, "epochs": epochs,
                "scaffold_delta_c": pickle.dumps(delta_c),
            }

        self._train(epochs, train_loader, class_weights)
        return get_parameters(self.model), len(train_loader.dataset), \
            {"deferred": False, "cid": self.cid, "epochs": epochs}

    def _save_agent_state(self):
        if self.state_dir:
            save_agent_state(self.state_dir, self.cid,
                             {"sigma": self.agent.sigma, "previous_loss": self.agent.previous_loss})

    def model_cfg_epochs(self):
        return getattr(self.model_cfg, "e_std", 3)

    # ---- Flower evaluate (Section 4.5.3, global test set) ----------------
    def evaluate(self, parameters, config):
        set_parameters(self.model, parameters)
        self.model.eval()
        crit = nn.CrossEntropyLoss()
        loss_sum, total, correct = 0.0, 0, 0
        all_pred, all_true = [], []
        with torch.no_grad():
            for xb, yb in self.test_loader:
                xb, yb = xb.to(self.device), yb.to(self.device)
                out = self.model(xb)
                loss_sum += crit(out, yb).item() * len(yb)
                pred = out.argmax(1)
                correct += (pred == yb).sum().item()
                total += len(yb)
                all_pred.extend(pred.cpu().numpy().tolist())
                all_true.extend(yb.cpu().numpy().tolist())
        acc = correct / max(total, 1)
        macro_f1 = f1_score(all_true, all_pred, average="macro", zero_division=0)
        return loss_sum / max(total, 1), total, {"accuracy": acc, "macro_f1": macro_f1}
