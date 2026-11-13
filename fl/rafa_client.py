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
from agent.reactive_agent import ReactiveAgent, Action
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
                 state_dir=None, resource_seed=0, exp_seed=0, observe_only=False):
        self.cid = cid
        self.observe_only = observe_only
        self.exp_seed = int(exp_seed)
        self.last_train_sq_mean = None
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
                self.agent.previous_loss = state.get("previous_loss")
                self.agent.last_delta = state.get("last_delta")

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
        # SCAFFOLD's control-variate correction (Option II) is derived for plain
        # SGD. A constant per-step correction term compounds through a momentum
        # buffer's exponential moving average (roughly 1/(1-momentum) = 10x at
        # momentum=0.9), which blows up the weights within a few dozen rounds --
        # confirmed empirically (accuracy collapsed to chance level ~round 20).
        # Momentum is disabled specifically for the SCAFFOLD-corrected path.
        momentum = 0.0 if scaffold_control is not None else self.model_cfg.momentum
        opt = torch.optim.SGD(self.model.parameters(), lr=self.model_cfg.lr, momentum=momentum)
        # Per-sample (unweighted) losses from the final epoch, for OORT's
        # statistical utility sqrt(sum_j loss_j^2 / |B|) (Section 4.7.5).
        sample_crit = nn.CrossEntropyLoss(reduction="none")
        sq_sum, sq_n = 0.0, 0
        steps, loss_sum = 0, 0.0
        for ep in range(epochs):
            for xb, yb in train_loader:
                xb, yb = xb.to(self.device), yb.to(self.device)
                opt.zero_grad()
                out = self.model(xb)
                loss = crit(out, yb)
                loss_sum += loss.item()
                if ep == epochs - 1:
                    with torch.no_grad():
                        per = sample_crit(out, yb)
                        sq_sum += float((per ** 2).sum().item())
                        sq_n += len(yb)
                if proximal_mu > 0 and proximal_global is not None:
                    loss = loss + proximal_penalty(self.model, proximal_global, proximal_mu)
                loss.backward()
                if scaffold_control is not None:
                    with torch.no_grad():
                        for p, c, ci in zip(self.model.parameters(), scaffold_control, scaffold_local):
                            p.grad += torch.tensor(c - ci, device=p.device, dtype=p.grad.dtype)
                opt.step()
                steps += 1
        self.last_train_sq_mean = sq_sum / max(sq_n, 1)
        return steps, loss_sum / max(steps, 1)

    def _drift_labels(self, y_active, server_round):
        """Labels the drift score is computed on: a seeded batch, or all labels."""
        b = self.agent.cfg.drift_batch
        if b is None or b >= len(y_active):
            return y_active
        rng = np.random.RandomState(self._seed_round(server_round, stream=2))
        return rng.choice(y_active, size=b, replace=False)

    def _seed_round(self, server_round, stream=0):
        """
        Deterministic per-(experiment seed, client, round) seed. Seeds torch
        (DataLoader shuffling, dropout) inside whichever Ray worker runs this
        fit(), since parent-process seeding does not reach worker processes.
        SeedSequence avoids the collisions of seed+cid (e.g. 42+81 == 123+0).
        """
        return int(np.random.SeedSequence(
            [self.exp_seed, int(self.cid), int(server_round), stream]).generate_state(1)[0])

    # ---- Flower fit (Section 4.5.2) ---------------------------------------
    def fit(self, parameters, config):
        set_parameters(self.model, parameters)
        server_round = int(config.get("server_round", 1))
        torch.manual_seed(self._seed_round(server_round, stream=0))
        y_tr_active, y_val_active = self._active_labels(server_round)
        train_loader = _loader(self.X_tr, y_tr_active, self.model_cfg.batch_size, True)
        val_loader = _loader(self.X_val, y_val_active, self.model_cfg.batch_size, False)
        class_weights = _class_weights(y_tr_active, self.model_cfg.num_classes).to(self.device)

        # Non-agent baselines: standard fixed-epoch training, always communicate.
        if not self.use_agent:
            return self._fit_baseline(parameters, config, train_loader, class_weights)

        # Agent path. delta_t is the post-training improvement recorded at this
        # client's previous communicating round (None until two have happened);
        # no pre-training evaluation of the received global model is needed.
        delta_t = self.agent.compute_delta()
        drift_t = self.agent.compute_drift(self._drift_labels(y_tr_active, server_round))
        self.agent.rng = np.random.RandomState(self._seed_round(server_round, stream=1))
        rho_t = self.agent.sample_resource()

        if self.observe_only:
            # Pilot mode: record the same (delta, d, rho) observations as the
            # main runs, but act as FedAvg (E_std epochs, always communicate)
            # so the observations do not depend on uncalibrated thresholds.
            action = Action(1, self.agent.cfg.e_std, 1, "OBSERVE")
        else:
            action = self.agent.decide(delta_t, drift_t, rho_t)
        delta_log = float(delta_t) if delta_t is not None else float("nan")

        if action.participate == 0:                     # defer triple (0,0,0)
            self._save_agent_state()
            # NOTE: Flower still requires parameters in FitRes, so a deferred
            # client returns the (unchanged) global model. The server excludes
            # it from aggregation and from the logical-transmission count; no
            # network traffic saving is measured by this simulation.
            return get_parameters(self.model), 0, {
                "deferred": True, "cid": self.cid, "sigma": self.agent.sigma,
                "drift": drift_t, "delta": delta_log, "rho": rho_t,
                "epochs": 0, "rule": action.rule,
            }

        self._train(action.epochs, train_loader, class_weights)
        post_loss = self._evaluate_local(val_loader, class_weights)
        self.agent.update_loss(post_loss)               # Algorithm 1, lines 24-25
        self._save_agent_state()
        return get_parameters(self.model), len(train_loader.dataset), {
            "deferred": False, "cid": self.cid, "sigma": 0, "drift": drift_t,
            "delta": delta_log, "rho": rho_t, "val_loss_post": post_loss,
            "epochs": action.epochs, "rule": action.rule,
            "train_loss_sq_mean": self.last_train_sq_mean,
        }

    def _fit_baseline(self, parameters, config, train_loader, class_weights):
        epochs = self.model_cfg_epochs()

        if self.method == "fedprox":
            _, avg_loss = self._train(epochs, train_loader, class_weights,
                                      proximal_global=parameters, proximal_mu=EXPERIMENT.fedprox_mu)
            return get_parameters(self.model), len(train_loader.dataset), \
                {"deferred": False, "cid": self.cid, "epochs": epochs, "train_loss": avg_loss,
                "train_loss_sq_mean": self.last_train_sq_mean}

        if self.method == "scaffold":
            shapes = [p.shape for p in parameters]
            c_global = (pickle.loads(config["global_control"])
                       if "global_control" in config else [np.zeros(s, dtype=np.float32) for s in shapes])
            c_local = load_control_variate(self.state_dir, self.cid, shapes)
            steps, avg_loss = self._train(epochs, train_loader, class_weights,
                                          scaffold_control=c_global, scaffold_local=c_local)
            w_local = get_parameters(self.model)
            denom = max(steps, 1) * self.model_cfg.lr
            c_local_new = [ci - c + (wg - wl) / denom
                          for ci, c, wg, wl in zip(c_local, c_global, parameters, w_local)]
            delta_c = [new - old for new, old in zip(c_local_new, c_local)]
            save_control_variate(self.state_dir, self.cid, c_local_new)
            return w_local, len(train_loader.dataset), {
                "deferred": False, "cid": self.cid, "epochs": epochs, "train_loss": avg_loss,
                "train_loss_sq_mean": self.last_train_sq_mean,
                "scaffold_delta_c": pickle.dumps(delta_c),
            }

        _, avg_loss = self._train(epochs, train_loader, class_weights)
        return get_parameters(self.model), len(train_loader.dataset), \
            {"deferred": False, "cid": self.cid, "epochs": epochs, "train_loss": avg_loss,
                "train_loss_sq_mean": self.last_train_sq_mean}

    def _save_agent_state(self):
        if self.state_dir:
            save_agent_state(self.state_dir, self.cid,
                             {"sigma": self.agent.sigma,
                              "previous_loss": self.agent.previous_loss,
                              "last_delta": self.agent.last_delta})

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
