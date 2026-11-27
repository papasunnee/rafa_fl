"""
centralised.py — Centralised reference upper bound (not a federated method).

Trains the same HAR_MLP on the pooled UCI HAR training set, with the same
optimiser, learning rate, batch size and class-weighted loss as the federated
clients, so the only difference from the federated runs is that the data is
not partitioned. It answers one question for the thesis: how much accuracy
does federation cost? It is a reference row, not part of the hypothesis tests.

Protocol (per seed):
  - 20% of the training set is held out (stratified, seeded) as a validation
    set, mirroring the clients' local 80/20 train/val split;
  - train for --epochs epochs; the reported model is the one from the epoch
    with the lowest validation loss (the test set is never used for selection);
  - report test accuracy and macro-F1 of that model, plus the final epoch's.

Outputs: <out>/centralised_s<seed>.json per seed and <out>/summary.json
(mean and std over seeds). The default --out is results/centralised/, a
subfolder, so it is never mixed into the federated results.

Usage:
    python -m experiments.centralised
"""

import os, sys, json, argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import f1_score

from config import MODEL, EXPERIMENT
from data.har_loader import load_har, stratified_val_split
from models.mlp import HAR_MLP
from experiments.run_experiment import set_all_seeds, provenance


def _class_weights(y, num_classes):
    # identical to fl/rafa_client.py::_class_weights
    counts = np.bincount(y, minlength=num_classes).astype(np.float64) + 1.0
    inv = 1.0 / counts
    return torch.tensor(inv / inv.sum() * num_classes, dtype=torch.float32)


def _evaluate(model, X, y, crit=None):
    model.eval()
    with torch.no_grad():
        out = model(torch.tensor(X))
        pred = out.argmax(1).numpy()
        loss = crit(out, torch.tensor(y, dtype=torch.long)).item() if crit else None
    return {"loss": loss, "accuracy": float((pred == y).mean()),
            "macro_f1": float(f1_score(y, pred, average="macro", zero_division=0))}


def run_seed(seed, epochs, X_train, y_train, X_test, y_test):
    set_all_seeds(seed)
    rng = np.random.RandomState(seed)
    tr_idx, val_idx = stratified_val_split(np.arange(len(y_train)), y_train,
                                           EXPERIMENT.val_fraction, rng)
    Xtr, ytr = X_train[tr_idx], y_train[tr_idx]
    Xval, yval = X_train[val_idx], y_train[val_idx]

    torch.manual_seed(seed)                       # same init as the federated runs
    model = HAR_MLP(MODEL)
    w = _class_weights(ytr, MODEL.num_classes)
    crit = nn.CrossEntropyLoss(weight=w)
    opt = torch.optim.SGD(model.parameters(), lr=MODEL.lr, momentum=MODEL.momentum)
    loader = DataLoader(TensorDataset(torch.tensor(Xtr), torch.tensor(ytr, dtype=torch.long)),
                        batch_size=MODEL.batch_size, shuffle=True)

    curve, best = [], None
    for ep in range(1, epochs + 1):
        model.train()
        for xb, yb in loader:
            opt.zero_grad()
            crit(model(xb), yb).backward()
            opt.step()
        val = _evaluate(model, Xval, yval, crit)
        test = _evaluate(model, X_test, y_test)
        curve.append({"epoch": ep, "val_loss": val["loss"],
                      "test_accuracy": test["accuracy"], "test_macro_f1": test["macro_f1"]})
        if best is None or val["loss"] < best["val_loss"]:
            best = curve[-1]

    return {
        "method": "centralised", "seed": seed, "epochs": epochs,
        "train_samples": int(len(ytr)), "val_samples": int(len(yval)),
        "selected_epoch": best["epoch"],
        "test_accuracy": best["test_accuracy"], "test_macro_f1": best["test_macro_f1"],
        "final_epoch_test_accuracy": curve[-1]["test_accuracy"],
        "final_epoch_test_macro_f1": curve[-1]["test_macro_f1"],
        "curve": curve,
        "provenance": provenance(X_train, y_train),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--out", default=os.path.join("results", "centralised"))
    args = ap.parse_args()

    X_train, y_train, X_test, y_test = load_har(EXPERIMENT.dataset_root)
    os.makedirs(args.out, exist_ok=True)
    rows = []
    for seed in EXPERIMENT.seeds:
        r = run_seed(seed, args.epochs, X_train, y_train, X_test, y_test)
        with open(os.path.join(args.out, f"centralised_s{seed}.json"), "w") as f:
            json.dump(r, f, indent=2)
        print(f"seed {seed}: acc {r['test_accuracy']:.4f}  F1 {r['test_macro_f1']:.4f}  "
              f"(epoch {r['selected_epoch']})")
        rows.append(r)

    acc = np.array([r["test_accuracy"] for r in rows])
    f1 = np.array([r["test_macro_f1"] for r in rows])
    summary = {"method": "centralised", "seeds": EXPERIMENT.seeds, "epochs": args.epochs,
               "accuracy_mean": float(acc.mean()), "accuracy_std": float(acc.std(ddof=1)),
               "macro_f1_mean": float(f1.mean()), "macro_f1_std": float(f1.std(ddof=1)),
               "selected_epochs": [r["selected_epoch"] for r in rows]}
    with open(os.path.join(args.out, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
