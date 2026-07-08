# RAFA-FL Experiment Runner

Reference implementation for the RAFA-FL thesis (Chapters 3–4). Reactive-agent
federated learning with drift-aware joint governance of participation, local
training intensity, and communication.

## Environment

```bash
python3 -m venv venv && source venv/bin/activate
pip install torch flwr[simulation] numpy scipy scikit-learn
```

Tested targets: Python 3.10, PyTorch 2.0, Flower (flwr) 1.x.

## Dataset

Download the UCI HAR dataset and set `EXPERIMENT.dataset_root` in `config.py`
to its path. Expected layout:

```
UCI_HAR_Dataset/
  train/X_train.txt  train/y_train.txt
  test/X_test.txt    test/y_test.txt
```

## Workflow

1. **Validate the agent logic** (no dataset or torch needed):
   ```bash
   python agent/test_agent.py
   ```

2. **Run the pilot** to set thresholds and confirm epoch-function variation:
   ```bash
   python -m experiments.pilot
   ```
   Paste the printed `theta_delta` and `theta_d` into `config.py`.
   If it warns that the epoch function collapsed, retune `lam` / `e_max`
   before proceeding.

3. **Smoke test** the full pipeline on a small grid:
   ```bash
   python -m experiments.run_all --quick --out results/
   ```

4. **Full grid** (methods × α × N × seeds × drift, plus ablations):
   ```bash
   python -m experiments.run_all --out results/
   ```

5. **Analyse** the JSON logs in `results/` with `analysis/metrics.py` and
   `analysis/stats.py` to produce the Chapter 5 tables.

## Module map (matches Chapter 4)

| File | Thesis section |
|------|----------------|
| `config.py` | 3.9, 3.15 |
| `data/har_loader.py` | 4.2.1 |
| `data/dirichlet.py` | 3.3, 3.4 |
| `models/mlp.py` | 3.7 |
| `agent/drift_score.py` | 3.6.1, 4.4.3 |
| `agent/reactive_agent.py` | 3.6, 4.4 |
| `fl/rafa_client.py` | 4.5 |
| `fl/rafa_strategy.py` | 4.6 |
| `fl/baselines.py` | 4.7 |
| `experiments/*` | 4.8 |
| `analysis/*` | 3.11, 3.12 |

## Notes on baseline fidelity

FedCS and OORT use documented simulation simplifications (Sections 4.7.4–4.7.5):
FedCS ranks by the resource proxy in place of a wall-clock deadline; OORT uses
its statistical-utility component only, excluding system speed. Both simplifications
are conservative — they favour the baseline — so any RAFA-FL advantage is a lower bound.

FedProx and SCAFFOLD client-side hooks (proximal term, control variates) are
provided in `fl/baselines.py`; wire them into the client factory in
`run_experiment.py` when running those baselines (the current factory runs the
shared training loop, which is correct for FedAvg/FedCS/OORT).
```
