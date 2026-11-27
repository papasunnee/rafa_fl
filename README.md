# RAFA-FL Experiment Runner

Reference implementation for the RAFA-FL thesis (Chapters 3–4). Reactive-agent
federated learning with drift-aware joint governance of participation, local
training intensity, and communication.

## Environment

```bash
python -m venv venv && venv\Scriptsctivate        # Windows (source venv/bin/activate elsewhere)
pip install -r requirements.txt                       # pins flwr[simulation]==1.30.0
```

Tested: Python 3.10, PyTorch 2.x, Flower 1.30.0, Windows 11 (Ray local cluster).

## Dataset

Download the UCI HAR dataset and set `EXPERIMENT.dataset_root` in `config.py`
to its path. Expected layout:

```
UCI_HAR_Dataset/
  train/X_train.txt  train/y_train.txt
  test/X_test.txt    test/y_test.txt
```

## Workflow

1. **Tests** (agent logic, partitioning, OORT, state persistence, determinism):
   ```bash
   python agent/test_agent.py
   python fl/test_pipeline.py
   ```

2. **Pilot** (observe-only federated run that calibrates theta_delta and theta_d):
   ```bash
   python -m experiments.pilot --write_config
   ```
   Writes `experiments/pilot_output/pilot_run.json` and `pilot_report.json`
   and freezes both thresholds into `config.py`, tagged with the report's
   SHA-256. It refuses to write if the pilot run is invalid or too small.

3. **Smoke test** into a throwaway folder (git-ignored), never into `results/`:
   ```bash
   python -m experiments.run_all --quick --out results_test/
   ```

4. **Experiments** (each skips runs whose JSON already exists, so it can be
   stopped and restarted):
   ```bash
   python -m experiments.run_all --out results/                # 540 main + 25 ablation runs
   python -m experiments.run_all --out results/ --sensitivity  # 30 theta_rho sweep runs
   python -m experiments.centralised                           # centralised reference -> results/centralised/
   ```
   `--only_n 10|50|100` runs one N value, so the grid can be split across
   processes (two at a time was stable on a 16 GB laptop; three was not).

5. **Tables and figures** for Chapter 5:
   ```bash
   python -m analysis.thesis_outputs     # results/ -> tables/ (CSV + thesis_tables.md)
   python -m analysis.plots              # results/ -> figures/
   python -m analysis.report --metric post_drift_mean_accuracy --full_grid   # ad-hoc checks
   python ui/scripts/export_results.py   # refresh the dashboard snapshot
   ```

## Output folders

| Folder | Contents |
|--------|----------|
| `results/` | one JSON per run (accuracy curves, per-round and per-client logs, validity flag, provenance: git commit, package versions, dataset hash, config) |
| `results/centralised/` | centralised reference, one JSON per seed + `summary.json` |
| `results/_logs/` | run logs (git-ignored) |
| `tables/` | Chapter 5 tables as CSV, plus `thesis_tables.md` |
| `figures/` | Chapter 5 figures (PNG) |
| `experiments/pilot_output/` | pilot run and threshold report |

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

FedProx (proximal term) and SCAFFOLD (control variates) run inside the shared
client in `fl/rafa_client.py`. Every method uses the same optimiser: plain SGD
(momentum 0, `ModelConfig.momentum`), because SCAFFOLD's correction assumes
plain SGD and momentum 0.9 destabilised training on skewed clients.

## Reproducibility note

The seed fixes the data partition, the initial model, which clients drift,
local training randomness, resource draws and FedCS/OORT selection. Random
client selection (FedAvg, FedProx, SCAFFOLD, RAFA-FL) goes through Flower's
`SimpleClientManager.sample`, which uses Python's global `random` and client
registration order, so the exact clients chosen can differ between reruns.
`tables/table5_10_replicate_noise.csv` measures the resulting run-to-run noise.
