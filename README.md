

## Structure

```
configs/         Hydra configs (data / model / experiment / tracking)
src/isbm/        Python package
  models/        CGS_IRM sampler
  data/          Synthetic-graph generator
  metrics/       ARI, held-out predictive log-likelihood
  utils/         Seeding helper
  experiments/   Hydra entry point (run.py)
Notebooks/       Exploratory notebooks
outputs/         Hydra per-run dirs (gitignored)
mlruns/          MLflow local store (gitignored)
```

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## Running experiments

Run from the project root with `src/` on the Python path:

```bash
export PYTHONPATH=src   # or prefix each command

# Default run (Hydra reads configs/config.yaml)
python -m isbm.experiments.run

# Override any leaf parameter
python -m isbm.experiments.run model.n_iter=1000 data.N=300

# Sweep
python -m isbm.experiments.run -m model.alpha1=0.5,1.0,2.0

# Skip MLflow tracking for a quick run
python -m isbm.experiments.run tracking=none

# Browse tracked runs
mlflow ui
```

Each run writes to `outputs/<date>/<time>/` containing `.hydra/config.yaml` (full resolved config) and the Hydra log.
