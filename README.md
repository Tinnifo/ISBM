

## Structure

```
configs/         Hydra configs (data / model / experiment / tracking)
src/isbm/        Python package
  models/        CGS_IRM (baseline) + DA_IRM (delayed acceptance)
                 marginal.py (shared Beta-Bernoulli marginal), surrogates.py
  data/          Synthetic-graph generator
  metrics/       ARI, held-out predictive log-likelihood, ESS
  utils/         Seeding helper
  experiments/   Hydra entry point (run.py) + benchmark.py
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

## Delayed-acceptance sampler

`DA_IRM` (`configs/model/da_irm.yaml`) is a delayed-acceptance / Metropolis-within-Gibbs
variant of the collapsed Gibbs sampler. Instead of evaluating the exact
Beta-Bernoulli collapsed marginal for *every* candidate cluster of a node (the
`O(N * K * K_other)` `gammaln` scan of the baseline), each reassignment is a
two-stage move:

1. A cheap **surrogate** conditional `q(k) ~ prior(k) * surrogate_lik(k)` is
   built over all candidate clusters and a proposal `k*` is drawn from it.
   Because we propose from the surrogate directly, the delayed-acceptance
   stage-1 screen (`alpha_1`) is identically 1.
2. The proposal is corrected with the **exact** marginal at only the current
   cluster `k_old` and `k*`:

   ```
   alpha_2 = min(1, [L(k*)/L~(k*)] * [L~(k_old)/L(k_old)])
   ```

This is an independence Metropolis-Hastings step targeting the exact collapsed
full conditional, so it leaves the **same posterior** as the baseline invariant
while cutting the exact-marginal evaluations per node from `K + 1` to at most 2.

Two surrogates are available:

- `surrogate=plugin` (default): plug-in posterior-mean Bernoulli likelihood
  (`theta_hat = (a + n) / (a + b + n + Nm)`), fully vectorized and `gammaln`-free.
- `surrogate=subsample`: the exact `gammaln` marginal on a random `subsample_frac`
  of opposite-domain clusters, rescaled to estimate the full sum.

```bash
# Run the delayed-acceptance sampler
python -m isbm.experiments.run model=da_irm

# Choose the surrogate and its knob
python -m isbm.experiments.run model=da_irm model.surrogate=subsample model.subsample_frac=0.25
```

### Benchmark

Compare the baseline against the DA variants (wall-clock, exact-evaluation
count, ESS of the pseudo-LL trace, and correctness parity via ARI / mean
pseudo-LL) on the same synthetic graphs:

```bash
python -m isbm.experiments.benchmark --N 150 --n-iter 300 --seeds 3
python -m isbm.experiments.benchmark --N 150 --n-iter 300 --seeds 3 --alpha 40   # many small clusters
```

### Does it improve the Gibbs sampler?

The correctness of the kernel is confirmed on small graphs: where the plug-in
surrogate is accurate (few, small clusters), the DA co-clustering distribution
matches the exact Gibbs baseline and the stage-2 acceptance is high (~0.6).

At scale the answer is nuanced and surrogate-dependent:

- **`plugin`** is the cheapest (fully vectorized, `gammaln`-free) and gives a
  ~1.7x wall-clock speedup with ~10x fewer exact evaluations, but its per-block
  approximation error *accumulates* across the many blocks of a large model, so
  the stage-2 acceptance collapses (~0.002) and the chain mixes poorly / stays
  biased away from the high-probability region. It does **not** improve mixing
  at scale despite being faster per sweep.
- **`subsample`** keeps each retained block exact, so its bias is small: it
  reaches near-baseline pseudo-LL and ARI while running ~1.3-1.4x faster than the
  baseline. This is the more promising variant and exposes a genuine
  speed/quality knob via `subsample_frac`.

The broader takeaway matches the delayed-acceptance literature: surrogate
screening helps when the exact likelihood is expensive relative to the surrogate
*and* the surrogate is accurate in the model's effective dimension. For a cheap
conjugate collapsed model these margins are thin, so the plug-in screen degrades
mixing while the exact-on-subset screen offers only a modest net gain.
