

## Structure

```
configs/         Hydra configs (data / model / experiment / tracking)
src/isbm/        Python package
  models/        CGS_IRM (baseline), DA_IRM (delayed acceptance), SMC_IRM
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

Three surrogates are available:

- `surrogate=plugin` (default): plug-in posterior-mean Bernoulli likelihood
  (`theta_hat = (a + n) / (a + b + n + Nm)`), fully vectorized and `gammaln`-free.
- `surrogate=subsample`: the exact `gammaln` marginal on a random `subsample_frac`
  of opposite-domain clusters, rescaled to estimate the full sum.
- `surrogate=learned`: a machine-learning surrogate (DAphyloSMC / adamcmcpaper
  style) that learns the *residual* between the exact marginal and the plug-in
  baseline from cheap move features, trained online from the exact evaluations
  the kernel already computes. `learned_regressor` selects `linear` (fast numpy
  poly-ridge, default), `rf` (random forest), or `gp` (Gaussian process).

```bash
# Run the delayed-acceptance sampler
python -m isbm.experiments.run model=da_irm

# Choose the surrogate and its knob
python -m isbm.experiments.run model=da_irm model.surrogate=subsample model.subsample_frac=0.25
python -m isbm.experiments.run model=da_irm model.surrogate=learned model.learned_regressor=linear
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

- **`learned`** confirms the plug-in's failure is a *learnable bias*: an online
  regressor on the exact-vs-plug-in residual restores acceptance dramatically
  (e.g. random forest: ~0.003 -> ~0.41, recovering near-baseline mixing) using
  ~50x fewer exact evaluations. The catch is inference cost -- an RF accurate
  enough to help is far too slow per query for this cheap model (orders of
  magnitude slower wall-clock), and the fast `linear` variant only partially
  closes the gap (acceptance ~0.14, still short of baseline mixing). A learned
  surrogate only pays off when the exact likelihood is much more expensive than
  a model query.

The broader takeaway matches the delayed-acceptance literature: surrogate
screening helps when the exact likelihood is expensive relative to the surrogate
*and* the surrogate is accurate in the model's effective dimension. For a cheap
conjugate collapsed model these margins are thin, so the plug-in screen degrades
mixing, the learned surrogate is accurate but not cheap enough, and only the
exact-on-subset screen offers a modest net gain.

## SMC-over-data sampler

`SMC_IRM` (`configs/model/smc_irm.yaml`) is a sequential Monte Carlo sampler
over data (the DAphyloSMC / Bon et al. blueprint), for a single symmetric
partition IRM (CRP prior + directed Beta-Bernoulli blocks) -- matching the
data-generating process of `generate_irm_graph`, so `z1_` is directly comparable
via ARI. Nodes are revealed one at a time; each particle extends its partition
by assigning the new node, weighted by the collapsed predictive of that node's
edges to the already-placed nodes. Particles are resampled (systematic) when the
effective sample size drops below `resample_threshold * n_particles` and
rejuvenated with `n_rejuv` collapsed-Gibbs sweeps. The sampler also returns an
unbiased log marginal likelihood estimate (`model.logZ_`).

The delayed-acceptance idea plugs into the proposal:

- `proposal=exact`: the locally optimal proposal, scoring every candidate
  cluster with the exact predictive (`O(K)` exact predictives per node).
- `proposal=surrogate`: a plug-in predictive builds the proposal and only the
  *sampled* candidate is scored exactly (`O(1)` exact predictives per node), with
  the incremental importance weight corrected so the SMC estimator stays valid.

```bash
python -m isbm.experiments.run model=smc_irm
python -m isbm.experiments.run model=smc_irm model.proposal=surrogate
python -m isbm.experiments.benchmark --N 120 --seeds 2 --smc --n-particles 200 --n-rejuv 2
```

### Does the SMC route improve things?

- The surrogate proposal transfers the acceleration into SMC: it roughly halves
  the exact predictive evaluations and runs ~1.6-1.7x faster than exact SMC, at
  a modest cost in proposal quality (slightly lower ARI / `logZ`).
- SMC quality improves as expected with more particles and, especially, more
  rejuvenation sweeps (ARI climbs toward the Gibbs value).
- For point-estimate clustering on this cheap conjugate model, single-pass SMC
  is not competitive with collapsed Gibbs on wall-clock (each particle repeats
  the collapsed work, so total exact evaluations are much higher). Its real
  value is the marginal likelihood estimate and embarrassingly-parallel
  particles; the surrogate screen makes it cheaper without changing the target.
