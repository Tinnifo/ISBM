# ISBM: infinite stochastic block models

Study code for the infinite relational model (IRM) and its inference, following
Ishiguro, Sato & Ueda, *Collapsed Variational Bayes Inference of Infinite
Relational Model* (arXiv:1409.4757, 2014; published as *Averaged Collapsed
Variational Bayes Inference*, JMLR 18, 2017).

## Structure

```
src/isbm/            Python package (plain modules)
  irm.py             generate_irm_graph: single-domain IRM generator
  cgs.py             CGS_IRM: collapsed Gibbs sampler, two-domain IRM
  metrics.py         ari, heldout_test_ll
Notebooks/           marimo notebooks
  unipartite.py                 single-domain IRM + plain Gibbs (current study)
  Ising_model.py                Ising model, Boltzmann distribution, sampling
  gibbs_sampling.py             Gibbs sampling from scratch (Pólya)
  ad_posterior_walkthrough.py   4-person exact posterior vs. Gibbs chain
  learning_bayes.py             outline of inference methods
```

## Setup

```bash
uv sync          # creates .venv and installs isbm (editable) + dependencies
```

## Use

```bash
uv run marimo edit Notebooks/unipartite.py
```

```python
from isbm import generate_irm_graph, CGS_IRM
from isbm.metrics import ari

X, z = generate_irm_graph(N=100, seed=0)
model = CGS_IRM(n_iter=200, burnin=100, seed=0).fit(X)
print(ari(z, model.z1_))
```
