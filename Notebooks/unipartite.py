# /// script
# dependencies = [
#     "marimo",
#     "matplotlib==3.11.2",
#     "numpy==2.5.4",
#     "scipy==1.18.1",
# ]
# requires-python = ">=3.12"
# ///

import marimo

__generated_with = "0.25.2"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Infinite Relational Model (forward pass)

    Single-domain IRM, Ishiguro, Sato & Ueda (2014), Eqs. (10)–(12), p. 7:

    $$\theta_{k,l} \mid a_{k,l}, b_{k,l} \sim \mathrm{Beta}(a_{k,l}, b_{k,l}) \qquad (10)$$

    $$z_i \mid \alpha \sim \mathrm{CRP}(\alpha) \qquad (11)$$

    $$x_{i,j} \mid Z, \{\theta\} \sim \mathrm{Bernoulli}\big(\theta_{z_i, z_j}\big) \qquad (12)$$

    - $z_i$: cluster of node $i$, for $i = 1, \dots, N$
    - $\theta_{k,l}$: probability of a link from cluster $k$ to cluster $l$; a $K \times K$ table, with $K$ the number of distinct labels in $z$
    - $x_{i,j}$: link (1) or no link (0) from node $i$ to node $j$; an $N \times N$ matrix

    Sampling order in code: $z$ first (it fixes $K$), then $\theta$, then $X$.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1. Settings
    Size of the network and the prior parameters.
    """)
    return


@app.cell
def _():
    import numpy as np
    import matplotlib.pyplot as plt

    rng = np.random.default_rng(0)
    N = 60            # number of nodes
    alpha = 2.0       # CRP concentration: larger alpha -> more clusters
    a, b = 0.5, 0.5   # Beta(a, b) prior for every theta_{k,l}
    return N, a, alpha, b, np, plt, rng


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2. Explore the priors
    How $\alpha$ and $N$ shape the clusters, and how $a$ and $b$ shape the link probabilities.
    These plots use their own random numbers, so they do not change the data below.
    """)
    return


@app.cell
def _(N, alpha, mo):
    crp_alpha = mo.ui.slider(0.1, 20.0, step=0.1, value=alpha, label="α (concentration)", show_value=True)
    crp_N = mo.ui.slider(10, 200, step=10, value=N, label="N (nodes)", show_value=True)
    return crp_N, crp_alpha


@app.cell(hide_code=True)
def _(crp_N, crp_alpha, mo, np, plt):
    # CRP explorer: 300 draws of z ~ CRP(alpha) for N nodes (own rng, so the data below is unaffected)
    def _crp_counts(n, a, rng):
        counts = []
        for _i in range(n):
            w = np.array(counts + [a], dtype=float)
            k = rng.choice(len(w), p=w / w.sum())
            if k == len(counts):
                counts.append(1)
            else:
                counts[k] += 1
        return np.array(counts)

    _rng = np.random.default_rng(1)
    _draws = [_crp_counts(crp_N.value, crp_alpha.value, _rng) for _ in range(300)]
    _K = np.array([len(c) for c in _draws])
    _EK = np.sum(crp_alpha.value / (crp_alpha.value + np.arange(crp_N.value)))  # E[K] = sum_i alpha/(alpha+i)

    _fig, (_ax1, _ax2) = plt.subplots(1, 2, figsize=(9, 3.2))
    _ax1.hist(_K, bins=np.arange(_K.min(), _K.max() + 2) - 0.5, color="grey")
    _ax1.axvline(_EK, color="k", ls="--", label=f"E[K] = {_EK:.1f}")
    _ax1.set(xlabel="number of clusters K", ylabel="count (300 draws)", title="How many clusters?")
    _ax1.legend(frameon=False)
    _ax2.bar(range(len(_draws[0])), np.sort(_draws[0])[::-1], color="grey")
    _ax2.set(xlabel="cluster (largest first)", ylabel="nodes", title="Cluster sizes in one draw")
    _fig.tight_layout()
    mo.vstack([mo.md(r"**CRP prior** $z_i \mid \alpha \sim \mathrm{CRP}(\alpha)$"), mo.hstack([crp_alpha, crp_N]), _fig])
    return


@app.cell
def _(a, b, mo):
    beta_a = mo.ui.slider(0.1, 10.0, step=0.1, value=a, label="a", show_value=True)
    beta_b = mo.ui.slider(0.1, 10.0, step=0.1, value=b, label="b", show_value=True)
    return beta_a, beta_b


@app.cell(hide_code=True)
def _(K_true, beta_a, beta_b, mo, np, plt, z_true):
    # Beta explorer: density of one theta_{k,l}, one K x K draw of theta, and the X it gives,
    # sorted by z_true (own rng, so the data in sections 3-5 is unaffected)
    from scipy.stats import beta as _beta_dist

    _a, _b = beta_a.value, beta_b.value
    _x = np.linspace(0.001, 0.999, 500)
    _pdf = _beta_dist.pdf(_x, _a, _b)
    _rng = np.random.default_rng(2)
    _theta = _rng.beta(_a, _b, size=(K_true, K_true))
    _X = _rng.binomial(1, _theta[z_true][:, z_true])          # same labels z_true, new theta
    _order = np.argsort(z_true, kind="stable")
    _edges = np.cumsum(np.bincount(z_true))[:-1] - 0.5         # cluster borders after sorting

    _fig, (_ax1, _ax2, _ax3) = plt.subplots(1, 3, figsize=(11, 3.4), gridspec_kw={"width_ratios": [1.5, 1, 1]},
                                            constrained_layout=True)
    _ax1.plot(_x, _pdf, color="k")
    _ax1.fill_between(_x, _pdf, color="grey", alpha=0.3)
    _ax1.axvline(_a / (_a + _b), color="k", ls="--")
    _ax1.set(xlabel=r"$\theta_{k,l}$", ylabel="density",
             title=f"Beta({_a:.1f}, {_b:.1f}): mean {_a / (_a + _b):.2f} (dashed), a+b = {_a + _b:.1f}",
             ylim=(0, min(_pdf.max() * 1.05, 6)))
    _ax2.imshow(_theta, cmap="Greys", vmin=0, vmax=1)
    for _k in range(K_true):
        for _l in range(K_true):
            _ax2.text(_l, _k, f"{_theta[_k, _l]:.1f}", ha="center", va="center", fontsize=6,
                      color="w" if _theta[_k, _l] > 0.5 else "k")
    _ax2.set(title=f"one {K_true}×{K_true} draw of θ", xticks=[], yticks=[])
    _ax3.imshow(_X[_order][:, _order], cmap="Greys", vmin=0, vmax=1, interpolation="nearest")
    for _e in _edges:
        _ax3.axhline(_e, color="tab:red", lw=0.5)
        _ax3.axvline(_e, color="tab:red", lw=0.5)
    _ax3.set(title="X from this θ, sorted by z", xticks=[], yticks=[])
    mo.vstack([mo.md(r"**Beta prior** $\theta_{k,l} \mid a, b \sim \mathrm{Beta}(a, b)$"), mo.hstack([beta_a, beta_b]), _fig])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3. Cluster labels $z \sim \mathrm{CRP}(\alpha)$
    One label per node. The number of distinct labels is $K$.

    Eq. (11) written out, one node at a time. Node $i$ ($i = 0, \dots, N-1$, counted from 0 as in the code) arrives after $i$ nodes are seated. Let $n_k$ be the number of those earlier nodes at table $k$, and $K_i$ the number of tables opened so far:

    $$P(z_i = k \mid z_0, \dots, z_{i-1}) =
    \begin{cases}
    \dfrac{n_k}{i + \alpha} & k = 0, \dots, K_i - 1 \quad \text{(join table } k\text{)} \\[2ex]
    \dfrac{\alpha}{i + \alpha} & k = K_i \quad \text{(open a new table)}
    \end{cases}$$

    With nodes counted from 1, as in the papers, the denominator reads $i - 1 + \alpha$.

    | Math | Code |
    |---|---|
    | $n_k$ | `counts[k]` |
    | $K_i$ | `len(counts)` |
    | $(n_0, \dots, n_{K_i-1}, \alpha)$ | `weights = counts + [alpha]` |
    | $i + \alpha$ | `weights.sum()` |
    | $z_i = K_i$ (new table) | `k == len(counts)` |
    | $K = \max_i z_i + 1$ | `K_true = z_true.max() + 1` |

    Expected number of tables after $N$ nodes: $E[K] = \sum_{i=0}^{N-1} \dfrac{\alpha}{\alpha + i}$ (about 7.4 for $N = 60$, $\alpha = 2$).
    """)
    return


@app.cell
def _(N, alpha, np, rng):
    def sample_crp(N, alpha, rng):
        z = np.zeros(N, dtype=int)
        counts = []                       # counts[k] = nodes seated in cluster k so far
        for i in range(N):
            weights = np.array(counts + [alpha], dtype=float)  # existing clusters, then "new"
            k = rng.choice(len(weights), p=weights / weights.sum())
            if k == len(counts):
                counts.append(1)          # node i opens a new cluster
            else:
                counts[k] += 1            # node i joins cluster k
            z[i] = k
        return z

    z_true = sample_crp(N, alpha, rng)
    K_true = z_true.max() + 1
    print("K =", K_true, "| cluster sizes:", np.bincount(z_true))
    return K_true, z_true


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 4. Link probabilities $\theta_{k,l} \sim \mathrm{Beta}(a, b)$
    One probability per pair of clusters: a $K \times K$ table.
    """)
    return


@app.cell
def _(K_true, a, b, np, rng):
    theta_true = rng.beta(a, b, size=(K_true, K_true))
    print(np.round(theta_true, 2))
    return (theta_true,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 5. Links $x_{i,j} \sim \mathrm{Bernoulli}(\theta_{z_i, z_j})$
    One coin flip per pair of nodes: an $N \times N$ matrix.
    """)
    return


@app.cell
def _(rng, theta_true, z_true):
    P = theta_true[z_true][:, z_true]   # N x N: the theta that each cell of X uses
    X = rng.binomial(1, P)
    print("X shape:", X.shape, "| link density:", X.mean().round(3))
    return (X,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 6. Look at the data
    Left: what the sampler will see. Right: the same matrix sorted by the true labels.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.vstack([
        mo.image(src=str(mo.notebook_dir() / "assets" / "ishiguro2014_fig1.png"), width=700),
        mo.md(r"""
    *Figure 1 of Ishiguro, Sato & Ueda (2014), p. 7. (a) The observed matrix $X$. (b) The same matrix with rows and columns sorted by the inferred clusters $Z$: blocks appear.*
    Our plot below shows the same idea for our data. One difference: the paper's example is **two-domain** (rows and columns have separate clusters $k$ and $l$), while ours is **single-domain**, so rows and columns are sorted by the same $z$.
    """),
    ])
    return


@app.cell
def _(X, np, plt, z_true):
    order = np.argsort(z_true, kind="stable")
    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    axes[0].imshow(X, cmap="Greys"); axes[0].set_title("X as observed")
    axes[1].imshow(X[order][:, order], cmap="Greys"); axes[1].set_title("X sorted by true z")
    fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Check on the true labels.** With $z$ = `z_true`, large blocks should give $\theta$ close to `theta_true`; blocks with few pairs stay uncertain.
    """)
    return


@app.cell
def _(
    K_true,
    X,
    a,
    b,
    block_counts,
    gibbs_rng,
    sample_theta,
    theta_true,
    z_true,
):
    _n1, _n0 = block_counts(X, z_true, K_true)
    _draw = sample_theta(X, z_true, K_true, a, b, gibbs_rng)
    _mean = (a + _n1) / (a + b + _n1 + _n0)
    print("total pairs:", (_n1 + _n0).sum(), "| total links:", _n1.sum(), "=", X.sum())
    for (k, l) in [(0, 0), (2, 1), (0, 1), (3, 3), (5, 5), (5, 6)]:
        print((k, l), "pairs", _n1[k, l] + _n0[k, l], "links", _n1[k, l],
              "| true", theta_true[k, l].round(2), "post mean", _mean[k, l].round(2), "draw", _draw[k, l].round(2))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # MCMC
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Metropolis-Hastings
    - This is the foundation of Gibbs Sampler
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Gibbs sampling
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Ising Model
    """)
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
