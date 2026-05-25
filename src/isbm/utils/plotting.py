import matplotlib.pyplot as plt
import numpy as np


def sorted_adjacency_plot(X, z, ax=None, title=None):
    """Show adjacency matrix X with rows/cols sorted by cluster label z."""
    labels = np.asarray(z)
    idx = np.argsort(labels)
    X_sorted = X[np.ix_(idx, idx)]
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(X_sorted, cmap="Greys", interpolation="nearest", aspect="equal")
    if title:
        ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])
    return ax


def ll_trace_plot(ll_trace, ax=None, burnin=None):
    """Plot pseudo log-likelihood over Gibbs iterations."""
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 4))
    ax.plot(ll_trace)
    if burnin is not None:
        ax.axvline(burnin, linestyle="--", color="grey", label=f"burnin={burnin}")
        ax.legend()
    ax.set_xlabel("iteration")
    ax.set_ylabel("pseudo log-likelihood")
    return ax
