"""Cheap surrogate likelihoods for the delayed-acceptance IRM sampler.

A surrogate approximates the per-cluster collapsed log marginal likelihood
contribution used to build the proposal distribution over candidate clusters.
Because the delayed-acceptance kernel corrects every proposal against the exact
marginal (see :mod:`isbm.models.marginal`), the surrogate only needs to be a
cheap *guide*, not exact -- any bias is removed by the second-stage acceptance
so the target posterior stays invariant.

All methods operate over the *active clusters of the opposite domain* (the last
array axis). ``nk`` / ``Nmk`` may be 1D (a single candidate cluster) or 2D
(``n_candidates x n_active_other``); the return has the leading shape.
"""

import numpy as np

from isbm.models.marginal import existing_log_marginal


class Surrogate:
    """Base surrogate interface."""

    def __init__(self, a, b, rng=None):
        self.a = float(a)
        self.b = float(b)
        self.rng = rng if rng is not None else np.random.default_rng()

    def existing_loglik(self, nk, Nmk, n_plus, N_plus):
        """Surrogate log marginal for joining existing cluster(s)."""
        raise NotImplementedError

    def new_loglik(self, n_plus, N_plus):
        """Surrogate log marginal for forming a brand-new (empty) cluster."""
        raise NotImplementedError


class PlugInSurrogate(Surrogate):
    """Plug-in posterior-mean Bernoulli likelihood.

    Uses ``theta_hat = (a + nk) / (a + b + nk + Nmk)`` and scores the joining
    object with ``sum(n_plus * log theta_hat + N_plus * log(1 - theta_hat))``.
    This avoids every ``gammaln`` call in the exact marginal while staying close
    to it, making it the default cheap screen.
    """

    _EPS = 1e-12

    def existing_loglik(self, nk, Nmk, n_plus, N_plus):
        nk = np.asarray(nk, dtype=np.float64)
        Nmk = np.asarray(Nmk, dtype=np.float64)
        theta = (self.a + nk) / (self.a + self.b + nk + Nmk)
        theta = np.clip(theta, self._EPS, 1.0 - self._EPS)
        return np.sum(
            n_plus * np.log(theta) + N_plus * np.log1p(-theta), axis=-1
        )

    def new_loglik(self, n_plus, N_plus):
        theta = self.a / (self.a + self.b)
        theta = min(max(theta, self._EPS), 1.0 - self._EPS)
        return float(
            np.sum(n_plus) * np.log(theta) + np.sum(N_plus) * np.log1p(-theta)
        )


class SubsampledMarginalSurrogate(Surrogate):
    """Exact ``gammaln`` marginal evaluated on a random subset of opposite-domain
    clusters, rescaled to estimate the full sum.

    This trades bias for speed via ``frac``: with ``frac`` of the active
    opposite-domain clusters sampled, the subset log-marginal is scaled by
    ``n_total / n_subset`` to form an (approximately) unbiased estimate of the
    full-sum log marginal.
    """

    def __init__(self, a, b, frac=0.5, rng=None):
        super().__init__(a, b, rng=rng)
        self.frac = float(frac)

    def _subset_idx(self, n_active):
        if n_active == 0:
            return np.empty(0, dtype=int), 1.0
        n_sub = max(1, int(round(self.frac * n_active)))
        if n_sub >= n_active:
            return np.arange(n_active), 1.0
        idx = self.rng.choice(n_active, size=n_sub, replace=False)
        scale = n_active / n_sub
        return idx, scale

    def existing_loglik(self, nk, Nmk, n_plus, N_plus):
        nk = np.atleast_2d(np.asarray(nk, dtype=np.float64))
        Nmk = np.atleast_2d(np.asarray(Nmk, dtype=np.float64))
        n_plus = np.asarray(n_plus, dtype=np.float64)
        N_plus = np.asarray(N_plus, dtype=np.float64)
        n_active = n_plus.shape[-1]
        idx, scale = self._subset_idx(n_active)
        out = np.array(
            [
                existing_log_marginal(
                    nk[r, idx], Nmk[r, idx], n_plus[idx], N_plus[idx], self.a, self.b
                )
                for r in range(nk.shape[0])
            ]
        )
        out *= scale
        return out

    def new_loglik(self, n_plus, N_plus):
        from isbm.models.marginal import new_log_marginal

        n_plus = np.asarray(n_plus, dtype=np.float64)
        N_plus = np.asarray(N_plus, dtype=np.float64)
        n_active = n_plus.shape[-1]
        idx, scale = self._subset_idx(n_active)
        return scale * new_log_marginal(n_plus[idx], N_plus[idx], self.a, self.b)


_SURROGATES = {
    "plugin": PlugInSurrogate,
    "subsample": SubsampledMarginalSurrogate,
}


def build_surrogate(name, a, b, rng=None, subsample_frac=0.5):
    """Factory used by the sampler / Hydra config.

    ``name`` is one of ``{"plugin", "subsample"}``.
    """
    name = (name or "plugin").lower()
    if name not in _SURROGATES:
        raise ValueError(
            f"Unknown surrogate {name!r}; choose from {sorted(_SURROGATES)}"
        )
    if name == "subsample":
        return SubsampledMarginalSurrogate(a, b, frac=subsample_frac, rng=rng)
    return PlugInSurrogate(a, b, rng=rng)
