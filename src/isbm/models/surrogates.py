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

from isbm.models.marginal import existing_log_marginal_rows, new_log_marginal


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

    # --- optional online-training hooks (no-ops for analytic surrogates) ---
    def observe_existing(self, nk_row, Nmk_row, n_plus, N_plus, exact_value):
        """Record an exact evaluation for an existing-cluster move (for learning)."""

    def observe_new(self, n_plus, N_plus, exact_value):
        """Record an exact evaluation for a new-cluster move (for learning)."""

    def maybe_refit(self):
        """Refit the surrogate from collected data if it is a learning surrogate."""


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
        out = existing_log_marginal_rows(
            nk[:, idx], Nmk[:, idx], n_plus[idx], N_plus[idx], self.a, self.b
        )
        return out * scale

    def new_loglik(self, n_plus, N_plus):
        n_plus = np.asarray(n_plus, dtype=np.float64)
        N_plus = np.asarray(N_plus, dtype=np.float64)
        n_active = n_plus.shape[-1]
        idx, scale = self._subset_idx(n_active)
        return scale * new_log_marginal(n_plus[idx], N_plus[idx], self.a, self.b)


class _PolyRidge:
    """Tiny numpy ridge regressor on standardized degree-2 polynomial features.

    Kept dependency-free and vectorized so that ``predict`` (a single matmul) is
    orders of magnitude cheaper per call than a scikit-learn estimator -- which
    matters because the surrogate is queried once per node update.
    """

    def __init__(self, degree=2, alpha=1.0):
        self.degree = degree
        self.alpha = alpha
        self.w = None
        self.mu = None
        self.sd = None

    def _expand(self, X):
        Xs = (X - self.mu) / self.sd
        parts = [np.ones((Xs.shape[0], 1)), Xs]
        if self.degree >= 2:
            n = Xs.shape[1]
            parts.append(np.hstack([Xs[:, i : i + 1] * Xs[:, i:] for i in range(n)]))
        return np.hstack(parts)

    def fit(self, X, y):
        self.mu = X.mean(axis=0)
        self.sd = X.std(axis=0) + 1e-8
        P = self._expand(X)
        A = P.T @ P + self.alpha * np.eye(P.shape[1])
        self.w = np.linalg.solve(A, P.T @ y)

    def predict(self, X):
        return self._expand(X) @ self.w


class LearnedSurrogate(Surrogate):
    """Machine-learning surrogate that learns the exact-vs-plug-in *residual*.

    Following the DAphyloSMC (random forest) / adamcmcpaper (Gaussian process)
    template, a regressor is trained online to predict the gap between the exact
    collapsed marginal and a cheap plug-in baseline, from cheap summary features
    of the move. The surrogate returns ``plugin + predicted_residual``; because
    the plug-in already captures most of the signal, the regressor only has to
    model the (smooth) correction, and the surrogate sharpens as the sampler
    collects more exact evaluations -- exactly the "approximation improves as we
    go" idea. As with every surrogate, the delayed-acceptance stage-2 step still
    corrects against the exact marginal, so the posterior stays invariant.

    Training targets are the exact evaluations the delayed-acceptance kernel
    already computes (two per node), fed back through :meth:`observe_existing` /
    :meth:`observe_new`.
    """

    def __init__(
        self,
        a,
        b,
        regressor="linear",
        warmup=800,
        refit_every=800,
        max_train=12000,
        gp_max_train=1500,
        rng=None,
    ):
        super().__init__(a, b, rng=rng)
        self.base = PlugInSurrogate(a, b, rng=rng)
        self.regressor = regressor.lower()
        self.warmup = int(warmup)
        self.refit_every = int(refit_every)
        self.max_train = int(max_train)
        self.gp_max_train = int(gp_max_train)
        self._X = []
        self._y = []
        self._model = None
        self._n_seen = 0
        self._since_refit = 0

    def _features(self, nk, Nmk, n_plus, N_plus):
        """Return ``(features (E, F), plugin (E,))`` for one or more clusters."""
        nk = np.atleast_2d(np.asarray(nk, dtype=np.float64))
        Nmk = np.atleast_2d(np.asarray(Nmk, dtype=np.float64))
        n_plus = np.asarray(n_plus, dtype=np.float64)
        N_plus = np.asarray(N_plus, dtype=np.float64)
        plugin = self.base.existing_loglik(nk, Nmk, n_plus, N_plus)
        E = nk.shape[0]
        A = float(nk.shape[-1])
        sum_nk = nk.sum(axis=-1)
        sum_Nmk = Nmk.sum(axis=-1)
        snp = float(n_plus.sum())
        sNp = float(N_plus.sum())
        feats = np.column_stack(
            [
                plugin,
                sum_nk,
                sum_Nmk,
                sum_nk + sum_Nmk,
                np.full(E, snp),
                np.full(E, sNp),
                np.full(E, A),
            ]
        )
        return feats, np.atleast_1d(plugin)

    def _features_new(self, n_plus, N_plus):
        n_plus = np.asarray(n_plus, dtype=np.float64)
        N_plus = np.asarray(N_plus, dtype=np.float64)
        plugin = float(self.base.new_loglik(n_plus, N_plus))
        snp = float(n_plus.sum())
        sNp = float(N_plus.sum())
        A = float(len(n_plus))
        feats = np.array([[plugin, 0.0, 0.0, 0.0, snp, sNp, A]])
        return feats, plugin

    def existing_loglik(self, nk, Nmk, n_plus, N_plus):
        feats, plugin = self._features(nk, Nmk, n_plus, N_plus)
        if self._model is None:
            return plugin
        return plugin + self._model.predict(feats)

    def new_loglik(self, n_plus, N_plus):
        feats, plugin = self._features_new(n_plus, N_plus)
        if self._model is None:
            return plugin
        return plugin + float(self._model.predict(feats)[0])

    def observe_existing(self, nk_row, Nmk_row, n_plus, N_plus, exact_value):
        feats, plugin = self._features(nk_row, Nmk_row, n_plus, N_plus)
        self._add(feats[0], float(exact_value) - float(plugin[0]))

    def observe_new(self, n_plus, N_plus, exact_value):
        feats, plugin = self._features_new(n_plus, N_plus)
        self._add(feats[0], float(exact_value) - plugin)

    def _add(self, x, y):
        self._X.append(x)
        self._y.append(y)
        self._n_seen += 1
        self._since_refit += 1
        if len(self._X) > self.max_train:
            self._X.pop(0)
            self._y.pop(0)

    def maybe_refit(self):
        if self._n_seen < self.warmup:
            return
        if self._model is not None and self._since_refit < self.refit_every:
            return
        self._fit()

    def _fit(self):
        X = np.asarray(self._X)
        y = np.asarray(self._y)
        model = self._make_model(X, y)
        model.fit(X, y)
        self._model = model
        self._since_refit = 0

    def _make_model(self, X, y):
        if self.regressor in ("linear", "poly", "ridge"):
            return _PolyRidge(degree=2, alpha=1.0)
        if self.regressor == "gp":
            from sklearn.gaussian_process import GaussianProcessRegressor
            from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel

            if X.shape[0] > self.gp_max_train:
                sel = self.rng.choice(X.shape[0], size=self.gp_max_train, replace=False)
                self._X = [self._X[i] for i in sel]
                self._y = [self._y[i] for i in sel]
            kernel = ConstantKernel(1.0) * RBF(length_scale=np.ones(X.shape[1])) + WhiteKernel()
            return GaussianProcessRegressor(kernel=kernel, normalize_y=True, alpha=1e-6)

        from sklearn.ensemble import RandomForestRegressor

        seed = int(self.rng.integers(0, 2**31 - 1))
        return RandomForestRegressor(
            n_estimators=60, max_depth=12, min_samples_leaf=3, n_jobs=1, random_state=seed
        )


_SURROGATES = {
    "plugin": PlugInSurrogate,
    "subsample": SubsampledMarginalSurrogate,
    "learned": LearnedSurrogate,
}


def build_surrogate(
    name,
    a,
    b,
    rng=None,
    subsample_frac=0.5,
    learned_regressor="linear",
    warmup=800,
    refit_every=800,
):
    """Factory used by the sampler / Hydra config.

    ``name`` is one of ``{"plugin", "subsample", "learned"}``. For ``learned``,
    ``learned_regressor`` selects ``{"linear", "rf", "gp"}``.
    """
    name = (name or "plugin").lower()
    if name not in _SURROGATES:
        raise ValueError(
            f"Unknown surrogate {name!r}; choose from {sorted(_SURROGATES)}"
        )
    if name == "subsample":
        return SubsampledMarginalSurrogate(a, b, frac=subsample_frac, rng=rng)
    if name == "learned":
        return LearnedSurrogate(
            a, b, regressor=learned_regressor, warmup=warmup, refit_every=refit_every, rng=rng
        )
    return PlugInSurrogate(a, b, rng=rng)
