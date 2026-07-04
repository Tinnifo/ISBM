"""Sequential Monte Carlo over data for the collapsed IRM.

This is the "SMC-over-data" blueprint (Fearnhead-style particle filter for the
CRP / SBM, with the surrogate-acceleration idea of DAphyloSMC / Bon et al.).
Nodes are revealed one at a time; each particle extends its partition by
assigning the new node, weighted by the collapsed Beta-Bernoulli predictive of
that node's edges to the already-placed nodes. Particles are resampled when the
effective sample size drops and (optionally) rejuvenated with collapsed Gibbs
sweeps.

Model: a single symmetric partition ``z`` over ``N`` nodes (CRP prior with
concentration ``alpha``), directed Beta-Bernoulli blocks (self-loops excluded).
This matches the data-generating process in
:func:`isbm.data.synthetic.generate_irm_graph` (one true ``Z``), so the recovered
``z1_`` is directly comparable via ARI.

Two proposals are provided:

* ``proposal="exact"`` -- the locally optimal proposal, evaluating the exact
  predictive for every candidate cluster (``O(K)`` exact predictives per node).
* ``proposal="surrogate"`` -- a plug-in predictive builds the proposal and only
  the *sampled* candidate is scored with the exact predictive (``O(1)`` exact
  predictives per node). The incremental importance weight is corrected so the
  SMC estimator stays valid, i.e. this is a delayed-acceptance-style acceleration
  of the SMC sampler rather than an approximation of a different target.
"""

import numpy as np
from scipy.special import gammaln

from isbm.models.cgs import _compact


def _logsumexp(a):
    a = np.asarray(a, dtype=np.float64)
    m = a.max()
    if not np.isfinite(m):
        return m
    return m + np.log(np.sum(np.exp(a - m)))


def _block_marg(nkl, Nmkl, no, No, a, b, mode):
    """Log predictive of adding ``(no, No)`` edge/non-edge counts to block(s)
    with current counts ``(nkl, Nmkl)``. Arrays are broadcast."""
    nkl = np.asarray(nkl, dtype=np.float64)
    Nmkl = np.asarray(Nmkl, dtype=np.float64)
    if mode == "exact":
        return (
            gammaln(a + nkl + no)
            + gammaln(b + Nmkl + No)
            - gammaln(a + b + nkl + Nmkl + no + No)
            - gammaln(a + nkl)
            - gammaln(b + Nmkl)
            + gammaln(a + b + nkl + Nmkl)
        )
    theta = (a + nkl) / (a + b + nkl + Nmkl)
    theta = np.clip(theta, 1e-12, 1.0 - 1e-12)
    return no * np.log(theta) + No * np.log1p(-theta)


def _block_marg_new(no, No, a, b, mode):
    no = np.asarray(no, dtype=np.float64)
    No = np.asarray(No, dtype=np.float64)
    if mode == "exact":
        return (
            gammaln(a + no)
            + gammaln(b + No)
            - gammaln(a + b + no + No)
            - gammaln(a)
            - gammaln(b)
            + gammaln(a + b)
        )
    theta = a / (a + b)
    theta = min(max(theta, 1e-12), 1.0 - 1e-12)
    return no * np.log(theta) + No * np.log1p(-theta)


class _Particle:
    __slots__ = ("z", "n", "Nm", "m")

    def __init__(self, N):
        self.z = np.full(N, -1, dtype=int)
        self.n = np.zeros((N, N), dtype=float)
        self.Nm = np.zeros((N, N), dtype=float)
        self.m = np.zeros(N, dtype=float)

    def clone(self):
        p = _Particle.__new__(_Particle)
        p.z = self.z.copy()
        p.n = self.n.copy()
        p.Nm = self.Nm.copy()
        p.m = self.m.copy()
        return p


def _node_counts(i, X, z, K, placed_idx):
    """Per-cluster edge/non-edge counts between node ``i`` and placed nodes."""
    lab = z[placed_idx]
    x_out = X[i, placed_idx]
    x_in = X[placed_idx, i]
    n_out = np.zeros(K)
    n_in = np.zeros(K)
    cnt = np.zeros(K)
    np.add.at(n_out, lab, x_out)
    np.add.at(n_in, lab, x_in)
    np.add.at(cnt, lab, 1.0)
    return n_out, cnt - n_out, n_in, cnt - n_in


def _log_pred(p, n_out, N_out, n_in, N_in, active, a, b, mode):
    """Log predictive (excluding CRP prior) for each existing candidate cluster
    and the new-cluster option."""
    no_l = n_out[active]
    No_l = N_out[active]
    ni_l = n_in[active]
    Ni_l = N_in[active]
    lp = np.empty(active.size + 1, dtype=float)
    for pos, k in enumerate(active):
        out_terms = _block_marg(p.n[k, active], p.Nm[k, active], no_l, No_l, a, b, mode)
        in_terms = _block_marg(p.n[active, k], p.Nm[active, k], ni_l, Ni_l, a, b, mode)
        total = out_terms.sum() + in_terms.sum()
        # Fix the k == k block: outgoing and incoming edges share block (k, k)
        # and must be scored as a single update, not two independent ones.
        total -= out_terms[pos] + in_terms[pos]
        total += _block_marg(
            p.n[k, k], p.Nm[k, k], no_l[pos] + ni_l[pos], No_l[pos] + Ni_l[pos], a, b, mode
        )
        lp[pos] = total
    lp[-1] = (
        _block_marg_new(no_l, No_l, a, b, mode).sum()
        + _block_marg_new(ni_l, Ni_l, a, b, mode).sum()
    )
    return lp


def _log_pred_single(p, k, n_out, N_out, n_in, N_in, active, a, b, mode):
    """Exact/plug-in log predictive for one existing candidate cluster ``k``."""
    no_l = n_out[active]
    No_l = N_out[active]
    ni_l = n_in[active]
    Ni_l = N_in[active]
    out_terms = _block_marg(p.n[k, active], p.Nm[k, active], no_l, No_l, a, b, mode)
    in_terms = _block_marg(p.n[active, k], p.Nm[active, k], ni_l, Ni_l, a, b, mode)
    total = out_terms.sum() + in_terms.sum()
    pos = int(np.where(active == k)[0][0])
    total -= out_terms[pos] + in_terms[pos]
    total += _block_marg(
        p.n[k, k], p.Nm[k, k], no_l[pos] + ni_l[pos], No_l[pos] + Ni_l[pos], a, b, mode
    )
    return float(total)


def _apply(p, i, k, n_out, N_out, n_in, N_in, active):
    p.z[i] = k
    p.m[k] += 1
    if active.size:
        p.n[k, active] += n_out[active]
        p.Nm[k, active] += N_out[active]
        p.n[active, k] += n_in[active]
        p.Nm[active, k] += N_in[active]


def _remove(p, i, X, K):
    k = p.z[i]
    placed_idx = np.where(p.z >= 0)[0]
    placed_idx = placed_idx[placed_idx != i]
    n_out, N_out, n_in, N_in = _node_counts(i, X, p.z, K, placed_idx)
    active = np.where(p.m > 0)[0]
    # remove k's own membership contribution before subtracting blocks
    p.m[k] -= 1
    p.z[i] = -1
    if active.size:
        p.n[k, active] -= n_out[active]
        p.Nm[k, active] -= N_out[active]
        p.n[active, k] -= n_in[active]
        p.Nm[active, k] -= N_in[active]
    return k


def _systematic_resample(w, rng):
    P = w.size
    positions = (rng.random() + np.arange(P)) / P
    idx = np.searchsorted(np.cumsum(w), positions)
    return np.clip(idx, 0, P - 1)


class SMC_IRM:
    def __init__(
        self,
        n_particles=200,
        alpha=1.0,
        a=1.0,
        b=1.0,
        proposal="exact",
        resample_threshold=0.5,
        n_rejuv=1,
        seed=None,
    ):
        self.n_particles = n_particles
        self.alpha = alpha
        self.a = a
        self.b = b
        self.proposal = proposal
        self.resample_threshold = resample_threshold
        self.n_rejuv = n_rejuv
        self.seed = seed
        self.burnin = 0

    def fit(self, X):
        rng = np.random.default_rng(self.seed)
        X = np.asarray(X, dtype=np.float64)
        N = X.shape[0]
        K = N
        P = self.n_particles
        a, b, alpha = self.a, self.b, self.alpha
        mode = "plugin" if self.proposal == "surrogate" else "exact"

        particles = [_Particle(N) for _ in range(P)]
        log_wnorm = np.full(P, -np.log(P))
        order = rng.permutation(N)

        self.ll_trace_ = []
        self.n_exact_evals_ = 0
        n_resamples = 0
        logZ = 0.0

        for step, i in enumerate(order):
            s = step  # number of nodes already placed
            log_denom = np.log(s + alpha) if s > 0 else np.log(alpha)
            log_inc = np.empty(P, dtype=float)

            for pi, p in enumerate(particles):
                placed_idx = np.where(p.z >= 0)[0]
                n_out, N_out, n_in, N_in = _node_counts(i, X, p.z, K, placed_idx)
                active = np.where(p.m > 0)[0]

                log_prior = np.empty(active.size + 1)
                log_prior[:-1] = np.log(p.m[active]) - log_denom
                log_prior[-1] = np.log(alpha) - log_denom

                lp_prop = _log_pred(p, n_out, N_out, n_in, N_in, active, a, b, mode)
                log_w = log_prior + lp_prop
                log_inc_surr = _logsumexp(log_w)
                q = np.exp(log_w - log_inc_surr)
                q /= q.sum()
                choice = rng.choice(active.size + 1, p=q)
                k = int(active[choice]) if choice < active.size else int(np.where(p.m == 0)[0][0])

                if mode == "exact":
                    log_inc[pi] = log_inc_surr
                    self.n_exact_evals_ += active.size + 1
                else:
                    if choice < active.size:
                        exact_k = _log_pred_single(
                            p, k, n_out, N_out, n_in, N_in, active, a, b, "exact"
                        )
                    else:
                        exact_k = (
                            _block_marg_new(n_out[active], N_out[active], a, b, "exact").sum()
                            + _block_marg_new(n_in[active], N_in[active], a, b, "exact").sum()
                        )
                    log_inc[pi] = log_inc_surr + exact_k - lp_prop[choice]
                    self.n_exact_evals_ += 1

                _apply(p, i, k, n_out, N_out, n_in, N_in, active)

            logw = log_wnorm + log_inc
            ls = _logsumexp(logw)
            logZ += ls
            log_wnorm = logw - ls
            w = np.exp(log_wnorm)
            ess = 1.0 / np.sum(w**2)

            if ess < self.resample_threshold * P:
                idx = _systematic_resample(w, rng)
                particles = [particles[j].clone() for j in idx]
                log_wnorm = np.full(P, -np.log(P))
                n_resamples += 1
                if self.n_rejuv > 0:
                    for p in particles:
                        self._rejuvenate(p, X, K, rng)

            self.ll_trace_.append(logZ)

        w = np.exp(log_wnorm - _logsumexp(log_wnorm))
        best = int(np.argmax(w))
        bp = particles[best]

        self.logZ_ = logZ
        self.n_resamples_ = n_resamples
        self.z1_ = _compact(bp.z)
        self.z2_ = self.z1_
        self.n_ = bp.n
        self.Nm_ = bp.Nm
        self.m1_ = bp.m
        self.m2_ = bp.m
        self.z1_samples_ = [_compact(p.z) for p in particles]
        self.z2_samples_ = self.z1_samples_
        return self

    def _rejuvenate(self, p, X, K, rng):
        a, b = self.a, self.b
        alpha = self.alpha
        for _ in range(self.n_rejuv):
            placed = np.where(p.z >= 0)[0]
            s = placed.size
            if s <= 1:
                return
            for i in rng.permutation(placed):
                _remove(p, i, X, K)
                placed_idx = np.where(p.z >= 0)[0]
                n_out, N_out, n_in, N_in = _node_counts(i, X, p.z, K, placed_idx)
                active = np.where(p.m > 0)[0]
                log_denom = np.log((s - 1) + alpha)
                log_prior = np.empty(active.size + 1)
                log_prior[:-1] = np.log(p.m[active]) - log_denom
                log_prior[-1] = np.log(alpha) - log_denom
                lp = _log_pred(p, n_out, N_out, n_in, N_in, active, a, b, "exact")
                self.n_exact_evals_ += active.size + 1
                log_w = log_prior + lp
                log_w -= log_w.max()
                q = np.exp(log_w)
                q /= q.sum()
                choice = rng.choice(active.size + 1, p=q)
                k = int(active[choice]) if choice < active.size else int(np.where(p.m == 0)[0][0])
                _apply(p, i, k, n_out, N_out, n_in, N_in, active)

    def predict(self, X, i_test, j_test):
        z1, z2 = self.z1_, self.z2_
        log_ll = 0.0
        for i, j in zip(i_test, j_test):
            k, l_cluster = z1[i], z2[j]
            a_post = self.a + self.n_[k, l_cluster]
            b_post = self.b + self.Nm_[k, l_cluster]
            theta = a_post / (a_post + b_post)
            xij = X[i, j]
            log_ll += xij * np.log(theta + 1e-300) + (1 - xij) * np.log(1 - theta + 1e-300)
        return log_ll / max(len(i_test), 1)
