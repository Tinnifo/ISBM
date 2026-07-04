"""Delayed-acceptance IRM sampler.

A Metropolis-within-Gibbs variant of :class:`isbm.models.cgs.CGS_IRM`. Instead
of evaluating the exact Beta-Bernoulli collapsed marginal for *every* candidate
cluster of a node (the ``O(N * K * K_other)`` ``gammaln`` scan of the baseline),
each node reassignment is a two-stage delayed-acceptance move:

1. Build a cheap surrogate conditional ``q(k) ~ prior(k) * surrogate_lik(k)`` over
   all candidate clusters (no ``gammaln``) and propose ``k*`` from it. Because we
   propose directly from the surrogate, the delayed-acceptance stage-1 screen
   (``alpha_1``) is identically 1.
2. Correct with the stage-2 ratio using the *exact* marginal at only the current
   cluster ``k_old`` and the proposal ``k*``::

       alpha_2 = min(1, [L(k*)/L~(k*)] * [L~(k_old)/L(k_old)])

   where ``L`` is the exact collapsed marginal and ``L~`` the surrogate.

The move is an independence Metropolis-Hastings step targeting the exact
collapsed full conditional (the proposal ``q`` does not depend on the current
state once the node is removed), so the sampler leaves the *same* posterior as
the baseline invariant -- only the exact-evaluation cost per node drops from
``K + 1`` to at most 2.
"""

import numpy as np

from isbm.models.cgs import _compact
from isbm.models.marginal import existing_log_marginal, new_log_marginal
from isbm.models.surrogates import build_surrogate


class DA_IRM:
    def __init__(
        self,
        alpha1=1.0,
        alpha2=1.0,
        a=1.0,
        b=1.0,
        n_iter=500,
        burnin=250,
        surrogate="plugin",
        subsample_frac=0.5,
        seed=None,
    ):
        self.alpha1 = alpha1
        self.alpha2 = alpha2
        self.a = a
        self.b = b
        self.n_iter = n_iter
        self.burnin = burnin
        self.surrogate = surrogate
        self.subsample_frac = subsample_frac
        self.seed = seed

    def fit(self, X):
        rng = np.random.default_rng(self.seed)
        surrogate = build_surrogate(
            self.surrogate, self.a, self.b, rng=rng, subsample_frac=self.subsample_frac
        )

        X = np.asarray(X, dtype=np.float64)
        N1, N2 = X.shape

        z1 = np.arange(N1, dtype=int)
        z2 = np.arange(N2, dtype=int)
        K = max(N1, N2)

        m1 = np.zeros(K, dtype=float)
        m2 = np.zeros(K, dtype=float)
        n = np.zeros((K, K), dtype=float)
        Nm = np.zeros((K, K), dtype=float)

        for i in range(N1):
            m1[z1[i]] += 1
        for j in range(N2):
            m2[j] += 1
        for i in range(N1):
            for j in range(N2):
                n[z1[i], z2[j]] += X[i, j]
                Nm[z1[i], z2[j]] += 1.0 - X[i, j]

        self.z1_samples_ = []
        self.z2_samples_ = []
        self.ll_trace_ = []
        self.accept_trace_ = []
        self.exact_evals_trace_ = []
        self.n_exact_evals_ = 0

        for it in range(self.n_iter):
            n_proposed = 0
            n_accept = 0
            n_exact = 0

            for i in rng.permutation(N1):
                k_old = z1[i]

                m1[k_old] -= 1
                np.add.at(n[k_old], z2, -X[i, :])
                np.add.at(Nm[k_old], z2, -(1.0 - X[i, :]))

                n_plus = np.zeros(K, dtype=float)
                N_plus = np.zeros(K, dtype=float)
                np.add.at(n_plus, z2, X[i, :])
                np.add.at(N_plus, z2, 1.0 - X[i, :])

                active2 = np.where(m2 > 0)[0]
                existing1 = np.where(m1 > 0)[0]
                np_ = n_plus[active2]
                Np_ = N_plus[active2]
                n_rows = n[np.ix_(existing1, active2)]
                Nm_rows = Nm[np.ix_(existing1, active2)]

                chosen, accepted, ne, proposed = _da_decide(
                    existing1,
                    m1,
                    self.alpha1,
                    n_rows,
                    Nm_rows,
                    np_,
                    Np_,
                    k_old,
                    surrogate,
                    self.a,
                    self.b,
                    rng,
                )
                if chosen == -1:
                    chosen = int(np.where(m1 == 0)[0][0])

                z1[i] = chosen
                m1[chosen] += 1
                np.add.at(n[chosen], z2, X[i, :])
                np.add.at(Nm[chosen], z2, 1.0 - X[i, :])

                n_proposed += int(proposed)
                n_accept += int(accepted)
                n_exact += ne

            for j in rng.permutation(N2):
                l_old = z2[j]

                m2[l_old] -= 1
                np.add.at(n[:, l_old], z1, -X[:, j])
                np.add.at(Nm[:, l_old], z1, -(1.0 - X[:, j]))

                n_plus = np.zeros(K, dtype=float)
                N_plus = np.zeros(K, dtype=float)
                np.add.at(n_plus, z1, X[:, j])
                np.add.at(N_plus, z1, 1.0 - X[:, j])

                active1 = np.where(m1 > 0)[0]
                existing2 = np.where(m2 > 0)[0]
                np_ = n_plus[active1]
                Np_ = N_plus[active1]
                n_rows = n[np.ix_(active1, existing2)].T
                Nm_rows = Nm[np.ix_(active1, existing2)].T

                chosen, accepted, ne, proposed = _da_decide(
                    existing2,
                    m2,
                    self.alpha2,
                    n_rows,
                    Nm_rows,
                    np_,
                    Np_,
                    l_old,
                    surrogate,
                    self.a,
                    self.b,
                    rng,
                )
                if chosen == -1:
                    chosen = int(np.where(m2 == 0)[0][0])

                z2[j] = chosen
                m2[chosen] += 1
                np.add.at(n[:, chosen], z1, X[:, j])
                np.add.at(Nm[:, chosen], z1, 1.0 - X[:, j])

                n_proposed += int(proposed)
                n_accept += int(accepted)
                n_exact += ne

            ll = self._pseudo_ll(n, Nm, m1, m2)
            self.ll_trace_.append(ll)
            self.accept_trace_.append(n_accept / max(n_proposed, 1))
            self.exact_evals_trace_.append(n_exact)
            self.n_exact_evals_ += n_exact

            if it >= self.burnin:
                self.z1_samples_.append(_compact(z1))
                self.z2_samples_.append(_compact(z2))

        self.z1_ = _compact(z1)
        self.z2_ = _compact(z2)
        self.n_ = n
        self.Nm_ = Nm
        self.m1_ = m1
        self.m2_ = m2
        self.acceptance_rate_ = float(np.mean(self.accept_trace_))
        return self

    def predict(self, X, i_test, j_test):
        """Marginal test log-likelihood using final sample."""
        z1, z2 = self.z1_, self.z2_
        log_ll = 0.0
        for i, j in zip(i_test, j_test):
            k, l_cluster = z1[i], z2[j]
            a_post = self.a + self.n_[k, l_cluster]
            b_post = self.b + self.Nm_[k, l_cluster]
            theta = a_post / (a_post + b_post)
            xij = X[i, j]
            log_ll += xij * np.log(theta + 1e-300) + (1 - xij) * np.log(
                1 - theta + 1e-300
            )
        return log_ll / max(len(i_test), 1)

    def _pseudo_ll(self, n, Nm, m1, m2):
        from scipy.special import gammaln

        a, b = self.a, self.b
        act1 = np.where(m1 > 0)[0]
        act2 = np.where(m2 > 0)[0]
        nk = n[np.ix_(act1, act2)]
        Nmk = Nm[np.ix_(act1, act2)]
        return float(
            np.sum(
                gammaln(a + nk)
                + gammaln(b + Nmk)
                - gammaln(a + b + nk + Nmk)
                - gammaln(a)
                - gammaln(b)
                + gammaln(a + b)
            )
        )


def _da_decide(
    existing_ids,
    m_self,
    alpha_self,
    n_rows,
    Nm_rows,
    n_plus,
    N_plus,
    k_old,
    surrogate,
    a,
    b,
    rng,
):
    """One delayed-acceptance node move.

    ``existing_ids`` are the occupied self-domain clusters (``m > 0`` after the
    node was removed); ``n_rows`` / ``Nm_rows`` are their cross-counts over the
    active opposite-domain clusters (shape ``(E, A)``). ``n_plus`` / ``N_plus``
    are the removed node's counts over those same opposite-domain clusters.

    Returns ``(chosen_id, accepted, n_exact_evals, proposed_change)`` where
    ``chosen_id == -1`` means a brand-new cluster and ``proposed_change`` is
    ``False`` when the surrogate proposed the current state (a null move needing
    no exact correction).
    """
    E = existing_ids.shape[0]

    surr_existing = np.atleast_1d(
        surrogate.existing_loglik(n_rows, Nm_rows, n_plus, N_plus)
    )
    surr_new = float(surrogate.new_loglik(n_plus, N_plus))

    log_prior = np.empty(E + 1, dtype=float)
    log_prior[:E] = np.log(m_self[existing_ids])
    log_prior[E] = np.log(alpha_self)

    surr = np.empty(E + 1, dtype=float)
    surr[:E] = surr_existing
    surr[E] = surr_new

    # Candidate ids: existing clusters then a "new cluster" sentinel (-1).
    cand_ids = np.empty(E + 1, dtype=int)
    cand_ids[:E] = existing_ids
    cand_ids[E] = -1

    # Surrogate proposal q ~ prior * exp(surrogate loglik).
    log_q = log_prior + surr
    log_q -= log_q.max()
    q = np.exp(log_q)
    q /= q.sum()
    j_star = rng.choice(E + 1, p=q)

    # Current-state candidate index: an occupied k_old maps to its existing
    # slot, otherwise (k_old emptied by the removal) it is the "new" candidate.
    old_pos = np.where(existing_ids == k_old)[0]
    j_old = int(old_pos[0]) if old_pos.size else E

    def exact_loglik(idx):
        if cand_ids[idx] == -1:
            return new_log_marginal(n_plus, N_plus, a, b)
        return existing_log_marginal(
            n_rows[idx], Nm_rows[idx], n_plus, N_plus, a, b
        )

    if j_star == j_old:
        # Proposing the current state: not a genuine move, no correction needed.
        return int(cand_ids[j_old]), False, 0, False

    exact_star = exact_loglik(j_star)
    exact_old = exact_loglik(j_old)
    n_exact = 2

    # log alpha_2 = [L(k*) - L~(k*)] - [L(k_old) - L~(k_old)].
    log_alpha = (exact_star - surr[j_star]) - (exact_old - surr[j_old])

    if np.log(rng.random()) < log_alpha:
        return int(cand_ids[j_star]), True, n_exact, True
    return int(cand_ids[j_old]), False, n_exact, True
