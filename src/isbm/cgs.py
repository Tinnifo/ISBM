r"""Collapsed Gibbs sampling for the two-domain IRM.

Based on Ishiguro, Sato & Ueda, "Collapsed Variational Bayes Inference of
Infinite Relational Model", https://arxiv.org/abs/1409.4757 (Section 3).
Notes carried over verbatim from the former notebook Notebooks/CGS.py, whose
CGS_IRM class gave identical samples to this one (checked 2026-11-05).

1. Environment and Log-Gamma Setup
    In a collapsed sampler, we analytically integrate out the probability
    parameters ($\theta$). This integration results in ratios of Beta functions,
    denoted as $B(x, y)$ in the paper.

    Because multiplying many probabilities together leads to numerical underflow,
    we perform our math in log-space. The log of a Beta function can be calculated
    using the log-Gamma function (`gammaln`):

    $$\log B(x,y) = \log\Gamma(x) + \log\Gamma(y) - \log\Gamma(x+y)$$

2. Counting Statistics & Initialization
    Equation (13) - Link Counts:
    $$n_{k,l} = \sum_i \sum_j z_{1,i,k} z_{2,j,l} x_{i,j} \ , \quad
      N_{k,l} = \sum_i \sum_j z_{1,i,k} z_{2,j,l} (1 - x_{i,j}) \tag{13}$$
    where $n_{k,l}$ and $N_{k,l}$ denote the number of positive and negative
    relation observations in the $(k,l)$-cluster pairs.

    Equation (14) - Cluster Memberships:
    $$m_{1,k} = \sum_i z_{1,i,k} \ , \quad m_{2,l} = \sum_j z_{2,j,l} \tag{14}$$
    which are the number of objects assigned to each cluster in Domain 1 ($m_1$)
    and Domain 2 ($m_2$).

    The model is initialized with every object in its own singleton cluster, and
    then the initial state of these 4 tracking matrices is built.

3. The Gibbs Sweep: Domain 1 ($Z_1$)
    In a single Gibbs step, we remove object $i$ from the network, recalculate the
    statistics (denoted as $-(1,i)$ in the paper), and assess the probability of
    placing it in cluster $k$. The probability equation is derived in Equation (20).
    Because we are taking the $\log()$ of Equation (20) to prevent underflow, the
    products become sums, and the division of Beta functions $B(x,y) / B(w,z)$
    becomes subtraction in log-space. Equation 20's likelihood term

    $\prod_l \frac{B(a + n_{k,l}^{-i} + n_{k,l}^{+i}, b + N_{k,l}^{-i} + N_{k,l}^{+i})}
                  {B(a + n_{k,l}^{-i}, b + N_{k,l}^{-i})}$

    becomes the `gammaln` expression in the `lp` calculation.

4. The Gibbs Sweep: Domain 2 ($Z_2$)
    The derivation for the second domain is completely symmetric to the first
    domain. The probability formulation is defined in Equation (24) of the paper.
    The code mirrors the Domain 1 block, iterating over `N2`, modifying `m2` and
    assessing `z2`.

5. Helper Functions
    The helpers manage the pseudo-log-likelihood tracking across iterations, and
    compact the cluster indices (removing gaps like [0, 2, 4] -> [0, 1, 2]) at the
    end of the sampling process.
"""

import numpy as np
from scipy.special import gammaln

# The Gibbs Sampler implementation is based on Section 3 of Collapsed Variational Bayes Inference of Infinite Relational Model: https://arxiv.org/abs/1409.4757


class CGS_IRM:
    def __init__(
        self,
        alpha1=1.0,  # hyperparameter of CRP for z_1 (eq. 7)
        alpha2=1.0,  # hyperparameter of CRP for z_2 (eq. 8)
        a=1.0,  # Hyperparameter of the beta distribution (eq. 6)
        b=1.0,  # Hyperparameter of the beta distribution (eq. 6)
        n_iter=500,  # Number of iterations
        burnin=250,  # The sweeps discarded before collecting samples (To allow the Markov chain to reach convergence)
        seed=None,
    ):
        self.alpha1 = alpha1
        self.alpha2 = alpha2
        self.a = a
        self.b = b
        self.n_iter = n_iter
        self.burnin = burnin
        self.seed = seed

    def fit(self, X):

        rng = np.random.default_rng(self.seed)  # random sampling initialization
        X = np.asarray(
            X, dtype=np.float64
        )  # convert input matrix X to a 2D array of floats
        N1, N2 = (
            X.shape
        )  # Extrtact dimensions. N1 is the objects in the first domain, and N2 is the objects in the second domain.

        # singleton initialization of
        z1 = np.arange(N1, dtype=int)
        z2 = np.arange(N2, dtype=int)
        K = max(N1, N2)  # Upper bound of clusters

        # Initialize (eq. 13) and (eq. 14) counters
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
        self.z1_samples_ = []  # append accepted cluster assignments
        self.z2_samples_ = []  # append accepted cluster assignments
        self.ll_trace_ = []  # pseudo log likelihood

        for it in range(self.n_iter):
            for i in rng.permutation(N1):
                k_old = z1[i]

                # Remove i's contribution to n, Nm, m1
                m1[k_old] -= 1
                np.add.at(n[k_old], z2, -X[i, :])
                np.add.at(Nm[k_old], z2, -(1.0 - X[i, :]))

                n_plus = np.zeros(K, dtype=float)
                N_plus = np.zeros(K, dtype=float)
                np.add.at(n_plus, z2, X[i, :])
                np.add.at(N_plus, z2, 1.0 - X[i, :])

                active2 = np.where(m2 > 0)[0]

                existing1 = np.where(m1 > 0)[0]
                log_p = []
                ids = []
                for k in existing1:
                    nk = n[k, active2]
                    Nmk = Nm[k, active2]
                    np_ = n_plus[active2]
                    Np_ = N_plus[active2]
                    # Translating the Beta functions of Eq 20 into log-Gamma space
                    lp = np.log(m1[k]) + np.sum(
                        gammaln(self.a + nk + np_)
                        + gammaln(self.b + Nmk + Np_)
                        - gammaln(self.a + self.b + nk + Nmk + np_ + Np_)
                        - gammaln(self.a + nk)
                        - gammaln(self.b + Nmk)
                        + gammaln(self.a + self.b + nk + Nmk)
                    )
                    log_p.append(lp)
                    ids.append(k)

                np_ = n_plus[active2]
                Np_ = N_plus[active2]
                lp_new = np.log(self.alpha1) + np.sum(
                    gammaln(self.a + np_)
                    + gammaln(self.b + Np_)
                    - gammaln(self.a + self.b + np_ + Np_)
                    - gammaln(self.a)
                    - gammaln(self.b)
                    + gammaln(self.a + self.b)
                )
                log_p.append(lp_new)
                ids.append(-1)

                # Final Normalization (Log-Sum-Exp Trick)
                log_p = np.array(log_p)
                log_p -= log_p.max()  # Subtract maximum to prevent np.exp() overflow
                p = np.exp(log_p)
                p /= p.sum()  # Normalize
                chosen = ids[rng.choice(len(ids), p=p)]

                if chosen == -1:
                    chosen = int(np.where(m1 == 0)[0][0])

                z1[i] = chosen
                m1[chosen] += 1
                np.add.at(n[chosen], z2, X[i, :])
                np.add.at(Nm[chosen], z2, 1.0 - X[i, :])

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
                log_p = []
                ids = []
                for l2 in existing2:
                    nl = n[active1, l2]
                    Nml = Nm[active1, l2]
                    np_ = n_plus[active1]
                    Np_ = N_plus[active1]
                    lp = np.log(m2[l2]) + np.sum(
                        gammaln(self.a + nl + np_)
                        + gammaln(self.b + Nml + Np_)
                        - gammaln(self.a + self.b + nl + Nml + np_ + Np_)
                        - gammaln(self.a + nl)
                        - gammaln(self.b + Nml)
                        + gammaln(self.a + self.b + nl + Nml)
                    )
                    log_p.append(lp)
                    ids.append(l2)

                np_ = n_plus[active1]
                Np_ = N_plus[active1]
                lp_new = np.log(self.alpha2) + np.sum(
                    gammaln(self.a + np_)
                    + gammaln(self.b + Np_)
                    - gammaln(self.a + self.b + np_ + Np_)
                    - gammaln(self.a)
                    - gammaln(self.b)
                    + gammaln(self.a + self.b)
                )
                log_p.append(lp_new)
                ids.append(-1)

                log_p = np.array(log_p)
                log_p -= log_p.max()
                p = np.exp(log_p)
                p /= p.sum()
                chosen = ids[rng.choice(len(ids), p=p)]

                if chosen == -1:
                    chosen = int(np.where(m2 == 0)[0][0])

                z2[j] = chosen
                m2[chosen] += 1
                np.add.at(n[:, chosen], z1, X[:, j])
                np.add.at(Nm[:, chosen], z1, 1.0 - X[:, j])

            # Pseudo log-likelihood
            ll = self._pseudo_ll(n, Nm, m1, m2)
            self.ll_trace_.append(ll)

            if it >= self.burnin:
                self.z1_samples_.append(_compact(z1))
                self.z2_samples_.append(_compact(z2))

        self.z1_ = _compact(z1)
        self.z2_ = _compact(z2)
        self.n_ = n
        self.Nm_ = Nm
        self.m1_ = m1
        self.m2_ = m2
        return self

    # NOTE: possible bug. z1_/z2_ are re-indexed by _compact (0, 1, 2, ...), but n_/Nm_
    # are still indexed by the pre-compaction cluster ids. predict() can then read an
    # empty row and fall back to the prior (theta = a / (a + b)).
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


def _compact(z):
    mapping = {}
    out = np.empty_like(z)
    for i, k in enumerate(z):
        if k not in mapping:
            mapping[k] = len(mapping)
        out[i] = mapping[k]
    return out
