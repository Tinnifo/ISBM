"""Shared Beta-Bernoulli block marginal likelihood helpers.

These functions implement the collapsed marginal likelihood contribution of a
single object joining a cluster, summed over the clusters of the opposite
domain. They are the exact expressions used by the collapsed Gibbs sampler
(Section 3 of https://arxiv.org/abs/1409.4757, eq. 20) and are shared by both
the baseline sampler and the delayed-acceptance sampler so the two stay
numerically consistent.

Notation (all arrays are indexed over the *active clusters of the opposite
domain*):
    nk, Nmk      current edge / non-edge counts of the target cluster
    n_plus, N_plus  edge / non-edge counts contributed by the object joining
"""

import numpy as np
from scipy.special import gammaln


def existing_log_marginal(nk, Nmk, n_plus, N_plus, a, b):
    """Log marginal likelihood of adding an object to an existing cluster.

    This excludes the CRP prior term (``log m`` for the cluster size); callers
    add that separately.
    """
    return float(
        np.sum(
            gammaln(a + nk + n_plus)
            + gammaln(b + Nmk + N_plus)
            - gammaln(a + b + nk + Nmk + n_plus + N_plus)
            - gammaln(a + nk)
            - gammaln(b + Nmk)
            + gammaln(a + b + nk + Nmk)
        )
    )


def new_log_marginal(n_plus, N_plus, a, b):
    """Log marginal likelihood of an object forming a brand-new cluster.

    This excludes the CRP prior term (``log alpha``); callers add that
    separately. It is the ``nk = Nmk = 0`` special case of
    :func:`existing_log_marginal`.
    """
    return float(
        np.sum(
            gammaln(a + n_plus)
            + gammaln(b + N_plus)
            - gammaln(a + b + n_plus + N_plus)
            - gammaln(a)
            - gammaln(b)
            + gammaln(a + b)
        )
    )
