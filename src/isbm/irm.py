r"""Single-domain IRM generative model.

Based on Ishiguro, Sato & Ueda, "Collapsed Variational Bayes Inference of
Infinite Relational Model", https://arxiv.org/abs/1409.4757.
Notes carried over verbatim from the former notebook Notebooks/IRM.py.

The single-domain IRM generative model has three parts:

* The Partitions (Chinese Restaurant Process)
* The Block Probabilities (Beta Prior)
* The Network Links (Bernoulli Likelihood)

The Partitions (Chinese Restaurant Process)
    $$z_i|\alpha \sim \text{CRP}(\alpha) \qquad (11)$$

    Because the CRP dynamically groups our $N$ nodes into clusters, it naturally
    determines the final number of occupied clusters, $K$. Once the CRP is finished
    seating all the "customers" (nodes), we can count how many "tables" (clusters)
    are occupied. That exact number becomes the $K$ we use to build our $K \times K$
    matrix for the block probabilities, $\theta$.

    $$p(z_i = k | z_{\backslash i}, \alpha) = \begin{cases}
      \frac{m_{\backslash i, k}}{N - 1 + \alpha} & m_{\backslash i, k} > 0 \text{ (existing cluster)} \\
      \frac{\alpha}{N - 1 + \alpha} & m_{\backslash i, k} = 0 \text{ (new cluster)}
    \end{cases} \qquad (2)$$

The Block Probabilities (Beta Prior)
    $$\theta_{k,l}|a_{k,l}, b_{k,l} \sim \text{Beta}( a_{k,l}, b_{k,l} ) \qquad (10)$$

    We can define a 2D array of theta values sampled from the beta distribution by
    defining the size of the matrix, given hyperparamters $a$ and $b$, this is faster
    than a for loop since we perform vectorization.

The Network Links (Bernoulli Likelihood)
    $$x_{i, j}|Z, \{\theta\} \sim \text{Bernoulli}( \theta_{z_i,z_j} ) \qquad (12)$$

    We now have $z$ whcich is our list of lenght $N$ where $z[i]$ tells us which
    cluster node $i$ belongs to. And we have the theta_matrix which is our $K \times K$
    matrix where theta_matrix[k,l] gives the probability of a link from clyster $k$
    to cluster $l$. Now we need to generate our final $N \times N$ adjacency matrix,
    $X$, where X[i, j] = 1 if there is a link from node i to node j, and 0 otherwise.

Note: this generator departs from the paper by using two Beta priors, one for
within-cluster (block) pairs and one for between-cluster pairs.
"""

import numpy as np


def generate_irm_graph(
    N=150,
    alpha=5.0,
    a_block=10.0,
    b_block=1.0,
    a_non_block=1.0,
    b_non_block=15.0,
    seed=None,
):
    """Sample an N x N adjacency matrix from an Infinite Relational Model.

    Cluster assignments are drawn from a Chinese Restaurant Process with
    concentration `alpha`. Within-cluster edge probabilities are sampled from
    Beta(a_block, b_block); between-cluster probabilities from
    Beta(a_non_block, b_non_block). Returns (X, Z) where X is the binary
    adjacency matrix and Z is the cluster assignment vector.
    """
    rng = np.random.default_rng(seed)

    cluster_counts = []
    Z = []
    for _ in range(N):
        weights = cluster_counts + [alpha]
        total = sum(weights)
        probabilities = [w / total for w in weights]
        chosen_index = int(rng.choice(len(weights), p=probabilities))
        Z.append(chosen_index)
        if chosen_index == len(cluster_counts):
            cluster_counts.append(1)
        else:
            cluster_counts[chosen_index] += 1

    K = len(cluster_counts)

    theta_matrix_block = rng.beta(a_block, b_block, size=(K, K))
    theta_matrix_non_block = rng.beta(a_non_block, b_non_block, size=(K, K))

    X = np.zeros((N, N), dtype=np.float64)
    Z_arr = np.asarray(Z)
    for i in range(N):
        for j in range(N):
            if Z[i] == Z[j]:
                p_link = theta_matrix_block[Z[i], Z[j]]
            else:
                p_link = theta_matrix_non_block[Z[i], Z[j]]
            X[i, j] = rng.binomial(n=1, p=p_link)

    return X, Z_arr
