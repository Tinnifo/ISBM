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
