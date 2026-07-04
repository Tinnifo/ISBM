import numpy as np
from sklearn.metrics import adjusted_rand_score


def ari(z_true, z_pred):
    """Adjusted Rand Index between two cluster assignment vectors."""
    return float(adjusted_rand_score(z_true, z_pred))


def heldout_test_ll(model, X, i_test, j_test):
    """Mean held-out predictive log-likelihood using the model's final sample."""
    return float(model.predict(X, i_test, j_test))


def _autocovariance(x):
    """Biased autocovariance of a 1D series via FFT (index 0 is the variance)."""
    n = x.size
    x = x - x.mean()
    fft = np.fft.fft(x, n=2 * n)
    acov = np.fft.ifft(fft * np.conjugate(fft))[:n].real
    return acov / n


def effective_sample_size(x):
    """Effective sample size of an MCMC trace.

    Uses Geyer's initial positive sequence estimator of the integrated
    autocorrelation time: ``ESS = n / (1 + 2 * sum_k rho_k)`` where the sum over
    lag-autocorrelations ``rho_k`` is truncated at the first non-positive
    consecutive pair. Higher ESS means better mixing.
    """
    x = np.asarray(x, dtype=float)
    n = x.size
    if n < 4:
        return float(n)
    acov = _autocovariance(x)
    # A (near-)constant trace has no information about mixing; treating its ESS
    # as ``n`` would flatter a chain that is simply stuck. Report ESS as
    # undefined (NaN) in that degenerate case instead.
    scale = abs(x.mean()) + 1e-12
    if acov[0] <= (1e-10 * scale) ** 2:
        return float("nan")
    rho = acov / acov[0]
    tau = 1.0
    for t in range(1, n - 1, 2):
        pair = rho[t] + rho[t + 1]
        if pair <= 0:
            break
        tau += 2.0 * pair
    if tau <= 0:
        return float(n)
    return float(min(n / tau, n))
