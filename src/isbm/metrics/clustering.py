from sklearn.metrics import adjusted_rand_score


def ari(z_true, z_pred):
    """Adjusted Rand Index between two cluster assignment vectors."""
    return float(adjusted_rand_score(z_true, z_pred))


def heldout_test_ll(model, X, i_test, j_test):
    """Mean held-out predictive log-likelihood using the model's final sample."""
    return float(model.predict(X, i_test, j_test))
