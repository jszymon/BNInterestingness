"""Conditional probability tables given by logistic regression.

The parents are one-hot encoded (the first value of each parent is the
reference category) and, optionally, standardized.  The model is fitted
on the counts of all parent/child value combinations, so the data does
not need to be scanned again.  The result is the full conditional
probability table predicted by the model."""

import warnings

import numpy as np


def _encode(parent_sizes):
    """One-hot (drop first) encoding of all parent configurations.

    Returns an array with one row per configuration, in np.ndindex order."""
    configs = list(np.ndindex(*parent_sizes))
    X = np.zeros((len(configs), sum(s - 1 for s in parent_sizes)))
    for r, x in enumerate(configs):
        offset = 0
        for v, s in zip(x, parent_sizes):
            if v > 0:
                X[r, offset + v - 1] = 1.0
            offset += s - 1
    return X


def logistic_cpt(counts, C = np.inf, standardize = True):
    """Fit a logistic regression CPT.

    counts has shape parent_sizes + [number of child values].  C is the
    inverse regularization strength as in scikit-learn (np.inf means
    no regularization).  Returns an array of the same shape as counts
    holding P(child | parents)."""
    counts = np.asarray(counts, dtype=float)
    parent_sizes = counts.shape[:-1]
    r = counts.shape[-1]
    total = counts.reshape(-1, r).sum(axis=0)
    if total.sum() == 0:
        return np.full(counts.shape, 1.0 / r)
    observed = np.flatnonzero(total)
    # no informative features or a single observed class: the
    # (unpenalized) intercept alone gives the marginal frequencies
    X_all = _encode(parent_sizes)
    if X_all.shape[1] == 0 or len(observed) == 1:
        return np.broadcast_to(total / total.sum(), counts.shape).copy()

    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.exceptions import ConvergenceWarning

    flat = counts.reshape(-1, r)
    rows, ys = np.nonzero(flat)
    X = X_all[rows]
    w = flat[rows, ys]
    if standardize:
        scaler = StandardScaler().fit(X, sample_weight=w)
        X = scaler.transform(X)
        X_all = scaler.transform(X_all)
    model = LogisticRegression(C=C, max_iter=1000)
    with warnings.catch_warnings():
        # without regularization separable data makes coefficients diverge;
        # the predicted probabilities are still sensible (close to 0 or 1)
        warnings.simplefilter("ignore", ConvergenceWarning)
        model.fit(X, ys, sample_weight=w)
    proba = model.predict_proba(X_all)
    cpt = np.zeros((X_all.shape[0], r))
    cpt[:, model.classes_] = proba
    return cpt.reshape(counts.shape)
