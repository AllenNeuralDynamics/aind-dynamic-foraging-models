"""Match one fit's states to a reference fit's states (for interpretation only).

State labels are arbitrary: a per-subject fit started from the global fit usually keeps its
labels, but nothing forces it to. Held-out likelihood does not depend on labels, so alignment
matters only when states are compared across subjects or with the global model.
"""

import numpy as np
from scipy.optimize import linear_sum_assignment


def match_by_posterior_overlap(posteriors, reference_posteriors):
    """Permutation of ``posteriors``' states that best overlaps the reference posteriors.

    Both arguments are posteriors over the same trials (e.g. a subject's fit and the global
    fit, smoothed on that subject's sessions).

    Parameters
    ----------
    posteriors : np.ndarray
        Shape (N, K), rows summing to 1.
    reference_posteriors : np.ndarray
        Shape (N, K).

    Returns
    -------
    tuple
        ``(permutation, quality, overlap)``. ``permutation[j]`` is the state of
        ``posteriors`` matched to reference state ``j``. ``quality`` is the probability
        that a state drawn from each posterior on a random trial is a matched pair: 1 only
        when both are the same hard assignment, so compare it with the reference's own
        value (``match_by_posterior_overlap(reference, reference)``), which is below 1
        whenever the posteriors are soft. ``overlap`` is the (K, K) expected co-occupancy
        matrix, normalised to sum to 1.
    """
    overlap = np.asarray(posteriors).T @ np.asarray(reference_posteriors)
    overlap = overlap / overlap.sum()
    rows, cols = linear_sum_assignment(-overlap)
    permutation = np.empty_like(rows)
    permutation[cols] = rows
    return permutation, float(overlap[rows, cols].sum()), overlap


def match_by_weights(weights, reference_weights):
    """Permutation of ``weights``' states with the smallest total squared weight distance.

    Parameters
    ----------
    weights : np.ndarray
        Per-state weights, shape (K, ...).
    reference_weights : np.ndarray
        Same shape.

    Returns
    -------
    tuple
        ``(permutation, cost)``: ``permutation[j]`` is the state of ``weights`` matched to
        reference state ``j``, and ``cost`` (K, K) the squared distances.
    """
    K = weights.shape[0]
    flat = np.asarray(weights).reshape(K, -1)
    reference = np.asarray(reference_weights).reshape(K, -1)
    cost = ((flat[:, None, :] - reference[None, :, :]) ** 2).sum(-1)
    rows, cols = linear_sum_assignment(cost)
    permutation = np.empty_like(rows)
    permutation[cols] = rows
    return permutation, cost
