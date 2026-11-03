"""Comparisons: the gap between a model and a subject, the noise floor, and the floor ratio.

Study 15 ADR 0003. For one subject and statistic, the gap is the mean of |model - subject| over
the statistic's bins, weighted by the subject's own count in each bin; bins with fewer than
``min_trials`` of the subject's trials are dropped. The noise floor is the gap a perfect model
would still have because the subject has finite trials: half the gap between two random halves
of the subject's sessions, averaged over splits.
"""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np

from .spec import DISTRIBUTION_STATISTICS
from .tally import OBS, PRED, N, add, empty_tally

MIN_TRIALS = 5


def values(tally: dict[str, np.ndarray], stat: str, column: int = OBS) -> np.ndarray:
    """Per-bin value: a switch rate (``column`` / N) or, for a distribution, the share of units.

    Bins with no trials are NaN.
    """
    table = tally[stat]
    n = table[:, N]
    with np.errstate(invalid="ignore", divide="ignore"):
        if stat in DISTRIBUTION_STATISTICS:
            if column == PRED:
                raise ValueError(f"{stat} has no replay prediction; replay tests it through hazard")
            total = n.sum()
            return np.where(n > 0, n / total, np.nan) if total > 0 else np.full(len(n), np.nan)
        return np.where(n > 0, table[:, column] / n, np.nan)


def _weighted_gap(model: np.ndarray, subject: np.ndarray, weight: np.ndarray, keep: np.ndarray):
    """Mean |model - subject| over kept, finite bins, weighted by ``weight``."""
    ok = keep & np.isfinite(model) & np.isfinite(subject)
    if not ok.any():
        return np.nan
    return float(np.sum(weight[ok] * np.abs(model[ok] - subject[ok])) / np.sum(weight[ok]))


def kept_bins(subject: dict[str, np.ndarray], stat: str, min_trials: int = MIN_TRIALS):
    """The bins a subject keeps: those with at least ``min_trials`` of its own trials."""
    return subject[stat][:, N] >= min_trials


def replay_gap(tally: dict[str, np.ndarray], stat: str, min_trials: int = MIN_TRIALS) -> float:
    """Gap between the model's predicted and the subject's observed rate, on the same trials."""
    keep = kept_bins(tally, stat, min_trials)
    return _weighted_gap(
        values(tally, stat, PRED), values(tally, stat, OBS), tally[stat][:, N], keep
    )


def rollout_gap(
    model: dict[str, np.ndarray],
    subject: dict[str, np.ndarray],
    stat: str,
    min_trials: int = MIN_TRIALS,
) -> float:
    """Gap between a rollout tally and the subject's tally (bins the rollout never visits drop)."""
    keep = kept_bins(subject, stat, min_trials)
    return _weighted_gap(values(model, stat), values(subject, stat), subject[stat][:, N], keep)


def noise_floor(
    session_tallies: Sequence[dict[str, np.ndarray]],
    stat: str,
    n_splits: int = 20,
    seed: int = 0,
    min_trials: int = MIN_TRIALS,
) -> float:
    """Half the gap between two random halves of a subject's sessions, averaged over splits.

    The bins and weights are the full subject's, so the floor is computed over the same bins as
    the subject's gap. Needs at least two sessions; otherwise NaN.
    """
    k = len(session_tallies)
    if k < 2:
        return np.nan
    full = empty_tally()
    for one in session_tallies:
        full = add(full, one)
    keep = kept_bins(full, stat, min_trials)
    weight = full[stat][:, N]
    rng = np.random.default_rng(seed)
    gaps = []
    for _ in range(n_splits):
        order = rng.permutation(k)
        half_a, half_b = empty_tally(), empty_tally()
        for j, i in enumerate(order):
            if j < k // 2:
                half_a = add(half_a, session_tallies[i])
            else:
                half_b = add(half_b, session_tallies[i])
        gaps.append(_weighted_gap(values(half_a, stat), values(half_b, stat), weight, keep))
    gaps = np.asarray(gaps, dtype=float)
    return float(0.5 * np.nanmean(gaps)) if np.isfinite(gaps).any() else np.nan


def floor_ratio(gaps: Sequence[float], floors: Optional[Sequence[float]]) -> float:
    """Subject-balanced gap over subject-balanced floor, on subjects where both are finite."""
    g = np.asarray(gaps, dtype=float)
    f = np.asarray(floors, dtype=float)
    ok = np.isfinite(g) & np.isfinite(f)
    if not ok.any():
        return np.nan
    return float(np.mean(g[ok]) / np.mean(f[ok]))
