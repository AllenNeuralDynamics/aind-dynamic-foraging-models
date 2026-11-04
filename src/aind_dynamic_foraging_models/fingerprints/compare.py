"""Comparisons: the gap between a model and a subject, the noise floors, and the floor ratio.

Study 15 ADR 0003 and 0008. For one subject and statistic, the gap is the mean of
|model - subject| over the statistic's bins, weighted by the subject's own count in each bin;
bins with fewer than ``min_trials`` of the subject's trials are dropped, and bins a rollout never
visits are left out of its gap and counted by :func:`coverage`.

A noise floor is the gap a perfect model would still have, and what is left over depends on the
mode:

- replay compares model and subject on the same trials, so only coin-flip noise is left:
  :func:`binomial_floor`;
- a rollout makes new sessions, so coin-flip noise and the differences between sessions are
  left: :func:`noise_floor`, the split-half (test-retest) floor.
"""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
from scipy.stats import binom

from .spec import DISTRIBUTION_STATISTICS
from .tally import OBS, PRED, N, add, empty_tally

MIN_TRIALS = 5


def values(tally: dict[str, np.ndarray], stat: str, column: int = OBS) -> np.ndarray:
    """Per-bin value: a switch rate (``column`` / N) or, for a distribution, the share of units.

    A rate bin with no trials is NaN. A distribution's empty bin is a share of 0, which is a real
    value; only an entirely empty distribution is NaN.
    """
    table = tally[stat]
    n = table[:, N]
    with np.errstate(invalid="ignore", divide="ignore"):
        if stat in DISTRIBUTION_STATISTICS:
            if column == PRED:
                raise ValueError(f"{stat} has no replay prediction; replay tests it through hazard")
            total = n.sum()
            return n / total if total > 0 else np.full(len(n), np.nan)
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


def coverage(
    model: dict[str, np.ndarray],
    subject: dict[str, np.ndarray],
    stat: str,
    min_trials: int = MIN_TRIALS,
) -> float:
    """Share of the subject's kept-bin weight that the rollout visits (1 when it visits all).

    A distribution has a value in every bin once the rollout has any units, so its coverage is 1
    then and 0 otherwise.
    """
    keep = kept_bins(subject, stat, min_trials)
    weight = subject[stat][:, N]
    if not keep.any():
        return np.nan
    if stat in DISTRIBUTION_STATISTICS:
        return float(model[stat][:, N].sum() > 0)
    visited = keep & (model[stat][:, N] > 0)
    return float(weight[visited].sum() / weight[keep].sum())


def _split_scale(half_a: dict[str, np.ndarray], half_b: dict[str, np.ndarray], stat: str):
    """Per-bin factor from the gap between two halves to a perfect model's full-data gap.

    Under binomial noise, E|a - b| grows as sqrt(1/n_a + 1/n_b) and the full data's gap as
    sqrt(1/(n_a + n_b)); their ratio is sqrt(n_a * n_b) / (n_a + n_b), which is 1/2 for equal
    halves. A distribution's shares use the halves' totals.
    """
    n_a, n_b = half_a[stat][:, N], half_b[stat][:, N]
    if stat in DISTRIBUTION_STATISTICS:
        n_a, n_b = np.full_like(n_a, n_a.sum()), np.full_like(n_b, n_b.sum())
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.sqrt(n_a * n_b) / (n_a + n_b)


def _expected_abs_deviation(n: int, r: float) -> float:
    """E|X/n - r| for X ~ Binomial(n, r): a perfect model's expected miss in one bin."""
    k = np.arange(n + 1)
    return float(np.sum(binom.pmf(k, n, r) * np.abs(k / n - r)))


def binomial_floor(tally: dict[str, np.ndarray], stat: str, min_trials: int = MIN_TRIALS) -> float:
    """The replay floor: the gap a perfect model would still have from coin-flip noise alone.

    In replay the model predicts each of the subject's real trials, so if it were the true process
    each outcome would be a coin flip with its probability. For each bin the subject keeps, with
    ``n`` of its trials and observed rate ``r``, the expected miss is E|X/n - r| for
    X ~ Binomial(n, r); bins are weighted by ``n``, as the gap is. It uses only the subject's data,
    so every model shares it. A bin with ``r`` of 0 or 1 adds 0. Rate statistics only: a
    distribution has no replay prediction. NaN when the subject keeps no bin.
    """
    if stat in DISTRIBUTION_STATISTICS:
        raise ValueError(f"{stat} has no replay prediction, so no replay floor")
    keep = kept_bins(tally, stat, min_trials)
    if not keep.any():
        return np.nan
    n = tally[stat][keep, N]
    r = tally[stat][keep, OBS] / n
    miss = np.array([_expected_abs_deviation(int(round(nb)), rb) for nb, rb in zip(n, r)])
    return float(np.sum(n * miss) / n.sum())


def noise_floor(
    session_tallies: Sequence[dict[str, np.ndarray]],
    stat: str,
    n_splits: int = 20,
    seed: int = 0,
    min_trials: int = MIN_TRIALS,
    max_draws_per_split: int = 10,
) -> float:
    """The rollout floor: the gap between two random halves of a subject's sessions, scaled to
    the full data.

    Each bin's difference between the halves is rescaled by :func:`_split_scale`, so halves with
    unequal counts in a bin are not over-weighted. The bins and weights are the full subject's, so
    the floor is computed over the same bins as the subject's gap. It includes differences between
    sessions, which a rollout's new sessions also have; a replay can track them, so replay uses
    :func:`binomial_floor` instead. A split counts only if both halves
    have trials in every kept bin, so the floor compares the same bins as the gap; up to
    ``max_draws_per_split * n_splits`` splits are drawn to find ``n_splits`` such splits. NaN with
    fewer than two sessions or no such split.
    """
    k = len(session_tallies)
    if k < 2:
        return np.nan
    full = empty_tally()
    for one in session_tallies:
        full = add(full, one)
    keep = kept_bins(full, stat, min_trials)
    if not keep.any():
        return np.nan
    weight = full[stat][:, N]
    rng = np.random.default_rng(seed)
    gaps = []
    for _ in range(max_draws_per_split * n_splits):
        if len(gaps) == n_splits:
            break
        order = rng.permutation(k)
        half_a, half_b = empty_tally(), empty_tally()
        for j, i in enumerate(order):
            if j < k // 2:
                half_a = add(half_a, session_tallies[i])
            else:
                half_b = add(half_b, session_tallies[i])
        scale = _split_scale(half_a, half_b, stat)
        diff = scale * (values(half_a, stat) - values(half_b, stat))
        if not np.isfinite(diff[keep]).all():
            continue  # a kept bin is missing from one half: this split cannot compare it
        gaps.append(_weighted_gap(diff, np.zeros_like(diff), weight, keep))
    return float(np.mean(gaps)) if gaps else np.nan


def floor_ratio(gaps: Sequence[float], floors: Optional[Sequence[float]]) -> float:
    """Subject-balanced gap over subject-balanced floor, on subjects where both are finite."""
    g = np.asarray(gaps, dtype=float)
    f = np.asarray(floors, dtype=float)
    ok = np.isfinite(g) & np.isfinite(f)
    if not ok.any():
        return np.nan
    return float(np.mean(g[ok]) / np.mean(f[ok]))
