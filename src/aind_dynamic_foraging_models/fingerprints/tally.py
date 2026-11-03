"""Tallies: for every bin of every statistic, the trial count, the summed prediction and the
summed outcome.

A tally is a dict ``{statistic: array of shape (n_bins, 3)}`` with columns ``N, PRED, OBS``.
Animal sessions and rollouts fill ``OBS`` with observed outcomes; replay also fills ``PRED`` with
the model's probability of that outcome. Tallies add, so a subject's tally is the sum of its
sessions' tallies, and any coarser bin is the sum of finer ones.
"""

from __future__ import annotations

from typing import Iterable, Optional, Sequence

import numpy as np

from .spec import BINS, CAP, PATTERN_OFFSETS, RUN_LENGTH_EDGES, SPEC_VERSION, STATISTICS

N, PRED, OBS = 0, 1, 2


def empty_tally() -> dict[str, np.ndarray]:
    """A tally with every bin at zero."""
    return {stat: np.zeros((len(BINS[stat]), 3)) for stat in STATISTICS}


def add(a: dict[str, np.ndarray], b: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """The sum of two tallies (a new tally; neither input changes)."""
    return {stat: a[stat] + b[stat] for stat in STATISTICS}


def drop_ignores(choices, rewards, p_right=None):
    """Remove ignored trials (choice NaN or 2), so the previous trial is the previous engaged one.

    Returns ``(choices, rewards)`` or, if ``p_right`` is given, ``(choices, rewards, p_right)``.
    """
    c = np.asarray(choices, dtype=float)
    keep = np.isfinite(c) & (c != 2)
    out = (c[keep].astype(np.int64), np.asarray(rewards, dtype=float)[keep])
    if p_right is None:
        return out
    return out + (np.asarray(p_right, dtype=float)[keep],)


def _bincount(keys: np.ndarray, pred: Optional[np.ndarray], obs: np.ndarray, n_bins: int):
    """Count, summed prediction and summed outcome per bin; keys below 0 are skipped."""
    ok = keys >= 0
    k = keys[ok]
    table = np.zeros((n_bins, 3))
    table[:, N] = np.bincount(k, minlength=n_bins)
    table[:, OBS] = np.bincount(k, weights=obs[ok], minlength=n_bins)
    if pred is not None:
        table[:, PRED] = np.bincount(k, weights=pred[ok], minlength=n_bins)
    return table


def session_tally(choices, rewards, p_right=None) -> dict[str, np.ndarray]:
    """Tally one session of engaged trials (0 = left, 1 = right; reward > 0 counts as rewarded).

    ``p_right``, if given, is the model's probability of a right choice on each trial, predicted
    from the trials before it; it fills ``PRED`` for the switch statistics.
    """
    c = np.asarray(choices, dtype=np.int64)
    r = (np.asarray(rewards, dtype=float) > 0).astype(np.int64)
    out = empty_tally()
    n_trials = len(c)
    if n_trials < 2:
        return out

    new_run = np.r_[True, c[1:] != c[:-1]]
    idx = np.arange(n_trials)
    run_start = np.maximum.accumulate(np.where(new_run, idx, 0))
    run = idx - run_start + 1  # length of the same-side run ending at each trial
    new_streak = new_run | np.r_[True, r[:-1] == 1]
    streak_start = np.maximum.accumulate(np.where(new_streak, idx, 0))
    losses = np.where(r == 0, idx - streak_start + 1, 0)  # unrewarded trials ending here

    t = idx[1:]
    prev = t - 1
    switch = (c[t] != c[prev]).astype(float)
    pred = None
    if p_right is not None:
        p = np.asarray(p_right, dtype=float)
        pred = np.where(c[prev] == 1, 1.0 - p[t], p[t])

    loss_key = np.where(losses[prev] > 0, np.minimum(losses[prev], CAP) - 1, -1)
    out["losses"] = _bincount(loss_key, pred, switch, len(BINS["losses"]))

    out["hazard"] = _bincount(
        r[prev] * CAP + np.minimum(run[prev], CAP) - 1, pred, switch, len(BINS["hazard"])
    )

    pattern_key = np.full((3, len(t)), -1)
    code = r[prev].copy()
    pattern_key[0] = PATTERN_OFFSETS[0] + code
    for depth in (2, 3):
        lag = t - depth
        valid = lag >= 0
        lag = np.maximum(lag, 0)
        digit = 2 * (c[lag] == c[prev]) + r[lag]
        code = code + 2 * 4 ** (depth - 2) * digit
        pattern_key[depth - 1] = np.where(valid, PATTERN_OFFSETS[depth - 1] + code, -1)
    tables = [_bincount(pattern_key[d], pred, switch, len(BINS["pattern"])) for d in range(3)]
    out["pattern"] = tables[0] + tables[1] + tables[2]

    before = np.maximum(t - 2, 0)
    after_switch = (t >= 2) & (c[prev] != c[before])
    post_key = np.where(after_switch, r[prev] * CAP + np.minimum(run[before], CAP) - 1, -1)
    out["post_switch"] = _bincount(post_key, pred, switch, len(BINS["post_switch"]))

    completed = run[:-1][new_run[1:]]  # a run is complete when the next trial switches
    run_key = np.searchsorted(RUN_LENGTH_EDGES, completed, side="right") - 1
    out["run_length"] = _bincount(run_key, None, np.ones(len(run_key)), len(BINS["run_length"]))
    return out


def tally_sessions(
    choices: Sequence, rewards: Sequence, p_right: Optional[Sequence] = None
) -> dict[str, np.ndarray]:
    """The summed tally of several sessions (each already free of ignored trials)."""
    total = empty_tally()
    for i, (c, r) in enumerate(zip(choices, rewards)):
        total = add(total, session_tally(c, r, None if p_right is None else p_right[i]))
    return total


def tally_groups(
    choices: Sequence,
    rewards: Sequence,
    groups: Iterable[str],
    p_right: Optional[Sequence] = None,
) -> dict[str, dict[str, np.ndarray]]:
    """Tallies per group label (for example ``"final|Coupled Baiting"``) and over ``"all"``.

    ``"all"`` is reserved for the total, so a session labelled ``"all"`` is refused.
    """
    out: dict[str, dict[str, np.ndarray]] = {"all": empty_tally()}
    for i, (c, r, g) in enumerate(zip(choices, rewards, groups)):
        if g == "all":
            raise ValueError('"all" is the total over every group; give the session another label')
        one = session_tally(c, r, None if p_right is None else p_right[i])
        out["all"] = add(out["all"], one)
        out[g] = add(out.get(g, empty_tally()), one)
    return out


def to_json(tally: dict[str, np.ndarray]) -> dict:
    """A JSON-ready tally: ``{"spec": SPEC_VERSION, stat: {label: [n, pred, obs]}}``.

    Only bins with trials are written.
    """
    out: dict = {"spec": SPEC_VERSION}
    for stat in STATISTICS:
        out[stat] = {
            label: [float(v) for v in row]
            for label, row in zip(BINS[stat], tally[stat])
            if row[N] > 0
        }
    return out


def from_json(data: dict) -> dict[str, np.ndarray]:
    """Read a tally written by :func:`to_json`; refuses another spec version."""
    if data.get("spec") != SPEC_VERSION:
        raise ValueError(f"tally is {data.get('spec')!r}, this library reads {SPEC_VERSION!r}")
    tally = empty_tally()
    for stat in STATISTICS:
        index = {label: i for i, label in enumerate(BINS[stat])}
        for label, row in data.get(stat, {}).items():
            tally[stat][index[label]] = row
    return tally
