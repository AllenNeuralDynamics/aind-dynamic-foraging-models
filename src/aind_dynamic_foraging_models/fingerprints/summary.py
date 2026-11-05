"""One-call fingerprints: per subject and session set, the gap, the noise floor, the floor ratio
and the fingerprint score; and their subject-balanced aggregate.

Study 15 of aind-dynamic-foraging-bfm-studies (ADR 0003, 0008, 0009). A *set* is a named group of
session labels whose tallies are summed (for example every group except warm-ups). The
fingerprint score of a set is the geometric mean of its floor ratios over the mode's statistics:
the rate statistics in replay, which a rollout joins with the run-length distribution. Replay's
floor is binomial, a rollout's the split-half floor, and a rollout also reports its coverage.
These functions only assemble ``tally_groups``, ``replay_gap``, ``binomial_floor``,
``rollout_gap``, ``noise_floor``, ``coverage`` and ``floor_ratio``; no statistic, bin or floor is
defined here.
"""

from __future__ import annotations

from typing import Iterable, Mapping, Optional, Sequence

import numpy as np

from .compare import (
    MIN_TRIALS,
    binomial_floor,
    coverage,
    floor_ratio,
    noise_floor,
    replay_gap,
    rollout_gap,
)
from .spec import RATE_STATISTICS, SPEC_VERSION, STATISTICS
from .tally import add, empty_tally, session_tally, tally_groups


def fingerprint_score(ratios: Iterable[float]) -> float:
    """The geometric mean of floor ratios (ADR 0009); NaN if any of them is not a finite,
    positive number (a statistic with no ratio makes the score undefined, not better)."""
    r = np.asarray(list(ratios), dtype=float)
    if r.size == 0 or not np.all(np.isfinite(r) & (r > 0)):
        return float("nan")
    return float(np.exp(np.mean(np.log(r))))


def default_sets(groups: Iterable[str]) -> dict[str, set[str]]:
    """``all`` (every group) and each group alone."""
    labels = sorted(set(groups) - {"all"})
    return {"all": set(labels), **{g: {g} for g in labels}}


def _summed(tallies: Mapping[str, dict], groups: Iterable[str]):
    # each group once ("all" is their sum, and a repeated label would count twice)
    present = [tallies[g] for g in dict.fromkeys(groups) if g in tallies and g != "all"]
    if not present:
        return None
    total = empty_tally()
    for t in present:
        total = add(total, t)
    return total


def compare_replay(
    tallies: Mapping[str, dict],
    sets: Optional[Mapping[str, Iterable[str]]] = None,
    min_trials: int = MIN_TRIALS,
) -> dict:
    """One subject's replay comparison from its per-group replay tallies (``tally_groups`` with
    ``p_right``; an ``all`` entry is ignored, sets name the groups they sum).

    Returns ``{set: {"statistics": {stat: {"gap", "floor", "floor_ratio"}}, "score"}}``; a set
    with none of the subject's groups has NaN everywhere.
    """
    groups = [g for g in tallies if g != "all"]
    sets = default_sets(groups) if sets is None else sets
    out = {}
    for name, members in sets.items():
        t = _summed(tallies, members)
        stats = {}
        for stat in RATE_STATISTICS:
            gap = np.nan if t is None else replay_gap(t, stat, min_trials)
            floor = np.nan if t is None else binomial_floor(t, stat, min_trials)
            ok = np.isfinite(gap) and np.isfinite(floor) and floor > 0
            stats[stat] = {
                "gap": float(gap),
                "floor": float(floor),
                "floor_ratio": float(gap / floor) if ok else float("nan"),
            }
        out[name] = {
            "statistics": stats,
            "score": fingerprint_score(s["floor_ratio"] for s in stats.values()),
        }
    return out


def replay_fingerprints(
    choices: Sequence,
    rewards: Sequence,
    groups: Sequence[str],
    p_right: Sequence,
    sets: Optional[Mapping[str, Iterable[str]]] = None,
    min_trials: int = MIN_TRIALS,
) -> dict:
    """One subject's replay fingerprints in one call.

    ``choices``, ``rewards`` and ``p_right`` hold one array per session (engaged trials only),
    ``groups`` one label per session. Returns ``{"spec", "tallies": {group: tally},
    "sets": compare_replay(...)}``; ``tallies`` includes ``all``.
    """
    tallies = tally_groups(choices, rewards, groups, p_right=p_right)
    return {
        "spec": SPEC_VERSION,
        "tallies": tallies,
        "sets": compare_replay(tallies, sets, min_trials),
    }


def _ratio(gap: float, floor: float) -> float:
    ok = np.isfinite(gap) and np.isfinite(floor) and floor > 0
    return float(gap / floor) if ok else float("nan")


def compare_rollout(
    model_tallies: Mapping[str, dict],
    subject_sessions: Sequence[dict],
    subject_groups: Sequence[str],
    sets: Optional[Mapping[str, Iterable[str]]] = None,
    min_trials: int = MIN_TRIALS,
    n_splits: int = 20,
    seed: int = 0,
) -> dict:
    """One subject's rollout comparison: the model's rollout tallies per group (``tally_groups``
    without ``p_right``; an ``all`` entry is ignored) against the subject's own sessions.

    ``subject_sessions`` holds one ``session_tally`` per real session and ``subject_groups`` its
    label. Per set and statistic (``STATISTICS``, the run-length distribution included): the gap
    (``rollout_gap``), the split-half floor of the set's real sessions (``noise_floor``, with
    ``n_splits`` and ``seed``), their ratio and the coverage. Returns ``{set: {"statistics": {stat:
    {"gap", "floor", "floor_ratio", "coverage"}}, "score"}}``; a set the model or the subject has
    no sessions in has NaN everywhere.
    """
    subject_groups = list(subject_groups)
    if len(subject_groups) != len(subject_sessions):
        raise ValueError(f"{len(subject_sessions)} sessions but {len(subject_groups)} labels")
    if sets is None:
        sets = default_sets(set(subject_groups) | set(model_tallies))
    out = {}
    for name, members in sets.items():
        members = set(members) - {"all"}
        model = _summed(model_tallies, members)
        own = [t for t, g in zip(subject_sessions, subject_groups) if g in members]
        subject = None
        for t in own:
            subject = add(empty_tally() if subject is None else subject, t)
        stats = {}
        for stat in STATISTICS:
            if model is None or subject is None:
                gap = floor = cov = np.nan
            else:
                gap = rollout_gap(model, subject, stat, min_trials)
                floor = noise_floor(own, stat, n_splits, seed, min_trials)
                cov = coverage(model, subject, stat, min_trials)
            stats[stat] = {
                "gap": float(gap),
                "floor": float(floor),
                "floor_ratio": _ratio(gap, floor),
                "coverage": float(cov),
            }
        out[name] = {
            "statistics": stats,
            "score": fingerprint_score(s["floor_ratio"] for s in stats.values()),
        }
    return out


def rollout_fingerprints(
    rollout_choices: Sequence,
    rollout_rewards: Sequence,
    rollout_groups: Sequence[str],
    choices: Sequence,
    rewards: Sequence,
    groups: Sequence[str],
    sets: Optional[Mapping[str, Iterable[str]]] = None,
    min_trials: int = MIN_TRIALS,
    n_splits: int = 20,
    seed: int = 0,
) -> dict:
    """One subject's rollout fingerprints in one call.

    ``rollout_*`` hold one array per rollout session (``rollout``'s choices and rewards; several
    rollouts of one real session each count as a session) and its group label; ``choices``,
    ``rewards`` and ``groups`` the subject's real sessions (engaged trials only). Returns
    ``{"spec", "tallies": {"rollout": {group: tally}, "subject": {group: tally}}, "sets":
    compare_rollout(...)}``; both tallies include ``all``.
    """
    model = tally_groups(rollout_choices, rollout_rewards, rollout_groups)
    subject = tally_groups(choices, rewards, groups)
    sessions = [session_tally(c, r) for c, r in zip(choices, rewards)]
    return {
        "spec": SPEC_VERSION,
        "tallies": {"rollout": model, "subject": subject},
        "sets": compare_rollout(model, sessions, groups, sets, min_trials, n_splits, seed),
    }


def _cell(subject: Mapping[str, dict], name: str, stat: str, key: str) -> float:
    cell = subject.get(name, {}).get("statistics", {}).get(stat)
    return np.nan if cell is None else cell.get(key, np.nan)


def aggregate_subjects(per_subject: Sequence[Mapping[str, dict]]) -> dict:
    """Subject-balanced aggregate of ``compare_replay`` or ``compare_rollout`` results (or the
    ``"sets"`` of ``replay_fingerprints`` / ``rollout_fingerprints``), one per subject.

    The statistics are the ones the results carry (the rate statistics for a replay, the
    run-length distribution too for a rollout). Per set and statistic: the mean gap and mean floor
    over the subjects where both are finite, their count, and the floor ratio of those means
    (``floor_ratio``); for a rollout, also the mean coverage over every subject that has one,
    including those whose rollout visits none of their bins (coverage 0, gap NaN). Per set, the
    score of those floor ratios.
    """
    names = sorted({name for subject in per_subject for name in subject})
    out = {}
    for name in names:
        # The statistics the results carry: the rate statistics in replay; a rollout adds more.
        present = {st for s in per_subject if name in s for st in s[name]["statistics"]}
        stats = {}
        order = [st for st in STATISTICS if st in present] + sorted(present - set(STATISTICS))
        for stat in order:
            g = np.array([_cell(s, name, stat, "gap") for s in per_subject])
            f = np.array([_cell(s, name, stat, "floor") for s in per_subject])
            ok = np.isfinite(g) & np.isfinite(f)
            stats[stat] = {
                "n_subjects": int(ok.sum()),
                "gap": float(g[ok].mean()) if ok.any() else float("nan"),
                "floor": float(f[ok].mean()) if ok.any() else float("nan"),
                "floor_ratio": floor_ratio(g, f),
            }
            if any("coverage" in s.get(name, {}).get("statistics", {}).get(stat, {})
                   for s in per_subject):
                c = np.array([_cell(s, name, stat, "coverage") for s in per_subject])
                has = np.isfinite(c)
                stats[stat]["coverage"] = float(c[has].mean()) if has.any() else float("nan")
        out[name] = {
            "statistics": stats,
            "score": fingerprint_score(s["floor_ratio"] for s in stats.values()),
        }
    return out
