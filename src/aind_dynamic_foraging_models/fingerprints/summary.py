"""One-call fingerprints: per subject and session set, the gap, the noise floor, the floor ratio
and the fingerprint score; and their subject-balanced aggregate.

Study 15 of aind-dynamic-foraging-bfm-studies (ADR 0003, 0008, 0009). A *set* is a named group of
session labels whose tallies are summed (for example every group except warm-ups). The
fingerprint score of a set is the geometric mean of its floor ratios over the mode's statistics.
These functions only assemble ``tally_groups``, ``replay_gap``, ``binomial_floor`` and
``floor_ratio``; no statistic, bin or floor is defined here.
"""

from __future__ import annotations

from typing import Iterable, Mapping, Optional, Sequence

import numpy as np

from .compare import MIN_TRIALS, binomial_floor, floor_ratio, replay_gap
from .spec import RATE_STATISTICS, SPEC_VERSION, STATISTICS
from .tally import add, empty_tally, tally_groups


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
    present = [tallies[g] for g in groups if g in tallies]
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


def _cell(subject: Mapping[str, dict], name: str, stat: str, key: str) -> float:
    cell = subject.get(name, {}).get("statistics", {}).get(stat)
    return np.nan if cell is None else cell[key]


def aggregate_subjects(per_subject: Sequence[Mapping[str, dict]]) -> dict:
    """Subject-balanced aggregate of ``compare_replay`` (or ``replay_fingerprints(...)["sets"]``)
    results, one per subject.

    The statistics are the ones the results carry (the rate statistics for a replay), so a
    comparison with more statistics aggregates through the same function. Per set and statistic:
    the mean gap and mean floor over the subjects where both are finite,
    their count, and the floor ratio of those means (``floor_ratio``); per set, the score of
    those floor ratios.
    """
    names = sorted({name for subject in per_subject for name in subject})
    out = {}
    for name in names:
        # The statistics the results carry: the rate statistics in replay; a rollout adds more.
        present = {st for s in per_subject if name in s for st in s[name]["statistics"]}
        stats = {}
        for stat in [st for st in STATISTICS if st in present]:
            g = np.array([_cell(s, name, stat, "gap") for s in per_subject])
            f = np.array([_cell(s, name, stat, "floor") for s in per_subject])
            ok = np.isfinite(g) & np.isfinite(f)
            stats[stat] = {
                "n_subjects": int(ok.sum()),
                "gap": float(g[ok].mean()) if ok.any() else float("nan"),
                "floor": float(f[ok].mean()) if ok.any() else float("nan"),
                "floor_ratio": floor_ratio(g, f),
            }
        out[name] = {
            "statistics": stats,
            "score": fingerprint_score(s["floor_ratio"] for s in stats.values()),
        }
    return out
