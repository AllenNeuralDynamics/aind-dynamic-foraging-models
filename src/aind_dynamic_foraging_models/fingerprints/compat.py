"""The two older definitions, computed from fingerprints-v1 tallies.

- MEvolve (aind-dynamic-foraging-mevolve ``fingerprints.py``, main 0dfddad): per statistic, the
  pooled trial-weighted |predicted - observed| switch rate over its bins (no minimum count); the
  three depths of ``pattern`` pooled into one statistic; its ``run`` is ``hazard`` here; the
  replay gap is the mean of the three. Its gate rule is fixed (study 40 ADR 0001, 0007), so these
  numbers must not change.
- The wrapper (aind-dynamic-foraging-bfm-wrapper ``post_training_analysis/generative_analysis.py``):
  its ``abstract`` history pattern names the oldest trial ``A``; it lists trials oldest first,
  upper case for rewarded. Its post-switch statistic splits run length into 1 and more than 1.
"""

from __future__ import annotations

import numpy as np

from .spec import CAP
from .tally import OBS, PRED, N

MEVOLVE_STATISTICS = (("losses", "losses"), ("pattern", "pattern"), ("run", "hazard"))
WRAPPER_POST_SWITCH_ROWS = (
    ("rewarded", "run_length_1"),
    ("rewarded", "run_length_gt1"),
    ("unrewarded", "run_length_1"),
    ("unrewarded", "run_length_gt1"),
)


def mevolve_replay_gaps(tallies: dict[str, dict[str, np.ndarray]]) -> dict:
    """MEvolve's replay fingerprints from replay tallies per group (see :func:`tally_groups`).

    Returns ``{group: {"losses", "pattern", "run", "replay_gap"}}``; a statistic with no trials in
    a group is left out, as MEvolve leaves it out.
    """
    out: dict = {}
    for group, tally in tallies.items():
        stats = {}
        for name, stat in MEVOLVE_STATISTICS:
            table = tally[stat]
            n = table[:, N]
            if n.sum() == 0:
                continue
            stats[name] = float(np.sum(np.abs(table[:, PRED] - table[:, OBS])) / n.sum())
        if stats:
            stats["replay_gap"] = float(np.mean(list(stats.values())))
            out[group] = stats
    return out


def wrapper_abstract_label(label: str) -> str:
    """The wrapper's ``abstract`` name of a fingerprints-v1 history label.

    ``"S+O-"`` (last trial same side rewarded, the one before other side unrewarded) becomes
    ``"aB"``: oldest first, the oldest trial's side is ``A``, upper case when rewarded.
    """
    trials = [(label[i] == "S", label[i + 1] == "+") for i in range(0, len(label), 2)]
    oldest_same = trials[-1][0]
    letters = []
    for same, rewarded in reversed(trials):
        letter = "A" if same == oldest_same else "B"
        letters.append(letter if rewarded else letter.lower())
    return "".join(letters)


def wrapper_post_switch(tally: dict[str, np.ndarray]) -> np.ndarray:
    """The post-switch tally merged into the wrapper's four bins, in its row order.

    Rows follow ``WRAPPER_POST_SWITCH_ROWS``; columns are N, PRED, OBS.
    """
    table = tally["post_switch"]
    rewarded, unrewarded = table[CAP:], table[:CAP]
    return np.vstack([rewarded[0], rewarded[1:].sum(0), unrewarded[0], unrewarded[1:].sum(0)])
