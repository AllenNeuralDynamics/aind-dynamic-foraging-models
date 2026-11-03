"""Fingerprint spec v1: the statistics, their bins and the bins' labels.

The spec is decided in study 15 of aind-dynamic-foraging-bfm-studies (ADR 0004). Every output
names ``SPEC_VERSION``; changing a key, a cap or a statistic makes a new version.

A past trial is coded relative to the last choice: ``S`` (same side) or ``O`` (other side), then
``+`` (rewarded) or ``-`` (unrewarded). A history label lists the most recent trial first, so
``S+O-S-`` reads "last trial rewarded; the one before on the other side, unrewarded; the one before
that on the same side, unrewarded". The most recent trial is ``S`` by definition.

Statistics whose outcome is a switch (trial t's choice differs from trial t - 1's):

- ``losses``: consecutive unrewarded trials on the current side before t, 1 to 5, then 6 or more;
  trials after a rewarded trial have no bin;
- ``pattern``: the last 1, 2 and 3 trials (2 + 8 + 32 bins; a trial fills one bin per depth);
- ``hazard``: the last trial's reward by the current run length (1 to 5, then 6 or more);
- ``post_switch``: trials right after a switch, by the switch trial's reward and the length of
  the run before the switch (study 01 r9's headline statistic, with finer run bins).

Statistic whose unit is a completed run of same-side choices:

- ``run_length``: how many runs end at each length (1 to 5, 6-10, 11-20, 21-40, 41 or more).
  The last run of a session is not complete, so it is not counted.
"""

from __future__ import annotations

SPEC_VERSION = "fingerprints-v1"
CAP = 6
RUN_LENGTH_EDGES = (1, 2, 3, 4, 5, 6, 11, 21, 41)

RATE_STATISTICS = ("losses", "pattern", "hazard", "post_switch")
DISTRIBUTION_STATISTICS = ("run_length",)
STATISTICS = RATE_STATISTICS + DISTRIBUTION_STATISTICS

PATTERN_DEPTHS = (1, 2, 3)
PATTERN_OFFSETS = (0, 2, 10)


def _capped(k: int) -> str:
    """A count's label, with counts of ``CAP`` or more sharing one bin."""
    return str(k) if k < CAP else f"{CAP}+"


def _pattern_labels() -> tuple[str, ...]:
    """History labels for depths 1, 2 and 3, in bin order."""
    labels = []
    for depth in PATTERN_DEPTHS:
        for code in range(2 * 4 ** (depth - 1)):
            parts = ["S" + "-+"[code % 2]]
            rest = code // 2
            for _ in range(depth - 1):
                digit, rest = rest % 4, rest // 4
                parts.append("OS"[digit // 2] + "-+"[digit % 2])
            labels.append("".join(parts))
    return tuple(labels)


def _run_length_labels() -> tuple[str, ...]:
    """Run-length bin labels from ``RUN_LENGTH_EDGES``."""
    labels = []
    for lo, hi in zip(RUN_LENGTH_EDGES, RUN_LENGTH_EDGES[1:] + (None,)):
        if hi is None:
            labels.append(f"{lo}+")
        elif hi == lo + 1:
            labels.append(str(lo))
        else:
            labels.append(f"{lo}-{hi - 1}")
    return tuple(labels)


_REWARD_BY_RUN = tuple(f"{'-+'[r]}{_capped(k)}" for r in (0, 1) for k in range(1, CAP + 1))

BINS: dict[str, tuple[str, ...]] = {
    "losses": tuple(_capped(k) for k in range(1, CAP + 1)),
    "pattern": _pattern_labels(),
    "hazard": _REWARD_BY_RUN,
    "post_switch": _REWARD_BY_RUN,
    "run_length": _run_length_labels(),
}
