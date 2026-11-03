"""Tests for fingerprints: hand-checked bins, and parity with the two older implementations."""

import unittest
from collections import defaultdict

import numpy as np

from aind_dynamic_foraging_models.fingerprints import (
    BINS,
    OBS,
    PRED,
    SPEC_VERSION,
    N,
    coverage,
    drop_ignores,
    empty_tally,
    floor_ratio,
    from_json,
    mevolve_replay_gaps,
    noise_floor,
    replay_gap,
    rollout_gap,
    session_tally,
    tally_groups,
    tally_sessions,
    to_json,
    values,
    wrapper_abstract_label,
    wrapper_post_switch,
)

# ---------------------------------------------------------------------------------------------
# Reference 1: MEvolve's replay fingerprints, copied from aind-dynamic-foraging-mevolve
# src/mevolve/fingerprints.py at main 0dfddad (sessions as (choices, rewards, group) tuples).
# ---------------------------------------------------------------------------------------------
_MEVOLVE_CAP = 6


def _mevolve_features(choices, rewards):
    """MEvolve's bin keys for every trial t >= 1."""
    out = {"losses": [], "pattern": [], "run": []}
    run, losses = 0, 0
    for t in range(1, len(choices)):
        prev = choices[t - 1]
        run = run + 1 if t > 1 and choices[t - 1] == choices[t - 2] else 1
        losses = (
            0
            if rewards[t - 1]
            else (losses + 1 if t > 1 and choices[t - 1] == choices[t - 2] else 1)
        )
        out["losses"].append(min(losses, _MEVOLVE_CAP) if losses else None)
        hist = tuple(
            (int(choices[t - lag] == prev), int(rewards[t - lag]))
            for lag in range(1, min(3, t) + 1)
        )
        out["pattern"].append([hist[:n] for n in range(1, len(hist) + 1)])
        out["run"].append((int(rewards[t - 1]), min(run, _MEVOLVE_CAP)))
    return out


def _mevolve_fingerprints(sessions, p_right):
    """MEvolve's gaps per statistic, overall and per group."""
    tallies = defaultdict(lambda: defaultdict(lambda: np.zeros(3)))
    for (choices, rewards, group), pr in zip(sessions, p_right):
        c, r = np.asarray(choices), np.asarray(rewards)
        if len(c) < 2:
            continue
        switch = (c[1:] != c[:-1]).astype(float)
        pred = np.where(c[:-1] == 1, 1.0 - pr[1:], pr[1:])
        feats = _mevolve_features(c, r)
        for g in ("all", group):
            for i in range(len(switch)):
                row = np.array([1.0, pred[i], switch[i]])
                if feats["losses"][i] is not None:
                    tallies[(g, "losses")][feats["losses"][i]] += row
                for key in feats["pattern"][i]:
                    tallies[(g, "pattern")][key] += row
                tallies[(g, "run")][feats["run"][i]] += row
    out = {}
    for (g, stat), bins in tallies.items():
        n = np.array([b[0] for b in bins.values()])
        gap = np.array([abs(b[1] - b[2]) / b[0] for b in bins.values()])
        out.setdefault(g, {})[stat] = float(np.sum(n * gap) / np.sum(n))
    for g, stats in out.items():
        stats["replay_gap"] = float(
            np.mean([stats[k] for k in ("losses", "pattern", "run") if k in stats])
        )
    return out


# ---------------------------------------------------------------------------------------------
# Reference 2: the wrapper's counting rules, from aind-dynamic-foraging-bfm-wrapper
# code/post_training_analysis/generative_analysis.py at main 5affee5
# (_extract_history_pattern_count_record, _history_pattern_to_abstract,
# extract_switches_and_run_lengths, _extract_subject_metric_counts).
# ---------------------------------------------------------------------------------------------
def _wrapper_abstract_counts(choices, rewards, n_back):
    """The wrapper's (total, switches) per abstract pattern at one depth."""

    def encode(c, r):
        """L/l/R/r: side, upper case when rewarded."""
        return ("L" if r > 0 else "l") if c == 0 else ("R" if r > 0 else "r")

    def abstract(p):
        """Rename sides so the oldest trial is A."""
        if p[0].upper() == "L":
            m = {"L": "A", "l": "a", "R": "B", "r": "b"}
        else:
            m = {"L": "B", "l": "b", "R": "A", "r": "a"}
        return "".join(m[ch] for ch in p)

    counts = defaultdict(lambda: [0, 0])  # total, switches
    enc = [encode(c, r) for c, r in zip(choices, rewards)]
    for t in range(n_back, len(enc)):
        start = t - n_back
        key = abstract("".join(enc[start:t]))
        counts[key][0] += 1
        counts[key][1] += int(choices[t] != choices[t - 1])
    return counts


def _wrapper_post_switch_counts(choices, rewards):
    """The wrapper's (n, switches) after a switch, per reward x run 1 / >1."""
    switch_idx = [i for i in range(1, len(choices)) if choices[i] != choices[i - 1]]
    runs, prev = [], 0
    for s in switch_idx:
        runs.append(s - prev)
        prev = s
    counts = defaultdict(lambda: [0, 0])  # n, successes
    for pos, s in enumerate(switch_idx):
        if s <= 0 or s >= len(rewards) or s + 1 >= len(choices):
            continue
        key = (
            "rewarded" if rewards[s] > 0 else "unrewarded",
            "run_length_1" if runs[pos] == 1 else "run_length_gt1",
        )
        counts[key][0] += 1
        counts[key][1] += int(choices[s + 1] != choices[s])
    return counts


def _random_sessions(seed, n_sessions=6, n_trials=300):
    """Sticky random sessions with random rewards, p(right) and groups."""
    rng = np.random.default_rng(seed)
    out = []
    for k in range(n_sessions):
        stay = rng.uniform(0.6, 0.95)
        c = [int(rng.integers(2))]
        for _ in range(n_trials - 1):
            c.append(c[-1] if rng.random() < stay else 1 - c[-1])
        r = (rng.random(n_trials) < 0.4).astype(float)
        p = rng.uniform(0.02, 0.98, n_trials)
        out.append((np.array(c), r, p, ["final|Coupled Baiting", "earlier|Uncoupled"][k % 2]))
    return out


class TestBins(unittest.TestCase):
    """Bin labels and the mapping onto the wrapper's labels."""

    def test_labels(self):
        """Pattern bins are 2 + 8 + 32 labels, most recent first, the last trial always S."""
        self.assertEqual(len(BINS["pattern"]), 42)
        self.assertEqual(BINS["pattern"][:2], ("S-", "S+"))
        self.assertTrue(all(label[0] == "S" for label in BINS["pattern"]))
        self.assertEqual(BINS["hazard"][0], "-1")
        self.assertEqual(BINS["hazard"][-1], "+6+")
        self.assertEqual(
            BINS["run_length"], ("1", "2", "3", "4", "5", "6-10", "11-20", "21-40", "41+")
        )

    def test_wrapper_abstract_is_a_bijection(self):
        """The 32 depth-3 labels map onto 32 different wrapper names, each starting at A."""
        names = {wrapper_abstract_label(label) for label in BINS["pattern"][10:]}
        self.assertEqual(len(names), 32)
        self.assertTrue(all(name[0] in "Aa" for name in names))
        self.assertEqual(wrapper_abstract_label("S+O-"), "aB")


class TestTally(unittest.TestCase):
    """Bins of a hand-checked session."""

    def setUp(self):
        """L+ L- R- R+ R- L- L+."""
        self.t = session_tally([0, 0, 1, 1, 1, 0, 0], [1, 0, 0, 1, 0, 0, 1])

    def _row(self, stat, label):
        """(count, summed outcome) of one bin."""
        return list(self.t[stat][BINS[stat].index(label)][[0, OBS]])

    def test_losses(self):
        """Four trials follow one loss on the current side; two of them switch."""
        self.assertEqual(self._row("losses", "1"), [4, 2])
        self.assertEqual(self.t["losses"][:, 0].sum(), 4)

    def test_hazard(self):
        """Reward x run length of the last trial."""
        self.assertEqual(self._row("hazard", "-1"), [2, 0])
        self.assertEqual(self._row("hazard", "-2"), [1, 1])
        self.assertEqual(self._row("hazard", "-3"), [1, 1])
        self.assertEqual(self._row("hazard", "+1"), [1, 0])
        self.assertEqual(self._row("hazard", "+2"), [1, 0])

    def test_post_switch(self):
        """Two switches are followed by a trial; neither is followed by another switch."""
        self.assertEqual(self._row("post_switch", "-2"), [1, 0])
        self.assertEqual(self._row("post_switch", "-3"), [1, 0])
        self.assertEqual(self.t["post_switch"][:, 0].sum(), 2)

    def test_pattern_and_runs(self):
        """Depth-1 bins, and the two completed runs (the last run is not counted)."""
        self.assertEqual(self._row("pattern", "S+"), [2, 0])
        self.assertEqual(self._row("pattern", "S-"), [4, 2])
        self.assertEqual(self._row("run_length", "2"), [1, 1])
        self.assertEqual(self._row("run_length", "3"), [1, 1])
        self.assertEqual(self.t["run_length"][:, 0].sum(), 2)

    def test_short_and_ignores(self):
        """A session under two trials tallies nothing; ignored trials are removed first."""
        self.assertEqual(sum(v.sum() for v in session_tally([1], [1]).values()), 0)
        c, r, p = drop_ignores([0, np.nan, 2, 1], [1, 0, 0, 0], [0.1, 0.2, 0.3, 0.4])
        self.assertEqual(list(c), [0, 1])
        self.assertEqual(list(p), [0.1, 0.4])
        self.assertEqual(len(drop_ignores([0, 1], [0, 1])), 2)


class TestParity(unittest.TestCase):
    """The library reproduces MEvolve's numbers and the wrapper's counts."""

    def test_mevolve(self):
        """Same replay gaps as MEvolve's own code, overall and per group."""
        sessions = _random_sessions(1)
        ref = _mevolve_fingerprints(
            [(c, r, g) for c, r, _, g in sessions], [p for *_, p, _ in sessions]
        )
        tallies = tally_groups(
            [s[0] for s in sessions],
            [s[1] for s in sessions],
            [s[3] for s in sessions],
            p_right=[s[2] for s in sessions],
        )
        got = mevolve_replay_gaps(tallies)
        self.assertEqual(set(got), set(ref))
        for group in ref:
            for stat, value in ref[group].items():
                self.assertAlmostEqual(got[group][stat], value, places=12)

    def test_wrapper_patterns(self):
        """Same per-pattern counts as the wrapper's abstract encoding, at each depth."""
        for c, r, _, _ in _random_sessions(2):
            t = session_tally(c, r)
            for depth, offset in ((1, 0), (2, 2), (3, 10)):
                ref = _wrapper_abstract_counts(c, r, depth)
                for i in range(offset, offset + 2 * 4 ** (depth - 1)):
                    name = wrapper_abstract_label(BINS["pattern"][i])
                    self.assertEqual(list(t["pattern"][i][[0, OBS]]), ref.get(name, [0, 0]))

    def test_wrapper_post_switch(self):
        """Same post-switch counts as the wrapper's four bins."""
        for c, r, _, _ in _random_sessions(3):
            ref = _wrapper_post_switch_counts(c, r)
            table = wrapper_post_switch(session_tally(c, r))
            for row, key in zip(
                table,
                (
                    ("rewarded", "run_length_1"),
                    ("rewarded", "run_length_gt1"),
                    ("unrewarded", "run_length_1"),
                    ("unrewarded", "run_length_gt1"),
                ),
            ):
                self.assertEqual([row[0], row[OBS]], ref.get(key, [0, 0]))


class TestCompare(unittest.TestCase):
    """Gaps, the noise floor and JSON."""

    def test_replay_gap(self):
        """A model that predicts each outcome exactly has no gap; a constant one does."""
        sessions = _random_sessions(4)
        exact = [c.astype(float) for c, *_ in sessions]  # p(right) = the choice made
        t = tally_sessions([s[0] for s in sessions], [s[1] for s in sessions], exact)
        self.assertAlmostEqual(replay_gap(t, "hazard"), 0.0)
        half = tally_sessions(
            [s[0] for s in sessions],
            [s[1] for s in sessions],
            [np.full(len(s[0]), 0.5) for s in sessions],
        )
        self.assertGreater(replay_gap(half, "pattern"), 0.1)
        self.assertTrue(np.isnan(replay_gap(empty_tally(), "losses")))

    def test_rollout_gap_and_floor(self):
        """A tally against itself has no gap; the floor is positive, seeded, needs 2 sessions."""
        sessions = _random_sessions(5, n_sessions=10)
        per = [session_tally(c, r) for c, r, _, _ in sessions]
        whole = tally_sessions([s[0] for s in sessions], [s[1] for s in sessions])
        for stat in ("hazard", "run_length"):
            self.assertAlmostEqual(rollout_gap(whole, whole, stat), 0.0)
            floor = noise_floor(per, stat, n_splits=5, seed=1)
            self.assertGreater(floor, 0)
            self.assertEqual(floor, noise_floor(per, stat, n_splits=5, seed=1))
        self.assertTrue(np.isnan(noise_floor(per[:1], "hazard")))
        self.assertAlmostEqual(floor_ratio([0.2, 0.4, np.nan], [0.1, 0.1, 0.1]), 3.0)
        self.assertTrue(np.isnan(floor_ratio([np.nan], [0.1])))

    def test_floor_with_unequal_sessions(self):
        """With unequal sessions the rescaled floor matches a perfect model's expected gap."""
        rng = np.random.default_rng(8)
        lengths = (900, 100, 60, 30, 10)
        floors, expected = [], []
        for _ in range(40):
            per = []
            for n in lengths:  # i.i.d. 50/50 choices: every bin's switch rate is 0.5
                per.append(session_tally(rng.integers(0, 2, n), rng.integers(0, 2, n)))
            whole = per[0]
            for one in per[1:]:
                whole = {k: whole[k] + one[k] for k in whole}
            n_bin = whole["hazard"][:, N]
            keep = n_bin >= 5
            # E|p_hat - p| for a perfect model, normal approximation, weighted by the bin counts
            e_gap = np.sqrt(2 / np.pi) * 0.5 / np.sqrt(n_bin[keep])
            expected.append(np.sum(n_bin[keep] * e_gap) / n_bin[keep].sum())
            floors.append(noise_floor(per, "hazard", n_splits=10, seed=1))
        self.assertAlmostEqual(np.mean(floors) / np.mean(expected), 1.0, delta=0.15)

    def test_coverage(self):
        """A rollout that never visits some of the subject's bins covers less than all of them."""
        c, r, _, _ = _random_sessions(9, n_sessions=1)[0]
        subject = session_tally(c, r)
        self.assertAlmostEqual(coverage(subject, subject, "pattern"), 1.0)
        never_rewarded = session_tally(c, np.zeros(len(c)))
        self.assertLess(coverage(never_rewarded, subject, "pattern"), 0.6)
        self.assertTrue(np.isnan(coverage(subject, empty_tally(), "pattern")))

    def test_distribution_zero_bins(self):
        """An empty distribution bin is a share of 0, so a shifted distribution has a gap."""
        subject, model = empty_tally(), empty_tally()
        subject["run_length"][:3, N] = [5, 5, 0]
        model["run_length"][:3, N] = [0, 5, 5]
        self.assertAlmostEqual(rollout_gap(model, subject, "run_length"), 0.25)
        self.assertEqual(coverage(model, subject, "run_length"), 1.0)
        self.assertEqual(coverage(empty_tally(), subject, "run_length"), 0.0)

    def test_reserved_group(self):
        """A session cannot be labelled "all", which would count it twice."""
        c, r, _, _ = _random_sessions(10, n_sessions=1)[0]
        with self.assertRaises(ValueError):
            tally_groups([c], [r], ["all"])

    def test_values(self):
        """A distribution's shares sum to 1 and have no replay prediction."""
        c, r, _, _ = _random_sessions(6, n_sessions=1)[0]
        t = session_tally(c, r)
        self.assertAlmostEqual(np.nansum(values(t, "run_length")), 1.0)
        with self.assertRaises(ValueError):
            values(t, "run_length", PRED)
        self.assertTrue(np.isnan(values(empty_tally(), "run_length")).all())

    def test_json(self):
        """A tally survives JSON; another spec version is refused."""
        c, r, p, _ = _random_sessions(7, n_sessions=1)[0]
        t = session_tally(c, r, p)
        data = to_json(t)
        self.assertEqual(data["spec"], SPEC_VERSION)
        back = from_json(data)
        for stat in t:
            np.testing.assert_allclose(back[stat], t[stat])
        with self.assertRaises(ValueError):
            from_json({"spec": "fingerprints-v0"})


if __name__ == "__main__":
    unittest.main()
