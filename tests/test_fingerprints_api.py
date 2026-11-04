"""Tests for the fingerprints' public interface: one-call replay fingerprints, the score,
and the replay driver with its adapters."""

import unittest
from types import SimpleNamespace

import numpy as np

from aind_dynamic_foraging_models.fingerprints import (
    RATE_STATISTICS,
    ForagerAgent,
    GLMAgent,
    ProgramAgent,
    aggregate_subjects,
    binomial_floor,
    compare_replay,
    fingerprint_score,
    floor_ratio,
    replay,
    replay_fingerprints,
    replay_gap,
    tally_groups,
)
from aind_dynamic_foraging_models.generative_model import ForagerCollection

# The hand-checked session of test_fingerprints.py.
CHOICES = np.array([0, 0, 1, 1, 1, 0, 0])
REWARDS = np.array([1, 0, 0, 1, 0, 0, 1])


def _session(rng, n):
    return rng.integers(0, 2, n), (rng.random(n) < 0.4).astype(float)


def _subject(rng, n_sessions=6):
    """Sessions in two groups, with a p(right) that depends on the last choice and reward."""
    choices, rewards, groups, p_right = [], [], [], []
    for i in range(n_sessions):
        c, r = _session(rng, int(rng.integers(150, 400)))
        prev = np.r_[0, c[:-1]]
        stay = np.where(np.r_[0, r[:-1]] > 0, 0.85, 0.7)
        p_right.append(np.where(prev == 1, stay, 1 - stay))
        choices.append(c)
        rewards.append(r)
        groups.append("g1" if i % 2 else "g0")
    return choices, rewards, groups, p_right


class TestFingerprintScore(unittest.TestCase):
    def test_geometric_mean(self):
        self.assertAlmostEqual(fingerprint_score([2.0, 0.5]), 1.0, places=15)
        self.assertAlmostEqual(fingerprint_score([1.0, 4.0, 2.0]), 2.0, places=14)

    def test_undefined(self):
        for ratios in ([1.5, float("nan")], [1.5, 0.0], [1.5, float("inf")], []):
            self.assertTrue(np.isnan(fingerprint_score(ratios)), ratios)


class TestReplayFingerprints(unittest.TestCase):
    def test_hand_checked_session(self):
        """One session: the set's numbers are the existing gap and floor of its tally."""
        p = np.full(len(CHOICES), 0.3)
        out = replay_fingerprints([CHOICES], [REWARDS], ["g"], [p], min_trials=1)
        tally = tally_groups([CHOICES], [REWARDS], ["g"], p_right=[p])["g"]
        self.assertEqual(set(out["sets"]), {"all", "g"})
        for stat in RATE_STATISTICS:
            cell = out["sets"]["g"]["statistics"][stat]
            self.assertEqual(cell["gap"], replay_gap(tally, stat, 1))
            self.assertEqual(cell["floor"], binomial_floor(tally, stat, 1))
            np.testing.assert_equal(out["sets"]["all"]["statistics"][stat], cell)  # NaN == NaN

    def test_sets_sum_groups(self):
        choices, rewards, groups, p_right = _subject(np.random.default_rng(0))
        sets = {"both": {"g0", "g1"}, "g1": {"g1"}, "missing": {"g9"}}
        out = replay_fingerprints(choices, rewards, groups, p_right, sets=sets)["sets"]
        whole = tally_groups(choices, rewards, groups, p_right=p_right)["all"]
        ratios = []
        for stat in RATE_STATISTICS:
            gap, floor = replay_gap(whole, stat), binomial_floor(whole, stat)
            self.assertAlmostEqual(
                out["both"]["statistics"][stat]["floor_ratio"], gap / floor, places=12
            )
            ratios.append(gap / floor)
            self.assertTrue(np.isnan(out["missing"]["statistics"][stat]["gap"]))
        self.assertAlmostEqual(out["both"]["score"], fingerprint_score(ratios), places=12)
        self.assertTrue(np.isnan(out["missing"]["score"]))

    def test_aggregate_is_subject_balanced(self):
        rng = np.random.default_rng(1)
        subjects = [replay_fingerprints(*_subject(rng))["sets"] for _ in range(5)]
        agg = aggregate_subjects(subjects)
        for stat in RATE_STATISTICS:
            g = [s["all"]["statistics"][stat]["gap"] for s in subjects]
            f = [s["all"]["statistics"][stat]["floor"] for s in subjects]
            cell = agg["all"]["statistics"][stat]
            self.assertEqual(cell["floor_ratio"], floor_ratio(g, f))
            self.assertEqual(cell["n_subjects"], 5)
        self.assertAlmostEqual(
            agg["all"]["score"],
            fingerprint_score(agg["all"]["statistics"][s]["floor_ratio"] for s in RATE_STATISTICS),
            places=15,
        )

    def test_aggregate_carries_any_statistics(self):
        """A comparison with an extra statistic (as a rollout's run lengths) aggregates as is."""
        cell = {"gap": 0.02, "floor": 0.01, "floor_ratio": 2.0}
        stats = {"hazard": dict(cell), "run_length": dict(cell), "kernel": dict(cell)}
        agg = aggregate_subjects([{"s": {"statistics": stats}}] * 3)["s"]
        self.assertEqual(list(agg["statistics"]), ["hazard", "run_length", "kernel"])
        self.assertAlmostEqual(agg["score"], 2.0, places=15)

    def test_tallies_are_tally_groups(self):
        """The one-call entry point tallies as ``tally_groups`` does (the golden tests)."""
        choices, rewards, groups, p_right = _subject(np.random.default_rng(5))
        got = replay_fingerprints(choices, rewards, groups, p_right)["tallies"]
        want = tally_groups(choices, rewards, groups, p_right=p_right)
        self.assertEqual(set(got), set(want))
        for g in want:
            for stat in want[g]:
                np.testing.assert_array_equal(got[g][stat], want[g][stat])

    def test_compare_replay_ignores_all(self):
        choices, rewards, groups, p_right = _subject(np.random.default_rng(2))
        tallies = tally_groups(choices, rewards, groups, p_right=p_right)
        self.assertEqual(set(compare_replay(tallies)), {"all", "g0", "g1"})
        # A set naming every key, "all" included, still counts each session once.
        explicit = compare_replay(tallies, {"every": set(tallies)})["every"]
        np.testing.assert_equal(explicit, compare_replay(tallies)["all"])
        repeated = compare_replay(tallies, {"twice": ["g0", "g0"]})["twice"]
        np.testing.assert_equal(repeated, compare_replay(tallies)["g0"])


class TestReplayDriver(unittest.TestCase):
    def test_forager_adapter_reproduces_every_preset(self):
        """The adapter steps a forager exactly as ``perform_closed_loop`` does."""
        rng = np.random.default_rng(3)
        collection = ForagerCollection()
        for preset in collection.presets:
            for _ in range(2):
                c, r = _session(rng, 300)
                reference = collection.get_preset_forager(preset, seed=0)
                reference.perform_closed_loop(c, r)
                agent = ForagerAgent(collection.get_preset_forager(preset, seed=0))
                p = replay(agent, c, r)
                want = reference.choice_prob[1] / reference.choice_prob.sum(axis=0)
                np.testing.assert_allclose(p, want, atol=1e-6, err_msg=preset)

    def test_glm_adapter_reproduces_logistic_regression(self):
        """The lag buffer gives the Su2022 regression's own probabilities, trial by trial."""
        from sklearn.linear_model import LogisticRegression

        from aind_dynamic_foraging_models.logistic_regression.model import (
            prepare_logistic_design_matrix,
        )

        rng = np.random.default_rng(4)
        c, r = _session(rng, 800)
        design = prepare_logistic_design_matrix(c, r, logistic_model="Su2022", n_trial_back=15)
        x, y = design.X.to_numpy(), design.Y.to_numpy().ravel()
        fit = LogisticRegression(C=1.0).fit(x, y)
        # Columns run RewC at lags 15..1, then UnrC at lags 15..1; the agent takes lag 1 first.
        w = fit.coef_[0]
        agent = GLMAgent(w[:15][::-1], w[15:][::-1], fit.intercept_[0])
        p = replay(agent, c, r)
        np.testing.assert_allclose(p[15:], fit.predict_proba(x)[:, 1], atol=1e-6)
        # Before lag 15 the missing history is zero, as in study 15's freeze_p1_glm.py.
        self.assertAlmostEqual(p[0], 1 / (1 + np.exp(-fit.intercept_[0])), places=12)

    def test_program_adapter(self):
        """A MEvolve-style program: logits from the latent variables, softmax in ``predict``."""
        program = SimpleNamespace(
            init_latents=lambda p, session: np.zeros(2),
            predict=lambda lat, p: p["beta"] * lat,
            update=lambda lat, choice, reward, p: lat
            + p["alpha"] * (np.eye(2)[choice] * (reward - lat)),
        )
        agent = ProgramAgent(program, {"alpha": 0.3, "beta": 4.0})
        c, r = np.array([1, 1, 0, 1]), np.array([1.0, 0.0, 1.0, 1.0])
        p = replay(agent, c, r)
        lat, want = np.zeros(2), []
        for ci, ri in zip(c, r):
            e = np.exp(4.0 * lat)
            want.append(e[1] / e.sum())
            lat = lat + 0.3 * (np.eye(2)[ci] * (ri - lat))
        np.testing.assert_allclose(p, want, atol=1e-15)
        self.assertEqual(p[0], 0.5)

    def test_program_adapter_fills_session_facts(self):
        """Facts the caller omits are 0, as in MEvolve's program.replay."""
        seen = {}
        program = SimpleNamespace(
            init_latents=lambda p, session: seen.update(session) or np.zeros(2),
            predict=lambda lat, p: lat,
            update=lambda lat, c, r, p: lat,
        )
        replay(ProgramAgent(program, {}), [0, 1], [1.0, 0.0], session={"stage": 2})
        self.assertEqual(seen, {"stage": 2, "coupled": 0, "baited": 0, "index": 0})

    def test_empty_session(self):
        """An all-ignored session gives no predictions, even for a forager."""
        agent = ForagerAgent(ForagerCollection().get_preset_forager("Bari2019", seed=0))
        self.assertEqual(len(replay(agent, [], [])), 0)

    def test_lengths_must_match(self):
        with self.assertRaises(ValueError):
            replay(GLMAgent([0.0], [0.0], 0.0), [0, 1], [1.0])


if __name__ == "__main__":
    unittest.main()
