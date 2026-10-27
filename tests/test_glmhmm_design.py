"""GLM-HMM design matrices: hand-checked values and parity with the Su2022 regression."""

import unittest

import numpy as np

from tests._glmhmm_deps import DEPS_PRESENT, assert_deps_present

assert_deps_present()
if DEPS_PRESENT:
    from aind_dynamic_foraging_models.glm_hmm.design import (
        build_inputs,
        column_names,
        column_slices,
        default_design,
    )
    from aind_dynamic_foraging_models.logistic_regression import prepare_logistic_design_matrix


@unittest.skipUnless(DEPS_PRESENT, "needs the glmhmm extra")
class TestDefaultDesign(unittest.TestCase):
    """Column values on a short session that contains an ignore."""

    def setUp(self):
        """Choices R, L, I, R with rewards 1, 0, (1 -> forced 0), 1; two lags."""
        self.choices = np.array([1, 0, 2, 1])
        self.rewards = np.array([1, 0, 1, 1])
        self.X = default_design(self.choices, self.rewards, n_lags=2)

    def test_shape_and_bias(self):
        """4L regressor columns plus a bias of ones."""
        self.assertEqual(self.X.shape, (4, 9))
        np.testing.assert_array_equal(self.X[:, -1], 1.0)

    def test_first_trial_has_no_history(self):
        """Trial 0 sees only the bias."""
        np.testing.assert_array_equal(self.X[0, :-1], 0.0)

    def test_side_lags_are_signed_and_ignore_is_zero(self):
        """RewC/UnrC: +1 right, -1 left, 0 on the ignored trial."""
        names = column_names(2)
        row = dict(zip(names, self.X[3]))  # trial 3: lag 1 = ignore, lag 2 = unrewarded left
        self.assertEqual(row["RewC_1"], 0.0)
        self.assertEqual(row["UnrC_1"], 0.0)
        self.assertEqual(row["UnrC_2"], -1.0)
        self.assertEqual(row["RewC_2"], 0.0)
        row = dict(zip(names, self.X[1]))  # trial 1: lag 1 = rewarded right
        self.assertEqual(row["RewC_1"], 1.0)

    def test_engagement_lags_and_ignored_reward(self):
        """An ignored trial counts as Ignored and is never Rewarded."""
        row = dict(zip(column_names(2), self.X[3]))
        self.assertEqual(row["Ignored_1"], 1.0)
        self.assertEqual(row["Rewarded_1"], 0.0)  # reward on the ignore was forced to 0
        self.assertEqual(row["Ignored_2"], 0.0)

    def test_inputs_append_mask_and_slices_cover_columns(self):
        """build_inputs adds a mask of ones; the slices index the documented blocks."""
        inputs = build_inputs(self.choices, self.rewards, 2)
        cols = column_slices(2)
        self.assertEqual(inputs.shape, (4, 10))
        np.testing.assert_array_equal(inputs[:, cols["mask"]], 1.0)
        np.testing.assert_array_equal(inputs[:, cols["regressors"]], self.X)

    def test_session_shorter_than_the_history(self):
        """Fewer trials than lags: the design builds, with the unreachable lags all zero."""
        X = default_design(np.array([1, 0, 2]), np.array([1, 0, 0]), n_lags=5)
        self.assertEqual(X.shape, (3, 21))
        names = column_names(5)
        np.testing.assert_array_equal(X[:, names.index("RewC_3")], 0.0)
        self.assertEqual(X[2, names.index("RewC_2")], 1.0)  # trial 2, two back: rewarded right
        np.testing.assert_array_equal(default_design([2], [0], 3)[0, :-1], 0.0)

    def test_rejects_bad_codes_and_unknown_design(self):
        """Out-of-range choices or rewards, and unknown designs, fail loudly."""
        with self.assertRaises(ValueError):
            default_design([0, 3], [0, 0], 1)
        with self.assertRaises(ValueError):
            default_design([0, 1], [0, 2], 1)
        with self.assertRaises(KeyError):
            build_inputs([0, 1], [0, 1], 1, design="nope")

    def test_rejects_fractional_choices(self):
        """A fractional choice is malformed input, not a left or right code after truncation."""
        for bad in ([0, 1.9], [0, -0.5], [2.5, 1]):
            with self.assertRaises(ValueError):
                default_design(bad, [0, 0], 1)
        np.testing.assert_array_equal(default_design([0.0, 1.0], [0, 1], 1),
                                      default_design([0, 1], [0, 1], 1))


@unittest.skipUnless(DEPS_PRESENT, "needs the glmhmm extra")
class TestSu2022Parity(unittest.TestCase):
    """Without ignores, the side block is exactly the Su2022 RewC/UnrC design."""

    def test_side_block_matches_prepare_logistic_design_matrix(self):
        """Same values on every trial Su2022 keeps (it drops the first n_trial_back)."""
        rng = np.random.default_rng(0)
        choices = rng.integers(0, 2, 200)
        rewards = rng.integers(0, 2, 200)
        n_back = 5
        ours = default_design(choices, rewards, n_back)
        theirs = prepare_logistic_design_matrix(
            choices, rewards, logistic_model="Su2022", n_trial_back=n_back
        )
        trials = theirs.index.to_numpy()
        for family, offset in (("RewC", 0), ("UnrC", n_back)):
            for lag in range(1, n_back + 1):
                np.testing.assert_array_equal(
                    ours[trials, offset + lag - 1],
                    theirs["X", family, lag].to_numpy(),
                    err_msg=f"{family} lag {lag}",
                )


if __name__ == "__main__":
    unittest.main()
