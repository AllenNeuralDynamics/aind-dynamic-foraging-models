"""Testing the softmax used by action selection"""

import unittest

import numpy as np

from aind_dynamic_foraging_models.generative_model.act_functions import softmax


class TestSoftmax(unittest.TestCase):
    """Testing softmax, including its branch for logits above 700"""

    def test_overflow_branch_favours_the_larger_logit(self):
        """Above 700 the larger logit gets the probability, whatever the rng draws (#96)."""
        for seed in range(200):
            rng = np.random.default_rng(seed)
            np.testing.assert_array_equal(softmax(np.array([800.0, -790.0]), rng=rng), [1.0, 0.0])
            np.testing.assert_array_equal(softmax(np.array([-790.0, 800.0]), rng=rng), [0.0, 1.0])

    def test_overflow_branch_is_the_stable_softmax(self):
        """Above 700 the result is the softmax itself, computed without overflow."""
        p = softmax(np.array([800.0, 799.0]))
        np.testing.assert_allclose(p, [np.e / (1 + np.e), 1 / (1 + np.e)], rtol=1e-12)
        np.testing.assert_allclose(softmax(np.array([800.0, 800.0])), [0.5, 0.5], rtol=1e-12)

    def test_continuous_at_700(self):
        """Just below and just above the switch give the same probabilities."""
        below = softmax(np.array([700.0 - 1e-9, 699.0 - 1e-9]))
        above = softmax(np.array([700.0 + 1e-9, 699.0 + 1e-9]))
        np.testing.assert_allclose(below, above, rtol=1e-9)

    def test_below_700_unchanged(self):
        """Below 700 the result is exp(x) / sum(exp(x)), as before, to the last bit."""
        rng = np.random.default_rng(0)
        for _ in range(100):
            x = rng.uniform(-50, 650, size=2)
            np.testing.assert_array_equal(softmax(x), np.exp(x) / np.sum(np.exp(x)))


if __name__ == "__main__":
    unittest.main(verbosity=2)
