"""State matching by posterior overlap and by weight distance."""

import unittest

import numpy as np

from tests._glmhmm_deps import DEPS_PRESENT, assert_deps_present

assert_deps_present()
if DEPS_PRESENT:
    from aind_dynamic_foraging_models.glm_hmm import (
        match_by_posterior_overlap,
        match_by_weights,
    )


@unittest.skipUnless(DEPS_PRESENT, "needs the glmhmm extra")
class TestAlignment(unittest.TestCase):
    """A known relabelling is undone."""

    def test_posterior_overlap_recovers_a_permutation(self):
        """Permuted, slightly noisy posteriors map back with high quality."""
        rng = np.random.default_rng(0)
        reference = rng.dirichlet(np.ones(3) * 0.2, size=500)
        relabel = np.array([2, 0, 1])  # new state i is reference state relabel[i]
        noisy = reference[:, relabel] + 0.01 * rng.random((500, 3))
        noisy /= noisy.sum(1, keepdims=True)
        permutation, quality, overlap = match_by_posterior_overlap(noisy, reference)
        np.testing.assert_allclose(noisy[:, permutation], reference, atol=0.02)
        _, self_quality, _ = match_by_posterior_overlap(reference, reference)
        self.assertAlmostEqual(quality, self_quality, delta=0.02)
        self.assertAlmostEqual(overlap.sum(), 1.0)

    def test_weight_distance_recovers_a_permutation(self):
        """Permuted weights map back exactly."""
        reference = np.arange(12.0).reshape(3, 2, 2)
        permutation, cost = match_by_weights(reference[[1, 2, 0]], reference)
        np.testing.assert_array_equal(reference[[1, 2, 0]][permutation], reference)
        self.assertEqual(cost.shape, (3, 3))


if __name__ == "__main__":
    unittest.main()
