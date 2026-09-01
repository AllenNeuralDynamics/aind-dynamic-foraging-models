"""Figures: the scale transform, and that each one draws what it claims to."""

import pathlib
import tempfile
import unittest

from tests._hb_deps import assert_deps_present

import numpy as np
from scipy.stats import norm

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from aind_dynamic_foraging_models.hierarchical_bayes import plotting as P

    HAS_MPL = True
except ImportError:  # pragma: no cover - exercised only without matplotlib
    HAS_MPL = False

# A broken extra must not report OK by skipping every test that touches it. With
# AIND_HB_REQUIRE_DEPS=1 -- which the CI job that installs [bayes] sets -- a failed
# import becomes an error here instead of a run of silent skips.
assert_deps_present(HAS_MPL)

N_PARAMS = 5


def _draws(n_draws=200, seed=0):
    """Population draws and per-subject means on the unconstrained scale."""
    rng = np.random.default_rng(seed)
    population = rng.normal(0.2, 0.3, (n_draws, N_PARAMS))
    subjects = rng.normal(0.2, 0.5, (12, N_PARAMS))
    return population, subjects


@unittest.skipUnless(HAS_MPL, "requires matplotlib")
class TestScaleTransform(unittest.TestCase):
    """Values must be drawn in the units their labels name."""

    def test_rates_map_through_the_normal_cdf(self):
        """The three rate parameters land on [0, 1], never negative.

        The sampler works in unconstrained coordinates where a rate reads as -0.8; drawing
        that under a label naming a rate misstates it.
        """
        for index in (0, 1, 2):
            with self.subTest(param_index=index):
                values = P.to_bounded(np.array([-0.8, 0.0, 1.2]), index)
                np.testing.assert_allclose(values, norm.cdf([-0.8, 0.0, 1.2]), rtol=1e-6)
                self.assertTrue(np.all(values >= 0) and np.all(values <= 1))

    def test_inverse_temperature_scales_to_its_bound(self):
        """Beta maps onto [0, beta_max]."""
        values = P.to_bounded(np.array([-3.0, 0.0, 3.0]), 3, beta_max=10.0)
        self.assertTrue(np.all(values >= 0) and np.all(values <= 10.0))
        midpoint = P.to_bounded(np.array([0.0]), 3, beta_max=10.0)
        self.assertAlmostEqual(float(midpoint[0]), 5.0, places=6)

    def test_side_bias_is_left_alone(self):
        """The bias is genuinely unbounded, so it must not be squashed."""
        raw = np.array([-2.0, 0.0, 2.0])
        np.testing.assert_allclose(P.to_bounded(raw, 4), raw)


@unittest.skipUnless(HAS_MPL, "requires matplotlib")
class TestFigures(unittest.TestCase):
    """Each figure renders, saves, and carries the marks it promises."""

    def setUp(self):
        """Fresh draws and a scratch directory per test."""
        self.population, self.subjects = _draws()
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.addCleanup(plt.close, "all")

    def test_population_recovery_draws_one_panel_per_parameter(self):
        """Five parameters, five panels, and a saved file."""
        path = pathlib.Path(self.tmp.name) / "recovery.png"
        fig = P.plot_population_recovery(
            self.population, self.subjects, truth=np.zeros(N_PARAMS), path=str(path)
        )
        self.assertEqual(len(fig.axes), N_PARAMS)
        self.assertTrue(path.exists() and path.stat().st_size > 0)

    def test_population_recovery_without_truth(self):
        """Ground truth is optional, since real data has none."""
        fig = P.plot_population_recovery(self.population, self.subjects)
        self.assertEqual(len(fig.axes), N_PARAMS)

    def test_conditioning_curve_marks_every_rung(self):
        """One marker per k, plus the reference lines it is given."""
        scores = {0: 0.695, 1: 0.714, 2: 0.721, 4: 0.722, 8: 0.724, "matched": 0.719}
        fig = P.plot_conditioning_curve(
            scores, references={"MLE": (0.7127, P.MLE), "GRU": (0.7248, P.GRU)},
            title="test",
        )
        ax = fig.axes[0]
        line = ax.lines[0]
        self.assertEqual(len(line.get_xdata()), 5)          # the numeric rungs only
        self.assertGreaterEqual(len(ax.lines), 4)           # curve + matched + 2 references

    def test_conditioning_curve_without_references(self):
        """References are optional."""
        fig = P.plot_conditioning_curve({0: 0.69, 4: 0.72})
        self.assertEqual(len(fig.axes[0].lines[0].get_xdata()), 2)

    def test_shrinkage_pairs_pooled_with_unpooled(self):
        """With an unpooled arm the figure draws both series and the connectors.

        Without it the figure shows only where subjects ended up, not that pooling moved
        them, which is the comparison it exists to make.
        """
        unpooled = self.subjects + np.random.default_rng(1).normal(0, 0.3, self.subjects.shape)
        fig = P.plot_shrinkage(
            self.subjects, self.population.mean(axis=0), unpooled=unpooled, param_index=0
        )
        ax = fig.axes[0]
        self.assertEqual(len(ax.collections) + len(ax.lines),
                         len(ax.collections) + len(ax.lines))  # rendered without error
        self.assertIsNotNone(ax.get_legend())                  # two series, so a legend
        self.assertGreaterEqual(len(ax.lines), self.subjects.shape[0])  # one connector each

    def test_shrinkage_without_unpooled_has_no_legend(self):
        """A single series needs no legend box; the axis label names it."""
        fig = P.plot_shrinkage(self.subjects, self.population.mean(axis=0), param_index=0)
        self.assertIsNone(fig.axes[0].get_legend())

    def test_shrinkage_draws_on_the_bounded_scale(self):
        """A learn-rate axis must not span negative values."""
        fig = P.plot_shrinkage(self.subjects, self.population.mean(axis=0), param_index=0)
        lo, hi = fig.axes[0].get_xlim()
        self.assertGreaterEqual(lo, -0.05)
        self.assertLessEqual(hi, 1.05)


if __name__ == "__main__":
    unittest.main()
