"""Held-out subjects fitted alone: the fixed hyperprior and the multi-chain fit.

``FIXED_HYPERPRIOR`` claims that the batched adaptation model, given it in place of a population,
is ``hattori2019_two_level``. That claim is what lets a study fit held-out subjects without a
population level, so it is checked exactly, on the log density, rather than by eye.
"""

import unittest

from tests._hb_deps import assert_deps_present

import numpy as np

try:
    import jax
    import jax.numpy as jnp
    from numpyro.infer.util import log_density

    from aind_dynamic_foraging_models.hierarchical_bayes.heldout import (
        FIXED_HYPERPRIOR,
        adapt_subjects_batched,
        fit_heldout_two_level,
    )
    from aind_dynamic_foraging_models.hierarchical_bayes.model import hattori2019_two_level

    HAS_JAX = True
except ImportError:  # pragma: no cover - exercised only without the bayes extra
    HAS_JAX = False

assert_deps_present(HAS_JAX)

N_PARAMS = 5


def _sessions(n_subjects, n_sessions, n_trials, seed=0):
    """Random histories; the density check does not need realistic behaviour."""
    rng = np.random.default_rng(seed)
    choices = rng.integers(0, 2, size=(n_subjects, n_sessions, n_trials))
    rewards = (rng.random((n_subjects, n_sessions, n_trials)) < 0.4).astype(float)
    return choices, rewards


@unittest.skipUnless(HAS_JAX, "requires the 'bayes' extra (jax, numpyro)")
class TestFixedHyperpriorIsTheTwoLevelModel(unittest.TestCase):
    """Same joint density, up to the Jacobian of sampling log sigma instead of sigma."""

    def test_log_densities_match(self):
        """At matched parameters the two log densities differ only by sum(log sigma).

        ``hattori2019_two_level`` puts a log-normal on ``sigma``; the batched model puts a
        normal on ``log sigma``. Those are one distribution written on two scales, whose
        densities differ by the Jacobian ``|d log sigma / d sigma| = 1 / sigma``.
        """
        choices, rewards = _sessions(1, 4, 50)
        rng = np.random.default_rng(1)
        mu_p = rng.normal(0.0, 1.0, N_PARAMS)
        log_sigma = rng.normal(-1.0, 1.0, N_PARAMS)
        theta_raw = rng.normal(0.0, 1.0, (4, N_PARAMS))

        two_level, _ = log_density(
            hattori2019_two_level, (choices[0], rewards[0]), {},
            {"mu_p": mu_p, "sigma": np.exp(log_sigma), "theta_raw": theta_raw},
        )
        batched, _ = log_density(
            adapt_subjects_batched, (choices, rewards, FIXED_HYPERPRIOR), {},
            {
                "mu_raw": mu_p[None],
                "log_sigma_raw": (log_sigma - FIXED_HYPERPRIOR["log_sigma_mean"])[None],
                "theta_raw": theta_raw[None],
            },
        )
        self.assertAlmostEqual(float(two_level) + float(np.sum(log_sigma)), float(batched),
                               delta=1e-3 * abs(float(batched)))

    def test_hyperprior_matches_the_two_level_defaults(self):
        """Standard normal on mu_p; log sigma centred at -1 with unit scale."""
        np.testing.assert_array_equal(FIXED_HYPERPRIOR["population_mean"], np.zeros(N_PARAMS))
        np.testing.assert_array_equal(FIXED_HYPERPRIOR["population_scale"], np.ones(N_PARAMS))
        np.testing.assert_array_equal(FIXED_HYPERPRIOR["log_sigma_mean"], -np.ones(N_PARAMS))
        np.testing.assert_array_equal(FIXED_HYPERPRIOR["log_sigma_spread"], np.ones(N_PARAMS))


@unittest.skipUnless(HAS_JAX, "requires the 'bayes' extra (jax, numpyro)")
class TestMultiChainFit(unittest.TestCase):
    """The fit pools chains for scoring and reports diagnostics per subject."""

    @classmethod
    def setUpClass(cls):
        """A short two-chain fit of three subjects, one with a padded session slot."""
        cls.n_subjects, n_context, n_trials = 3, 3, 80
        choices, rewards = _sessions(cls.n_subjects, n_context, n_trials, seed=2)
        session_mask = np.ones((cls.n_subjects, n_context), dtype=bool)
        session_mask[2, 2] = False
        cls.samples, cls.diagnostics = fit_heldout_two_level(
            choices, rewards, rng_key=jax.random.PRNGKey(0), session_mask=session_mask,
            num_warmup=150, num_samples=100, num_chains=2,
        )

    def test_draws_pool_chains_with_a_subject_axis(self):
        """Chains are pooled, ready for the batched scorers."""
        self.assertEqual(self.samples["mu_p"].shape, (200, self.n_subjects, N_PARAMS))
        self.assertEqual(self.samples["log_sigma"].shape, (200, self.n_subjects, N_PARAMS))

    def test_diagnostics_are_per_subject(self):
        """One worst-case R-hat and ESS per subject, plus a joint divergence count."""
        self.assertEqual(self.diagnostics["max_r_hat"].shape, (self.n_subjects,))
        self.assertEqual(self.diagnostics["min_ess_bulk"].shape, (self.n_subjects,))
        self.assertTrue(np.all(np.isfinite(self.diagnostics["max_r_hat"])))
        self.assertTrue(np.all(self.diagnostics["min_ess_bulk"] > 0))
        self.assertIsInstance(self.diagnostics["divergences"], int)

    def test_r_hat_flags_chains_that_disagree(self):
        """Two chains stuck in different places give a large R-hat for that subject only."""
        from aind_dynamic_foraging_models.hierarchical_bayes.heldout import subject_diagnostics

        rng = np.random.default_rng(3)
        draws = rng.normal(0.0, 1.0, (2, 200, 2, N_PARAMS))
        draws[1, :, 1, 0] += 5.0   # subject 1's chains disagree on one parameter
        diagnostics = subject_diagnostics(_StubRun({"mu_p": jnp.asarray(draws),
                                                    "log_sigma": jnp.asarray(draws)}))
        self.assertLess(diagnostics["max_r_hat"][0], 1.05)
        self.assertGreater(diagnostics["max_r_hat"][1], 1.5)
        self.assertIsNone(diagnostics["divergences"])

    def test_r_hat_flags_chains_that_differ_only_in_scale(self):
        """Same centre, three times the spread: only the folded, rank-normalised R-hat sees it."""
        from aind_dynamic_foraging_models.hierarchical_bayes.heldout import subject_diagnostics

        rng = np.random.default_rng(4)
        draws = rng.normal(0.0, 1.0, (4, 500, 2, N_PARAMS))
        draws[2:, :, 1, 0] *= 3.0   # subject 1: half the chains three times wider
        diagnostics = subject_diagnostics(_StubRun({"mu_p": draws, "log_sigma": draws}))
        self.assertLess(diagnostics["max_r_hat"][0], 1.02)
        self.assertGreater(diagnostics["max_r_hat"][1], 1.05)

    def test_ess_is_the_bulk_ess(self):
        """ESS agrees with ArviZ's bulk ESS computed parameter by parameter."""
        import arviz as az

        from aind_dynamic_foraging_models.hierarchical_bayes.heldout import subject_diagnostics

        rng = np.random.default_rng(5)
        draws = np.cumsum(rng.normal(0.0, 1.0, (4, 300, 2, N_PARAMS)), axis=1) * 0.05 \
            + rng.normal(0.0, 1.0, (4, 300, 2, N_PARAMS))     # autocorrelated
        diagnostics = subject_diagnostics(_StubRun({"mu_p": draws, "log_sigma": draws + 1.0}))
        for subject in range(2):
            reference = min(float(az.ess(draws[:, :, subject, k], method="bulk"))
                            for k in range(N_PARAMS))
            self.assertAlmostEqual(diagnostics["min_ess_bulk"][subject], reference, delta=1e-6)


class _StubRun:
    """Stands in for a finished MCMC run with chosen draws, shaped (chains, draws, ...)."""

    def __init__(self, grouped):
        """Hold the draws."""
        self.grouped = grouped

    def get_samples(self, group_by_chain=False):
        """Return the held draws."""
        return self.grouped

    def get_extra_fields(self):
        """No divergence field, as when it was not collected."""
        return {}


if __name__ == "__main__":
    unittest.main()
