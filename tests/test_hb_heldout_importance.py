"""Refit and rescore tools for held-out subjects fitted alone (study 13).

Two claims are checked. The centred session level is the same model as the non-centred one,
so a refit with it changes the sampler's geometry and nothing else. And the importance-sampled
session-marginal estimates the same integral as the plain draw-average, with weights that
stay well behaved on a long session where the plain average starves.
"""

import unittest

from tests._hb_deps import assert_deps_present

import numpy as np

try:
    import jax
    from numpyro.infer.util import log_density

    from aind_dynamic_foraging_models.hierarchical_bayes.heldout import (
        FIXED_HYPERPRIOR,
        adapt_subjects_batched,
        batched_heldout_scores,
        importance_sampled_session_marginal,
    )
    from tests.test_hb_heldout import _simulate_sessions

    HAS_JAX = True
except ImportError:  # pragma: no cover - exercised only without the bayes extra
    HAS_JAX = False

assert_deps_present(HAS_JAX)

N_PARAMS = 5
TRUE_MU = np.array([0.8, -0.6, -0.8, 0.6, 0.1])
SESSION_SD = 0.4


@unittest.skipUnless(HAS_JAX, "requires the 'bayes' extra (jax, numpyro)")
class TestCentredSessionLevel(unittest.TestCase):
    """Centred and non-centred densities differ only by the Jacobian of theta = mu + sigma z."""

    def test_same_density_up_to_the_jacobian(self):
        rng = np.random.default_rng(0)
        choices = rng.integers(0, 2, size=(1, 3, 40))
        rewards = (rng.random((1, 3, 40)) < 0.4).astype(float)
        mu_raw = rng.normal(size=(1, N_PARAMS))
        log_sigma_raw = rng.normal(size=(1, N_PARAMS))
        theta_raw = rng.normal(size=(1, 3, N_PARAMS))
        log_sigma = FIXED_HYPERPRIOR["log_sigma_mean"] + log_sigma_raw
        theta = mu_raw[:, None, :] + np.exp(log_sigma)[:, None, :] * theta_raw

        args = (choices, rewards, FIXED_HYPERPRIOR)
        shared = {"mu_raw": mu_raw, "log_sigma_raw": log_sigma_raw}
        non_centred, _ = log_density(adapt_subjects_batched, args, {},
                                     {**shared, "theta_raw": theta_raw})
        centred, _ = log_density(adapt_subjects_batched, args, {"centered": True},
                                 {**shared, "theta": theta})
        jacobian = 3 * float(np.sum(log_sigma))      # one log sigma per session and parameter
        self.assertAlmostEqual(float(centred) + jacobian, float(non_centred),
                               delta=1e-4 * abs(float(non_centred)))


@unittest.skipUnless(HAS_JAX, "requires the 'bayes' extra (jax, numpyro)")
class TestImportanceSampledSessionMarginal(unittest.TestCase):
    """Same integral as the plain scorer; healthier weights where the plain one starves."""

    @classmethod
    def setUpClass(cls):
        rng = np.random.default_rng(1)
        n_draws = 400
        cls.mu_p = TRUE_MU + 0.1 * rng.standard_normal((n_draws, N_PARAMS))
        cls.log_sigma = np.log(SESSION_SD) + 0.05 * rng.standard_normal((n_draws, N_PARAMS))
        theta = TRUE_MU + SESSION_SD * rng.standard_normal((2, N_PARAMS))
        short_c, short_r = _simulate_sessions(theta[:1], n_trials=60, seed_offset=10)
        long_c, long_r = _simulate_sessions(theta[1:], n_trials=800, seed_offset=20)
        cls.sessions = {"short": (short_c, short_r), "long": (long_c, long_r)}

    def _plain(self, name, n_theta_per_draw):
        c, r = self.sessions[name]
        samples = {"mu_p": self.mu_p[:, None, :], "log_sigma": self.log_sigma[:, None, :]}
        return batched_heldout_scores(samples, np.zeros(1, dtype=int), c, r,
                                      rng_key=jax.random.PRNGKey(2),
                                      n_theta_per_draw=n_theta_per_draw)

    def _importance(self, name):
        c, r = self.sessions[name]
        return importance_sampled_session_marginal(self.mu_p, self.log_sigma, c, r,
                                                   rng_key=jax.random.PRNGKey(3))

    def test_agrees_with_the_plain_scorer_where_both_are_healthy(self):
        """On a short session both are trustworthy, so they must agree within their errors."""
        plain = self._plain("short", n_theta_per_draw=250)        # 100,000 draws
        importance = self._importance("short")
        self.assertTrue(importance["mode_found"][0])
        self.assertLess(plain["pareto_k"][0], 0.7)
        self.assertLess(importance["pareto_k"][0], 0.7)
        error = np.hypot(plain["session_marginal_se"][0], importance["session_marginal_se"][0])
        self.assertLess(abs(plain["session_marginal"][0] - importance["session_marginal"][0]),
                        4 * error + 0.02)

    def test_weights_stay_healthy_on_a_long_session(self):
        """The plain average rests on fewer draws as sessions lengthen; the proposal does not."""
        plain = self._plain("long", n_theta_per_draw=60)          # 24,000 draws, as IS
        importance = self._importance("long")
        self.assertLess(importance["pareto_k"][0], 0.7)
        self.assertGreater(importance["weight_ess"][0], 5 * plain["weight_ess"][0])
        self.assertEqual(importance["n_trials"][0], 800)
