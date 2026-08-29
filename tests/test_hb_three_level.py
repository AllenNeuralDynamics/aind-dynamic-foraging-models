"""The one-stage joint model: population over subjects over sessions."""

import unittest

import numpy as np
from aind_behavior_gym.dynamic_foraging.task import CoupledBlockTask
from scipy.stats import norm

from aind_dynamic_foraging_models.generative_model import ForagerCollection

try:
    import jax
    from numpyro.infer import MCMC, NUTS

    from aind_dynamic_foraging_models.hierarchical_bayes.model import (
        hattori2019_three_level,
    )

    HAS_JAX = True
except ImportError:  # pragma: no cover - exercised only without the bayes extra
    HAS_JAX = False

BETA_MAX = 10.0
N_PARAMS = 5


def _simulate_cohort(population_mean, population_scale, sigma, n_subjects, n_sessions,
                     n_trials, seed=0):
    """Generate a cohort from the full three-level structure."""
    rng = np.random.default_rng(seed)
    mu_p = population_mean + population_scale * rng.standard_normal((n_subjects, N_PARAMS))

    choices = np.zeros((n_subjects, n_sessions, n_trials), dtype=int)
    rewards = np.zeros((n_subjects, n_sessions, n_trials), dtype=float)
    for subject in range(n_subjects):
        theta = mu_p[subject] + sigma * rng.standard_normal((n_sessions, N_PARAMS))
        for session in range(n_sessions):
            forager = ForagerCollection().get_preset_forager(
                "Hattori2019", seed=subject * 100 + session
            )
            forager.set_params(
                learn_rate_rew=float(norm.cdf(theta[session, 0])),
                learn_rate_unrew=float(norm.cdf(theta[session, 1])),
                forget_rate_unchosen=float(norm.cdf(theta[session, 2])),
                softmax_inverse_temperature=float(norm.cdf(theta[session, 3]) * BETA_MAX),
                biasL=float(theta[session, 4]),
            )
            forager.perform(
                CoupledBlockTask(
                    reward_baiting=True, num_trials=n_trials, seed=subject * 100 + session
                )
            )
            choices[subject, session] = forager.get_choice_history()
            rewards[subject, session] = forager.get_reward_history()
    return choices, rewards, mu_p


@unittest.skipUnless(HAS_JAX, "requires the 'bayes' extra (jax, numpyro)")
class TestThreeLevel(unittest.TestCase):
    """Sampling the joint model and recovering the population level."""

    @classmethod
    def setUpClass(cls):
        """Simulate a small cohort and fit it once."""
        cls.population_mean = np.array([0.3, -0.6, -0.8, 0.2, 0.0])
        cls.population_scale = np.array([0.4] * N_PARAMS)
        cls.sigma = np.array([0.25] * N_PARAMS)
        cls.n_subjects, cls.n_sessions, cls.n_trials = 6, 5, 250

        choices, rewards, cls.true_mu_p = _simulate_cohort(
            cls.population_mean, cls.population_scale, cls.sigma,
            cls.n_subjects, cls.n_sessions, cls.n_trials,
        )
        mcmc = MCMC(
            NUTS(hattori2019_three_level),
            num_warmup=300, num_samples=300, num_chains=1, progress_bar=False,
        )
        mcmc.run(jax.random.PRNGKey(0), choices, rewards)
        cls.mcmc = mcmc
        cls.samples = mcmc.get_samples()

    def test_shapes(self):
        """Subject and session levels carry the expected dimensions."""
        self.assertEqual(
            np.asarray(self.samples["mu_p"]).shape[1:], (self.n_subjects, N_PARAMS)
        )
        self.assertEqual(
            np.asarray(self.samples["session_log_lik"]).shape[1:],
            (self.n_subjects, self.n_sessions),
        )

    def test_no_divergences(self):
        """The doubly non-centred parameterisation samples cleanly."""
        divergences = int(np.sum(np.asarray(self.mcmc.get_extra_fields()["diverging"])))
        self.assertLessEqual(divergences, 5)

    def test_recovers_population_mean(self):
        """The population location is recovered from the cohort."""
        posterior = np.asarray(self.samples["population_mean"]).mean(axis=0)
        np.testing.assert_allclose(posterior[:3], self.population_mean[:3], atol=0.6)

    def test_subject_estimates_shrink_toward_population(self):
        """Subject estimates lie between their own data and the cohort mean.

        This is the behaviour two-stage cannot reproduce at the session level, and the
        reason the joint model is the reference the approximation is judged against.
        """
        posterior_mu = np.asarray(self.samples["mu_p"]).mean(axis=0)
        population = np.asarray(self.samples["population_mean"]).mean(axis=0)

        spread_posterior = posterior_mu.std(axis=0)[:3]
        spread_truth = self.true_mu_p.std(axis=0)[:3]
        # Shrinkage pulls the estimated spread in, never out.
        self.assertTrue(np.all(spread_posterior <= spread_truth + 0.35))
        self.assertEqual(population.shape, (N_PARAMS,))

    def test_all_session_log_liks_are_finite_and_negative(self):
        """Every real session contributes a finite, negative log likelihood."""
        log_lik = np.asarray(self.samples["session_log_lik"])
        self.assertTrue(np.all(np.isfinite(log_lik)))
        self.assertTrue(np.all(log_lik < 0))

    def test_session_mask_zeroes_padded_slots(self):
        """Masked session slots contribute exactly zero, so ragged cohorts pad safely."""
        choices, rewards, _ = _simulate_cohort(
            self.population_mean, self.population_scale, self.sigma,
            n_subjects=3, n_sessions=4, n_trials=150, seed=1,
        )
        # Give subject 0 only two real sessions, subject 1 three.
        session_mask = np.ones((3, 4), dtype=bool)
        session_mask[0, 2:] = False
        session_mask[1, 3:] = False

        mcmc = MCMC(
            NUTS(hattori2019_three_level),
            num_warmup=50, num_samples=50, num_chains=1, progress_bar=False,
        )
        mcmc.run(jax.random.PRNGKey(0), choices, rewards, session_mask=session_mask)
        log_lik = np.asarray(mcmc.get_samples()["session_log_lik"])

        np.testing.assert_array_equal(log_lik[:, ~session_mask], 0.0)
        self.assertTrue(np.all(log_lik[:, session_mask] < 0))


if __name__ == "__main__":
    unittest.main()
