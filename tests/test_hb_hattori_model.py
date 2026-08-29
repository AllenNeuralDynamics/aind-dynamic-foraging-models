"""The NumPyro two-level Hattori2019 model: transforms, sampling, and recovery."""

import unittest

import numpy as np
from aind_behavior_gym.dynamic_foraging.task import CoupledBlockTask
from scipy.stats import norm

from aind_dynamic_foraging_models.generative_model import ForagerCollection

try:
    import jax
    import jax.numpy as jnp
    from numpyro.infer import MCMC, NUTS

    from aind_dynamic_foraging_models.hierarchical_bayes.model import (
        HATTORI2019_PARAMS,
        hattori2019_session_params,
        hattori2019_two_level,
    )

    HAS_JAX = True
except ImportError:  # pragma: no cover - exercised only without the bayes extra
    HAS_JAX = False

BETA_MAX = 10.0


def _to_bounded(theta, beta_max=BETA_MAX):
    """Reference transform in numpy, independent of the JAX implementation."""
    return {
        "learn_rate_rew": norm.cdf(theta[..., 0]),
        "learn_rate_unrew": norm.cdf(theta[..., 1]),
        "forget_rate_unchosen": norm.cdf(theta[..., 2]),
        "softmax_inverse_temperature": norm.cdf(theta[..., 3]) * beta_max,
        "bias_l": theta[..., 4],
    }


def _simulate_subject(theta, n_trials=300):
    """Simulate one session per row of ``theta`` with the numpy forager."""
    bounded = _to_bounded(theta)
    choices, rewards = [], []
    for session in range(theta.shape[0]):
        forager = ForagerCollection().get_preset_forager("Hattori2019", seed=session)
        forager.set_params(
            learn_rate_rew=float(bounded["learn_rate_rew"][session]),
            learn_rate_unrew=float(bounded["learn_rate_unrew"][session]),
            forget_rate_unchosen=float(bounded["forget_rate_unchosen"][session]),
            softmax_inverse_temperature=float(
                bounded["softmax_inverse_temperature"][session]
            ),
            biasL=float(bounded["bias_l"][session]),
        )
        forager.perform(CoupledBlockTask(reward_baiting=True, num_trials=n_trials, seed=session))
        choices.append(forager.get_choice_history())
        rewards.append(forager.get_reward_history())
    return np.stack(choices), np.stack(rewards)


@unittest.skipUnless(HAS_JAX, "requires the 'bayes' extra (jax, numpyro)")
class TestHattoriTransforms(unittest.TestCase):
    """The unconstrained-to-bounded mapping."""

    def test_matches_numpy_reference(self):
        """The JAX transform agrees with an independent numpy implementation."""
        rng = np.random.default_rng(0)
        theta = rng.standard_normal((20, len(HATTORI2019_PARAMS)))
        actual = hattori2019_session_params(jnp.asarray(theta), beta_max=BETA_MAX)
        expected = _to_bounded(theta)
        for name in HATTORI2019_PARAMS:
            np.testing.assert_allclose(
                np.asarray(actual[name]), expected[name], rtol=1e-5, atol=1e-6
            )

    def test_zero_maps_to_range_midpoints(self):
        """An unconstrained zero sits at the middle of each bounded range."""
        params = hattori2019_session_params(jnp.zeros(len(HATTORI2019_PARAMS)), beta_max=BETA_MAX)
        self.assertAlmostEqual(float(params["learn_rate_rew"]), 0.5, places=5)
        self.assertAlmostEqual(float(params["forget_rate_unchosen"]), 0.5, places=5)
        self.assertAlmostEqual(float(params["softmax_inverse_temperature"]), BETA_MAX / 2, places=5)
        self.assertAlmostEqual(float(params["bias_l"]), 0.0, places=5)

    def test_standard_normal_becomes_uniform(self):
        """A standard normal on the unconstrained scale is a uniform bounded prior.

        This is what makes the published model's "non-informative (uniform)" priors and its
        ``mu_p ~ normal(0, 1)`` code the same statement rather than a contradiction.

        Checked by comparing empirical to uniform quantiles with a tolerance, rather than by
        a hypothesis test: a fixed p-value threshold rejects a correct transform a few
        percent of the time purely by sampling luck.
        """
        rng = np.random.default_rng(1)
        theta = rng.standard_normal((20000, len(HATTORI2019_PARAMS)))
        learn_rate = np.asarray(hattori2019_session_params(jnp.asarray(theta))["learn_rate_rew"])

        probs = np.linspace(0.05, 0.95, 19)
        np.testing.assert_allclose(np.quantile(learn_rate, probs), probs, atol=0.02)


@unittest.skipUnless(HAS_JAX, "requires the 'bayes' extra (jax, numpyro)")
class TestHattoriTwoLevelFit(unittest.TestCase):
    """Sampling the two-level model and recovering the subject-level parameters."""

    @classmethod
    def setUpClass(cls):
        """Simulate one subject and fit it once, reusing the result across tests."""
        rng = np.random.default_rng(0)
        cls.true_mu_p = np.array([0.3, -0.6, -0.8, 0.2, 0.0])
        theta = cls.true_mu_p + 0.25 * rng.standard_normal((8, len(HATTORI2019_PARAMS)))
        choices, rewards = _simulate_subject(theta, n_trials=300)

        mcmc = MCMC(
            NUTS(hattori2019_two_level),
            num_warmup=300,
            num_samples=300,
            num_chains=1,
            progress_bar=False,
        )
        mcmc.run(jax.random.PRNGKey(0), choices, rewards, beta_max=BETA_MAX)
        cls.mcmc = mcmc
        cls.samples = mcmc.get_samples()

    def test_no_divergences(self):
        """The non-centred parameterisation samples without divergences."""
        divergences = int(np.sum(np.asarray(self.mcmc.get_extra_fields()["diverging"])))
        self.assertEqual(divergences, 0)

    def test_recovers_subject_level_parameters(self):
        """Subject-level posterior means land near the generating values."""
        truth = _to_bounded(self.true_mu_p)
        for name in ("learn_rate_rew", "learn_rate_unrew", "forget_rate_unchosen"):
            with self.subTest(param=name):
                posterior_mean = float(self.samples[f"subject_{name}"].mean())
                self.assertAlmostEqual(posterior_mean, float(truth[name]), delta=0.2)

        beta = float(self.samples["subject_softmax_inverse_temperature"].mean())
        self.assertAlmostEqual(beta, float(truth["softmax_inverse_temperature"]), delta=2.0)

    def test_session_parameters_stay_in_range(self):
        """Every posterior draw of every session parameter respects its bounds."""
        for name in ("learn_rate_rew", "learn_rate_unrew", "forget_rate_unchosen"):
            with self.subTest(param=name):
                values = np.asarray(self.samples[name])
                self.assertTrue(np.all((values >= 0.0) & (values <= 1.0)))
        beta = np.asarray(self.samples["softmax_inverse_temperature"])
        self.assertTrue(np.all((beta >= 0.0) & (beta <= BETA_MAX)))

    def test_session_log_lik_shape(self):
        """One log likelihood per session per draw, all finite and negative."""
        log_lik = np.asarray(self.samples["session_log_lik"])
        self.assertEqual(log_lik.shape, (300, 8))
        self.assertTrue(np.all(np.isfinite(log_lik)))
        self.assertTrue(np.all(log_lik < 0))


if __name__ == "__main__":
    unittest.main()
