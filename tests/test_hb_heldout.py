"""Held-out adaptation and posterior-predictive scoring."""

import unittest

import numpy as np
from aind_behavior_gym.dynamic_foraging.task import CoupledBlockTask
from scipy.stats import norm

from aind_dynamic_foraging_models.generative_model import ForagerCollection

try:
    import jax

    from aind_dynamic_foraging_models.hierarchical_bayes.heldout import (
        fit_adaptation,
        pointwise_log_predictive_density,
        posterior_predictive_choice_prob,
    )

    HAS_JAX = True
except ImportError:  # pragma: no cover - exercised only without the bayes extra
    HAS_JAX = False

BETA_MAX = 10.0
N_PARAMS = 5

POPULATION = {
    "population_mean": np.array([0.3, -0.6, -0.8, 0.2, 0.0]),
    "population_scale": np.array([0.4] * N_PARAMS),
    "log_sigma_mean": np.array([-1.4] * N_PARAMS),
    "log_sigma_spread": np.array([0.3] * N_PARAMS),
}


def _simulate_sessions(theta, n_trials=250, seed_offset=0):
    """Simulate one session per row of ``theta``."""
    choices, rewards = [], []
    for i in range(theta.shape[0]):
        forager = ForagerCollection().get_preset_forager("Hattori2019", seed=seed_offset + i)
        forager.set_params(
            learn_rate_rew=float(norm.cdf(theta[i, 0])),
            learn_rate_unrew=float(norm.cdf(theta[i, 1])),
            forget_rate_unchosen=float(norm.cdf(theta[i, 2])),
            softmax_inverse_temperature=float(norm.cdf(theta[i, 3]) * BETA_MAX),
            biasL=float(theta[i, 4]),
        )
        forager.perform(
            CoupledBlockTask(reward_baiting=True, num_trials=n_trials, seed=seed_offset + i)
        )
        choices.append(forager.get_choice_history())
        rewards.append(forager.get_reward_history())
    return np.stack(choices), np.stack(rewards)


@unittest.skipUnless(HAS_JAX, "requires the 'bayes' extra (jax, numpyro)")
class TestAdaptation(unittest.TestCase):
    """Conditioning a held-out subject on k context sessions."""

    @classmethod
    def setUpClass(cls):
        """Simulate one atypical held-out subject and its context sessions."""
        rng = np.random.default_rng(0)
        # Deliberately far from the population mean, so adaptation has to move.
        cls.true_mu = np.array([1.2, -1.2, -0.8, 0.9, 0.0])
        theta = cls.true_mu + 0.2 * rng.standard_normal((10, N_PARAMS))
        cls.choices, cls.rewards = _simulate_sessions(theta, n_trials=250)

    def _adapt(self, k):
        """Fit the adaptation model on the first k context sessions."""
        return fit_adaptation(
            self.choices[:k], self.rewards[:k], POPULATION,
            rng_key=jax.random.PRNGKey(0), num_warmup=250, num_samples=250,
        )

    def test_zero_shot_returns_population_prior(self):
        """With no context the posterior is the population predictive prior."""
        samples = self._adapt(0)
        mu_p = np.asarray(samples["mu_p"])
        self.assertEqual(mu_p.shape[1], N_PARAMS)
        # Prior mean and scale, within sampling noise of 250 draws.
        np.testing.assert_allclose(
            mu_p.mean(axis=0), POPULATION["population_mean"], atol=0.25
        )
        np.testing.assert_allclose(
            mu_p.std(axis=0), POPULATION["population_scale"], atol=0.2
        )

    def test_context_moves_posterior_toward_the_subject(self):
        """More context pulls the estimate away from the cohort toward this subject."""
        far_at = {}
        for k in (0, 8):
            mu_p = np.asarray(self._adapt(k)["mu_p"]).mean(axis=0)
            far_at[k] = float(np.abs(mu_p[0] - self.true_mu[0]))
        self.assertLess(far_at[8], far_at[0])

    def test_posterior_narrows_with_more_context(self):
        """Adding context sessions reduces posterior uncertainty."""
        width_0 = float(np.asarray(self._adapt(0)["mu_p"]).std(axis=0)[0])
        width_8 = float(np.asarray(self._adapt(8)["mu_p"]).std(axis=0)[0])
        self.assertLess(width_8, width_0)


@unittest.skipUnless(HAS_JAX, "requires the 'bayes' extra (jax, numpyro)")
class TestScoring(unittest.TestCase):
    """Posterior-predictive scoring of a held-out session."""

    @classmethod
    def setUpClass(cls):
        """Adapt on context, hold out a disjoint session."""
        rng = np.random.default_rng(1)
        true_mu = np.array([0.5, -0.5, -0.9, 0.4, 0.0])
        theta = true_mu + 0.2 * rng.standard_normal((6, N_PARAMS))
        choices, rewards = _simulate_sessions(theta, n_trials=200, seed_offset=50)
        cls.context = (choices[:4], rewards[:4])
        cls.test_choices, cls.test_rewards = choices[4], rewards[4]

        cls.samples = fit_adaptation(
            *cls.context, POPULATION,
            rng_key=jax.random.PRNGKey(0), num_warmup=250, num_samples=250,
        )
        cls.choice_prob = posterior_predictive_choice_prob(
            cls.samples, cls.test_choices, cls.test_rewards, rng_key=jax.random.PRNGKey(1)
        )

    def test_choice_prob_is_a_distribution(self):
        """Probabilities are valid and normalised over the two actions."""
        self.assertEqual(self.choice_prob.shape, (2, len(self.test_choices)))
        np.testing.assert_allclose(self.choice_prob.sum(axis=0), 1.0, atol=1e-5)
        self.assertTrue(np.all(self.choice_prob >= 0))

    def test_beats_chance(self):
        """The adapted model predicts a held-out session better than a coin flip."""
        total, n = pointwise_log_predictive_density(self.choice_prob, self.test_choices)
        self.assertGreater(np.exp(total / n), 0.5)

    def test_probability_space_averaging_exceeds_log_space(self):
        """Averaging draws before the log gives a larger score than after it.

        Jensen's inequality guarantees this, and the gap is why the averaging order is
        fixed by ADR-0003: doing it in log space silently understates every model.
        """
        prob_space, n = pointwise_log_predictive_density(self.choice_prob, self.test_choices)

        from aind_dynamic_foraging_models.hierarchical_bayes.likelihood import (
            hattori2019_choice_prob,
        )
        from aind_dynamic_foraging_models.hierarchical_bayes.model import (
            hattori2019_session_params,
        )

        mu_p = np.asarray(self.samples["mu_p"])
        sigma = np.exp(np.asarray(self.samples["log_sigma"]))
        noise = np.asarray(jax.random.normal(jax.random.PRNGKey(1), mu_p.shape))
        params = hattori2019_session_params(mu_p + sigma * noise, beta_max=BETA_MAX)

        per_draw = []
        for draw in range(mu_p.shape[0]):
            probs = np.asarray(hattori2019_choice_prob(
                self.test_choices, self.test_rewards,
                learn_rate_rew=float(params["learn_rate_rew"][draw]),
                learn_rate_unrew=float(params["learn_rate_unrew"][draw]),
                forget_rate_unchosen=float(params["forget_rate_unchosen"][draw]),
                softmax_inverse_temperature=float(
                    params["softmax_inverse_temperature"][draw]
                ),
                bias_l=float(params["bias_l"][draw]),
            ))
            total, _ = pointwise_log_predictive_density(probs, self.test_choices)
            per_draw.append(total)
        log_space = float(np.mean(per_draw))

        self.assertGreater(prob_space, log_space)


if __name__ == "__main__":
    unittest.main()
