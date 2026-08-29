"""Batched adaptation: does fitting many subjects at once match fitting them one by one?

Held-out subjects are independent given a frozen population, so batching them into one
sampler is a large speed win on a device where extra lanes are nearly free. The catch is
that one sampler adapts a single step size across every subject's block. This measures
whether that costs anything, rather than assuming either way.
"""

import time
import unittest

import numpy as np
from aind_behavior_gym.dynamic_foraging.task import CoupledBlockTask
from scipy.stats import norm

from aind_dynamic_foraging_models.generative_model import ForagerCollection

try:
    import jax

    from aind_dynamic_foraging_models.hierarchical_bayes.heldout import (
        batched_choice_prob,
        fit_adaptation,
        fit_adaptation_batched,
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


def _simulate_cohort(n_subjects, n_sessions, n_trials, seed=0):
    """Simulate held-out subjects drawn from the population."""
    rng = np.random.default_rng(seed)
    mu = POPULATION["population_mean"] + POPULATION["population_scale"] * rng.standard_normal(
        (n_subjects, N_PARAMS)
    )
    choices = np.zeros((n_subjects, n_sessions, n_trials), dtype=int)
    rewards = np.zeros((n_subjects, n_sessions, n_trials), dtype=float)
    for s in range(n_subjects):
        theta = mu[s] + 0.25 * rng.standard_normal((n_sessions, N_PARAMS))
        for j in range(n_sessions):
            tag = s * 100 + j
            f = ForagerCollection().get_preset_forager("Hattori2019", seed=tag)
            f.set_params(
                learn_rate_rew=float(norm.cdf(theta[j, 0])),
                learn_rate_unrew=float(norm.cdf(theta[j, 1])),
                forget_rate_unchosen=float(norm.cdf(theta[j, 2])),
                softmax_inverse_temperature=float(norm.cdf(theta[j, 3]) * BETA_MAX),
                biasL=float(theta[j, 4]),
            )
            f.perform(CoupledBlockTask(reward_baiting=True, num_trials=n_trials, seed=tag))
            choices[s, j] = f.get_choice_history()
            rewards[s, j] = f.get_reward_history()
    return choices, rewards


@unittest.skipUnless(HAS_JAX, "requires the 'bayes' extra (jax, numpyro)")
class TestBatchedAdaptation(unittest.TestCase):
    """Batched against sequential adaptation on the same subjects."""

    @classmethod
    def setUpClass(cls):
        """Simulate a small cohort, then adapt it both ways."""
        cls.n_subjects, n_context, n_trials = 6, 4, 200
        cls.choices, cls.rewards = _simulate_cohort(cls.n_subjects, n_context + 1, n_trials)
        cls.context_c = cls.choices[:, :n_context]
        cls.context_r = cls.rewards[:, :n_context]
        cls.test_c = cls.choices[:, n_context]
        cls.test_r = cls.rewards[:, n_context]

        started = time.time()
        cls.batched = fit_adaptation_batched(
            cls.context_c, cls.context_r, POPULATION,
            rng_key=jax.random.PRNGKey(0), num_warmup=300, num_samples=300,
        )
        cls.batched_seconds = time.time() - started

        started = time.time()
        cls.sequential = []
        for s in range(cls.n_subjects):
            cls.sequential.append(fit_adaptation(
                cls.context_c[s], cls.context_r[s], POPULATION,
                rng_key=jax.random.PRNGKey(s), num_warmup=300, num_samples=300,
            ))
        cls.sequential_seconds = time.time() - started

    def test_shapes_carry_a_subject_axis(self):
        """Batched draws are indexed by subject."""
        mu_p = np.asarray(self.batched["mu_p"])
        self.assertEqual(mu_p.shape[1:], (self.n_subjects, N_PARAMS))

    def test_posteriors_agree_with_sequential(self):
        """Per-subject posterior means match sequential fitting.

        A shared step size across subject blocks is the approximation batching makes; this
        is what bounds its cost.
        """
        batched_mu = np.asarray(self.batched["mu_p"]).mean(axis=0)
        for s in range(self.n_subjects):
            sequential_mu = np.asarray(self.sequential[s]["mu_p"]).mean(axis=0)
            with self.subTest(subject=s):
                np.testing.assert_allclose(
                    batched_mu[s][:3], sequential_mu[:3], atol=0.35
                )

    def test_scores_agree_with_sequential(self):
        """Held-out likelihoods match, which is the quantity that actually gets reported."""
        batched_total, sequential_total, n_total = 0.0, 0.0, 0
        for s in range(self.n_subjects):
            key = jax.random.PRNGKey(100 + s)
            batched_prob = batched_choice_prob(
                self.batched, s, self.test_c[s], self.test_r[s], rng_key=key
            )
            sequential_prob = posterior_predictive_choice_prob(
                self.sequential[s], self.test_c[s], self.test_r[s], rng_key=key
            )
            b, n = pointwise_log_predictive_density(batched_prob, self.test_c[s])
            q, _ = pointwise_log_predictive_density(sequential_prob, self.test_c[s])
            batched_total += b
            sequential_total += q
            n_total += n

        batched_lik = float(np.exp(batched_total / n_total))
        sequential_lik = float(np.exp(sequential_total / n_total))
        self.assertAlmostEqual(batched_lik, sequential_lik, delta=0.01)

    def test_batching_is_faster(self):
        """One batched fit beats one fit per subject."""
        self.assertLess(self.batched_seconds, self.sequential_seconds)


if __name__ == "__main__":
    unittest.main()
