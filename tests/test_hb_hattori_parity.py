"""Parity between the JAX Hattori2019 likelihood and the numpy forager.

This is the correctness anchor for the hierarchical Bayesian subpackage: the JAX
dynamics are written independently of ``generative_model``, so agreeing with it to
floating-point tolerance is real evidence that both are right.
"""

import unittest

import numpy as np
from aind_behavior_gym.dynamic_foraging.task import CoupledBlockTask

from aind_dynamic_foraging_models.generative_model import ForagerCollection

try:
    from aind_dynamic_foraging_models.hierarchical_bayes import (
        hattori2019_choice_prob,
        hattori2019_log_likelihood,
    )

    HAS_JAX = True
except ImportError:  # pragma: no cover - exercised only without the bayes extra
    HAS_JAX = False


PARAM_NAMES = [
    "learn_rate_rew",
    "learn_rate_unrew",
    "forget_rate_unchosen",
    "softmax_inverse_temperature",
    "bias_l",
]


def _simulate_session(params, n_trials=300, seed=42):
    """Run the numpy forager generatively and return its history and choice probs."""
    forager = ForagerCollection().get_preset_forager("Hattori2019", seed=seed)
    forager.set_params(
        learn_rate_rew=params["learn_rate_rew"],
        learn_rate_unrew=params["learn_rate_unrew"],
        forget_rate_unchosen=params["forget_rate_unchosen"],
        softmax_inverse_temperature=params["softmax_inverse_temperature"],
        biasL=params["bias_l"],
    )
    task = CoupledBlockTask(reward_baiting=True, num_trials=n_trials, seed=seed)
    forager.perform(task)

    choice_history = forager.get_choice_history()
    reward_history = forager.get_reward_history()

    # Teacher-forced replay is what the likelihood models, so score against that.
    forager.perform_closed_loop(choice_history, reward_history)
    return choice_history, reward_history, forager.choice_prob


@unittest.skipUnless(HAS_JAX, "requires the 'bayes' extra (jax)")
class TestHattoriParity(unittest.TestCase):
    """The JAX likelihood must reproduce the numpy forager exactly."""

    def test_choice_prob_matches_numpy_forager(self):
        """Per-trial choice probabilities agree across many parameter settings."""
        rng = np.random.default_rng(0)

        for trial_index in range(10):
            params = {
                "learn_rate_rew": float(rng.uniform(0.0, 1.0)),
                "learn_rate_unrew": float(rng.uniform(0.0, 1.0)),
                "forget_rate_unchosen": float(rng.uniform(0.0, 1.0)),
                "softmax_inverse_temperature": float(rng.uniform(0.1, 15.0)),
                "bias_l": float(rng.uniform(-2.0, 2.0)),
            }
            with self.subTest(case=trial_index, **params):
                choices, rewards, expected = _simulate_session(params, seed=trial_index)
                actual = hattori2019_choice_prob(choices, rewards, **params)
                np.testing.assert_allclose(np.asarray(actual), expected, rtol=1e-5, atol=1e-6)

    def test_boundary_parameters(self):
        """Parameter values at the edges of their ranges still agree."""
        edge_cases = [
            dict(zip(PARAM_NAMES, [0.0, 0.0, 0.0, 1.0, 0.0])),  # no learning, no forgetting
            dict(zip(PARAM_NAMES, [1.0, 1.0, 1.0, 1.0, 0.0])),  # full learning and forgetting
            dict(zip(PARAM_NAMES, [0.5, 0.1, 0.2, 0.0, 0.0])),  # zero temperature: uniform
            dict(zip(PARAM_NAMES, [0.5, 0.1, 0.2, 5.0, 3.0])),  # strong left bias
            dict(zip(PARAM_NAMES, [0.5, 0.1, 0.2, 5.0, -3.0])),  # strong right bias
        ]
        for case_index, params in enumerate(edge_cases):
            with self.subTest(case=case_index, **params):
                choices, rewards, expected = _simulate_session(params, seed=case_index)
                actual = hattori2019_choice_prob(choices, rewards, **params)
                np.testing.assert_allclose(np.asarray(actual), expected, rtol=1e-5, atol=1e-6)

    def test_log_likelihood_matches_manual_sum(self):
        """The log likelihood equals the summed log probability of observed choices."""
        params = dict(zip(PARAM_NAMES, [0.6, 0.2, 0.3, 8.0, 0.1]))
        choices, rewards, expected_prob = _simulate_session(params)

        expected = np.sum(np.log(expected_prob[choices.astype(int), np.arange(len(choices))]))
        actual = hattori2019_log_likelihood(choices, rewards, **params)
        np.testing.assert_allclose(float(actual), expected, rtol=1e-5)

    def test_valid_mask_excludes_trials(self):
        """Masked-out trials contribute nothing to the log likelihood."""
        params = dict(zip(PARAM_NAMES, [0.6, 0.2, 0.3, 8.0, 0.1]))
        choices, rewards, _ = _simulate_session(params)

        mask = np.ones(len(choices), dtype=bool)
        mask[::2] = False

        full = hattori2019_log_likelihood(choices, rewards, **params)
        masked = hattori2019_log_likelihood(choices, rewards, valid_mask=mask, **params)
        self.assertLess(float(masked), 0.0)
        self.assertGreater(float(masked), float(full))


if __name__ == "__main__":
    unittest.main()
