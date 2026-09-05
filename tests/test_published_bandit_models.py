"""Equation-level tests for published external-bandit baselines."""

import unittest

import numpy as np
from scipy.special import expit

from aind_dynamic_foraging_models.generative_model import (
    ForagerGrossmanMetaLearning,
    ForagerRLCK,
    ForagerZidHistoryKernel,
)


class TestPublishedBanditModels(unittest.TestCase):
    """Check each implementation against a hand-computed short trajectory."""

    def test_rlck_equations(self):
        agent = ForagerRLCK(seed=0)
        agent.set_params(
            learn_rate=0.5,
            choice_kernel_step_size=0.5,
            softmax_inverse_temperature=2.0,
            choice_kernel_inverse_temperature=1.0,
        )
        agent.perform_closed_loop(np.array([0, 0, 1]), np.array([1.0, 0.0, 1.0]))

        np.testing.assert_allclose(
            agent.q_value,
            np.array([[0.0, 0.5, 0.25, 0.25], [0.0, 0.0, 0.0, 0.5]]),
        )
        np.testing.assert_allclose(
            agent.choice_kernel,
            np.array([[0.0, 0.5, 0.75, 0.375], [0.0, 0.0, 0.0, 0.5]]),
        )
        np.testing.assert_allclose(
            agent.choice_prob,
            np.array(
                [
                    [0.5, expit(1.5), expit(1.25)],
                    [0.5, 1.0 - expit(1.5), 1.0 - expit(1.25)],
                ]
            ),
        )

    def test_zid_history_kernel_foraging_equations(self):
        agent = ForagerZidHistoryKernel(seed=0)
        agent.set_params(
            learn_rate=0.5,
            threshold=0.4,
            choice_kernel_step_size=0.5,
            softmax_inverse_temperature=2.0,
            choice_kernel_inverse_temperature=1.0,
        )
        agent.perform_closed_loop(
            np.array([0, 0, 1, 1]),
            np.array([1.0, 0.0, 1.0, 0.0]),
        )

        np.testing.assert_allclose(agent.value, np.array([1.0, 0.7, 0.35, 0.7, 0.35]))
        np.testing.assert_allclose(agent.state_history_kernel, np.array([0, 0, 0.5, 0, 0.5]))
        np.testing.assert_allclose(
            agent.choice_prob,
            np.array(
                [
                    [0.5, expit(0.6), expit(0.4), 1.0 - expit(0.6)],
                    [0.5, 1.0 - expit(0.6), 1.0 - expit(0.4), expit(0.6)],
                ]
            ),
        )

    def test_grossman_meta_learning_equations(self):
        agent = ForagerGrossmanMetaLearning(seed=0)
        agent.set_params(
            learn_rate_rew=0.5,
            learn_rate_unrew=0.2,
            forgetting_factor=0.8,
            expected_uncertainty_step_size=0.2,
            negative_learning_rate_step_size=0.5,
            choice_bias=0.0,
            softmax_inverse_temperature=2.0,
        )
        agent.perform_closed_loop(np.array([0, 0, 1]), np.array([1.0, 0.0, 0.0]))

        np.testing.assert_allclose(
            agent.q_value,
            np.array([[0.0, 0.5, 0.36, 0.288], [0.0, 0.0, 0.0, 0.0]]),
        )
        np.testing.assert_allclose(agent.expected_uncertainty, np.array([0, 0.2, 0.26, 0.208]))
        np.testing.assert_allclose(agent.unexpected_uncertainty, np.array([1.0, 0.3, -0.26]))
        np.testing.assert_allclose(agent.negative_learning_rate, np.array([0.2, 0.2, 0.35, 0.35]))
        np.testing.assert_allclose(
            agent.choice_prob,
            np.array(
                [
                    [0.5, expit(1.0), expit(0.72)],
                    [0.5, 1.0 - expit(1.0), 1.0 - expit(0.72)],
                ]
            ),
        )

    def test_grossman_divergent_proposal_does_not_crash_likelihood(self):
        agent = ForagerGrossmanMetaLearning(seed=0)
        agent.set_params(
            learn_rate_rew=1.0,
            learn_rate_unrew=1.0,
            forgetting_factor=1.0,
            expected_uncertainty_step_size=1.0,
            negative_learning_rate_step_size=1.0,
            choice_bias=0.0,
            softmax_inverse_temperature=10.0,
        )
        choices = np.tile(np.array([0, 1]), 1000)
        rewards = np.tile(np.array([1.0, 0.0, 0.0, 1.0]), 500)
        agent.perform_closed_loop(choices, rewards)
        self.assertTrue(np.isfinite(agent.choice_prob).all())


if __name__ == "__main__":
    unittest.main(verbosity=2)
