"""Equation-level tests for published external-bandit baselines."""

import unittest

import numpy as np
from scipy.special import expit

from aind_dynamic_foraging_models.generative_model import (
    ForagerBeronRFLR,
    ForagerEcksteinBI,
    ForagerEcksteinRL,
    ForagerGrossmanMetaLearning,
    ForagerLebedevaPR,
    ForagerMillerRHG,
    ForagerRLCK,
    ForagerZidHistoryKernel,
    filter_findling_weber_session,
    findling_weber_parameter_grid,
    fit_findling_weber_map,
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

        np.testing.assert_allclose(agent.value, np.array([1.0, 1.0, 0.5, 0.7, 0.35]))
        np.testing.assert_allclose(agent.state_history_kernel, np.array([0, 0, 0.5, 0, 0.5]))
        np.testing.assert_allclose(
            agent.choice_prob,
            np.array(
                [
                    [0.5, expit(1.2), expit(0.7), 1.0 - expit(0.6)],
                    [0.5, 1.0 - expit(1.2), 1.0 - expit(0.7), expit(0.6)],
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

        agent._reset()
        agent.q_value[:, 0] = [-np.inf, np.inf]
        _, choice_prob = agent.act(None)
        np.testing.assert_allclose(choice_prob, [0.5, 0.5])

    def test_lebedeva_pr_equations(self):
        agent = ForagerLebedevaPR(seed=0)
        agent.set_params(
            perseveration_learning_rate=0.5,
            reward_learning_rate=0.25,
            perseveration_weight=2.0,
            reward_weight=4.0,
            choice_bias=0.2,
        )
        agent.perform_closed_loop(np.array([1, 1, 0]), np.array([1.0, 0.0, 1.0]))

        np.testing.assert_allclose(agent.perseveration, [0.0, 1.0, 1.5, -0.25])
        np.testing.assert_allclose(agent.reward_seeking, [0.0, 1.0, -0.25, -1.1875])
        np.testing.assert_allclose(
            agent.choice_prob[1],
            [expit(0.2), expit(2.2), expit(1.45)],
        )

    def test_beron_rflr_equations(self):
        agent = ForagerBeronRFLR(seed=0)
        agent.set_params(
            choice_history_weight=0.7,
            reward_evidence_weight=2.0,
            evidence_time_constant=2.0,
        )
        agent.perform_closed_loop(np.array([1, 0, 1]), np.array([1.0, 0.0, 1.0]))

        decay = np.exp(-0.5)
        np.testing.assert_allclose(
            agent.reward_evidence,
            [0.0, 2.0, 2.0 * decay, 2.0 * decay**2 + 2.0],
        )
        np.testing.assert_allclose(
            agent.choice_prob[1],
            [0.5, expit(2.7), expit(2.0 * decay - 0.7)],
        )

    def test_miller_rhg_equations(self):
        agent = ForagerMillerRHG(seed=0)
        agent.set_params(
            reward_weight=1.0,
            habit_weight=0.5,
            gambler_fallacy_weight=-0.25,
            reward_retention_logit=0.0,
            habit_retention_logit=0.0,
            gambler_fallacy_retention_logit=0.0,
            choice_bias=0.1,
        )
        agent.perform_closed_loop(np.array([1, 1, 0]), np.array([1.0, 0.0, 0.0]))

        np.testing.assert_allclose(agent.reward_seeking, [0.0, 0.5, -0.25, 0.375])
        np.testing.assert_allclose(agent.habit, [0.0, 0.5, 0.75, -0.125])
        np.testing.assert_allclose(agent.gambler_fallacy, [0.0, 0.0, 1.0, -0.5])
        np.testing.assert_allclose(
            agent.choice_prob[1],
            [expit(0.2), expit(1.7), expit(-0.05)],
        )

    def test_eckstein_rl_equations(self):
        agent = ForagerEcksteinRL(seed=0)
        agent.set_params(
            positive_learning_rate=0.5,
            negative_learning_rate=0.25,
            softmax_inverse_temperature=2.0,
            perseveration_bonus=0.2,
        )
        agent.perform_closed_loop(np.array([1, 1, 0]), np.array([1.0, 0.0, 0.0]))

        np.testing.assert_allclose(
            agent.q_value,
            np.array([[0.5, 0.25, 0.4375, 0.328125], [0.5, 0.75, 0.5625, 0.671875]]),
        )
        squash = lambda value: 0.0001 + 0.9998 * expit(value)
        np.testing.assert_allclose(
            agent.choice_prob[1],
            [squash(0.0), squash(1.4), squash(0.65)],
        )

    def test_eckstein_bi_equations(self):
        agent = ForagerEcksteinBI(seed=0)
        agent.set_params(
            subjective_switch_probability=0.1,
            subjective_reward_probability=0.8,
            softmax_inverse_temperature=2.0,
            perseveration_bonus=0.05,
        )
        agent.perform_closed_loop(np.array([1, 1, 0]), np.array([1.0, 0.0, 1.0]))

        epsilon = agent.incorrect_choice_reward_probability
        posterior_1 = 0.8 / (0.8 + epsilon)
        belief_1 = 0.9 * posterior_1 + 0.1 * (1.0 - posterior_1)
        posterior_2 = ((1.0 - 0.8) * belief_1) / (
            (1.0 - 0.8) * belief_1 + (1.0 - epsilon) * (1.0 - belief_1)
        )
        belief_2 = 0.9 * posterior_2 + 0.1 * (1.0 - posterior_2)
        squash = lambda value: 0.0001 + 0.9998 * expit(value)
        np.testing.assert_allclose(agent.probability_right_correct[:3], [0.5, belief_1, belief_2])
        np.testing.assert_allclose(
            agent.choice_prob[1],
            [
                0.5,
                squash(2.0 * (2.0 * (belief_1 + 0.05) - 1.0)),
                squash(2.0 * (2.0 * (belief_2 + 0.05) - 1.0)),
            ],
        )

    def test_findling_grid_matches_released_sobol_prefix(self):
        grid = findling_weber_parameter_grid()
        self.assertEqual(grid.shape, (1000, 2))
        np.testing.assert_allclose(
            grid[:5],
            [[0.5, 0.5], [0.75, 0.25], [0.25, 0.75], [0.375, 0.375], [0.875, 0.875]],
        )
        np.testing.assert_allclose(grid.min(axis=0), [1 / 512, 1 / 1024])
        np.testing.assert_allclose(grid.max(axis=0), [1023 / 1024, 1023 / 1024])

    def test_findling_zero_noise_filter_is_exact_and_seed_independent(self):
        parameters = np.array([[1.0, 0.0]])
        choices = np.array([1, 1, 0])
        rewards = np.array([1, 0, 1])
        first = filter_findling_weber_session(choices, rewards, parameters, n_particles=2, seed=1)
        second = filter_findling_weber_session(
            choices, rewards, parameters, n_particles=8, seed=999
        )
        expected_probability_right = np.array([[0.5, 2 / 3, 0.5]])
        np.testing.assert_allclose(first.probability_right, expected_probability_right)
        np.testing.assert_allclose(second.probability_right, expected_probability_right)
        expected_log_likelihood = np.log(0.5) + np.log(2 / 3) + np.log(0.5)
        np.testing.assert_allclose(first.log_likelihood, [expected_log_likelihood])
        np.testing.assert_allclose(second.log_likelihood, [expected_log_likelihood])

    def test_findling_map_fit_is_deterministic(self):
        choices = [np.array([1, 1, 0, 1]), np.array([0, 0, 1])]
        rewards = [np.array([1, 0, 1, 1]), np.array([1, 1, 0])]
        grid = np.array([[1.0, 0.0], [0.5, 0.5], [0.25, 1.0]])
        first = fit_findling_weber_map(choices, rewards, parameters=grid, n_particles=4, seed=7)
        second = fit_findling_weber_map(choices, rewards, parameters=grid, n_particles=4, seed=7)
        np.testing.assert_allclose(first[0], second[0])
        np.testing.assert_allclose(first[1], second[1])
        self.assertEqual(first[2], second[2])


if __name__ == "__main__":
    unittest.main(verbosity=2)
