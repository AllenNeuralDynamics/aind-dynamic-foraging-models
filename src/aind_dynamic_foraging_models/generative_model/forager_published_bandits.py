"""Published behavioral baselines for external two-arm bandit datasets."""

from __future__ import annotations

import numpy as np
from scipy.special import expit, softmax

from .base import DynamicForagingAgentMLEBase
from .learn_functions import learn_choice_kernel
from .params.published_bandit_params import (
    generate_grossman_meta_learning_params,
    generate_rlck_params,
    generate_zid_foraging_params,
)


class ForagerRLCK(DynamicForagingAgentMLEBase):
    """Four-parameter RL plus choice-kernel model used by Chen and Zid.

    This keeps separate inverse-temperature parameters for learned action value
    and choice history, matching the published equations directly.
    """

    def __init__(self, params: dict = {}, **kwargs):
        self.agent_kwargs = {}
        super().__init__(agent_kwargs=self.agent_kwargs, params=params, **kwargs)

    def _get_params_model(self, _agent_kwargs):
        return generate_rlck_params()

    def get_agent_alias(self):
        return "RLCK"

    def _reset(self):
        super()._reset()
        self.q_value = np.zeros((self.n_actions, self.n_trials + 1), dtype=float)
        self.choice_kernel = np.zeros((self.n_actions, self.n_trials + 1), dtype=float)

    def act(self, _observation):
        decision_value = (
            float(self.params.softmax_inverse_temperature) * self.q_value[:, self.trial]
            + float(self.params.choice_kernel_inverse_temperature)
            * self.choice_kernel[:, self.trial]
        )
        choice_prob = softmax(decision_value)
        choice = self.rng.choice(self.n_actions, p=choice_prob)
        return choice, choice_prob

    def learn(self, _observation, choice, reward, _next_observation, _done):
        previous_q = self.q_value[:, self.trial - 1]
        self.q_value[:, self.trial] = previous_q
        self.q_value[choice, self.trial] = previous_q[choice] + float(
            self.params.learn_rate
        ) * (float(reward) - previous_q[choice])
        self.choice_kernel[:, self.trial] = learn_choice_kernel(
            choice=choice,
            choice_kernel_tminus1=self.choice_kernel[:, self.trial - 1],
            choice_kernel_step_size=float(self.params.choice_kernel_step_size),
        )

    def get_latent_variables(self):
        return {
            "q_value": self.q_value.tolist(),
            "choice_kernel": self.choice_kernel.tolist(),
            "choice_prob": self.choice_prob.tolist(),
        }


class ForagerZidHistoryKernel(DynamicForagingAgentMLEBase):
    """Zid et al. history-kernel-2 compare-to-threshold foraging-RL."""

    def __init__(self, params: dict = {}, **kwargs):
        self.agent_kwargs = {}
        super().__init__(agent_kwargs=self.agent_kwargs, params=params, **kwargs)

    def _get_params_model(self, _agent_kwargs):
        return generate_zid_foraging_params()

    def get_agent_alias(self):
        return "ZidHistoryKernel2ForagingRL"

    def _reset(self):
        super()._reset()
        self.value = np.full(self.n_trials + 1, np.nan, dtype=float)
        self.value[0] = 1.0
        self.state_history_kernel = np.zeros(self.n_trials + 1, dtype=float)

    def act(self, _observation):
        if self.trial == 0:
            choice_prob = np.array([0.5, 0.5], dtype=float)
        else:
            logit = (
                float(self.params.softmax_inverse_temperature)
                * (float(self.value[self.trial]) - float(self.params.threshold))
                + float(self.params.choice_kernel_inverse_temperature)
                * float(self.state_history_kernel[self.trial])
            )
            probability_stay = float(expit(logit))
            previous_choice = int(self.choice_history[self.trial - 1])
            choice_prob = np.empty(self.n_actions, dtype=float)
            choice_prob[previous_choice] = probability_stay
            choice_prob[1 - previous_choice] = 1.0 - probability_stay
        choice = self.rng.choice(self.n_actions, p=choice_prob)
        return choice, choice_prob

    def learn(self, _observation, choice, reward, _next_observation, _done):
        switched = self.trial == 1 or choice != self.choice_history[self.trial - 2]
        previous_value = (
            float(self.params.threshold) if switched else float(self.value[self.trial - 1])
        )
        self.value[self.trial] = previous_value + float(self.params.learn_rate) * (
            float(reward) - previous_value
        )
        if switched:
            self.state_history_kernel[self.trial] = 0.0
        else:
            previous_kernel = float(self.state_history_kernel[self.trial - 1])
            self.state_history_kernel[self.trial] = previous_kernel + float(
                self.params.choice_kernel_step_size
            ) * (1.0 - previous_kernel)

    def get_latent_variables(self):
        return {
            "value": self.value.tolist(),
            "state_history_kernel": self.state_history_kernel.tolist(),
            "choice_prob": self.choice_prob.tolist(),
        }


class ForagerGrossmanMetaLearning(DynamicForagingAgentMLEBase):
    """Grossman et al. uncertainty-dependent asymmetric meta-learning model."""

    def __init__(self, params: dict = {}, **kwargs):
        self.agent_kwargs = {}
        super().__init__(agent_kwargs=self.agent_kwargs, params=params, **kwargs)

    def _get_params_model(self, _agent_kwargs):
        return generate_grossman_meta_learning_params()

    def get_agent_alias(self):
        return "GrossmanMetaLearning"

    def _reset(self):
        super()._reset()
        self.q_value = np.zeros((self.n_actions, self.n_trials + 1), dtype=float)
        self.expected_uncertainty = np.zeros(self.n_trials + 1, dtype=float)
        self.unexpected_uncertainty = np.full(self.n_trials, np.nan, dtype=float)
        self.negative_learning_rate = np.full(self.n_trials + 1, np.nan, dtype=float)
        self.negative_learning_rate[0] = float(self.params.learn_rate_unrew)

    def act(self, _observation):
        q_left, q_right = self.q_value[:, self.trial]
        right_logit = float(self.params.softmax_inverse_temperature) * (
            float(q_right) - float(q_left) + float(self.params.choice_bias)
        )
        if not np.isfinite(right_logit):
            # Differential evolution can propose parameters whose long-horizon
            # meta-learning state diverges. Give that invalid tail chance
            # likelihood so the optimizer can reject it instead of crashing.
            probability_right = 0.5
        else:
            probability_right = float(expit(right_logit))
        choice_prob = np.array([1.0 - probability_right, probability_right], dtype=float)
        choice = self.rng.choice(self.n_actions, p=choice_prob)
        return choice, choice_prob

    def learn(self, _observation, choice, reward, _next_observation, _done):
        previous_q = self.q_value[:, self.trial - 1]
        prediction_error = float(reward) - float(previous_q[choice])
        previous_expected = float(self.expected_uncertainty[self.trial - 1])
        unexpected = abs(prediction_error) - previous_expected
        self.unexpected_uncertainty[self.trial - 1] = unexpected

        previous_negative_rate = float(self.negative_learning_rate[self.trial - 1])
        if prediction_error < 0.0:
            integration = float(self.params.negative_learning_rate_step_size)
            negative_rate = integration * (
                unexpected + float(self.params.learn_rate_unrew)
            ) + (1.0 - integration) * previous_negative_rate
            negative_rate = max(0.0, negative_rate)
            learning_rate = negative_rate
        else:
            negative_rate = previous_negative_rate
            learning_rate = float(self.params.learn_rate_rew)

        self.q_value[:, self.trial] = previous_q
        self.q_value[choice, self.trial] = float(previous_q[choice]) + learning_rate * (
            prediction_error * (1.0 - previous_expected)
        )
        self.q_value[1 - choice, self.trial] = float(self.params.forgetting_factor) * float(
            previous_q[1 - choice]
        )
        self.expected_uncertainty[self.trial] = previous_expected + float(
            self.params.expected_uncertainty_step_size
        ) * unexpected
        self.negative_learning_rate[self.trial] = negative_rate

    def get_latent_variables(self):
        return {
            "q_value": self.q_value.tolist(),
            "expected_uncertainty": self.expected_uncertainty.tolist(),
            "unexpected_uncertainty": self.unexpected_uncertainty.tolist(),
            "negative_learning_rate": self.negative_learning_rate.tolist(),
            "choice_prob": self.choice_prob.tolist(),
        }
