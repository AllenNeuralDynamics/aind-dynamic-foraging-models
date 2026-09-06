"""Published behavioral baselines for external two-arm bandit datasets."""

from __future__ import annotations

import numpy as np
from scipy.special import expit, softmax

from .base import DynamicForagingAgentMLEBase
from .learn_functions import learn_choice_kernel
from .params.published_bandit_params import (
    generate_beron_rflr_params,
    generate_eckstein_bi_params,
    generate_eckstein_rl_params,
    generate_grossman_meta_learning_params,
    generate_lebedeva_pr_params,
    generate_miller_rhg_params,
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
        self.q_value[choice, self.trial] = previous_q[choice] + float(self.params.learn_rate) * (
            float(reward) - previous_q[choice]
        )
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
            logit = float(self.params.softmax_inverse_temperature) * (
                float(self.value[self.trial]) - float(self.params.threshold)
            ) + float(self.params.choice_kernel_inverse_temperature) * float(
                self.state_history_kernel[self.trial]
            )
            probability_stay = float(expit(logit))
            previous_choice = int(self.choice_history[self.trial - 1])
            choice_prob = np.empty(self.n_actions, dtype=float)
            choice_prob[previous_choice] = probability_stay
            choice_prob[1 - previous_choice] = 1.0 - probability_stay
        choice = self.rng.choice(self.n_actions, p=choice_prob)
        return choice, choice_prob

    def learn(self, _observation, choice, reward, _next_observation, _done):
        first_update = self.trial == 1
        switched = not first_update and choice != self.choice_history[self.trial - 2]
        previous_value = (
            float(self.params.threshold) if switched else float(self.value[self.trial - 1])
        )
        self.value[self.trial] = previous_value + float(self.params.learn_rate) * (
            float(reward) - previous_value
        )
        if first_update or switched:
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
            negative_rate = (
                integration * (unexpected + float(self.params.learn_rate_unrew))
                + (1.0 - integration) * previous_negative_rate
            )
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
        self.expected_uncertainty[self.trial] = (
            previous_expected + float(self.params.expected_uncertainty_step_size) * unexpected
        )
        self.negative_learning_rate[self.trial] = negative_rate

    def get_latent_variables(self):
        return {
            "q_value": self.q_value.tolist(),
            "expected_uncertainty": self.expected_uncertainty.tolist(),
            "unexpected_uncertainty": self.unexpected_uncertainty.tolist(),
            "negative_learning_rate": self.negative_learning_rate.tolist(),
            "choice_prob": self.choice_prob.tolist(),
        }


class ForagerLebedevaPR(DynamicForagingAgentMLEBase):
    """Lebedeva et al. perseveration/reward-learning (PR) model."""

    def __init__(self, params: dict = {}, **kwargs):
        self.agent_kwargs = {}
        super().__init__(agent_kwargs=self.agent_kwargs, params=params, **kwargs)

    def _get_params_model(self, _agent_kwargs):
        return generate_lebedeva_pr_params()

    def get_agent_alias(self):
        return "LebedevaPR"

    def _reset(self):
        super()._reset()
        self.perseveration = np.zeros(self.n_trials + 1, dtype=float)
        self.reward_seeking = np.zeros(self.n_trials + 1, dtype=float)

    def act(self, _observation):
        right_logit = (
            float(self.perseveration[self.trial])
            + float(self.reward_seeking[self.trial])
            + float(self.params.choice_bias)
        )
        probability_right = float(expit(right_logit))
        choice_prob = np.array([1.0 - probability_right, probability_right], dtype=float)
        choice = self.rng.choice(self.n_actions, p=choice_prob)
        return choice, choice_prob

    def learn(self, _observation, choice, reward, _next_observation, _done):
        signed_choice = 2.0 * float(choice) - 1.0
        signed_feedback = 2.0 * float(reward) - 1.0
        previous_index = self.trial - 1

        perseveration_rate = float(self.params.perseveration_learning_rate)
        reward_rate = float(self.params.reward_learning_rate)
        self.perseveration[self.trial] = (1.0 - perseveration_rate) * self.perseveration[
            previous_index
        ] + perseveration_rate * float(self.params.perseveration_weight) * signed_choice
        self.reward_seeking[self.trial] = (1.0 - reward_rate) * self.reward_seeking[
            previous_index
        ] + reward_rate * float(self.params.reward_weight) * signed_choice * signed_feedback

    def get_latent_variables(self):
        return {
            "perseveration": self.perseveration.tolist(),
            "reward_seeking": self.reward_seeking.tolist(),
            "choice_prob": self.choice_prob.tolist(),
        }


class ForagerBeronRFLR(DynamicForagingAgentMLEBase):
    """Beron et al. recursively formulated logistic regression (RFLR)."""

    def __init__(self, params: dict = {}, **kwargs):
        self.agent_kwargs = {}
        super().__init__(agent_kwargs=self.agent_kwargs, params=params, **kwargs)

    def _get_params_model(self, _agent_kwargs):
        return generate_beron_rflr_params()

    def get_agent_alias(self):
        return "BeronRFLR"

    def _reset(self):
        super()._reset()
        self.reward_evidence = np.zeros(self.n_trials + 1, dtype=float)

    def act(self, _observation):
        if self.trial == 0:
            choice_prob = np.array([0.5, 0.5], dtype=float)
        else:
            previous_choice = 2.0 * float(self.choice_history[self.trial - 1]) - 1.0
            right_logit = (
                float(self.reward_evidence[self.trial])
                + float(self.params.choice_history_weight) * previous_choice
            )
            probability_right = float(expit(right_logit))
            choice_prob = np.array([1.0 - probability_right, probability_right], dtype=float)
        choice = self.rng.choice(self.n_actions, p=choice_prob)
        return choice, choice_prob

    def learn(self, _observation, choice, reward, _next_observation, _done):
        signed_choice = 2.0 * float(choice) - 1.0
        decay = np.exp(-1.0 / float(self.params.evidence_time_constant))
        self.reward_evidence[self.trial] = (
            decay * self.reward_evidence[self.trial - 1]
            + float(self.params.reward_evidence_weight) * float(reward) * signed_choice
        )

    def get_latent_variables(self):
        return {
            "reward_evidence": self.reward_evidence.tolist(),
            "choice_prob": self.choice_prob.tolist(),
        }


class ForagerMillerRHG(DynamicForagingAgentMLEBase):
    """Miller et al. reward-seeking/habit/gambler-fallacy (RHG) model."""

    def __init__(self, params: dict = {}, **kwargs):
        self.agent_kwargs = {}
        super().__init__(agent_kwargs=self.agent_kwargs, params=params, **kwargs)

    def _get_params_model(self, _agent_kwargs):
        return generate_miller_rhg_params()

    def get_agent_alias(self):
        return "MillerRHG"

    def _reset(self):
        super()._reset()
        self.reward_seeking = np.zeros(self.n_trials + 1, dtype=float)
        self.habit = np.zeros(self.n_trials + 1, dtype=float)
        self.gambler_fallacy = np.zeros(self.n_trials + 1, dtype=float)

    @staticmethod
    def _retention(logit: float) -> float:
        return float(expit(logit))

    def act(self, _observation):
        choice_term = (
            float(self.params.reward_weight) * self.reward_seeking[self.trial]
            + float(self.params.habit_weight) * self.habit[self.trial]
            + float(self.params.gambler_fallacy_weight) * self.gambler_fallacy[self.trial]
            + float(self.params.choice_bias)
        )
        # The published implementation returns logits [-choice_term,
        # +choice_term], hence the factor of two in the binary log-odds.
        probability_right = float(expit(2.0 * choice_term))
        choice_prob = np.array([1.0 - probability_right, probability_right], dtype=float)
        choice = self.rng.choice(self.n_actions, p=choice_prob)
        return choice, choice_prob

    def learn(self, _observation, choice, reward, _next_observation, _done):
        signed_choice = 2.0 * float(choice) - 1.0
        signed_feedback = 2.0 * float(reward) - 1.0
        previous_index = self.trial - 1

        reward_retention = self._retention(float(self.params.reward_retention_logit))
        habit_retention = self._retention(float(self.params.habit_retention_logit))
        gambler_retention = self._retention(float(self.params.gambler_fallacy_retention_logit))
        self.reward_seeking[self.trial] = (
            reward_retention * self.reward_seeking[previous_index]
            + (1.0 - reward_retention) * signed_feedback * signed_choice
        )
        self.habit[self.trial] = (
            habit_retention * self.habit[previous_index] + (1.0 - habit_retention) * signed_choice
        )
        self.gambler_fallacy[self.trial] = gambler_retention * self.gambler_fallacy[
            previous_index
        ] + (1.0 - gambler_retention) * (signed_choice - signed_feedback * signed_choice)

    def get_latent_variables(self):
        return {
            "reward_seeking": self.reward_seeking.tolist(),
            "habit": self.habit.tolist(),
            "gambler_fallacy": self.gambler_fallacy.tolist(),
            "choice_prob": self.choice_prob.tolist(),
        }


class ForagerEcksteinRL(DynamicForagingAgentMLEBase):
    """Eckstein et al. asymmetric, counterfactual RL with perseveration."""

    def __init__(self, params: dict = {}, **kwargs):
        self.agent_kwargs = {}
        super().__init__(agent_kwargs=self.agent_kwargs, params=params, **kwargs)

    def _get_params_model(self, _agent_kwargs):
        return generate_eckstein_rl_params()

    def get_agent_alias(self):
        return "EcksteinRL"

    def _reset(self):
        super()._reset()
        self.q_value = np.full((self.n_actions, self.n_trials + 1), 0.5, dtype=float)

    def act(self, _observation):
        values = self.q_value[:, self.trial].copy()
        if self.trial > 0:
            previous_choice = int(self.choice_history[self.trial - 1])
            values[previous_choice] += float(self.params.perseveration_bonus)
        right_logit = float(self.params.softmax_inverse_temperature) * (
            float(values[1]) - float(values[0])
        )
        probability_right = 0.0001 + 0.9998 * float(expit(right_logit))
        choice_prob = np.array([1.0 - probability_right, probability_right], dtype=float)
        choice = self.rng.choice(self.n_actions, p=choice_prob)
        return choice, choice_prob

    def learn(self, _observation, choice, reward, _next_observation, _done):
        previous_q = self.q_value[:, self.trial - 1]
        learning_rate = (
            float(self.params.positive_learning_rate)
            if float(reward) == 1.0
            else float(self.params.negative_learning_rate)
        )
        self.q_value[:, self.trial] = previous_q
        self.q_value[choice, self.trial] = previous_q[choice] + learning_rate * (
            float(reward) - previous_q[choice]
        )
        unchosen = 1 - int(choice)
        counterfactual_reward = 1.0 - float(reward)
        self.q_value[unchosen, self.trial] = previous_q[unchosen] + learning_rate * (
            counterfactual_reward - previous_q[unchosen]
        )

    def get_latent_variables(self):
        return {
            "q_value": self.q_value.tolist(),
            "choice_prob": self.choice_prob.tolist(),
        }


class ForagerEcksteinBI(DynamicForagingAgentMLEBase):
    """Eckstein et al. hidden-state Bayesian-inference model."""

    incorrect_choice_reward_probability = 1e-5

    def __init__(self, params: dict = {}, **kwargs):
        self.agent_kwargs = {}
        super().__init__(agent_kwargs=self.agent_kwargs, params=params, **kwargs)

    def _get_params_model(self, _agent_kwargs):
        return generate_eckstein_bi_params()

    def get_agent_alias(self):
        return "EcksteinBI"

    def _reset(self):
        super()._reset()
        self.probability_right_correct = np.full(self.n_trials + 1, 0.5, dtype=float)

    def act(self, _observation):
        if self.trial == 0:
            probability_right = 0.5
        else:
            signed_previous_choice = 2.0 * float(self.choice_history[self.trial - 1]) - 1.0
            biased_belief = float(
                self.probability_right_correct[self.trial]
            ) + signed_previous_choice * float(self.params.perseveration_bonus)
            right_logit = float(self.params.softmax_inverse_temperature) * (
                2.0 * biased_belief - 1.0
            )
            probability_right = 0.0001 + 0.9998 * float(expit(right_logit))
        choice_prob = np.array([1.0 - probability_right, probability_right], dtype=float)
        choice = self.rng.choice(self.n_actions, p=choice_prob)
        return choice, choice_prob

    def learn(self, _observation, choice, reward, _next_observation, _done):
        prior_right = float(self.probability_right_correct[self.trial - 1])
        p_reward = float(self.params.subjective_reward_probability)
        p_noisy = self.incorrect_choice_reward_probability

        if int(choice) == 1:
            reward_probability_if_right = p_reward
            reward_probability_if_left = p_noisy
        else:
            reward_probability_if_right = p_noisy
            reward_probability_if_left = p_reward
        likelihood_right = (
            reward_probability_if_right
            if float(reward) == 1.0
            else 1.0 - reward_probability_if_right
        )
        likelihood_left = (
            reward_probability_if_left if float(reward) == 1.0 else 1.0 - reward_probability_if_left
        )
        denominator = likelihood_right * prior_right + likelihood_left * (1.0 - prior_right)
        posterior_right = likelihood_right * prior_right / denominator if denominator > 0.0 else 0.5
        p_switch = float(self.params.subjective_switch_probability)
        self.probability_right_correct[self.trial] = (
            1.0 - p_switch
        ) * posterior_right + p_switch * (1.0 - posterior_right)

    def get_latent_variables(self):
        return {
            "probability_right_correct": self.probability_right_correct.tolist(),
            "choice_prob": self.choice_prob.tolist(),
        }
