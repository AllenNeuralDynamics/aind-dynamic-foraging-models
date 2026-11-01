"""Compare-to-threshold foraging model written in stay credit"""

from __future__ import annotations

import numpy as np
from aind_behavior_gym.dynamic_foraging.task import L, R

from .base import DynamicForagingAgentMLEBase
from .params.forager_compare_threshold_credit_params import (
    generate_pydantic_compare_threshold_credit_params,
)


class ForagerCompareThresholdCredit(DynamicForagingAgentMLEBase):
    """Compare-to-threshold model with stay credits as its parameters.

    ForagerCompareThreshold (one learning rate, no choice kernel, reset to threshold) in the
    coordinates c = beta * (value - threshold), the logit of staying:

      - after a win:    c <- c + learn_rate * (stay_credit_win - c)
      - after a loss:   c <- c + learn_rate * (stay_credit_loss - c)
      - after a switch: c <- 0, then that trial's update (value reset to threshold)
      - P(stay) = sigmoid(c + biasL if the last choice was Left else c)

    With stay_credit_win = beta * (1 - threshold) and stay_credit_loss = -beta * threshold the
    choice probabilities equal ForagerCompareThreshold's exactly. Only the parameters and their
    bounds differ: a positive stay_credit_loss reads directly as "a loss still pulls the subject
    to stay", and the bounds reach loss credits that the threshold bound of [-1, 1] excludes.
    """

    def __init__(self, params: dict = {}, **kwargs):
        """Initialize the agent; ``params`` holds fitted parameters, ``kwargs`` go to the base"""
        self.agent_kwargs = {}
        super().__init__(agent_kwargs=self.agent_kwargs, params=params, **kwargs)

    def _get_params_model(self, _agent_kwargs):
        return generate_pydantic_compare_threshold_credit_params()

    def get_agent_alias(self):
        """Get the agent alias string used in tables/plots."""
        return "CompareThresholdCredit"

    def _reset(self):
        super()._reset()
        # Stay credit has length n_trials + 1 to include the final post-trial update
        self.credit = np.full(self.n_trials + 1, np.nan)
        self.credit[0] = 0.0  # value starts at threshold

    def act(self, _observation):
        if self.trial == 0:
            choice_prob = np.array([0.5, 0.5], dtype=float)
            return self.rng.choice([L, R], p=choice_prob), choice_prob
        last_choice = int(self.choice_history[self.trial - 1])
        logit = float(self.credit[self.trial])
        if last_choice == L:
            logit += float(self.params.biasL)
        p_stay = float(np.clip(1.0 / (1.0 + np.exp(-logit)), 1e-12, 1.0 - 1e-12))
        choice_prob = np.zeros(self.n_actions, dtype=float)
        choice_prob[last_choice] = p_stay
        choice_prob[1 - last_choice] = 1.0 - p_stay
        return self.rng.choice([L, R], p=choice_prob), choice_prob

    def learn(self, _observation, choice, reward, _next_observation, _done):
        # self.trial has already advanced: write credit[trial] from credit[trial - 1].
        # The first choice counts as a switch, as in ForagerCompareThreshold.
        switched = self.trial == 1 or choice != self.choice_history[self.trial - 2]
        previous = 0.0 if switched else float(self.credit[self.trial - 1])
        target = float(self.params.stay_credit_win if reward else self.params.stay_credit_loss)
        self.credit[self.trial] = previous + float(self.params.learn_rate) * (target - previous)

    def get_latent_variables(self):
        return {"stay_credit": self.credit.tolist(), "choice_prob": self.choice_prob.tolist()}
