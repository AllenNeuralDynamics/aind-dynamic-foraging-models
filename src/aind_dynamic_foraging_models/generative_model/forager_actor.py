"""Maximum likelihood fitting of foraging models
"""

# %%
from typing import Literal

import numpy as np
from aind_behavior_gym.dynamic_foraging.task import L, R
from .act_functions import act_logistic
from .base import DynamicForagingAgentMLEBase
from .learn_functions import learn_actor, learn_choice_kernel
from .params.forager_actor_params import generate_pydantic_actor_params


class ForagerActor(DynamicForagingAgentMLEBase):
    """The familiy of simple Q-learning models."""

    def __init__(
        self,
        number_of_learning_rate: Literal[1, 2],
        number_of_forget_rate: Literal[0, 1],
        choice_kernel: Literal["none", "one_step", "full"] = "none",
        action_selection: Literal["act_logistic"] = "act_logistic",
        params: dict = {},
        **kwargs,
    ):
        """Init

        Parameters
        ----------
        number_of_learning_rate : Literal[1, 2], optional
            Number of learning rates, by default 2
            If 1, only one learn_rate will be included in the model.
            If 2, learn_rate_rew and learn_rate_unrew will be included in the model.
        number_of_forget_rate : Literal[0, 1], optional
            Number of forget_rates, by default 1.
            If 0, forget_rate_unchosen will not be included in the model.
            If 1, forget_rate_unchosen will be included in the model.
        action_selection : Literal["softmax", "epsilon-greedy"], optional
            Action selection type, by default "softmax"
        params: dict, optional
            Initial parameters of the model, by default {}.
            See the generated Pydantic model in forager_q_learning_params.py for the full
            list of parameters.
        """
        # -- Pack the agent_kwargs --
        self.agent_kwargs = dict(
            number_of_learning_rate=number_of_learning_rate,
            number_of_forget_rate=number_of_forget_rate,
            choice_kernel=choice_kernel,
            action_selection=action_selection,
        )  # Note that the class and self.agent_kwargs fully define the agent

        # -- Initialize the model parameters --
        super().__init__(agent_kwargs=self.agent_kwargs, params=params, **kwargs)

        # -- Some agent-family-specific variables --
        self.fit_choice_kernel = False

    def _get_params_model(self, agent_kwargs):
        """Implement the base class method to dynamically generate Pydantic models
        for parameters and fitting bounds for simple Q learning.
        """
        return generate_pydantic_actor_params(**agent_kwargs)

    def _reset(self):
        """Reset the agent"""
        # --- Call the base class reset ---
        super()._reset()

        # --- Agent family specific variables ---
        # Latent variables have n_trials + 1 length to capture the update
        # after the last trial (HH20210726)
        self.w = np.full([self.n_actions, self.n_trials], np.nan)
        self.w[:, 0] = 0  # Initial Q values as 0
        # Always initialize choice_kernel with nan, even if choice_kernel = "none"
        self.choice_kernel = np.full([self.n_actions, self.n_trials + 1], np.nan)
        self.choice_kernel[:, 0] = 0  # Initial choice kernel as 0

    def act(self, _):
        """Action selection"""
        # Handle choice kernel
        if self.agent_kwargs["choice_kernel"] == "none":
            choice_kernel = None
            choice_kernel_relative_weight = None
        else:
            choice_kernel = self.choice_kernel[:, self.trial]
            choice_kernel_relative_weight = self.params.choice_kernel_relative_weight
        # Action selection
        if self.agnet_kwargs["action_selection"] == "logistic":
            choice, choice_prob = act_logistic(
                w_t=self.q_value[:, self.trial],
                bias_terms=np.array([self.params.biasL, 0]),
                # -- Choice kernel --
                choice_kernel=choice_kernel,
                choice_kernel_relative_weight=choice_kernel_relative_weight,
                rng=self.rng,
            )
        return choice, choice_prob

    def learn(self, _observation, choice, choice_prob, reward, _next_observation, done):
        """Update Q values

        Note that self.trial already increased by 1 before learn() in the base class
        """

        # Handle params
        if self.agent_kwargs["number_of_learning_rate"] == 1:
            learn_rates = [self.params.learn_rate] * 2
        else:
            learn_rates = [self.params.learn_rate_rew, self.params.learn_rate_unrew]

        if self.agent_kwargs["number_of_forget_rate"] == 0:
            forget_rates = [0, 0]
        else:
            forget_rates = [self.params.forget_rate_unchosen, 0]

        # Update W
        self.w[:, self.trial] = learn_actor(
            choice=choice,
            choice_prob=choice,
            reward=reward,
            w_tminus1=self.w[:, self.trial - 1],
            learn_rates=learn_rates,
            forget_rates=forget_rates,
            forget_rates=forget_rates,
        )
        # Update choice kernel, if used
        if self.agent_kwargs["choice_kernel"] != "none":
            self.choice_kernel[:, self.trial] = learn_choice_kernel(
                choice=choice,
                choice_kernel_tminus1=self.choice_kernel[:, self.trial - 1],
                choice_kernel_step_size=self.params.choice_kernel_step_size,
            )

    def plot_latent_variables(self, ax, if_fitted=False):
        """Plot W"""
        if if_fitted:
            style = dict(lw=2, ls=":")
            prefix = "fitted_"
        else:
            style = dict(lw=0.5)
            prefix = ""

        x = np.arange(self.n_trials + 1) + 1  # When plotting, we start from 1
        ax.plot(x, self.w[L, :], label=f"{prefix}W(L)", color="red", **style)
        ax.plot(x, self.w[R, :], label=f"{prefix}W(R)", color="blue", **style)
