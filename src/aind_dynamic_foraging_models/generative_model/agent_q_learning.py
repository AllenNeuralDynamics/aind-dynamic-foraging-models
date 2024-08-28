"""Maximum likelihood fitting of foraging models
"""


# %%
from typing import Literal

import numpy as np

from .base import DynamicForagingAgentMLEBase
from .act_functions import act_softmax, act_epsilon_greedy
from .agent_q_learning_params import generate_pydantic_q_learning_params
from .learn_functions import learn_choice_kernel, learn_RWlike


class ForagerSimpleQ(DynamicForagingAgentMLEBase):
    """
    The familiy of simple Q-learning models.

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
    choice_kernel : Literal["none", "one_step", "full"], optional
        Choice kernel type, by default "none"
        If "none", no choice kernel will be included in the model.
        If "one_step", choice_step_size will be set to 1.0, i.e., only the previous choice
            affects the choice kernel. (Bari2019)
        If "full", both choice_step_size and choice_kernel_relative_weight will be included
    action_selection : Literal["softmax", "epsilon-greedy"], optional
        Action selection type, by default "softmax"
    params: dict, optional
        Initial parameters of the model, by default {}.
        See the generated Pydantic model in agent_q_learning_params.py for the full
        list of parameters.
    """

    def __init__(
        self,
        number_of_learning_rate: Literal[1, 2],
        number_of_forget_rate: Literal[0, 1],
        choice_kernel: Literal["none", "one_step", "full"],
        action_selection: Literal["softmax", "epsilon-greedy"],
        params: dict = {},
        **kwargs,
    ):
        """Init"""
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
        return generate_pydantic_q_learning_params(**agent_kwargs)

    def _reset(self):
        """Reset the agent"""
        # --- Call the base class reset ---
        super()._reset()
        
        # --- Agent family specific variables ---
        # Latent variables have n_trials + 1 length to capture the update
        # after the last trial (HH20210726)
        self.q_estimation = np.full([self.n_actions, self.n_trials + 1], np.nan)
        self.q_estimation[:, 0] = 0  # Initial Q values as 0

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
        if self.agent_kwargs["action_selection"] == "softmax":
            choice, choice_prob = act_softmax(
                q_estimation_t=self.q_estimation[:, self.trial],
                softmax_inverse_temperature=self.params.softmax_inverse_temperature,
                bias_terms=np.array([self.params.biasL, 0]),
                # -- Choice kernel --
                choice_kernel=choice_kernel,
                choice_kernel_relative_weight=choice_kernel_relative_weight,
                rng=self.rng,
            )
        elif self.agent_kwargs["action_selection"] == "epsilon-greedy":
            choice, choice_prob = act_epsilon_greedy(
                q_estimation_t=self.q_estimation[:, self.trial],
                epsilon=self.params.epsilon,
                bias_terms=np.array([self.params.biasL, 0]),
                # -- Choice kernel --
                choice_kernel=choice_kernel,
                choice_kernel_relative_weight=choice_kernel_relative_weight,
                rng=self.rng,
            )

        return choice, choice_prob

    def learn(self, _observation, choice, reward, _next_observation, done):
        """Update Q values"""

        # Handle params
        if self.agent_kwargs["number_of_learning_rate"] == 1:
            learn_rates = [self.params.learn_rate] * 2
        else:
            learn_rates = [self.params.learn_rate_rew, self.params.learn_rate_unrew]

        if self.agent_kwargs["number_of_forget_rate"] == 0:
            forget_rates = [0, 0]
        else:
            forget_rates = [self.params.forget_rate_unchosen, 0]

        # Update Q values
        self.q_estimation[:, self.trial] = learn_RWlike(
            choice=choice,
            reward=reward,
            q_estimation_tminus1=self.q_estimation[:, self.trial - 1],
            learn_rates=learn_rates,
            forget_rates=forget_rates,
        )

        # Update choice kernel, if used
        if self.agent_kwargs["choice_kernel"] != "none":
            self.choice_kernel[:, self.trial] = learn_choice_kernel(
                choice=choice,
                choice_kernel_tminus1=self.choice_kernel[:, self.trial - 1],
                choice_step_size=self.params.choice_step_size,
            )