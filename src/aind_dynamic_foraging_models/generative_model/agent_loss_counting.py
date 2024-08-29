"""Maximum likelihood fitting of foraging models
"""

# %%
from typing import Literal

import numpy as np

from .act_functions import act_loss_counting
from .base import DynamicForagingAgentMLEBase
from .learn_functions import learn_choice_kernel, learn_RWlike
from .params.agent_loss_counting_params import generate_pydantic_loss_counting_params


class ForagerLossCounting(DynamicForagingAgentMLEBase):
    """The familiy of loss counting models."""

    def __init__(
        self,
        params: dict = {},
        **kwargs,
    ):
        """Initialize the family of loss counting agents.
        
        Some special agents are:
        1. Never switch: loss_count_threshold_mean = inf
        2. Always switch: loss_count_threshold_mean = 0.0 & loss_count_threshold_std = 0.0
        3. Win-stay-lose-shift: loss_count_threshold_mean = 1.0 & loss_count_threshold_std = 0.0

        Parameters
        ----------
        params: dict, optional
            Initial parameters of the model, by default {}.
            In the loss counting model, the only two parameters are:
                - loss_count_threshold_mean: float
                - loss_count_threshold_std: float
        """
        # -- Pack the agent_kwargs --
        self.agent_kwargs = {}  # No hyperparameters for the loss counting model

        # -- Initialize the model parameters --
        super().__init__(agent_kwargs=self.agent_kwargs, params=params, **kwargs)

        # -- Some agent-family-specific variables --
        self.fit_choice_kernel = False

    def _get_params_model(self, agent_kwargs):
        """Get the params model of the agent
        """
        return generate_pydantic_loss_counting_params(**agent_kwargs)

    def _reset(self):
        """Reset the agent"""
        # --- Call the base class reset ---
        super()._reset()

        # --- Agent family specific variables ---
        self.loss_count = np.full(self.n_trials + 1, np.nan)
        self.loss_count[0] = 0  # Initial loss count as 0

    def act(self, _):
        """Action selection"""
        choice, choice_prob = act_loss_counting(
            previous_choice=self.choice_history[self.trial - 1] if self.trial > 0 else None,
            loss_count=self.loss_count[self.trial],
            loss_count_threshold_mean=self.params.loss_count_threshold_mean,
            loss_count_threshold_std=self.params.loss_count_threshold_std,
            rng=self.rng,
        )
        return choice, choice_prob

    def learn(self, _, choice, reward, __, done):
        """Update loss counter

        """
        if reward:
            self.loss_count[self.trial] = 0
            return

        # Note that self.trial already increased by 1 before learn() in the base class
        just_switched = self.trial == 1 or choice != self.choice_history[self.trial - 2]
        if just_switched:
            self.loss_count[self.trial] = 1
        else:
            self.loss_count[self.trial] = self.loss_count[self.trial - 1] + 1

    def plot_latent_variables(self, ax, if_fitted=False):
        """Plot Q values"""
        if if_fitted:
            # Don't need to plot loss_count since it is teacher-forced in loss counting agent
            return
        
        style = dict(lw=0.5)
        
        ax_loss_count = ax.twinx()

        x = np.arange(self.n_trials + 1) + 1  # When plotting, we start from 1
        ax_loss_count.plot(x, self.loss_count, label=f"loss_count", color="blue", **style)
        ax_loss_count.set(ylabel="Loss count")
        ax_loss_count.legend(loc="upper right", fontsize=6)