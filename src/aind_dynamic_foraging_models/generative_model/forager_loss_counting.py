"""Maximum likelihood fitting of foraging models
"""

# %%
from typing import Literal

import numpy as np
from aind_behavior_gym.dynamic_foraging.task import L, R

from .act_functions import act_loss_counting
from .base import DynamicForagingAgentMLEBase
from .learn_functions import learn_choice_kernel, learn_loss_counting
from .params.forager_loss_counting_params import generate_pydantic_loss_counting_params


class ForagerLossCounting(DynamicForagingAgentMLEBase):
    """The familiy of loss counting models."""

    def __init__(
        self,
        win_stay_lose_switch: Literal[False, True] = False,
        choice_kernel: Literal["none", "one_step", "full"] = "none",
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
        win_stay_lose_switch: bool, optional
            If True, the agent will be a win-stay-lose-shift agent
            (loss_count_threshold_mean and loss_count_threshold_std are fixed at 1 and 0),
            by default False
        choice_kernel : Literal["none", "one_step", "full"], optional
            Choice kernel type, by default "none"
            If "none", no choice kernel will be included in the model.
            If "one_step", choice_kernel_step_size will be set to 1.0, i.e., only the last choice
                affects the choice kernel. (Bari2019)
            If "full", both choice_kernel_step_size and choice_kernel_relative_weight
            will be included in fitting
        params: dict, optional
            Initial parameters of the model, by default {}.
            In the loss counting model, the only two parameters are:
                - loss_count_threshold_mean: float
                - loss_count_threshold_std: float
        """
        # -- Pack the agent_kwargs --
        self.agent_kwargs = dict(
            win_stay_lose_switch=win_stay_lose_switch,
            choice_kernel=choice_kernel,
        )

        # -- Initialize the model parameters --
        super().__init__(agent_kwargs=self.agent_kwargs, params=params, **kwargs)

        # -- Some agent-family-specific variables --
        self.fit_choice_kernel = False

    def _get_params_model(self, agent_kwargs):
        """Get the params model of the agent"""
        return generate_pydantic_loss_counting_params(**agent_kwargs)

    def get_agent_alias(self):
        """Get the agent alias"""
        _prefix = "WSLS" if self.agent_kwargs["win_stay_lose_switch"] else "LossCounting"
        _ck = {"none": "", "one_step": "_CK1", "full": "_CKfull"}[
            self.agent_kwargs["choice_kernel"]
        ]
        return _prefix + _ck

    def _reset(self):
        """Reset the agent"""
        # --- Call the base class reset ---
        super()._reset()

        # --- Agent family specific variables ---
        self.loss_count = np.full(self.n_trials + 1, np.nan)
        self.loss_count[0] = 0  # Initial loss count as 0

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

        choice, choice_prob = act_loss_counting(
            previous_choice=self.choice_history[self.trial - 1] if self.trial > 0 else None,
            loss_count=self.loss_count[self.trial],
            loss_count_threshold_mean=self.params.loss_count_threshold_mean,
            loss_count_threshold_std=self.params.loss_count_threshold_std,
            bias_terms=np.array([self.params.biasL, 0]),
            # -- Choice kernel --
            choice_kernel=choice_kernel,
            choice_kernel_relative_weight=choice_kernel_relative_weight,
            rng=self.rng,
        )
        return choice, choice_prob

    def learn(self, _, choice, reward, __, done):
        """Update loss counter

        Note that self.trial already increased by 1 before learn() in the base class
        """
        self.loss_count[self.trial] = learn_loss_counting(
            choice=choice,
            reward=reward,
            just_switched=(self.trial == 1 or choice != self.choice_history[self.trial - 2]),
            loss_count_tminus1=self.loss_count[self.trial - 1],
        )

        # Update choice kernel, if used
        if self.agent_kwargs["choice_kernel"] != "none":
            self.choice_kernel[:, self.trial] = learn_choice_kernel(
                choice=choice,
                choice_kernel_tminus1=self.choice_kernel[:, self.trial - 1],
                choice_kernel_step_size=self.params.choice_kernel_step_size,
            )

    def get_latent_variables(self):
        return {
            "loss_count": self.loss_count.tolist(),
            "choice_kernel": self.choice_kernel.tolist(),
            "choice_prob": self.choice_prob.tolist(),
        }

    def plot_latent_variables(self, ax, if_fitted=False):
        """Plot Q values"""
        if if_fitted:
            style = dict(lw=2, ls=":")
            prefix = "fitted_"
        else:
            style = dict(lw=0.5)
            prefix = ""

        x = np.arange(self.n_trials + 1) + 1  # When plotting, we start from 1

        if not if_fitted:
            # Only plot loss count if not fitted
            ax_loss_count = ax.twinx()

            ax_loss_count.plot(x, self.loss_count, label="loss_count", color="blue", **style)
            ax_loss_count.set(ylabel="Loss count")
            ax_loss_count.legend(loc="upper right", fontsize=6)

        # Add choice kernel, if used
        if self.agent_kwargs["choice_kernel"] != "none":
            ax.plot(
                x,
                self.choice_kernel[L, :],
                label=f"{prefix}choice_kernel(L)",
                color="purple",
                **style,
            )
            ax.plot(
                x,
                self.choice_kernel[R, :],
                label=f"{prefix}choice_kernel(R)",
                color="cyan",
                **style,
            )
