"""Actor-critic foraging model with MLE fitting.

This implements the classic (Sutton & Barto) actor-critic architecture adapted to a
two-armed dynamic foraging task, following the same design pattern as
:class:`~aind_dynamic_foraging_models.generative_model.forager_q_learning.ForagerQLearning`
so that it plugs directly into the existing MLE / differential-evolution fitting pipeline.

Two latent variables are tracked:

- **Actor** ``H``: a per-action preference. Actions are selected by a softmax over ``H``
  (plus an optional left bias and choice kernel).
- **Critic** ``V``: a single (state-less) state value that serves as a reward baseline.

Both are taught by the same TD error ``delta_t = reward_t - V_{t-1}`` (there is no
bootstrap term because the task is effectively a single-state bandit). See
:func:`~aind_dynamic_foraging_models.generative_model.learn_functions.learn_actor_critic`.
"""

# %%
from typing import Literal

import numpy as np
from aind_behavior_gym.dynamic_foraging.task import L, R

from .act_functions import act_softmax
from .base import DynamicForagingAgentMLEBase
from .learn_functions import learn_actor_critic, learn_choice_kernel
from .params.forager_actor_critic_params import generate_pydantic_actor_critic_params


class ForagerActorCritic(DynamicForagingAgentMLEBase):
    """The family of actor-critic models (softmax actor + state-value critic)."""

    def __init__(
        self,
        choice_kernel: Literal["none", "one_step", "full"] = "none",
        params: dict = {},
        **kwargs,
    ):
        """Init

        Parameters
        ----------
        choice_kernel : Literal["none", "one_step", "full"], optional
            Choice kernel type, by default "none"
            If "none", no choice kernel will be included in the model.
            If "one_step", choice_kernel_step_size will be set to 1.0, i.e., only the last choice
                affects the choice kernel. (Bari2019)
            If "full", both choice_kernel_step_size and choice_kernel_relative_weight
            will be included during fitting.
        params: dict, optional
            Initial parameters of the model, by default {}.
            See the generated Pydantic model in forager_actor_critic_params.py for the full
            list of parameters (learn_rate_actor, learn_rate_critic, biasL,
            softmax_inverse_temperature, and optionally choice-kernel parameters).

        Notes
        -----
        The action selection is always a softmax over the actor preferences, so the
        actor's learning-rate scale (``learn_rate_actor``) trades off with the softmax
        inverse temperature (``softmax_inverse_temperature``). These are not jointly
        identifiable; when fitting, it is recommended to clamp the inverse temperature,
        e.g. ``forager.fit(..., clamp_params={"softmax_inverse_temperature": 1.0})``.
        """
        # -- Pack the agent_kwargs --
        self.agent_kwargs = dict(
            choice_kernel=choice_kernel,
        )  # Note that the class and self.agent_kwargs fully define the agent

        # -- Initialize the model parameters --
        super().__init__(agent_kwargs=self.agent_kwargs, params=params, **kwargs)

        # -- Some agent-family-specific variables --
        self.fit_choice_kernel = False

    def _get_params_model(self, agent_kwargs):
        """Implement the base class method to dynamically generate Pydantic models
        for parameters and fitting bounds for the actor-critic agent.
        """
        return generate_pydantic_actor_critic_params(**agent_kwargs)

    def get_agent_alias(self):
        """Get the agent alias"""
        _ck = {"none": "", "one_step": "_CK1", "full": "_CKfull"}[
            self.agent_kwargs["choice_kernel"]
        ]
        return "ActorCritic" + _ck

    def _reset(self):
        """Reset the agent"""
        # --- Call the base class reset ---
        super()._reset()

        # --- Agent family specific variables ---
        # Latent variables have n_trials + 1 length to capture the update
        # after the last trial (consistent with ForagerQLearning).
        # Actor preference H (one per action)
        self.actor_preference = np.full([self.n_actions, self.n_trials + 1], np.nan)
        self.actor_preference[:, 0] = 0  # Initial actor preferences as 0

        # Critic state value V (scalar, state-less bandit)
        self.value = np.full(self.n_trials + 1, np.nan)
        self.value[0] = 0  # Initial value as 0

        # Always initialize choice_kernel with nan, even if choice_kernel = "none"
        self.choice_kernel = np.full([self.n_actions, self.n_trials + 1], np.nan)
        self.choice_kernel[:, 0] = 0  # Initial choice kernel as 0

    def act(self, _):
        """Action selection (softmax over actor preferences)"""
        # Handle choice kernel
        if self.agent_kwargs["choice_kernel"] == "none":
            choice_kernel = None
            choice_kernel_relative_weight = None
        else:
            choice_kernel = self.choice_kernel[:, self.trial]
            choice_kernel_relative_weight = self.params.choice_kernel_relative_weight

        # Action selection: the actor preference plays the role of "q_value" in the softmax
        choice, choice_prob = act_softmax(
            q_value_t=self.actor_preference[:, self.trial],
            softmax_inverse_temperature=self.params.softmax_inverse_temperature,
            bias_terms=np.array([self.params.biasL, 0]),
            # -- Choice kernel --
            choice_kernel=choice_kernel,
            choice_kernel_relative_weight=choice_kernel_relative_weight,
            rng=self.rng,
        )

        return choice, choice_prob

    def learn(self, _observation, choice, reward, _next_observation, done):
        """Update actor preferences and critic value.

        Note that self.trial already increased by 1 before learn() in the base class,
        so the policy used to make this trial's choice is choice_prob[:, self.trial - 1].
        """
        # The policy (choice probability) that was actually used to make this choice
        policy_prob = self.choice_prob[:, self.trial - 1]

        # Update actor preference and critic value
        self.actor_preference[:, self.trial], self.value[self.trial] = learn_actor_critic(
            choice=choice,
            reward=reward,
            actor_preference_tminus1=self.actor_preference[:, self.trial - 1],
            value_tminus1=self.value[self.trial - 1],
            choice_prob_t=policy_prob,
            learn_rates=[self.params.learn_rate_actor, self.params.learn_rate_critic],
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
            "actor_preference": self.actor_preference.tolist(),
            "value": self.value.tolist(),
            "choice_kernel": self.choice_kernel.tolist(),
            "choice_prob": self.choice_prob.tolist(),
        }

    def plot_latent_variables(self, ax, if_fitted=False):
        """Plot actor preferences and critic value"""
        if if_fitted:
            style = dict(lw=2, ls=":")
            prefix = "fitted_"
        else:
            style = dict(lw=0.5)
            prefix = ""

        x = np.arange(self.n_trials + 1) + 1  # When plotting, we start from 1

        # -- Actor preferences --
        ax.plot(x, self.actor_preference[L, :], label=f"{prefix}H(L)", color="red", **style)
        ax.plot(x, self.actor_preference[R, :], label=f"{prefix}H(R)", color="blue", **style)

        # -- Critic value (shared baseline) on a twin axis for readability --
        ax_value = ax.twinx() if not hasattr(ax, "_ac_value_twin") else ax._ac_value_twin
        ax._ac_value_twin = ax_value
        ax_value.plot(x, self.value, label=f"{prefix}V", color="black", **style)
        ax_value.set_ylabel("critic value V")

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
