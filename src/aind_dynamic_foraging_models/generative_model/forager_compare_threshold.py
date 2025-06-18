"""Compare-to-threshold foraging model implementation"""

from typing import Literal

import numpy as np
from aind_behavior_gym.dynamic_foraging.task import L, R

from .base import DynamicForagingAgentMLEBase
from .learn_functions import learn_choice_kernel
from .params.forager_compare_threshold_params import generate_pydantic_compare_threshold_params


class ForagerCompareThreshold(DynamicForagingAgentMLEBase):
    """Compare-to-threshold foraging model.

    This model only tracks a single value (for exploiting the current option) and
    makes decisions by comparing this value to a threshold.
    """

    def __init__(
        self,
        choice_kernel: Literal["none", "one_step", "full"] = "none",
        params: dict = {},
        **kwargs,
    ):
        """Initialize the compare-to-threshold foraging agent.

        Parameters
        ----------
        choice_kernel : Literal["none", "one_step", "full"], optional
            Choice kernel type, by default "none"
            If "none", no choice kernel will be included in the model.
            If "one_step", choice_kernel_step_size will be set to 1.0, i.e., only the last choice
                affects the choice kernel.
            If "full", both choice_kernel_step_size and choice_kernel_relative_weight
            will be included
        params : dict, optional
            Initial parameters of the model, by default {}.
        """
        # -- Pack the agent_kwargs --
        self.agent_kwargs = dict(
            choice_kernel=choice_kernel,
        )  # Note that the class and self.agent_kwargs fully define the agent

        # -- Initialize the model parameters --
        super().__init__(agent_kwargs=self.agent_kwargs, params=params, **kwargs)

    def _get_params_model(self, agent_kwargs):
        """Implement the base class method to dynamically generate Pydantic models
        for parameters and fitting bounds for the compare-to-threshold foraging model.
        """
        return generate_pydantic_compare_threshold_params(**agent_kwargs)

    def get_agent_alias(self):
        """Get the agent alias"""
        _ck = {"none": "", "one_step": "_CK1", "full": "_CKfull"}[
            self.agent_kwargs["choice_kernel"]
        ]
        return "ForagingCompareThreshold" + _ck

    def _reset(self):
        """Reset the agent"""
        # --- Call the base class reset ---
        super()._reset()

        # --- Agent family specific variables ---
        # Initialize a single value (for the exploit option) for all trials
        self.value = np.full(self.n_trials + 1, np.nan)
        self.value[0] = self.params.threshold  # Initialize to threshold

        # Track which option is currently active (True for exploit, False for explore)
        self.exploiting = np.full(self.n_trials, False)
        # Start with exploration for first trial
        self.current_option = "explore"

        # Always initialize choice kernel with nan, even if choice_kernel = "none"
        self.choice_kernel = np.full([self.n_actions, self.n_trials + 1], np.nan)
        self.choice_kernel[:, 0] = 0  # Initial choice kernel as 0

    def act(self, _):  # noqa: C901
        """Action selection using the options framework"""
        value = self.value[self.trial]
        threshold = self.params.threshold
        beta = self.params.softmax_inverse_temperature

        # prepare p_choice_given_explore
        base_prob = np.array([0.5, 0.5])
        # base_prob[L] += self.params.biasL
        # base_prob[R] -= self.params.biasL
        # base_prob = base_prob / np.sum(base_prob)  # Normalize

        # compute p(exploit) using softmax with comparison with threshold
        # p(exploit) = 1 / (1 + e^(-β(v-ρ)))
        if self.trial == 0:
            p_exploit = 1 / (1 + np.exp(-beta * (value - threshold)))
        # introduce sided bias
        else:
            # a_{t-1} is L
            if self.choice_history[self.trial - 1] == 0:
                p_exploit = 1 / (1 + np.exp(-beta * (value - threshold) - self.params.biasL))
            # a_{t-1} is R
            elif self.choice_history[self.trial - 1] == 1:
                p_exploit = 1 / (1 + np.exp(-beta * (value - threshold)))
            else:
                raise ValueError(f"incompatible choice type: {self.choice_history[self.trial - 1]}")

        # termination probabilities for each option
        beta_exploit = 1 - p_exploit  # Probability of terminating exploit
        beta_explore = p_exploit  # Probability of terminating explore

        # Check if current option should terminate
        terminate = False
        if self.current_option == "exploit":
            terminate = self.rng.random() < beta_exploit
        elif self.current_option == "explore":
            terminate = self.rng.random() < beta_explore
        else:
            raise ValueError(f"unrecognized current_option: {self.current_option}")
        # If terminating, switch to the other option
        if terminate:
            self.current_option = "explore" if self.current_option == "exploit" else "exploit"
        # Record if we're currently exploiting
        self.exploiting[self.trial] = self.current_option == "exploit"

        # select choice
        if self.trial == 0:
            choice = self.rng.choice([L, R], p=base_prob)
        elif self.current_option == "exploit":
            # use the previous choice
            choice = self.choice_history[self.trial - 1]
        elif self.current_option == "explore":
            # variant 1: explore means switch
            choice = 1 - self.choice_history[self.trial - 1]
            # # variant 2: explore means uniformly random choice
            # choice = self.rng.choice([L, R], p=base_prob))
        else:
            raise ValueError(f"unrecognized current_option: {self.current_option}")

        # compute likelihood of the choice
        if self.trial == 0:
            choice_prob = base_prob
        else:
            choice_prob = np.zeros(self.n_actions)

            # --- choice probability under exploitation: repeat the previous choice
            p_choice_given_exploit = np.zeros(self.n_actions)
            p_choice_given_exploit[self.choice_history[self.trial - 1]] = 1

            # --- choice probability under exploration: choose randomly among other options
            p_choice_given_explore = np.zeros(self.n_actions)
            # variant 1: explore means switch
            p_choice_given_explore[1 - self.choice_history[self.trial - 1]] = 1
            # # variant 2: explore means uniformly random choice
            # p_choice_given_explore = base_prob

            for action in range(self.n_actions):
                choice_prob[action] = (
                    p_exploit * p_choice_given_exploit[action]
                    + (1 - p_exploit) * p_choice_given_explore[action]
                )

            # --- Apply choice kernel influence if enabled
            if self.agent_kwargs["choice_kernel"] != "none":
                ck = self.choice_kernel[:, self.trial]
                ck_weight = self.params.choice_kernel_relative_weight

                # Mix choice probability with choice kernel
                choice_prob = (1 - ck_weight) * choice_prob + ck_weight * ck
                choice_prob = choice_prob / np.sum(choice_prob)  # Normalize

                # Re-sample choice based on adjusted probabilities
                if np.sum(choice_prob > 0) > 1:  # Only re-sample if there are multiple options
                    choice = self.rng.choice([L, R], p=choice_prob)

        return choice, choice_prob

    def learn(self, _observation, choice, reward, _next_observation, done):
        """Update value based on whether exploring or exploiting"""

        # update value based on wheteher exploiting
        # if not self.exploiting[self.trial-1]:
        #     # If we were exploring, reset value to threshold
        #     self.value[self.trial] = self.params.threshold
        # else:
        #     # If we were exploiting, update value using delta rule
        #     self.value[self.trial] = self.value[self.trial-1] + self.params.learn_rate *
        #         (reward - self.value[self.trial-1])

        # update value based on choice history
        if (self.trial == 1) or (choice != self.choice_history[self.trial - 2]):
            self.value[self.trial] = self.params.threshold + self.params.learn_rate * (
                reward - self.params.threshold
            )
            # print(f'reset value: {self.trial}')
            # self.value[self.trial] = self.params.threshold

        else:
            self.value[self.trial] = self.value[self.trial - 1] + self.params.learn_rate * (
                reward - self.value[self.trial - 1]
            )

        # Update choice kernel, if used
        if self.agent_kwargs["choice_kernel"] != "none":
            self.choice_kernel[:, self.trial] = learn_choice_kernel(
                choice=choice,
                choice_kernel_tminus1=self.choice_kernel[:, self.trial - 1],
                choice_kernel_step_size=self.params.choice_kernel_step_size,
            )

    def get_latent_variables(self):
        """Return latent variables for analysis"""
        return {
            "value": self.value.tolist(),
            "threshold": [self.params.threshold] * (self.n_trials + 1),
            "exploiting": self.exploiting.tolist(),
            "choice_kernel": self.choice_kernel.tolist(),
            "choice_prob": self.choice_prob.tolist(),
            "p_exploit": [
                1
                / (
                    1
                    + np.exp(-self.params.softmax_inverse_temperature * (v - self.params.threshold))
                )
                for v in self.value
            ],
        }

    def plot_latent_variables(self, ax, if_fitted=False):
        """Plot latent variables"""
        if if_fitted:
            style = dict(lw=2, ls=":")
            prefix = "fitted_"
        else:
            style = dict(lw=0.5)
            prefix = ""

        x = np.arange(self.n_trials + 1) + 1  # When plotting, we start from 1

        # Plot value
        ax.plot(x, self.value, label=f"{prefix}value", color="purple", **style)

        # Plot threshold as a horizontal line
        ax.axhline(
            y=self.params.threshold,
            color="black",
            linestyle="--",
            label=f"{prefix}threshold",
            **style,
        )

        # Calculate and plot p(exploit)
        p_exploit = [
            1 / (1 + np.exp(-self.params.softmax_inverse_temperature * (v - self.params.threshold)))
            for v in self.value
        ]
        ax.plot(x, p_exploit, label=f"{prefix}p(exploit)", color="cyan", **style)

        # Plot exploitation/exploration decisions on a secondary y-axis if not fitted
        # if not if_fitted:
        #     ax_exp = ax.twinx()
        #     # For the exploit/explore decisions, convert boolean to 0/1 for visualization
        #     exploit_data = np.array(self.exploiting, dtype=int)
        #     ax_exp.scatter(x_trials, exploit_data,
        #                 color="orange", alpha=0.5, s=20,
        #                 label="exploiting (1) vs exploring (0)")
        #     ax_exp.set_yticks([0, 1])
        #     ax_exp.set_yticklabels(["explore", "exploit"])
        #     ax_exp.set_ylabel("Current Option")
        #     ax_exp.set_ylim(-0.1, 1.1)

        #     # Add legend for the secondary axis
        #     handles, labels = ax_exp.get_legend_handles_labels()
        #     ax_exp.legend(handles, labels, loc='upper right', fontsize=6)

        # Add choice kernel, if used
        if self.agent_kwargs["choice_kernel"] != "none":
            ax.plot(
                x,
                self.choice_kernel[L, :],
                label=f"{prefix}choice_kernel(L)",
                color="red",
                **style,
            )
            ax.plot(
                x,
                self.choice_kernel[R, :],
                label=f"{prefix}choice_kernel(R)",
                color="blue",
                **style,
            )
