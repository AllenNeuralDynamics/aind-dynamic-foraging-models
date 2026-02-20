"""Compare-to-threshold foraging model implementation"""

from numpy._typing._array_like import NDArray
from numpy import float64
from typing import Any, Literal

import numpy as np
from aind_behavior_gym.dynamic_foraging.task import L, R

from .base import DynamicForagingAgentMLEBase
from .learn_functions import learn_choice_kernel
from .params.forager_compare_threshold_params import generate_pydantic_compare_threshold_params


class ForagerCompareThreshold(DynamicForagingAgentMLEBase):
    """Compare-to-threshold foraging model.

    This model only tracks a single value (for exploiting the current option) and
    makes decisions by comparing this value to a threshold.

    Key behavioral assumption (2 actions):
      - "exploit" means repeat the previous choice (stay)
      - "explore" means switch to the other side (leave)
    """

    def __init__(
        self,
        choice_kernel: Literal["none", "one_step", "full"] = "none",
        params: dict = {},
        reset_to_threshold: bool = True,
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
        reset_to_threshold : bool, optional
            If True, when a switch is detected the value update is "reset-like" toward threshold:
                v_t = threshold + alpha * (reward - threshold)
            If False, never reset; always do a standard delta update:
                v_t = v_{t-1} + alpha * (reward - v_{t-1})
        """
        # -- Pack the agent_kwargs --
        self.agent_kwargs = dict(
            choice_kernel=choice_kernel,
            reset_to_threshold=reset_to_threshold,
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
        _rt = "" if self.agent_kwargs.get("reset_to_threshold", True) else "_NoReset"
        return "ForagingCompareThreshold" + _ck + _rt

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

    def act(self, _) -> tuple[Any, NDArray[Any] | Any | NDArray[float64]]:  # noqa: C901
        """Action selection using the options framework."""
        value = self.value[self.trial]
        threshold = self.params.threshold
        beta = self.params.softmax_inverse_temperature

        # Uniform base probabilities for trial 0
        base_prob = np.array([0.5, 0.5], dtype=float)

        # ------------------------------------------------------------
        # Step A: compute p_exploit = P(stay) from value vs threshold
        # ------------------------------------------------------------
        # p_exploit = 1 / (1 + exp(-beta * (value - threshold))) [+ optional bias term]
        if self.trial == 0:
            logit = beta * (value - threshold)
        else:
            last_choice = self.choice_history[self.trial - 1]
            if last_choice == L:
                # Bias is applied only when last action was Left (as in your original code)
                logit = beta * (value - threshold) + self.params.biasL
            elif last_choice == R:
                logit = beta * (value - threshold)
            else:
                raise ValueError(f"incompatible choice type: {last_choice}")

        p_exploit = 1.0 / (1.0 + np.exp(-logit))
        p_exploit = float(np.clip(p_exploit, 1e-12, 1.0 - 1e-12))

        # ------------------------------------------------------------
        # Step B: option termination (exploit <-> explore) using p_exploit
        # ------------------------------------------------------------
        # - If currently exploiting, terminate with probability (1 - p_exploit)
        # - If currently exploring, terminate with probability (p_exploit)
        if self.current_option == "exploit":
            terminate = self.rng.random() < (1.0 - p_exploit)
        elif self.current_option == "explore":
            terminate = self.rng.random() < p_exploit
        else:
            raise ValueError(f"unrecognized current_option: {self.current_option}")

        if terminate:
            self.current_option = "explore" if self.current_option == "exploit" else "exploit"

        self.exploiting[self.trial] = (self.current_option == "exploit")

        # ------------------------------------------------------------
        # Step C: choose an action given the current option
        # ------------------------------------------------------------
        if self.trial == 0:
            choice = self.rng.choice([L, R], p=base_prob)
        else:
            last_choice = self.choice_history[self.trial - 1]
            if self.current_option == "exploit":
                choice = last_choice  # stay
            elif self.current_option == "explore":
                choice = 1 - last_choice  # switch
            else:
                raise ValueError(f"unrecognized current_option: {self.current_option}")

        # ------------------------------------------------------------
        # Step D (clarified): compute P(action) from "stay vs switch" probabilities
        # ------------------------------------------------------------
        # Because there are only 2 actions and:
        #   - exploit == repeat last action (stay)
        #   - explore == switch to the other action (switch)
        #
        # We have the direct mapping:
        #   P(a_t = last_choice)     = p_exploit
        #   P(a_t != last_choice)    = 1 - p_exploit
        #
        # This is exactly equivalent to the mixture form:
        #   P(a_t) = p_exploit * P(a_t|exploit) + (1-p_exploit) * P(a_t|explore)
        if self.trial == 0:
            choice_prob = base_prob.copy()
        else:
            last_choice = self.choice_history[self.trial - 1]
            other_choice = 1 - last_choice

            choice_prob = np.zeros(self.n_actions, dtype=float)
            choice_prob[last_choice] = p_exploit
            choice_prob[other_choice] = 1.0 - p_exploit

        # ------------------------------------------------------------
        # Optional: apply choice kernel influence (if enabled)
        # ------------------------------------------------------------
        if (self.trial > 0) and (self.agent_kwargs["choice_kernel"] != "none"):
            ck = self.choice_kernel[:, self.trial]
            ck_weight = float(self.params.choice_kernel_relative_weight)

            # Mix choice probability with choice kernel and normalize
            choice_prob = (1.0 - ck_weight) * choice_prob + ck_weight * ck
            s = float(np.sum(choice_prob))
            if s <= 0 or (not np.isfinite(s)):
                raise ValueError(f"choice_prob normalization failed: sum={s}")
            choice_prob = choice_prob / s

            # Re-sample choice based on adjusted probabilities only if not deterministic
            if np.sum(choice_prob > 0) > 1:
                choice = self.rng.choice([L, R], p=choice_prob)

        return choice, choice_prob

    def learn(self, _observation, choice, reward, _next_observation, done):
        """Update value and (optionally) choice kernel."""
        alpha = float(self.params.learn_rate)
        reset_to_threshold = bool(self.agent_kwargs.get("reset_to_threshold", True))

        # Decide whether a switch occurred (using your original switch-detection logic)
        switched = False
        if self.trial == 1:
            switched = True
        elif self.trial >= 2:
            switched = (choice != self.choice_history[self.trial - 2])

        # Update value
        if reset_to_threshold and switched:
            # Reset-like update toward threshold (original behavior)
            thr = float(self.params.threshold)
            self.value[self.trial] = thr + alpha * (reward - thr)
        else:
            # Standard delta rule from previous value (no reset)
            v_prev = float(self.value[self.trial - 1])
            self.value[self.trial] = v_prev + alpha * (reward - v_prev)

        # Update choice kernel, if used
        if self.agent_kwargs["choice_kernel"] != "none":
            self.choice_kernel[:, self.trial] = learn_choice_kernel(
                choice=choice,
                choice_kernel_tminus1=self.choice_kernel[:, self.trial - 1],
                choice_kernel_step_size=self.params.choice_kernel_step_size,
            )

    def get_latent_variables(self):
        """Return latent variables for analysis (consistent with actual decision rule)."""

        beta = self.params.softmax_inverse_temperature
        threshold = self.params.threshold
        biasL = getattr(self.params, "biasL", 0.0)

        p_exploit_all = []

        for t, v in enumerate(self.value):
            if t == 0:
                logit = beta * (v - threshold)
            else:
                last_choice = self.choice_history[t - 1]
                if last_choice == L:
                    logit = beta * (v - threshold) + biasL
                else:
                    logit = beta * (v - threshold)

            p = 1.0 / (1.0 + np.exp(-logit))
            p_exploit_all.append(float(p))

        return {
            "value": self.value.tolist(),
            "threshold": [threshold] * (self.n_trials + 1),
            "exploiting": self.exploiting.tolist(),
            "choice_kernel": self.choice_kernel.tolist(),
            "choice_prob": self.choice_prob.tolist(),
            "p_exploit": p_exploit_all   
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
