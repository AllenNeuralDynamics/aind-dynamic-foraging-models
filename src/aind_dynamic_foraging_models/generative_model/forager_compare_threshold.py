"""Compare-to-threshold foraging model implementation (MLE-ready).

This version adds **asymmetric learning rates** for rewarded vs unrewarded trials, in the
same design pattern as `ForagerQLearning`:

- A hyperparameter `number_of_learning_rate: Literal[1, 2]` lives in `agent_kwargs` so it
  shows up in the forager table / agent_kwargs listing and is enumerable by `ForagerCollection`.
- The parameter model (Pydantic) is still generated dynamically via
  `generate_pydantic_compare_threshold_params(...)`, and must support:
    - if number_of_learning_rate == 1: parameter `learn_rate`
    - if number_of_learning_rate == 2: parameters `learn_rate_rew`, `learn_rate_unrew`

IMPORTANT:
- `reset_to_threshold` remains a **hyperparameter** (agent_kwargs), NOT a fitted parameter.
  You should NOT set it via `set_params()`. Use the constructor or `set_agent_kwargs()`.
"""

from __future__ import annotations

from typing import Any, Literal

import numpy as np
from numpy import float64
from numpy._typing._array_like import NDArray
from aind_behavior_gym.dynamic_foraging.task import L, R

from .base import DynamicForagingAgentMLEBase
from .learn_functions import learn_choice_kernel
from .params.forager_compare_threshold_params import generate_pydantic_compare_threshold_params


class ForagerCompareThreshold(DynamicForagingAgentMLEBase):
    """Compare-to-threshold foraging model.

    This model tracks a single latent variable `value` and selects actions by comparing
    `value` to a threshold using a logistic (sigmoid) mapping (soft decision rule).

    Key behavioral assumption (2 actions):
      - "exploit" means repeat the previous choice (stay)
      - "explore" means switch to the other side (leave)
    """

    def __init__(
        self,
        number_of_learning_rate: Literal[1, 2] = 1,
        choice_kernel: Literal["none", "one_step", "full"] = "none",
        params: dict = {},
        reset_to_threshold: Literal[True, False] = True,
        **kwargs,
    ):
        """Initialize the compare-to-threshold foraging agent.

        Parameters
        ----------
        number_of_learning_rate : Literal[1, 2], optional
            Controls whether learning rate is symmetric (1) or asymmetric (2).
            - 1: include `learn_rate`
            - 2: include `learn_rate_rew` and `learn_rate_unrew`
            This is a structural hyperparameter (agent_kwargs), not a fitted parameter.
        choice_kernel : Literal["none", "one_step", "full"], optional
            Choice kernel type, by default "none".
            If "none", no choice kernel will be included in the model.
            If "one_step", choice_kernel_step_size will be set to 1.0, i.e., only the last choice
                affects the choice kernel.
            If "full", both choice_kernel_step_size and choice_kernel_relative_weight
                will be included.
        params : dict, optional
            Initial parameters of the model, by default {}.
            NOTE: This should only contain Pydantic parameters (fitted parameters), e.g.
              - threshold, softmax_inverse_temperature, biasL, learn_rate (or learn_rate_rew/unrew)
            Do NOT put hyperparameters like reset_to_threshold here.
        reset_to_threshold : Literal[True, False], optional
            Hyperparameter controlling value update rule at a switch.
            - True: when a switch is detected, do a reset-like update toward threshold:
                v_t = threshold + alpha * (reward - threshold)
            - False: never reset; always do a standard delta update:
                v_t = v_{t-1} + alpha * (reward - v_{t-1})
        **kwargs
            Passed to the base class (e.g., seed for rng).
        """

        # ---------------------------------------------------------------------
        # Pack the agent hyperparameters (agent_kwargs).
        #
        # IMPORTANT:
        # - These define the agent "structure" and should show up in the forager table.
        # - They are NOT validated by ParamModel and are NOT part of fitting bounds.
        # - The class + agent_kwargs together fully define the model variant.
        # ---------------------------------------------------------------------
        self.agent_kwargs = dict(
            number_of_learning_rate=number_of_learning_rate,
            choice_kernel=choice_kernel,
            reset_to_threshold=reset_to_threshold,
        )

        # Initialize the model parameters (Pydantic) via the base class
        super().__init__(agent_kwargs=self.agent_kwargs, params=params, **kwargs)

    def set_agent_kwargs(self, **agent_kwargs):
        """Update agent hyperparameters (agent_kwargs) after initialization.

        NOTE:
        - This does NOT rebuild ParamModel/ParamFitBoundModel.
        - Therefore, do NOT change hyperparameters that affect the parameter set
          (e.g., number_of_learning_rate or choice_kernel) after initialization.
          If you need a different structure, instantiate a new agent.
        """
        self.agent_kwargs.update(agent_kwargs)
        return self.agent_kwargs

    def _get_params_model(self, agent_kwargs):
        """Dynamically generate Pydantic models for parameters and fitting bounds.

        This must be consistent with the hyperparameters stored in agent_kwargs.
        """
        return generate_pydantic_compare_threshold_params(
            number_of_learning_rate=agent_kwargs["number_of_learning_rate"],
            choice_kernel=agent_kwargs["choice_kernel"],
        )

    def get_agent_alias(self):
        """Get the agent alias string used in tables/plots."""
        _lr = f"_L{self.agent_kwargs.get('number_of_learning_rate', 1)}"
        _ck = {"none": "", "one_step": "_CK1", "full": "_CKfull"}[
            self.agent_kwargs["choice_kernel"]
        ]
        _rt = "" if self.agent_kwargs.get("reset_to_threshold", True) else "_NoReset"
        return "ForagingCompareThreshold" + _lr + _ck + _rt

    def _reset(self):
        """Reset the agent state before running a session (generative or predictive)."""
        # Reset common MLE state (choice_prob, histories, etc.)
        super()._reset()

        # ---------------------------------------------------------------------
        # Agent-family state variables
        # ---------------------------------------------------------------------

        # Latent `value` has length n_trials + 1 to include the final post-trial update
        self.value = np.full(self.n_trials + 1, np.nan)
        self.value[0] = float(self.params.threshold)  # start at threshold

        # Track whether the agent is in exploit mode (True) or explore mode (False)
        self.exploiting = np.full(self.n_trials, False)

        # Track current option label
        self.current_option = "explore"  # start exploring at first trial

        # Choice kernel state (kept for convenience; only used when enabled)
        self.choice_kernel = np.full([self.n_actions, self.n_trials + 1], np.nan)
        self.choice_kernel[:, 0] = 0.0  # initial kernel

    def act(self, _) -> tuple[Any, NDArray[Any] | Any | NDArray[float64]]:  # noqa: C901
        """Select an action and return (choice, choice_prob).

        Design:
        - Compute p_exploit (= P(stay)) from value vs threshold through a logistic mapping.
        - Maintain an option state (exploit vs explore) that can terminate and switch
          according to p_exploit.
        - Translate the option state into actual left/right choice.
        - Compute the implied action probability vector [P(L), P(R)] for likelihood fitting.
        - Optionally mix with choice kernel if enabled.
        """
        value = float(self.value[self.trial])
        threshold = float(self.params.threshold)
        beta = float(self.params.softmax_inverse_temperature)

        # Uniform base probabilities for trial 0
        base_prob = np.array([0.5, 0.5], dtype=float)

        # ------------------------------------------------------------
        # Step A: compute p_exploit = P(stay) from value vs threshold
        # ------------------------------------------------------------
        # p_exploit = sigmoid(beta * (value - threshold) + bias_term_if_applicable)
        if self.trial == 0:
            logit = beta * (value - threshold)
        else:
            last_choice = int(self.choice_history[self.trial - 1])
            if last_choice == L:
                # Bias applied only when last action was Left (keeps your original convention)
                logit = beta * (value - threshold) + float(self.params.biasL)
            elif last_choice == R:
                logit = beta * (value - threshold)
            else:
                raise ValueError(f"incompatible choice type: {last_choice}")

        p_exploit = 1.0 / (1.0 + np.exp(-logit))
        # Clamp to avoid exact 0/1 which can break log-likelihood
        p_exploit = float(np.clip(p_exploit, 1e-12, 1.0 - 1e-12))

        # ------------------------------------------------------------
        # Step B: option termination (exploit <-> explore) using p_exploit
        # ------------------------------------------------------------
        # If exploiting, terminate with prob (1 - p_exploit)
        # If exploring, terminate with prob (p_exploit)
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
        # Step C: sample an action given the current option
        # ------------------------------------------------------------
        if self.trial == 0:
            # No "last_choice" exists, so we sample uniformly at trial 0
            choice = self.rng.choice([L, R], p=base_prob)
        else:
            last_choice = int(self.choice_history[self.trial - 1])
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
            last_choice = int(self.choice_history[self.trial - 1])
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

            # Mix probabilities and renormalize
            choice_prob = (1.0 - ck_weight) * choice_prob + ck_weight * ck
            s = float(np.sum(choice_prob))
            if s <= 0.0 or (not np.isfinite(s)):
                raise ValueError(f"choice_prob normalization failed: sum={s}")
            choice_prob = choice_prob / s

            # Resample choice according to the adjusted probabilities if not deterministic
            if np.sum(choice_prob > 0) > 1:
                choice = self.rng.choice([L, R], p=choice_prob)

        return choice, choice_prob

    def _get_alpha_for_trial(self, reward: float) -> float:
        """Return the learning rate alpha for the current trial.

        Logic:
        - If number_of_learning_rate == 1: alpha = learn_rate
        - If number_of_learning_rate == 2:
            alpha = learn_rate_rew if rewarded else learn_rate_unrew

        Rewarded criterion:
        - Here we use `reward > 0` as rewarded. This is robust if reward is 0/1.
          If your task uses other reward magnitudes, this still behaves sensibly.
        """
        n_lr = int(self.agent_kwargs.get("number_of_learning_rate", 1))
        if n_lr == 1:
            return float(self.params.learn_rate)
        if n_lr == 2:
            is_rewarded = float(reward) > 0.0
            return float(self.params.learn_rate_rew if is_rewarded else self.params.learn_rate_unrew)
        raise ValueError(f"number_of_learning_rate must be 1 or 2, got {n_lr}")

    def learn(self, _observation, choice, reward, _next_observation, done):
        """Update latent value and (optionally) the choice kernel.

        IMPORTANT: In this codebase, `self.trial` has already been incremented by 1
        before `learn()` is called. Therefore:
          - The update we write should populate `self.value[self.trial]`
          - The "previous" value is `self.value[self.trial - 1]`
        """
        reset_to_threshold = bool(self.agent_kwargs.get("reset_to_threshold", True))
        alpha = self._get_alpha_for_trial(reward)

        # ------------------------------------------------------------
        # Detect whether a switch occurred (same logic as your original)
        # ------------------------------------------------------------
        # trial == 1 means we just completed the first action; treat it as a "switch"
        # to initialize the latent dynamics similarly to your original behavior.
        switched = False
        if self.trial == 1:
            switched = True
        elif self.trial >= 2:
            # Compare current choice (from trial-1) to the choice at trial-2
            switched = (choice != self.choice_history[self.trial - 2])

        # ------------------------------------------------------------
        # Value update
        # ------------------------------------------------------------
        if reset_to_threshold and switched:
            # Reset-like update toward threshold
            thr = float(self.params.threshold)
            self.value[self.trial] = thr + alpha * (reward - thr)
        else:
            # Standard delta update from previous value
            v_prev = float(self.value[self.trial - 1])
            self.value[self.trial] = v_prev + alpha * (reward - v_prev)

        # ------------------------------------------------------------
        # Choice kernel update (optional)
        # ------------------------------------------------------------
        if self.agent_kwargs["choice_kernel"] != "none":
            self.choice_kernel[:, self.trial] = learn_choice_kernel(
                choice=choice,
                choice_kernel_tminus1=self.choice_kernel[:, self.trial - 1],
                choice_kernel_step_size=self.params.choice_kernel_step_size,
            )

    def get_latent_variables(self):
        """Return latent variables for analysis (consistent with decision rule).

        We also return p_exploit(t) computed from the stored `value` and parameters.
        """
        beta = float(self.params.softmax_inverse_temperature)
        threshold = float(self.params.threshold)
        biasL = float(getattr(self.params, "biasL", 0.0))

        p_exploit_all = []
        for t, v in enumerate(self.value):
            if t == 0:
                logit = beta * (float(v) - threshold)
            else:
                last_choice = int(self.choice_history[t - 1])
                if last_choice == L:
                    logit = beta * (float(v) - threshold) + biasL
                elif last_choice == R:
                    logit = beta * (float(v) - threshold)
                else:
                    raise ValueError(f"incompatible choice type: {last_choice}")

            p = 1.0 / (1.0 + np.exp(-logit))
            p_exploit_all.append(float(p))

        return {
            "value": self.value.tolist(),
            "threshold": [threshold] * (self.n_trials + 1),
            "exploiting": self.exploiting.tolist(),
            "choice_kernel": self.choice_kernel.tolist(),
            "choice_prob": self.choice_prob.tolist(),
            "p_exploit": p_exploit_all,
        }

    def plot_latent_variables(self, ax, if_fitted=False):
        """Plot latent variables.

        Notes:
        - This plot is mostly for visualization; it does not affect fitting.
        - Colors are kept as in your original snippet.
        """
        if if_fitted:
            style = dict(lw=2, ls=":")
            prefix = "fitted_"
        else:
            style = dict(lw=0.5)
            prefix = ""

        x = np.arange(self.n_trials + 1) + 1  # start at 1 for display

        # Plot value
        ax.plot(x, self.value, label=f"{prefix}value", color="purple", **style)

        # Plot threshold as a horizontal line
        ax.axhline(
            y=float(self.params.threshold),
            color="black",
            linestyle="--",
            label=f"{prefix}threshold",
            **style,
        )

        # Plot p(exploit) based on value-threshold (without bias term for quick visualization)
        p_exploit = [
            1.0 / (1.0 + np.exp(-float(self.params.softmax_inverse_temperature) * (float(v) - float(self.params.threshold))))
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
