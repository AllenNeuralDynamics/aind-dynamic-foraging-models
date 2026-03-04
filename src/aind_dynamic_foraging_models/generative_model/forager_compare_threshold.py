"""Compare-to-threshold foraging model implementation"""

from __future__ import annotations

from typing import Any, Literal, Optional

import numpy as np
from aind_behavior_gym.dynamic_foraging.task import L, R
from numpy import float64
from numpy._typing._array_like import NDArray

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

    Extensions:
      1) stay bias option:
         - If enabled, add an additive bias term on the *stay / exploit* logit.
         - This corresponds to a "perseveration" tendency beyond value/threshold.

      2) fixed threshold option:
         - If enabled, threshold is a fixed constant (NOT learnable).
         - Implementation: threshold lives only in `agent_kwargs` and is accessed via
           `_get_threshold_value()`. When fixed, `threshold` is NOT included in ParamModel,
           so it will NOT show up in free params / n_free_params / ρ symbol lists.
    """

    def __init__(
        self,
        number_of_learning_rate: Literal[1, 2] = 1,
        choice_kernel: Literal["none", "one_step", "full"] = "none",
        params: dict = {},
        reset_to_threshold: Literal[True, False] = True,
        # New options
        include_stay_bias: Literal[True, False] = False,
        fix_threshold: Literal[True, False] = False,
        threshold_fixed: Optional[float] = None,
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
        params : dict, optional
            Initial parameters of the model, by default {}.
            NOTE: This should only contain Pydantic parameters (fitted parameters), e.g.
              - softmax_inverse_temperature, biasL, learn_rate (or learn_rate_rew/unrew)
              - stay_bias (if include_stay_bias=True)
              - threshold ONLY if fix_threshold=False
        reset_to_threshold : Literal[True, False], optional
            Hyperparameter controlling value update rule at a switch.
        include_stay_bias : bool, optional
            If True, include a fitted parameter `stay_bias` that adds to the logit of P(stay).
        fix_threshold : bool, optional
            If True, the threshold is fixed (not learnable). Use `threshold_fixed`.
        threshold_fixed : Optional[float], optional
            Fixed threshold value when fix_threshold=True.
            If None, we will try to use:
              - params["threshold"] if present, otherwise
              - 0 (match generator default)
        **kwargs
            Passed to the base class (e.g., seed for rng).
        """

        # ---------------------------------------------------------------------
        # Resolve fixed threshold (if enabled)
        # ---------------------------------------------------------------------
        params = dict(params)  # defensive copy

        if fix_threshold:
            if threshold_fixed is None:
                if "threshold" in params:
                    threshold_fixed = float(params["threshold"])
                else:
                    threshold_fixed = 0  # keep consistent with generator default

            # IMPORTANT: threshold is NOT a fitted parameter when fixed
            params.pop("threshold", None)

        # ---------------------------------------------------------------------
        # Pack the agent hyperparameters (agent_kwargs).
        # ---------------------------------------------------------------------
        self.agent_kwargs = dict(
            number_of_learning_rate=number_of_learning_rate,
            choice_kernel=choice_kernel,
            reset_to_threshold=reset_to_threshold,
            include_stay_bias=bool(include_stay_bias),
            fix_threshold=bool(fix_threshold),
            threshold_fixed=float(threshold_fixed) if (fix_threshold and threshold_fixed is not None) else None,
        )

        # Initialize the model parameters (Pydantic) via the base class
        super().__init__(agent_kwargs=self.agent_kwargs, params=params, **kwargs)

    def set_agent_kwargs(self, **agent_kwargs):
        """Update agent hyperparameters (agent_kwargs) after initialization.

        NOTE:
        - This does NOT rebuild ParamModel/ParamFitBoundModel.
        - Therefore, do NOT change hyperparameters that affect the parameter set
          (e.g., number_of_learning_rate, choice_kernel, include_stay_bias, fix_threshold)
          after initialization. If you need a different structure, instantiate a new agent.
        """
        self.agent_kwargs.update(agent_kwargs)
        return self.agent_kwargs

    def _get_params_model(self, agent_kwargs):
        """Dynamically generate Pydantic models for parameters and fitting bounds.

        IMPORTANT:
        - We DO NOT pass threshold_fixed into the generator (not part of learnable params).
        - When fix_threshold=True, the generator must NOT include a 'threshold' field.
        """
        ParamModel, ParamFitBoundModel = generate_pydantic_compare_threshold_params(
            number_of_learning_rate=agent_kwargs["number_of_learning_rate"],
            choice_kernel=agent_kwargs["choice_kernel"],
            include_stay_bias=agent_kwargs.get("include_stay_bias", False),
            fix_threshold=agent_kwargs.get("fix_threshold", False),
        )
        return ParamModel, ParamFitBoundModel

    def get_agent_alias(self):
        """Get the agent alias string used in tables/plots."""
        parts = ["ForagingCompareThreshold"]

        # Learning rate structure
        parts.append(f"_L{self.agent_kwargs.get('number_of_learning_rate', 1)}")

        # Choice kernel
        ck = self.agent_kwargs["choice_kernel"]
        if ck == "one_step":
            parts.append("_CK1")
        elif ck == "full":
            parts.append("_CKfull")
        else:
            parts.append("_CKnone")

        # Reset flag (explicit)
        reset_flag = self.agent_kwargs.get("reset_to_threshold", True)
        parts.append(f"_Reset{'T' if reset_flag else 'F'}")

        # Stay bias flag (explicit)
        stay_flag = self.agent_kwargs.get("include_stay_bias", False)
        parts.append(f"_StayBias{'T' if stay_flag else 'F'}")

        # Fixed threshold flag (explicit)
        fix_flag = self.agent_kwargs.get("fix_threshold", False)
        parts.append(f"_FixThr{'T' if fix_flag else 'F'}")

        # If fixed threshold is True, include value
        if fix_flag:
            thr = self.agent_kwargs.get("threshold_fixed", None)
            if thr is not None:
                parts.append(f"{thr:.2f}")

        return "".join(parts)

    def _get_threshold_value(self) -> float:
        """Return the threshold value used by the model (fixed or learnable)."""
        if self.agent_kwargs.get("fix_threshold", False):
            thr_fixed = self.agent_kwargs.get("threshold_fixed", None)
            if thr_fixed is None:
                raise ValueError("fix_threshold=True but threshold_fixed is None.")
            return float(thr_fixed)

        # Learnable threshold must exist in ParamModel
        return float(self.params.threshold)

    def _get_stay_bias_value(self) -> float:
        """Return the stay bias used in the stay/exploit logit."""
        if self.agent_kwargs.get("include_stay_bias", False):
            return float(getattr(self.params, "stay_bias", 0.0))
        return 0.0

    def _reset(self):
        """Reset the agent state before running a session (generative or predictive)."""
        super()._reset()

        # ---------------------------------------------------------------------
        # Agent-family state variables
        # ---------------------------------------------------------------------
        self.value = np.full(self.n_trials + 1, np.nan)
        self.value[0] = float(self._get_threshold_value())  # start at threshold

        self.exploiting = np.full(self.n_trials, False)
        self.current_option = "explore"  # start exploring at first trial

        self.choice_kernel = np.full([self.n_actions, self.n_trials + 1], np.nan)
        self.choice_kernel[:, 0] = 0.0  # initial kernel

    def act(self, _) -> tuple[Any, NDArray[Any] | Any | NDArray[float64]]:  # noqa: C901
        """Select an action and return (choice, choice_prob)."""
        value = float(self.value[self.trial])
        threshold = float(self._get_threshold_value())
        beta = float(self.params.softmax_inverse_temperature)

        stay_bias = float(self._get_stay_bias_value())

        base_prob = np.array([0.5, 0.5], dtype=float)

        # ------------------------------------------------------------
        # Step A: compute p_exploit = P(stay)
        # ------------------------------------------------------------
        if self.trial == 0:
            logit = beta * (value - threshold)
        else:
            last_choice = int(self.choice_history[self.trial - 1])

            # Side bias convention: biasL applied only when last action was Left
            side_bias_term = 0.0
            if last_choice == L:
                side_bias_term = float(self.params.biasL)
            elif last_choice == R:
                side_bias_term = 0.0
            else:
                raise ValueError(f"incompatible choice type: {last_choice}")

            # Stay bias biases toward staying (exploit) on t>0
            logit = beta * (value - threshold) + side_bias_term + stay_bias

        p_exploit = 1.0 / (1.0 + np.exp(-logit))
        p_exploit = float(np.clip(p_exploit, 1e-12, 1.0 - 1e-12))

        # ------------------------------------------------------------
        # Step B: option termination (exploit <-> explore)
        # ------------------------------------------------------------
        if self.current_option == "exploit":
            terminate = self.rng.random() < (1.0 - p_exploit)
        elif self.current_option == "explore":
            terminate = self.rng.random() < p_exploit
        else:
            raise ValueError(f"unrecognized current_option: {self.current_option}")

        if terminate:
            self.current_option = "explore" if self.current_option == "exploit" else "exploit"

        self.exploiting[self.trial] = self.current_option == "exploit"

        # ------------------------------------------------------------
        # Step C: sample an action given the current option
        # ------------------------------------------------------------
        if self.trial == 0:
            choice = self.rng.choice([L, R], p=base_prob)
        else:
            last_choice = int(self.choice_history[self.trial - 1])
            if self.current_option == "exploit":
                choice = last_choice
            elif self.current_option == "explore":
                choice = 1 - last_choice
            else:
                raise ValueError(f"unrecognized current_option: {self.current_option}")

        # ------------------------------------------------------------
        # Step D: implied action probabilities
        # ------------------------------------------------------------
        if self.trial == 0:
            choice_prob = base_prob.copy()
        else:
            last_choice = int(self.choice_history[self.trial - 1])
            other_choice = 1 - last_choice
            choice_prob = np.zeros(self.n_actions, dtype=float)
            choice_prob[last_choice] = p_exploit
            choice_prob[other_choice] = 1.0 - p_exploit

        # ------------------------------------------------------------
        # Optional: apply choice kernel influence
        # ------------------------------------------------------------
        if (self.trial > 0) and (self.agent_kwargs["choice_kernel"] != "none"):
            ck = self.choice_kernel[:, self.trial]
            ck_weight = float(self.params.choice_kernel_relative_weight)

            choice_prob = (1.0 - ck_weight) * choice_prob + ck_weight * ck
            s = float(np.sum(choice_prob))
            if s <= 0.0 or (not np.isfinite(s)):
                raise ValueError(f"choice_prob normalization failed: sum={s}")
            choice_prob = choice_prob / s

            if np.sum(choice_prob > 0) > 1:
                choice = self.rng.choice([L, R], p=choice_prob)

        return choice, choice_prob

    def _get_alpha_for_trial(self, reward: float) -> float:
        """Return the learning rate alpha for the current trial."""
        n_lr = int(self.agent_kwargs.get("number_of_learning_rate", 1))
        if n_lr == 1:
            return float(self.params.learn_rate)
        if n_lr == 2:
            is_rewarded = float(reward) > 0.0
            return float(self.params.learn_rate_rew if is_rewarded else self.params.learn_rate_unrew)
        raise ValueError(f"number_of_learning_rate must be 1 or 2, got {n_lr}")

    def learn(self, _observation, choice, reward, _next_observation, done):
        """Update latent value and (optionally) the choice kernel."""
        reset_to_threshold = bool(self.agent_kwargs.get("reset_to_threshold", True))
        alpha = self._get_alpha_for_trial(reward)

        switched = False
        if self.trial == 1:
            switched = True
        elif self.trial >= 2:
            switched = choice != self.choice_history[self.trial - 2]

        threshold = float(self._get_threshold_value())
        if reset_to_threshold and switched:
            self.value[self.trial] = threshold + alpha * (reward - threshold)
        else:
            v_prev = float(self.value[self.trial - 1])
            self.value[self.trial] = v_prev + alpha * (reward - v_prev)

        if self.agent_kwargs["choice_kernel"] != "none":
            self.choice_kernel[:, self.trial] = learn_choice_kernel(
                choice=choice,
                choice_kernel_tminus1=self.choice_kernel[:, self.trial - 1],
                choice_kernel_step_size=self.params.choice_kernel_step_size,
            )

    def get_latent_variables(self):
        """Return latent variables for analysis (consistent with decision rule)."""
        beta = float(self.params.softmax_inverse_temperature)
        threshold = float(self._get_threshold_value())
        biasL = float(getattr(self.params, "biasL", 0.0))
        stay_bias = float(self._get_stay_bias_value())

        p_exploit_all = []
        for t, v in enumerate(self.value):
            if t == 0:
                logit = beta * (float(v) - threshold)
            else:
                last_choice = int(self.choice_history[t - 1])
                if last_choice == L:
                    side_bias_term = biasL
                elif last_choice == R:
                    side_bias_term = 0.0
                else:
                    raise ValueError(f"incompatible choice type: {last_choice}")

                logit = beta * (float(v) - threshold) + side_bias_term + stay_bias

            p = 1.0 / (1.0 + np.exp(-logit))
            p_exploit_all.append(float(p))

        out = {
            "value": self.value.tolist(),
            "threshold": [threshold] * (self.n_trials + 1),
            "exploiting": self.exploiting.tolist(),
            "choice_kernel": self.choice_kernel.tolist(),
            "choice_prob": self.choice_prob.tolist(),
            "p_exploit": p_exploit_all,
        }
        if self.agent_kwargs.get("include_stay_bias", False):
            out["stay_bias"] = [stay_bias] * (self.n_trials + 1)
        return out

    def plot_latent_variables(self, ax, if_fitted=False):
        """Plot latent variables."""
        if if_fitted:
            style = dict(lw=2, ls=":")
            prefix = "fitted_"
        else:
            style = dict(lw=0.5)
            prefix = ""

        x = np.arange(self.n_trials + 1) + 1
        threshold = float(self._get_threshold_value())

        ax.plot(x, self.value, label=f"{prefix}value", color="purple", **style)

        ax.axhline(
            y=threshold,
            color="black",
            linestyle="--",
            label=f"{prefix}threshold",
            **style,
        )

        p_exploit = [
            1.0
            / (
                1.0
                + np.exp(
                    -float(self.params.softmax_inverse_temperature) * (float(v) - threshold)
                )
            )
            for v in self.value
        ]
        ax.plot(x, p_exploit, label=f"{prefix}p(exploit)", color="cyan", **style)

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
