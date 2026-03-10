"""Compare-to-threshold foraging model implementation."""

from __future__ import annotations

from typing import Any, Literal, Optional

import numpy as np
from aind_behavior_gym.dynamic_foraging.task import L, R
from numpy import float64
from numpy._typing._array_like import NDArray

from .base import DynamicForagingAgentMLEBase
from .learn_functions import learn_choice_kernel
from .params.forager_compare_threshold_params import (
    generate_pydantic_compare_threshold_param,
)


def _sigmoid_stable(x: float) -> float:
    """Numerically stable sigmoid."""
    if x >= 0:
        z = np.exp(-x)
        return float(1.0 / (1.0 + z))
    z = np.exp(x)
    return float(z / (1.0 + z))


def _logit(p: float, eps: float = 1e-12) -> float:
    """Stable logit with clipping."""
    p = float(np.clip(p, eps, 1.0 - eps))
    return float(np.log(p) - np.log(1.0 - p))


class ForagerCompareThreshold(DynamicForagingAgentMLEBase):
    """Compare-to-threshold foraging model.

    This model tracks a single latent variable `value` and selects actions by comparing
    `value` to a threshold using a logistic mapping.

    Key behavioral assumption (2 actions):
      - "exploit" means repeat the previous choice
      - "explore" means switch to the other side

    Extensions:
      1) stay bias:
         - Additive term on the stay / exploit logit.

      2) side bias:
         - Additive term on logit(P(Left)) after constructing base side probabilities.

      3) choice kernel:
         - Added in logit space, analogous to the Q-learning model:
           logit(P(Left)) += beta * choice_kernel_relative_weight * (K_L - K_R)

      4) fixed threshold:
         - Threshold is fixed rather than learnable.
    """

    def __init__(
        self,
        number_of_learning_rate: Literal[1, 2] = 1,
        choice_kernel: Literal["none", "one_step", "full"] = "none",
        params: Optional[dict] = None,
        reset_to_threshold: Literal[True, False] = True,
        include_stay_bias: Literal[True, False] = False,
        include_side_bias: Literal[True, False] = True,
        fix_threshold: Literal[True, False] = False,
        threshold_fixed: Optional[float] = None,
        **kwargs,
    ):
        """Initialize the compare-to-threshold foraging agent."""
        params = {} if params is None else dict(params)

        if fix_threshold:
            if threshold_fixed is None:
                if "threshold" in params:
                    threshold_fixed = float(params["threshold"])
                else:
                    threshold_fixed = 0.0
            params.pop("threshold", None)

        self.agent_kwargs = dict(
            number_of_learning_rate=number_of_learning_rate,
            choice_kernel=choice_kernel,
            reset_to_threshold=reset_to_threshold,
            include_stay_bias=bool(include_stay_bias),
            include_side_bias=bool(include_side_bias),
            fix_threshold=bool(fix_threshold),
            threshold_fixed=(
                float(threshold_fixed)
                if (fix_threshold and threshold_fixed is not None)
                else None
            ),
        )

        super().__init__(agent_kwargs=self.agent_kwargs, params=params, **kwargs)

    def set_agent_kwargs(self, **agent_kwargs):
        """Update agent hyperparameters after initialization."""
        self.agent_kwargs.update(agent_kwargs)
        return self.agent_kwargs

    def _get_params_model(self, agent_kwargs):
        """Dynamically generate Pydantic models for parameters and fitting bounds."""
        ParamModel, ParamFitBoundModel = generate_pydantic_compare_threshold_params(
            number_of_learning_rate=agent_kwargs["number_of_learning_rate"],
            choice_kernel=agent_kwargs["choice_kernel"],
            include_stay_bias=agent_kwargs.get("include_stay_bias", False),
            include_side_bias=agent_kwargs.get("include_side_bias", False),
            fix_threshold=agent_kwargs.get("fix_threshold", False),
        )
        return ParamModel, ParamFitBoundModel

    def get_agent_alias(self):
        """Get the agent alias string used in tables/plots."""
        parts = ["ForagingCompareThreshold"]
        parts.append(f"_L{self.agent_kwargs.get('number_of_learning_rate', 1)}")

        ck = self.agent_kwargs["choice_kernel"]
        if ck == "one_step":
            parts.append("_CK1")
        elif ck == "full":
            parts.append("_CKfull")
        else:
            parts.append("_CKnone")

        reset_flag = self.agent_kwargs.get("reset_to_threshold", True)
        parts.append(f"_Reset{'T' if reset_flag else 'F'}")

        stay_flag = self.agent_kwargs.get("include_stay_bias", False)
        parts.append(f"_StayBias{'T' if stay_flag else 'F'}")

        side_flag = self.agent_kwargs.get("include_side_bias", True)
        parts.append(f"_SideBias{'T' if side_flag else 'F'}")

        fix_flag = self.agent_kwargs.get("fix_threshold", False)
        parts.append(f"_FixThr{'T' if fix_flag else 'F'}")

        if fix_flag:
            thr = self.agent_kwargs.get("threshold_fixed", None)
            if thr is not None:
                parts.append(f"{thr:.2f}")

        return "".join(parts)

    def _get_threshold_value(self) -> float:
        """Return the threshold value used by the model."""
        if self.agent_kwargs.get("fix_threshold", False):
            thr_fixed = self.agent_kwargs.get("threshold_fixed", None)
            if thr_fixed is None:
                raise ValueError("fix_threshold=True but threshold_fixed is None.")
            return float(thr_fixed)
        return float(self.params.threshold)

    def _get_stay_bias_value(self) -> float:
        """Return the stay bias used in the stay / exploit logit."""
        if self.agent_kwargs.get("include_stay_bias", False):
            return float(getattr(self.params, "stay_bias", 0.0))
        return 0.0

    def _reset(self):
        """Reset the agent state before running a session."""
        super()._reset()

        self.value = np.full(self.n_trials + 1, np.nan)
        self.value[0] = float(self._get_threshold_value())

        self.exploiting = np.full(self.n_trials, False)
        self.current_option = "explore"

        self.choice_kernel = np.full([self.n_actions, self.n_trials + 1], np.nan)
        self.choice_kernel[:, 0] = 0.0

    def act(self, _) -> tuple[Any, NDArray[Any] | Any | NDArray[float64]]:  # noqa: C901
        """Select an action and return (choice, choice_prob)."""
        value = float(self.value[self.trial])
        threshold = float(self._get_threshold_value())
        beta = float(self.params.softmax_inverse_temperature)

        stay_bias = float(self._get_stay_bias_value())
        include_side_bias = bool(self.agent_kwargs.get("include_side_bias", True))
        use_choice_kernel = bool(
            (self.trial > 0) and (self.agent_kwargs["choice_kernel"] != "none")
        )

        base_prob = np.array([0.5, 0.5], dtype=float)

        # Step A: compute p_exploit = P(stay)
        if self.trial == 0:
            exploit_logit = beta * (value - threshold)
        else:
            exploit_logit = beta * (value - threshold) + stay_bias

        p_exploit = _sigmoid_stable(float(exploit_logit))
        p_exploit = float(np.clip(p_exploit, 1e-12, 1.0 - 1e-12))

        # Step B: update exploit / explore state
        if self.current_option == "exploit":
            terminate = self.rng.random() < (1.0 - p_exploit)
        elif self.current_option == "explore":
            terminate = self.rng.random() < p_exploit
        else:
            raise ValueError(f"Unrecognized current_option: {self.current_option}")

        if terminate:
            self.current_option = (
                "explore" if self.current_option == "exploit" else "exploit"
            )

        self.exploiting[self.trial] = self.current_option == "exploit"

        # Step C: construct base action probabilities from exploit / explore rule
        if self.trial == 0:
            choice_prob = base_prob.copy()
        else:
            last_choice = int(self.choice_history[self.trial - 1])
            other_choice = 1 - last_choice

            choice_prob = np.zeros(self.n_actions, dtype=float)
            choice_prob[last_choice] = p_exploit
            choice_prob[other_choice] = 1.0 - p_exploit

            # Step D: move to left-choice logit space and add side bias / choice kernel
            total_logit = _logit(float(choice_prob[L]))

            if include_side_bias:
                total_logit += float(getattr(self.params, "biasL", 0.0))

            if use_choice_kernel:
                ck = self.choice_kernel[:, self.trial]
                ck_weight = float(self.params.choice_kernel_relative_weight)
                kernel_delta = float(ck[L] - ck[R])
                total_logit += beta * ck_weight * kernel_delta

            p_left = _sigmoid_stable(total_logit)
            p_left = float(np.clip(p_left, 1e-12, 1.0 - 1e-12))
            choice_prob[L] = p_left
            choice_prob[R] = 1.0 - p_left

        # Step E: sample final action from final probabilities
        choice = self.rng.choice([L, R], p=choice_prob)

        return choice, choice_prob

    def _get_alpha_for_trial(self, reward: float) -> float:
        """Return the learning rate alpha for the current trial."""
        n_lr = int(self.agent_kwargs.get("number_of_learning_rate", 1))
        if n_lr == 1:
            return float(self.params.learn_rate)
        if n_lr == 2:
            is_rewarded = float(reward) > 0.0
            return float(
                self.params.learn_rate_rew
                if is_rewarded
                else self.params.learn_rate_unrew
            )
        raise ValueError(f"number_of_learning_rate must be 1 or 2, got {n_lr}")

    def learn(self, _observation, choice, reward, _next_observation, done):
        """Update latent value and choice kernel."""
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
        """Return latent variables for analysis."""
        beta = float(self.params.softmax_inverse_temperature)
        threshold = float(self._get_threshold_value())
        stay_bias = float(self._get_stay_bias_value())
        include_side_bias = bool(self.agent_kwargs.get("include_side_bias", True))
        biasL = float(getattr(self.params, "biasL", 0.0))

        p_exploit_all = []
        for t, v in enumerate(self.value):
            if t == 0:
                exploit_logit = beta * (float(v) - threshold)
            else:
                exploit_logit = beta * (float(v) - threshold) + stay_bias
            p_exploit_all.append(float(_sigmoid_stable(float(exploit_logit))))

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
        if include_side_bias:
            out["biasL"] = [biasL] * (self.n_trials + 1)

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

        beta = float(self.params.softmax_inverse_temperature)
        stay_bias = float(self._get_stay_bias_value())

        p_exploit = []
        for t, v in enumerate(self.value):
            if t == 0:
                exploit_logit = beta * (float(v) - threshold)
            else:
                exploit_logit = beta * (float(v) - threshold) + stay_bias
            p_exploit.append(_sigmoid_stable(float(exploit_logit)))

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