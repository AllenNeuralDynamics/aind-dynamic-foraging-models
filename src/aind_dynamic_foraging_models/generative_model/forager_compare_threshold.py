"""Compare-to-threshold foraging model implementation"""

from __future__ import annotations

from typing import Any, Literal, Optional

import numpy as np
from aind_behavior_gym.dynamic_foraging.task import L, R
from numpy._typing import NDArray

from .base import DynamicForagingAgentMLEBase
from .learn_functions import learn_choice_kernel
from .params.forager_compare_threshold_params import (
    generate_pydantic_compare_threshold_params,
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
    `value` to a threshold using a logistic (sigmoid) mapping (soft decision rule).

    Key behavioral assumption (2 actions):
      - "exploit" means repeat the previous choice (stay)
      - "explore" means switch to the other side (leave)

    Extensions:
      1) stay bias option:
         - If enabled, add an additive bias term on the *stay / exploit* logit.
         - This corresponds to a "perseveration" tendency beyond value/threshold.

      2) side bias option:
         - If enabled, apply a side (Right-vs-Left) bias in logit space AFTER
           mapping stay/switch -> side choice probability.
         - This isolates spatial/motor bias from the stay/switch mechanism.

      3) fixed threshold option:
         - If enabled, threshold is a fixed constant (NOT learnable).
         - Implementation: threshold lives only in `agent_kwargs` and is accessed via
           `_get_threshold_value()`. When fixed, `threshold` is NOT included in ParamModel,
           so it will NOT show up in free params / n_free_params / ρ symbol lists.

    Important implementation note (indexing convention):
      - We maintain `value` with length (n_trials + 1).
      - `value[t]` is the latent available BEFORE acting on trial t.
      - After observing (choice, reward) on trial t, we write `value[t+1]`.
      - This is the standard RL convention and avoids overwriting the state used to act.
    """

    def __init__(
        self,
        number_of_learning_rate: Literal[1, 2] = 1,
        choice_kernel: Literal["none", "one_step", "full"] = "none",
        params: Optional[dict[str, Any]] = None,
        reset_to_threshold: Literal[True, False] = True,
        # Options
        include_stay_bias: Literal[True, False] = False,
        include_side_bias: Literal[True, False] = True,
        fix_threshold: Literal[True, False] = False,
        threshold_fixed: Optional[float] = None,
        **kwargs: Any,
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
            Initial parameters of the model, by default None.
            NOTE: This should only contain Pydantic parameters (fitted parameters), e.g.
              - softmax_inverse_temperature, biasL, learn_rate (or learn_rate_rew/unrew)
              - stay_bias (if include_stay_bias=True)
              - side_bias (if include_side_bias=True)
              - threshold ONLY if fix_threshold=False
        reset_to_threshold : Literal[True, False], optional
            Hyperparameter controlling value update rule at a switch.
        include_stay_bias : bool, optional
            If True, include a fitted parameter `stay_bias` that adds to the logit of P(stay).
        include_side_bias : bool, optional
            If True, include a fitted parameter `side_bias` applied to logit(P(R))
            after mapping stay/switch -> side probability.
        fix_threshold : bool, optional
            If True, the threshold is fixed (not learnable). Use `threshold_fixed`.
        threshold_fixed : Optional[float], optional
            Fixed threshold value when fix_threshold=True.
        **kwargs
            Passed to the base class (e.g., seed for rng).
        """
        # Defensive copy and avoid mutable default arguments.
        params = dict(params or {})

        # If threshold is fixed:
        # - choose a fixed threshold value (from threshold_fixed, or from params["threshold"], or default)
        # - remove "threshold" from fitted parameters (so ParamModel does not include it)
        if fix_threshold:
            if threshold_fixed is None:
                if "threshold" in params:
                    threshold_fixed = float(params["threshold"])
                else:
                    threshold_fixed = 0.0  # keep consistent with generator default
            params.pop("threshold", None)

        self.agent_kwargs = dict(
            number_of_learning_rate=int(number_of_learning_rate),
            choice_kernel=str(choice_kernel),
            reset_to_threshold=bool(reset_to_threshold),
            include_stay_bias=bool(include_stay_bias),
            include_side_bias=bool(include_side_bias),
            fix_threshold=bool(fix_threshold),
            threshold_fixed=float(threshold_fixed) if (fix_threshold and threshold_fixed is not None) else None,
        )

        super().__init__(agent_kwargs=self.agent_kwargs, params=params, **kwargs)

    def set_agent_kwargs(self, **agent_kwargs: Any):
        """Update agent hyperparameters (agent_kwargs) after initialization.

        NOTE:
        - This does NOT rebuild ParamModel/ParamFitBoundModel.
        - Therefore, do NOT change hyperparameters that affect the parameter set
          (e.g., number_of_learning_rate, choice_kernel, include_stay_bias,
           include_side_bias, fix_threshold) after initialization.
        """
        self.agent_kwargs.update(agent_kwargs)
        return self.agent_kwargs

    def _get_params_model(self, agent_kwargs: dict[str, Any]):
        """Dynamically generate Pydantic models for parameters and fitting bounds."""
        ParamModel, ParamFitBoundModel = generate_pydantic_compare_threshold_params(
            number_of_learning_rate=agent_kwargs["number_of_learning_rate"],
            choice_kernel=agent_kwargs["choice_kernel"],
            include_stay_bias=agent_kwargs.get("include_stay_bias", False),
            include_side_bias=agent_kwargs.get("include_side_bias", False),
            fix_threshold=agent_kwargs.get("fix_threshold", False),
        )
        return ParamModel, ParamFitBoundModel

    def get_agent_alias(self) -> str:
        """Get the agent alias string used in tables/plots."""
        parts: list[str] = ["ForagingCompareThreshold"]

        # Learning rate structure
        parts.append(f"_L{int(self.agent_kwargs.get('number_of_learning_rate', 1))}")

        # Choice kernel
        ck = str(self.agent_kwargs["choice_kernel"])
        if ck == "one_step":
            parts.append("_CK1")
        elif ck == "full":
            parts.append("_CKfull")
        else:
            parts.append("_CKnone")

        # Reset flag (explicit)
        reset_flag = bool(self.agent_kwargs.get("reset_to_threshold", True))
        parts.append(f"_Reset{'T' if reset_flag else 'F'}")

        # Stay bias flag (explicit)
        stay_flag = bool(self.agent_kwargs.get("include_stay_bias", False))
        parts.append(f"_StayBias{'T' if stay_flag else 'F'}")

        # Side bias flag (explicit)
        side_flag = bool(self.agent_kwargs.get("include_side_bias", False))
        parts.append(f"_SideBias{'T' if side_flag else 'F'}")

        # Fixed threshold flag (explicit)
        fix_flag = bool(self.agent_kwargs.get("fix_threshold", False))
        parts.append(f"_FixThr{'T' if fix_flag else 'F'}")

        # If fixed threshold is True, include the numeric value
        if fix_flag:
            thr = self.agent_kwargs.get("threshold_fixed", None)
            if thr is not None:
                parts.append(f"{float(thr):.2f}")

        return "".join(parts)

    def _get_threshold_value(self) -> float:
        """Return the threshold value used by the model (fixed or learnable)."""
        if bool(self.agent_kwargs.get("fix_threshold", False)):
            thr_fixed = self.agent_kwargs.get("threshold_fixed", None)
            if thr_fixed is None:
                raise ValueError("fix_threshold=True but threshold_fixed is None.")
            return float(thr_fixed)

        # Learnable threshold must exist in ParamModel
        return float(self.params.threshold)

    def _get_stay_bias_value(self) -> float:
        """Return the stay bias used in the stay/exploit logit."""
        if bool(self.agent_kwargs.get("include_stay_bias", False)):
            return float(getattr(self.params, "stay_bias", 0.0))
        return 0.0

    def _get_side_bias_value(self) -> float:
        """Return the side bias applied to logit(P(R)) after stay/switch mapping."""
        if bool(self.agent_kwargs.get("include_side_bias", False)):
            return float(getattr(self.params, "side_bias", 0.0))
        return 0.0

    def _reset(self) -> None:
        """Reset the agent state before running a session (generative or predictive)."""
        super()._reset()

        # ---------------------------------------------------------------------
        # Agent-family state variables
        # ---------------------------------------------------------------------
        # value has length (n_trials + 1) under RL indexing convention:
        # - value[t] used to act on trial t
        # - learn() writes value[t+1]
        self.value: NDArray[np.float64] = np.full(self.n_trials + 1, np.nan, dtype=float)
        self.value[0] = float(self._get_threshold_value())  # start at threshold

        # Track whether the agent is in exploit mode (per trial)
        self.exploiting: NDArray[np.bool_] = np.full(self.n_trials, False, dtype=bool)

        # Current high-level option state
        self.current_option: Literal["exploit", "explore"] = "explore"  # start exploring at first trial

        # Choice kernel (optional), shape (n_actions, n_trials+1) so it can be used at act(t)
        self.choice_kernel: NDArray[np.float64] = np.full((self.n_actions, self.n_trials + 1), np.nan, dtype=float)
        # Neutral initial kernel. If your original kernel logic assumes zeros, change this to 0.0.
        self.choice_kernel[:, 0] = 0.5

        # Record the final choice probabilities used to sample the action per trial
        self.choice_prob: NDArray[np.float64] = np.full((self.n_trials, self.n_actions), np.nan, dtype=float)

    def act(self, _) -> tuple[Any, NDArray[np.float64]]:  # noqa: C901
        """Select an action and return (choice, choice_prob)."""
        t = int(self.trial)

        value_t = float(self.value[t])
        threshold = float(self._get_threshold_value())
        beta = float(self.params.softmax_inverse_temperature)

        stay_bias = float(self._get_stay_bias_value())
        side_bias = float(self._get_side_bias_value())

        eps = 1e-12
        base_prob = np.array([0.5, 0.5], dtype=float)

        # ------------------------------------------------------------
        # Step A: compute p_stay = P(stay)
        #   NOTE: side bias is NOT applied here (by design).
        # ------------------------------------------------------------
        if t == 0:
            logit_stay = beta * (value_t - threshold)
        else:
            logit_stay = beta * (value_t - threshold) + stay_bias

        p_stay = _sigmoid_stable(logit_stay)
        p_stay = float(np.clip(p_stay, eps, 1.0 - eps))

        # ------------------------------------------------------------
        # Step B: option termination (exploit <-> explore)
        # ------------------------------------------------------------
        if self.current_option == "exploit":
            terminate = self.rng.random() < (1.0 - p_stay)
        elif self.current_option == "explore":
            terminate = self.rng.random() < p_stay
        else:
            raise ValueError(f"unrecognized current_option: {self.current_option}")

        if terminate:
            self.current_option = "explore" if self.current_option == "exploit" else "exploit"

        self.exploiting[t] = self.current_option == "exploit"

        # ------------------------------------------------------------
        # Step C: sample an action given the current option
        # ------------------------------------------------------------
        if t == 0:
            # No choice history exists at trial 0.
            # Start from unbiased base_prob, then apply side bias in logit space if enabled.
            pR_base = 0.5

            if bool(self.agent_kwargs.get("include_side_bias", False)):
                logit_pR = _logit(pR_base, eps=eps) + side_bias
                pR = _sigmoid_stable(logit_pR)
            else:
                pR = pR_base

            pR = float(np.clip(pR, eps, 1.0 - eps))
            choice_prob = np.array([1.0 - pR, pR], dtype=float)

            # Optional: apply choice kernel influence BEFORE sampling
            if str(self.agent_kwargs.get("choice_kernel", "none")) != "none":
                ck = np.asarray(self.choice_kernel[:, t], dtype=float)
                ck_weight = float(self.params.choice_kernel_relative_weight)
                choice_prob = (1.0 - ck_weight) * choice_prob + ck_weight * ck
                s = float(np.sum(choice_prob))
                if s <= 0.0 or (not np.isfinite(s)):
                    raise ValueError(f"choice_prob normalization failed: sum={s}")
                choice_prob = choice_prob / s

            choice = self.rng.choice([L, R], p=choice_prob)
            self.choice_prob[t, :] = choice_prob
            return choice, choice_prob

        # After trial 0, we can map stay/switch to L/R based on last_choice.
        last_choice = int(self.choice_history[t - 1])
        if last_choice not in (L, R):
            raise ValueError(f"incompatible choice type: {last_choice}")

        other_choice = 1 - last_choice

        # Base mapping (before side bias):
        # - exploit: stay on last_choice with probability p_stay
        # - explore: switch to other_choice with probability p_stay
        p_base = np.zeros(self.n_actions, dtype=float)
        if self.current_option == "exploit":
            p_base[last_choice] = p_stay
            p_base[other_choice] = 1.0 - p_stay
        elif self.current_option == "explore":
            p_base[last_choice] = 1.0 - p_stay
            p_base[other_choice] = p_stay
        else:
            raise ValueError(f"unrecognized current_option: {self.current_option}")

        # ------------------------------------------------------------
        # Step D: apply side bias AFTER mapping, in logit space on P(R)
        # ------------------------------------------------------------
        pR_base = float(p_base[R])
        if bool(self.agent_kwargs.get("include_side_bias", False)):
            logit_pR = _logit(pR_base, eps=eps) + side_bias
            pR = _sigmoid_stable(logit_pR)
        else:
            pR = pR_base

        pR = float(np.clip(pR, eps, 1.0 - eps))
        choice_prob = np.array([1.0 - pR, pR], dtype=float)

        # ------------------------------------------------------------
        # Optional: apply choice kernel influence BEFORE sampling
        # ------------------------------------------------------------
        if str(self.agent_kwargs.get("choice_kernel", "none")) != "none":
            ck = np.asarray(self.choice_kernel[:, t], dtype=float)
            ck_weight = float(self.params.choice_kernel_relative_weight)

            choice_prob = (1.0 - ck_weight) * choice_prob + ck_weight * ck
            s = float(np.sum(choice_prob))
            if s <= 0.0 or (not np.isfinite(s)):
                raise ValueError(f"choice_prob normalization failed: sum={s}")
            choice_prob = choice_prob / s

        # Sample choice from final choice_prob
        choice = self.rng.choice([L, R], p=choice_prob)

        # Store for analysis
        self.choice_prob[t, :] = choice_prob

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

    def learn(self, _observation, choice, reward, _next_observation, done) -> None:
        """Update latent value and (optionally) the choice kernel."""
        t = int(self.trial)

        reset_to_threshold = bool(self.agent_kwargs.get("reset_to_threshold", True))
        alpha = float(self._get_alpha_for_trial(float(reward)))
        threshold = float(self._get_threshold_value())

        # Detect whether CURRENT choice is a switch relative to previous trial
        if t == 0:
            switched = False
        else:
            prev_choice = int(self.choice_history[t - 1])
            switched = int(choice) != prev_choice

        # Update rule:
        # - if reset_to_threshold and switched: reset baseline to threshold then update toward reward
        # - else: standard delta-rule toward reward from previous value
        #
        # IMPORTANT: write into value[t+1] (do not overwrite value[t] used in act()).
        if reset_to_threshold and switched:
            self.value[t + 1] = threshold + alpha * (float(reward) - threshold)
        else:
            v_prev = float(self.value[t])
            self.value[t + 1] = v_prev + alpha * (float(reward) - v_prev)

        # Choice kernel update (if enabled):
        # Keep the kernel aligned with RL indexing as well:
        # - kernel[:, t] used in act(t)
        # - after observing choice(t), update kernel[:, t+1]
        if str(self.agent_kwargs.get("choice_kernel", "none")) != "none":
            self.choice_kernel[:, t + 1] = learn_choice_kernel(
                choice=choice,
                choice_kernel_tminus1=self.choice_kernel[:, t],
                choice_kernel_step_size=float(self.params.choice_kernel_step_size),
            )

    def get_latent_variables(self) -> dict[str, Any]:
        """Return latent variables for analysis (consistent with decision rule)."""
        beta = float(self.params.softmax_inverse_temperature)
        threshold = float(self._get_threshold_value())
        stay_bias = float(self._get_stay_bias_value())
        side_bias = float(self._get_side_bias_value())

        p_exploit_all: list[float] = []
        pR_all: list[float] = []

        eps = 1e-12

        # value has length (n_trials + 1); choice_history has length n_trials
        for t, v in enumerate(self.value):
            # Compute p(stay) from value vs threshold (+ optional stay_bias for t>0)
            if t == 0:
                logit_stay = beta * (float(v) - threshold)
            else:
                logit_stay = beta * (float(v) - threshold) + stay_bias

            p_exploit = _sigmoid_stable(logit_stay)
            p_exploit = float(np.clip(p_exploit, eps, 1.0 - eps))

            # Compute an implied P(R) from p(stay) and the last choice.
            # This is a diagnostic latent consistent with the stay/switch mapping.
            # Note: act() also depends on current_option; this p_right does not encode that.
            if t == 0:
                pR_base = 0.5
            else:
                last_choice = int(self.choice_history[t - 1])
                other_choice = 1 - last_choice

                p_base = np.zeros(2, dtype=float)
                p_base[last_choice] = p_exploit
                p_base[other_choice] = 1.0 - p_exploit
                pR_base = float(p_base[R])

            # Side bias in logit space on P(R)
            if bool(self.agent_kwargs.get("include_side_bias", False)):
                pR = _sigmoid_stable(_logit(pR_base, eps=eps) + side_bias)
            else:
                pR = pR_base

            pR = float(np.clip(pR, eps, 1.0 - eps))

            p_exploit_all.append(float(p_exploit))
            pR_all.append(float(pR))

        out: dict[str, Any] = {
            "value": self.value.tolist(),
            "threshold": [threshold] * (self.n_trials + 1),
            "exploiting": self.exploiting.tolist(),
            "choice_kernel": self.choice_kernel.tolist(),
            "choice_prob": self.choice_prob.tolist(),
            "p_exploit": p_exploit_all,
            "p_right": pR_all,
        }
        if bool(self.agent_kwargs.get("include_stay_bias", False)):
            out["stay_bias"] = [stay_bias] * (self.n_trials + 1)
        if bool(self.agent_kwargs.get("include_side_bias", False)):
            out["side_bias"] = [side_bias] * (self.n_trials + 1)
        return out

    def plot_latent_variables(self, ax, if_fitted: bool = False) -> None:
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
        ax.axhline(y=threshold, color="black", linestyle="--", label=f"{prefix}threshold", **style)

        beta = float(self.params.softmax_inverse_temperature)
        stay_bias = float(self._get_stay_bias_value())

        p_exploit = []
        for t, v in enumerate(self.value):
            if t == 0:
                logit_stay = beta * (float(v) - threshold)
            else:
                logit_stay = beta * (float(v) - threshold) + stay_bias
            p_exploit.append(_sigmoid_stable(logit_stay))

        ax.plot(x, p_exploit, label=f"{prefix}p(exploit)", color="cyan", **style)

        if str(self.agent_kwargs.get("choice_kernel", "none")) != "none":
            ax.plot(x, self.choice_kernel[L, :], label=f"{prefix}choice_kernel(L)", color="red", **style)
            ax.plot(x, self.choice_kernel[R, :], label=f"{prefix}choice_kernel(R)", color="blue", **style)