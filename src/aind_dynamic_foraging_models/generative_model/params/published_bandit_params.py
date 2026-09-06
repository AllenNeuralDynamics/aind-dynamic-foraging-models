"""Parameters for published two-arm bandit baseline models."""

from typing import Tuple, Type

from pydantic import BaseModel, Field

from .util import create_pydantic_models_dynamic


def generate_rlck_params() -> Tuple[Type[BaseModel], Type[BaseModel]]:
    """Return the four-parameter RLCK model used by Chen and Zid."""
    fields = {
        "learn_rate": (
            float,
            Field(default=0.5, ge=0.0, le=1.0, description="Reward learning rate"),
        ),
        "choice_kernel_step_size": (
            float,
            Field(default=0.5, ge=0.0, le=1.0, description="Choice-kernel learning rate"),
        ),
        "softmax_inverse_temperature": (
            float,
            Field(default=10.0, ge=0.0, description="Value inverse temperature"),
        ),
        "choice_kernel_inverse_temperature": (
            float,
            Field(default=1.0, ge=0.0, description="Choice-kernel inverse temperature"),
        ),
    }
    bounds = {
        "learn_rate": (0.0, 1.0),
        "choice_kernel_step_size": (0.0, 1.0),
        "softmax_inverse_temperature": (0.0, 100.0),
        "choice_kernel_inverse_temperature": (0.0, 100.0),
    }
    return create_pydantic_models_dynamic(fields, bounds)


def generate_zid_foraging_params() -> Tuple[Type[BaseModel], Type[BaseModel]]:
    """Return Zid et al.'s history-kernel-2 foraging-RL parameters."""
    fields = {
        "learn_rate": (
            float,
            Field(default=0.5, ge=0.0, le=1.0, description="Exploitation-value learning rate"),
        ),
        "threshold": (
            float,
            Field(default=0.5, ge=0.0, le=2.0, description="Exploitation threshold"),
        ),
        "choice_kernel_step_size": (
            float,
            Field(default=0.5, ge=0.0, le=1.0, description="State-history learning rate"),
        ),
        "softmax_inverse_temperature": (
            float,
            Field(default=10.0, ge=0.0, description="Value inverse temperature"),
        ),
        "choice_kernel_inverse_temperature": (
            float,
            Field(default=1.0, ge=0.0, description="State-history inverse temperature"),
        ),
    }
    bounds = {
        "learn_rate": (0.0, 1.0),
        "threshold": (0.0, 2.0),
        "choice_kernel_step_size": (0.0, 1.0),
        "softmax_inverse_temperature": (0.0, 100.0),
        "choice_kernel_inverse_temperature": (0.0, 100.0),
    }
    return create_pydantic_models_dynamic(fields, bounds)


def generate_grossman_meta_learning_params() -> Tuple[Type[BaseModel], Type[BaseModel]]:
    """Return Grossman et al.'s uncertainty-dependent meta-learning parameters."""
    fields = {
        "learn_rate_rew": (
            float,
            Field(default=0.5, ge=0.0, le=1.0, description="Positive-RPE learning rate"),
        ),
        "learn_rate_unrew": (
            float,
            Field(default=0.2, ge=0.0, le=1.0, description="Baseline negative-RPE rate"),
        ),
        "forgetting_factor": (
            float,
            Field(default=0.8, ge=0.0, le=1.0, description="Unchosen-value retention factor"),
        ),
        "expected_uncertainty_step_size": (
            float,
            Field(default=0.2, ge=0.0, le=1.0, description="Expected-uncertainty rate"),
        ),
        "negative_learning_rate_step_size": (
            float,
            Field(default=0.2, ge=0.0, le=1.0, description="Negative-rate integration"),
        ),
        "choice_bias": (
            float,
            Field(default=0.0, ge=-1.0, le=1.0, description="Right-choice value bias"),
        ),
        "softmax_inverse_temperature": (
            float,
            Field(default=5.0, ge=0.0, le=10.0, description="Inverse temperature"),
        ),
    }
    bounds = {
        "learn_rate_rew": (0.0, 1.0),
        "learn_rate_unrew": (0.0, 1.0),
        "forgetting_factor": (0.0, 1.0),
        "expected_uncertainty_step_size": (0.0, 1.0),
        "negative_learning_rate_step_size": (0.0, 1.0),
        "choice_bias": (-1.0, 1.0),
        "softmax_inverse_temperature": (0.0, 10.0),
    }
    return create_pydantic_models_dynamic(fields, bounds)


def generate_lebedeva_pr_params() -> Tuple[Type[BaseModel], Type[BaseModel]]:
    """Return Lebedeva et al.'s perseveration/reward-learning parameters."""
    fields = {
        "perseveration_learning_rate": (
            float,
            Field(default=0.2, ge=0.0, le=1.0, description="Perseveration learning rate"),
        ),
        "reward_learning_rate": (
            float,
            Field(default=0.2, ge=0.0, le=1.0, description="Reward-learning rate"),
        ),
        "perseveration_weight": (
            float,
            Field(default=1.0, description="Asymptotic perseveration magnitude"),
        ),
        "reward_weight": (
            float,
            Field(default=1.0, description="Asymptotic reward-seeking magnitude"),
        ),
        "choice_bias": (
            float,
            Field(default=0.0, description="Fixed right-choice log-odds bias"),
        ),
    }
    # The authors fit the three log-odds weights without finite bounds. The
    # shared differential-evolution fitter requires finite bounds, so use a
    # numerically saturated range (choice probabilities differ from 0/1 by
    # less than 2e-9 at either endpoint).
    bounds = {
        "perseveration_learning_rate": (0.0, 1.0),
        "reward_learning_rate": (0.0, 1.0),
        "perseveration_weight": (-20.0, 20.0),
        "reward_weight": (-20.0, 20.0),
        "choice_bias": (-20.0, 20.0),
    }
    return create_pydantic_models_dynamic(fields, bounds)


def generate_beron_rflr_params() -> Tuple[Type[BaseModel], Type[BaseModel]]:
    """Return Beron et al.'s recursively formulated logistic-regression parameters."""
    fields = {
        "choice_history_weight": (
            float,
            Field(default=1.0, description="One-trial choice-history weight"),
        ),
        "reward_evidence_weight": (
            float,
            Field(default=1.0, description="Choice-reward evidence weight"),
        ),
        "evidence_time_constant": (
            float,
            Field(default=2.0, gt=0.0, description="Evidence decay time constant in trials"),
        ),
    }
    bounds = {
        "choice_history_weight": (-20.0, 20.0),
        "reward_evidence_weight": (-20.0, 20.0),
        "evidence_time_constant": (0.05, 100.0),
    }
    return create_pydantic_models_dynamic(fields, bounds)


def generate_miller_rhg_params() -> Tuple[Type[BaseModel], Type[BaseModel]]:
    """Return Miller et al.'s reward/habit/gambler-fallacy parameters."""
    fields = {
        "reward_weight": (float, Field(default=1.0, description="Reward-seeking weight")),
        "habit_weight": (float, Field(default=1.0, description="Habit weight")),
        "gambler_fallacy_weight": (
            float,
            Field(default=1.0, description="Gambler's-fallacy weight"),
        ),
        "reward_retention_logit": (
            float,
            Field(default=0.0, description="Logit of reward-state retention"),
        ),
        "habit_retention_logit": (
            float,
            Field(default=0.0, description="Logit of habit-state retention"),
        ),
        "gambler_fallacy_retention_logit": (
            float,
            Field(default=0.0, description="Logit of gambler-state retention"),
        ),
        "choice_bias": (
            float,
            Field(default=0.0, description="Fixed right-choice half-logit bias"),
        ),
    }
    bounds = {
        "reward_weight": (-20.0, 20.0),
        "habit_weight": (-20.0, 20.0),
        "gambler_fallacy_weight": (-20.0, 20.0),
        "reward_retention_logit": (-10.0, 10.0),
        "habit_retention_logit": (-10.0, 10.0),
        "gambler_fallacy_retention_logit": (-10.0, 10.0),
        "choice_bias": (-20.0, 20.0),
    }
    return create_pydantic_models_dynamic(fields, bounds)


def generate_eckstein_rl_params() -> Tuple[Type[BaseModel], Type[BaseModel]]:
    """Return Eckstein et al.'s winning four-parameter RL model."""
    fields = {
        "positive_learning_rate": (
            float,
            Field(default=0.5, ge=0.0, le=1.0, description="Rewarded-trial learning rate"),
        ),
        "negative_learning_rate": (
            float,
            Field(default=0.5, ge=0.0, le=1.0, description="Unrewarded-trial learning rate"),
        ),
        "softmax_inverse_temperature": (
            float,
            Field(default=2.0, ge=0.0, le=15.0, description="Inverse temperature"),
        ),
        "perseveration_bonus": (
            float,
            Field(default=0.0, ge=-1.0, le=1.0, description="Previous-choice value bonus"),
        ),
    }
    bounds = {
        "positive_learning_rate": (0.0, 1.0),
        "negative_learning_rate": (0.0, 1.0),
        "softmax_inverse_temperature": (0.0, 15.0),
        "perseveration_bonus": (-1.0, 1.0),
    }
    return create_pydantic_models_dynamic(fields, bounds)


def generate_eckstein_bi_params() -> Tuple[Type[BaseModel], Type[BaseModel]]:
    """Return Eckstein et al.'s winning four-parameter Bayesian-inference model."""
    fields = {
        "subjective_switch_probability": (
            float,
            Field(default=0.05, ge=0.0, le=1.0, description="Subjective reversal rate"),
        ),
        "subjective_reward_probability": (
            float,
            Field(default=0.75, ge=0.0, le=1.0, description="Subjective correct-side reward rate"),
        ),
        "softmax_inverse_temperature": (
            float,
            Field(default=2.0, ge=0.0, le=15.0, description="Inverse temperature"),
        ),
        "perseveration_bonus": (
            float,
            Field(default=0.0, ge=-1.0, le=1.0, description="Previous-choice belief bonus"),
        ),
    }
    bounds = {
        "subjective_switch_probability": (0.0, 1.0),
        "subjective_reward_probability": (0.0, 1.0),
        "softmax_inverse_temperature": (0.0, 15.0),
        "perseveration_bonus": (-1.0, 1.0),
    }
    return create_pydantic_models_dynamic(fields, bounds)
