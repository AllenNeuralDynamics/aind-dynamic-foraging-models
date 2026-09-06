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
