"""Dynamically generate pydantic models for Q-learning agent parameters."""

# %%
from typing import Literal, Tuple, Type

from pydantic import BaseModel, Field

from .util import create_pydantic_models_dynamic


def generate_pydantic_q_learning_params(
    number_of_learning_rate: Literal[1, 2] = 2,
    number_of_forget_rate: Literal[0, 1] = 1,
    choice_kernel: Literal["none", "one_step", "full"] = "none",
    action_selection: Literal["softmax", "epsilon-greedy"] = "softmax",
) -> Tuple[Type[BaseModel], Type[BaseModel]]:
    """Dynamically generate Pydantic models for Q-learning agent parameters.

    All default values are hard-coded in this function. But when instantiating the model,
    you can always override the default values, both the params_fields and the fitting bounds.

    Parameters
    ----------
    number_of_learning_rate : Literal[1, 2], optional
        Number of learning rates, by default 2
        If 1, only one learn_rate will be included in the model.
        If 2, learn_rate_rew and learn_rate_unrew will be included in the model.
    number_of_forget_rate : Literal[0, 1], optional
        Number of forget_rates, by default 1.
        If 0, forget_rate_unchosen will not be included in the model.
        If 1, forget_rate_unchosen will be included in the model.
    choice_kernel : Literal["none", "one_step", "full"], optional
        Choice kernel type, by default "none"
        If "none", no choice kernel will be included in the model.
        If "one_step", choice_kernel_step_size will be set to 1.0, i.e., only the previous choice
            affects the choice kernel. (Bari2019)
        If "full", both choice_kernel_step_size and choice_kernel_relative_weight will be included
    action_selection : Literal["softmax", "epsilon-greedy"], optional
        Action selection type, by default "softmax"
    """

    # ====== Define common fields and constraints ======
    params_fields = {}
    fitting_bounds = {}

    # -- Handle learning rate fields --
    _add_learning_rate_fields(params_fields, fitting_bounds, number_of_learning_rate)

    # -- Handle forget rate field --
    _add_forget_rate_fields(params_fields, fitting_bounds, number_of_forget_rate)

    # -- Handle choice kernel fields --
    _add_choice_kernel_fields(params_fields, fitting_bounds, choice_kernel)

    # -- Handle action selection fields --
    _add_action_selection_fields(params_fields, fitting_bounds, action_selection)

    # ====== Dynamically create the pydantic models =====
    return create_pydantic_models_dynamic(params_fields, fitting_bounds)


def _add_learning_rate_fields(params_fields, fitting_bounds, number_of_learning_rate):
    """Add learning rate fields to the params_fields and fitting_bounds dictionaries."""
    assert number_of_learning_rate in [1, 2], "number_of_learning_rate must be 1 or 2"
    if number_of_learning_rate == 1:
        params_fields["learn_rate"] = (
            float,
            Field(default=0.5, ge=0.0, le=1.0, description="Learning rate"),
        )
        fitting_bounds["learn_rate"] = (0.0, 1.0)
    elif number_of_learning_rate == 2:
        params_fields["learn_rate_rew"] = (
            float,
            Field(default=0.5, ge=0.0, le=1.0, description="Learning rate for rewarded choice"),
        )
        fitting_bounds["learn_rate_rew"] = (0.0, 1.0)
        params_fields["learn_rate_unrew"] = (
            float,
            Field(default=0.1, ge=0.0, le=1.0, description="Learning rate for unrewarded choice"),
        )
        fitting_bounds["learn_rate_unrew"] = (0.0, 1.0)


def _add_forget_rate_fields(params_fields, fitting_bounds, number_of_forget_rate):
    """Add forget rate fields to the params_fields and fitting_bounds dictionaries."""
    assert number_of_forget_rate in [0, 1], "number_of_forget_rate must be 0 or 1"
    if number_of_forget_rate == 1:
        params_fields["forget_rate_unchosen"] = (
            float,
            Field(default=0.2, ge=0.0, le=1.0, description="Forgetting rate for unchosen side"),
        )
        fitting_bounds["forget_rate_unchosen"] = (0.0, 1.0)


def _add_choice_kernel_fields(params_fields, fitting_bounds, choice_kernel):
    """Add choice kernel fields to the params_fields and fitting_bounds dictionaries."""
    assert choice_kernel in [
        "none",
        "one_step",
        "full",
    ], "choice_kernel must be 'none', 'one_step', or 'full'"

    if choice_kernel == "none":
        return

    params_fields["choice_kernel_relative_weight"] = (
        float,
        Field(
            default=0.1,
            ge=0.0,
            description=(
                "Relative weight of choice kernel (very sensitive, should be quite small)"
            ),
        ),
    )
    fitting_bounds["choice_kernel_relative_weight"] = (0.0, 1.0)

    if choice_kernel == "full":
        params_fields["choice_kernel_step_size"] = (
            float,
            Field(
                default=0.1,
                ge=0.0,
                le=1.0,
                description="Step size for choice kernel (1.0 means only previous choice)",
            ),
        )
        fitting_bounds["choice_kernel_step_size"] = (0.0, 1.0)
    elif choice_kernel == "one_step":
        # If choice kernel is one-step (only the previous choice affects the choice kernel like
        # in Bari2019), set choice_kernel_step_size to 1.0
        params_fields["choice_kernel_step_size"] = (
            float,
            Field(
                default=1.0,
                ge=1.0,
                le=1.0,
                description="Step size for choice kernel == 1 (one-step choice kernel)",
                frozen=True,  # To indicate that this field is clamped by construction
            ),
        )
        fitting_bounds["choice_kernel_step_size"] = (1.0, 1.0)


def _add_action_selection_fields(params_fields, fitting_bounds, action_selection):
    """Add action selection fields to the params_fields and fitting_bounds dictionaries."""
    # Always include biasL
    params_fields["biasL"] = (float, Field(default=0.0, description="Left bias for softmax"))
    fitting_bounds["biasL"] = (-5.0, 5.0)

    if action_selection == "softmax":
        params_fields["softmax_inverse_temperature"] = (
            float,
            Field(default=10.0, ge=0.0, description="Softmax inverse temperature"),
        )
        fitting_bounds["softmax_inverse_temperature"] = (0.0, 100.0)
    elif action_selection == "epsilon-greedy":
        params_fields["epsilon"] = (
            float,
            Field(default=0.1, ge=0.0, le=1.0, description="Epsilon for epsilon-greedy"),
        )
        fitting_bounds["epsilon"] = (0.0, 1.0)
    else:
        raise ValueError("action_selection must be 'softmax' or 'epsilon-greedy'")
