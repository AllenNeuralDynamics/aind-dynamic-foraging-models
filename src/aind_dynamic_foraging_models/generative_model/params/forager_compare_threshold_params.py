"""Generate pydantic models for Compare-to-threshold foraging agent parameters."""

from typing import Literal, Tuple, Type

from pydantic import BaseModel, Field

from .forager_q_learning_params import _add_choice_kernel_fields
from .util import create_pydantic_models_dynamic


def generate_pydantic_compare_threshold_params(
    choice_kernel: Literal["none", "one_step", "full"] = "none",
) -> Tuple[Type[BaseModel], Type[BaseModel]]:
    """Generate Pydantic models for Compare-to-threshold foraging agent parameters.

    All default values are hard-coded in this function. But when instantiating the model,
    you can always override the default values, both the params and the fitting bounds.

    Parameters
    ----------
    choice_kernel : Literal["none", "one_step", "full"], optional
        Choice kernel type, by default "none"
        If "none", no choice kernel will be included in the model.
        If "one_step", choice_kernel_step_size will be set to 1.0, i.e., only the previous choice
            affects the choice kernel.
        If "full", both choice_kernel_step_size and choice_kernel_relative_weight will be included
    """

    # ====== Define common fields and constraints ======
    params_fields = {}
    fitting_bounds = {}

    # -- Basic model parameters --
    # Learning rate for value update
    params_fields["learn_rate"] = (
        float,
        Field(default=0.5, ge=0.0, le=1.0, description="Learning rate for value update"),
    )
    fitting_bounds["learn_rate"] = (0.0, 1.0)

    # Threshold (ρ) - the value to compare against
    params_fields["threshold"] = (
        float,
        Field(default=0.4, description="Threshold value for comparison (ρ)"),
    )
    fitting_bounds["threshold"] = (-1.0, 1.0)

    # Softmax inverse temperature (β)
    params_fields["softmax_inverse_temperature"] = (
        float,
        Field(default=10.0, ge=0.0, description="Softmax inverse temperature"),
    )
    fitting_bounds["softmax_inverse_temperature"] = (0.00000000001, 100.0)

    # Left bias term
    params_fields["biasL"] = (
        float,
        Field(default=0.0, description="Sticky bias for action selection"),
    )
    fitting_bounds["biasL"] = (-5.0, 5.0)
    # fitting_bounds["biasL"] = (-0.5, 0.5)

    # -- Add choice kernel fields if specified --
    _add_choice_kernel_fields(params_fields, fitting_bounds, choice_kernel)

    return create_pydantic_models_dynamic(params_fields, fitting_bounds)
