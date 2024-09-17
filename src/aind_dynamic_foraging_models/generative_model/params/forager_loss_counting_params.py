"""Generate pydantic models for Loss Count agent parameters."""

# %%
from typing import Literal, Tuple, Type

from pydantic import BaseModel, Field

from .forager_q_learning_params import _add_choice_kernel_fields
from .util import create_pydantic_models_dynamic


def generate_pydantic_loss_counting_params(
    win_stay_lose_switch: Literal[False, True] = False,
    choice_kernel: Literal["none", "one_step", "full"] = "none",
) -> Tuple[Type[BaseModel], Type[BaseModel]]:
    """Generate Pydantic models for Loss-counting agent parameters.

    All default values are hard-coded in this function. But when instantiating the model,
    you can always override the default values, both the params and the fitting bounds.

    Parameters
    ----------
    win_stay_lose_switch : bool, optional
        If True, the agent will be a win-stay-lose-shift agent
        (loss_count_threshold_mean and loss_count_threshold_std are fixed at 1 and 0),
        by default False
    choice_kernel : Literal["none", "one_step", "full"], optional
        Choice kernel type, by default "none"
        If "none", no choice kernel will be included in the model.
        If "one_step", choice_kernel_step_size will be set to 1.0, i.e., only the previous choice
            affects the choice kernel. (Bari2019)
        If "full", both choice_kernel_step_size and choice_kernel_relative_weight will be included
    """

    # ====== Define common fields and constraints ======
    params_fields = {}
    fitting_bounds = {}

    # -- Loss counting model parameters --
    if win_stay_lose_switch:
        params_fields["loss_count_threshold_mean"] = (
            float,
            Field(
                default=1.0,
                ge=1.0,
                le=1.0,
                frozen=True,  # To indicate that this field is clamped by construction
                description="Mean of the loss count threshold",
            ),
        )
        fitting_bounds["loss_count_threshold_mean"] = (1.0, 1.0)

        params_fields["loss_count_threshold_std"] = (
            float,
            Field(
                default=0.0,
                ge=0.0,
                le=0.0,
                frozen=True,  # To indicate that this field is clamped by construction
                description="Std of the loss count threshold",
            ),
        )
        fitting_bounds["loss_count_threshold_std"] = (0.0, 0.0)
    else:
        params_fields["loss_count_threshold_mean"] = (
            float,
            Field(
                default=1.0,
                ge=0.0,
                description="Mean of the loss count threshold",
            ),
        )
        fitting_bounds["loss_count_threshold_mean"] = (0.0, 10.0)

        params_fields["loss_count_threshold_std"] = (
            float,
            Field(
                default=0.0,
                ge=0.0,
                description="Std of the loss count threshold",
            ),
        )
        fitting_bounds["loss_count_threshold_std"] = (0.0, 10.0)

    # -- Always add a bias term --
    params_fields["biasL"] = (
        float,
        Field(default=0.0, ge=-1.0, le=1.0, description="Bias term for loss counting"),
    )  # Bias term for loss counting directly added to the choice probabilities
    fitting_bounds["biasL"] = (-1.0, 1.0)

    # -- Add choice kernel fields --
    _add_choice_kernel_fields(params_fields, fitting_bounds, choice_kernel)

    return create_pydantic_models_dynamic(params_fields, fitting_bounds)
