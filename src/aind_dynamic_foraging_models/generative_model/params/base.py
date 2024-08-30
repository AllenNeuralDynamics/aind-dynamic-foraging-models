from enum import Enum
from typing import Any, Dict, List, Tuple

from pydantic import ConfigDict, Field, create_model, model_validator


class ParamsSymbols(str, Enum):
    """Symbols for the parameters.

    The order determined the default order of parameters when output as a string.
    """

    loss_count_threshold_mean = R"$\mu_{LC}$"
    loss_count_threshold_std = R"$\sigma_{LC}$"
    learn_rate = R"$\alpha$"
    learn_rate_rew = R"$\alpha_{rew}$"
    learn_rate_unrew = R"$\alpha_{unr}$"
    forget_rate_unchosen = R"$\delta$"
    choice_kernel_step_size = R"$\alpha_{ck}$"
    choice_kernel_relative_weight = R"$w_{ck}$"
    biasL = R"$b_L$"
    softmax_inverse_temperature = R"$\beta$"
    epsilon = R"$\epsilon$"


def create_pydantic_models_dynamic(
    params_fields: Dict[str, Any],
    fitting_bounds: Dict[str, Tuple[float, float]],
):
    """Create Pydantic models dynamically based on the input fields and fitting bounds."""
    # -- params_model --
    params_model = create_model(
        "ParamsModel",
        **params_fields,
        __config__=ConfigDict(
            extra="forbid",
            validate_assignment=True,
        ),
    )

    # -- fitting_bounds_model --
    fitting_bounds_fields = {}
    for name, (lower, upper) in fitting_bounds.items():
        fitting_bounds_fields[name] = (
            List[float],
            Field(
                default=[lower, upper],
                min_length=2,
                max_length=2,
                description=f"Fitting bounds for {name}",
            ),
        )

    # Add a validator to check the fitting bounds
    def validate_bounds(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        for name, bounds in values.model_dump().items():
            lower_bound, upper_bound = bounds
            if lower_bound > upper_bound:
                raise ValueError(f"Lower bound for {name} must be <= upper bound")
        return values

    fitting_bounds_model = create_model(
        "FittingBoundsModel",
        **fitting_bounds_fields,
        __validators__={"validate_bounds": model_validator(mode="after")(validate_bounds)},
        __config__=ConfigDict(
            extra="forbid",
            validate_assignment=True,
        ),
    )

    return params_model, fitting_bounds_model
