from typing import Any, Dict, List, Tuple

from pydantic import ConfigDict, Field, create_model, model_validator


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
