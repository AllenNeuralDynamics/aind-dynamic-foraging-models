from typing import Any, Dict, List, Tuple

import numpy as np
from pydantic import ConfigDict, Field, create_model, model_validator

from aind_dynamic_foraging_models.generative_model.params import ParamsSymbols


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


def get_params_options(
    params_model,
    default_range=[-np.inf, np.inf],
    para_range_override={},
) -> dict:
    """Get options for the params fields.

    Useful for the Streamlit app.

    Parameters
    ----------
    params_model : Pydantic model
        The Pydantic model for the parameters.
    default_range : list, optional
        The default range for the parameters, by default [-np.inf, np.inf]
        If the range is not specified in the Pydantic model, this default range will be used.
    para_range_override : dict, optional
        The range override for user-specified parameters, by default {}

    Example
    >>> ParamsModel, FittingBoundsModel = generate_pydantic_q_learning_params(
            number_of_learning_rate=1,
            number_of_forget_rate=1,
            choice_kernel="one_step",
            action_selection="softmax",
        )
    >>> params_options = get_params_options(ParamsModel)
    {'learn_rate': {'para_range': [0.0, 1.0],
                    'para_default': 0.5,
                    'para_symbol': <ParamsSymbols.learn_rate: '$\\alpha$'>,
                    'para_desc': 'Learning rate'}, ...
    }

    """
    # Get the schema
    params_schema = params_model.model_json_schema()["properties"]

    # Extract ge and le constraints
    param_options = {}

    for para_name, para_field in params_schema.items():
        default = para_field.get("default", None)
        para_desc = para_field.get("description", "")

        if para_name in para_range_override:
            para_range = para_range_override[para_name]
        else:  # Get from pydantic schema
            para_range = default_range.copy()  # Default range
            # Override the range if specified
            if "minimum" in para_field:
                para_range[0] = para_field["minimum"]
            if "maximum" in para_field:
                para_range[1] = para_field["maximum"]
            para_range = [type(default)(x) for x in para_range]

        param_options[para_name] = dict(
            para_range=para_range,
            para_default=default,
            para_symbol=ParamsSymbols[para_name],
            para_desc=para_desc,
        )
    return param_options
