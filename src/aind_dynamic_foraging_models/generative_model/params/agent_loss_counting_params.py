"""Generate pydantic models for Loss Count agent parameters."""

# %%
from typing import Any, Dict, List, Literal, Tuple, Type

from pydantic import BaseModel, ConfigDict, Field, create_model, model_validator

from .base import create_pydantic_models_dynamic


def generate_pydantic_loss_counting_params(
    ) -> Tuple[Type[BaseModel], Type[BaseModel]]:
    """Generate Pydantic models for Loss-counting agent parameters.

    All default values are hard-coded in this function. But when instantiating the model,
    you can always override the default values, both the params and the fitting bounds.

    Parameters
    ----------
    No hyperparameters are needed for the loss counting model
    """

    # ====== Define common fields and constraints ======
    params_fields = {}
    fitting_bounds = {}

    # -- Loss counting model parameters --
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
            description="Mean of the loss count threshold",
        ),
    )
    fitting_bounds["loss_count_threshold_std"] = (0.0, 10.0)

    return create_pydantic_models_dynamic(params_fields, fitting_bounds)