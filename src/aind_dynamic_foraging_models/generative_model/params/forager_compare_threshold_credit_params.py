"""Generate pydantic models for the stay-credit form of the compare-to-threshold agent."""

from typing import Tuple, Type

from pydantic import BaseModel, Field

from .util import create_pydantic_models_dynamic


def generate_pydantic_compare_threshold_credit_params() -> Tuple[Type[BaseModel], Type[BaseModel]]:
    """Generate Pydantic models for ForagerCompareThresholdCredit parameters.

    The compare-to-threshold model written in stay credit c = beta * (value - threshold).
    Its win and loss credits map one to one onto CompareToThreshold's parameters:
    stay_credit_win = beta * (1 - threshold), stay_credit_loss = -beta * threshold, so
    beta = stay_credit_win - stay_credit_loss. The bounds are wide enough to reach the loss
    credits that CompareToThreshold's threshold bound of [-1, 1] excludes
    (stay_credit_loss > stay_credit_win / 2).

    Returns
    -------
    (ParamModel, ParamFitBoundModel) : Tuple[Type[BaseModel], Type[BaseModel]]
    """
    params_fields: dict = {}
    fitting_bounds: dict = {}

    params_fields["learn_rate"] = (
        float,
        Field(default=0.5, ge=0.0, le=1.0, description="Learning rate of the stay credit"),
    )
    fitting_bounds["learn_rate"] = (0.0, 1.0)

    params_fields["stay_credit_win"] = (
        float,
        Field(default=4.0, description="Stay credit a rewarded choice pulls toward"),
    )
    fitting_bounds["stay_credit_win"] = (-20.0, 20.0)

    params_fields["stay_credit_loss"] = (
        float,
        Field(default=1.5, description="Stay credit an unrewarded choice pulls toward"),
    )
    fitting_bounds["stay_credit_loss"] = (-20.0, 20.0)

    params_fields["biasL"] = (
        float,
        Field(
            default=0.0,
            description="Stay bias added to the logit when the last choice was Left",
        ),
    )
    fitting_bounds["biasL"] = (-5.0, 5.0)

    return create_pydantic_models_dynamic(params_fields, fitting_bounds)
