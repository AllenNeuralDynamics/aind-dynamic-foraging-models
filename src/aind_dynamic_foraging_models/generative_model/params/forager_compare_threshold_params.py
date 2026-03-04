"""Generate pydantic models for Compare-to-threshold foraging agent parameters."""

from typing import Literal, Tuple, Type

from pydantic import BaseModel, Field

from .forager_q_learning_params import _add_choice_kernel_fields
from .util import create_pydantic_models_dynamic


def generate_pydantic_compare_threshold_params(
    *,
    number_of_learning_rate: Literal[1, 2] = 1,
    choice_kernel: Literal["none", "one_step", "full"] = "none",
    include_stay_bias: Literal[True, False] = False,
    include_side_bias: Literal[True, False] = False,  # NEW
    fix_threshold: Literal[True, False] = False,
) -> Tuple[Type[BaseModel], Type[BaseModel]]:
    """Generate Pydantic models for Compare-to-threshold foraging agent parameters.

    All default values are defined here. When instantiating the returned ParamModel,
    you can override defaults by passing `params={...}` into the agent constructor.
    Similarly, bounds can be overridden during fitting via `fit_bounds_override={...}`.

    Parameters
    ----------
    number_of_learning_rate : Literal[1, 2], optional
        Controls whether learning rate is symmetric or asymmetric.
        - 1: include a single learning rate parameter `learn_rate`
        - 2: include `learn_rate_rew` and `learn_rate_unrew`

    choice_kernel : Literal["none", "one_step", "full"], optional
        Choice kernel type.
        - "none": no choice kernel parameters
        - "one_step": include choice_kernel_relative_weight and set step size to 1.0 (fixed)
        - "full": include both choice_kernel_step_size and choice_kernel_relative_weight

    include_stay_bias : bool, optional
        If True, include an additive stay/perseveration bias term `stay_bias`
        in the stay (exploit) logit.

    include_side_bias : bool, optional
        If True, include an additive side bias term `biasL`, which will be applied
        in logit space to P(R) after mapping stay/switch -> side probability.

    fix_threshold : bool, optional
        If True, threshold is fixed (not learnable) and should NOT be included as a
        fitted/free parameter. In this case, `threshold_fixed` is stored externally
        (agent_kwargs) and used by the agent code.

    Returns
    -------
    (ParamModel, ParamFitBoundModel) : Tuple[Type[BaseModel], Type[BaseModel]]
        ParamModel:
            A Pydantic model defining valid parameter names, defaults, and constraints.

        ParamFitBoundModel:
            A Pydantic model defining default fitting bounds for each parameter.
    """

    # -------------------------------------------------------------------------
    # Define parameter fields and default fitting bounds
    # -------------------------------------------------------------------------
    params_fields: dict = {}
    fitting_bounds: dict = {}

    # -------------------------------------------------------------------------
    # Learning rate(s) for value update
    # -------------------------------------------------------------------------
    if number_of_learning_rate == 1:
        params_fields["learn_rate"] = (
            float,
            Field(
                default=0.5,
                ge=0.0,
                le=1.0,
                description="Learning rate for value update (single alpha).",
            ),
        )
        fitting_bounds["learn_rate"] = (0.0, 1.0)

    elif number_of_learning_rate == 2:
        params_fields["learn_rate_rew"] = (
            float,
            Field(
                default=0.5,
                ge=0.0,
                le=1.0,
                description="Learning rate for rewarded trials (alpha_rew).",
            ),
        )
        fitting_bounds["learn_rate_rew"] = (0.0, 1.0)

        params_fields["learn_rate_unrew"] = (
            float,
            Field(
                default=0.2,
                ge=0.0,
                le=1.0,
                description="Learning rate for unrewarded trials (alpha_unrew).",
            ),
        )
        fitting_bounds["learn_rate_unrew"] = (0.0, 1.0)

    else:
        raise ValueError(f"number_of_learning_rate must be 1 or 2, got {number_of_learning_rate}")

    # -------------------------------------------------------------------------
    # Threshold (ρ): include ONLY when threshold is learnable
    # -------------------------------------------------------------------------
    if not fix_threshold:
        params_fields["threshold"] = (
            float,
            Field(
                default=0.4,
                description="Threshold value for comparison (ρ).",
            ),
        )
        fitting_bounds["threshold"] = (-1.0, 1.0)

    # -------------------------------------------------------------------------
    # Softmax inverse temperature (β)
    # -------------------------------------------------------------------------
    params_fields["softmax_inverse_temperature"] = (
        float,
        Field(
            default=10.0,
            ge=0.0,
            description="Softmax inverse temperature (β).",
        ),
    )
    fitting_bounds["softmax_inverse_temperature"] = (1e-11, 100.0)

    # -------------------------------------------------------------------------
    # Optional: side bias (Right-vs-Left bias in logit(P(R)))
    # -------------------------------------------------------------------------
    if include_side_bias:
        params_fields["biasL"] = (
            float,
            Field(
                default=0.0,
                description="Side bias parameter applied to logit(P(R)).",
            ),
        )
        fitting_bounds["biasL"] = (-5.0, 5.0)

    # -------------------------------------------------------------------------
    # Optional: stay bias (perseveration)
    # -------------------------------------------------------------------------
    if include_stay_bias:
        params_fields["stay_bias"] = (
            float,
            Field(
                default=0.0,
                description="Additive stay/perseveration bias on logit(P(stay)) (t>0).",
            ),
        )
        fitting_bounds["stay_bias"] = (-5.0, 5.0)

    # -- Add choice kernel fields if specified --
    _add_choice_kernel_fields(params_fields, fitting_bounds, choice_kernel)

    # -------------------------------------------------------------------------
    # Build the Pydantic models from field specifications and bounds
    # -------------------------------------------------------------------------
    return create_pydantic_models_dynamic(params_fields, fitting_bounds)