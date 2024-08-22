"""Dynamically generate pydantic models for Q-learning agent parameters."""

# %%
from pydantic import BaseModel, Field, create_model, model_validator, ValidationError, ConfigDict
from typing import Type, Optional, Dict, Any, Literal, List, Tuple


def generate_pydantic_q_learning_params(
    number_of_learning_rate: Literal[1, 2] = 2,
    number_of_forget_rate: Literal[0, 1] = 1,
    choice_kernel: Literal["none", "onestep", "full"] = "none",
    action_selection: Literal["softmax", "epsilon-greedy"] = "softmax",
) -> Tuple[Type[BaseModel], Type[BaseModel]]:
    """Dynamically generate Pydantic models for Q-learning agent parameters.

    All default values are hard-coded in this function. But when instantiating the model,
    you can always override the default values, both the params and the fitting bounds.
    
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
    choice_kernel : Literal["none", "onestep", "full"], optional
        Choice kernel type, by default "none"
        If "none", no choice kernel will be included in the model.
        If "onestep", choice_step_size will be set to 1.0, i.e., only the previous choice
            affects the choice kernel. (Bari2019)
        If "full", both choice_step_size and choice_softmax_inverse_temperature will be included
    action_selection : Literal["softmax", "epsilon-greedy"], optional
        Action selection type, by default "softmax"
    """

    # ====== Define common fields and constraints ======
    params = {}
    fitting_bounds = {}

    # -- Handle learning rate fields --
    assert number_of_learning_rate in [1, 2], "number_of_learning_rate must be 1 or 2"
    if number_of_learning_rate == 1:
        params["learn_rate"] = (
            float,
            Field(default=0.5, ge=0.0, le=1.0, description="Learning rate"),
        )
        fitting_bounds["learn_rate"] = (0.0, 1.0)
    elif number_of_learning_rate == 2:
        params["learn_rate_rew"] = (
            float,
            Field(default=0.5, ge=0.0, le=1.0, description="Learning rate for rewarded choice"),
        )
        fitting_bounds["learn_rate_rew"] = (0.0, 1.0)
        params["learn_rate_unrew"] = (
            float,
            Field(default=0.1, ge=0.0, le=1.0, description="Learning rate for unrewarded choice"),
        )
        fitting_bounds["learn_rate_unrew"] = (0.0, 1.0)

    # -- Handle forget rate field --
    assert number_of_forget_rate in [0, 1], "number_of_forget_rate must be 0 or 1"
    if number_of_forget_rate == 1:
        params["forget_rate_unchosen"] = (
            float,
            Field(default=0.2, ge=0.0, le=1.0, description="Forgetting rate for unchosen side"),
        )
        fitting_bounds["forget_rate_unchosen"] = (0.0, 1.0)

    # -- Handle choice kernel fields --
    assert (
        choice_kernel in ["none", "onestep", "full"], 
        "choice_kernel must be 'none', 'onestep', or 'full'"
        )
    if choice_kernel == "none":
        pass
    else:
        params["choice_softmax_inverse_temperature"] = (
            float,
            Field(default=10, ge=0.0, description="Softmax temperature for choice kernel"),
        )
        fitting_bounds["choice_softmax_inverse_temperature"] = (0.0, 100.0)
        
        if choice_kernel == "full":
            params["choice_step_size"] = (
                float,
                Field(default=0.1, ge=0.0, le=1.0, description="Step size for choice kernel"),
            )
            fitting_bounds["choice_step_size"] = (0.0, 1.0)
        elif choice_kernel == "onestep":
            # If choice kernel is one-step (only the previous choice affects the choice kernel like
            # in Bari2019), set choice_step_size to 1.0
            params["choice_step_size"] = (
                float,
                Field(
                    default=1.0,
                    ge=1.0,
                    le=1.0,
                    description="Step size for choice kernel == 1 (one-step choice kernel)",
                ),
            )
            fitting_bounds["choice_step_size"] = (1.0, 1.0)

    # -- Handle action selection fields --
    # Always include biasL
    params["biasL"] = (float, Field(default=0.0, description="Bias term for softmax"))
    fitting_bounds["biasL"] = (-5.0, 5.0)

    if action_selection == "softmax":
        params["softmax_inverse_temperature"] = (
            float,
            Field(default=10, ge=0.0, description="Softmax temperature"),
        )
        fitting_bounds["softmax_inverse_temperature"] = (0.0, 100.0)
    elif action_selection == "epsilon-greedy":
        params['epsilon'] = (
            float,
            Field(default=0.1, ge=0.0, le=1.0, description="Epsilon for epsilon-greedy"),
        )
        fitting_bounds['epsilon'] = (0.0, 1.0)
    else:
        raise ValueError("action_selection must be 'softmax' or 'epsilon-greedy'")

    # ====== Dynamically create the pydantic models =====
    params_model = create_model("ParamsModel", **params, 
                                __config__=ConfigDict(extra="forbid"))

    # Create the fitting bounds model
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
        __config__=ConfigDict(extra="forbid"),
    )

    return params_model, fitting_bounds_model