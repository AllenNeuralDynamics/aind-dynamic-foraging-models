"""Dynamically generate pydantic models for actor-critic agent parameters."""

# %%
from typing import Literal, Tuple, Type

from pydantic import BaseModel, Field

from .forager_q_learning_params import (
    _add_action_selection_fields,
    _add_choice_kernel_fields,
)
from .util import create_pydantic_models_dynamic


def generate_pydantic_actor_critic_params(
    choice_kernel: Literal["none", "one_step", "full"] = "none",
) -> Tuple[Type[BaseModel], Type[BaseModel]]:
    """Dynamically generate Pydantic models for actor-critic agent parameters.

    All default values are hard-coded in this function. But when instantiating the model,
    you can always override the default values, both the params_fields and the fitting bounds.

    Parameters
    ----------
    choice_kernel : Literal["none", "one_step", "full"], optional
        Choice kernel type, by default "none"
        If "none", no choice kernel will be included in the model.
        If "one_step", choice_kernel_step_size will be set to 1.0, i.e., only the previous choice
            affects the choice kernel. (Bari2019)
        If "full", both choice_kernel_step_size and choice_kernel_relative_weight will be included

    Notes
    -----
    The actor-critic agent always uses a softmax actor, so ``biasL`` and
    ``softmax_inverse_temperature`` (beta) are always included via the shared
    action-selection helper. Note that ``learn_rate_actor`` and ``beta`` are not
    jointly identifiable; it is recommended to clamp ``softmax_inverse_temperature``
    when fitting (e.g. ``clamp_params={"softmax_inverse_temperature": 1.0}``).
    """

    # ====== Define common fields and constraints ======
    params_fields = {}
    fitting_bounds = {}

    # -- Handle actor / critic learning-rate fields --
    _add_actor_critic_learning_rate_fields(params_fields, fitting_bounds)

    # -- Handle choice kernel fields (reuse the Q-learning helper) --
    _add_choice_kernel_fields(params_fields, fitting_bounds, choice_kernel)

    # -- Handle action selection fields (softmax; reuse the Q-learning helper) --
    _add_action_selection_fields(params_fields, fitting_bounds, "softmax")

    # ====== Dynamically create the pydantic models =====
    return create_pydantic_models_dynamic(params_fields, fitting_bounds)


def _add_actor_critic_learning_rate_fields(params_fields, fitting_bounds):
    """Add actor and critic learning-rate fields to params_fields and fitting_bounds."""
    params_fields["learn_rate_actor"] = (
        float,
        Field(default=0.3, ge=0.0, le=1.0, description="Actor (policy) learning rate"),
    )
    fitting_bounds["learn_rate_actor"] = (0.0, 1.0)

    params_fields["learn_rate_critic"] = (
        float,
        Field(default=0.3, ge=0.0, le=1.0, description="Critic (state-value) learning rate"),
    )
    fitting_bounds["learn_rate_critic"] = (0.0, 1.0)
