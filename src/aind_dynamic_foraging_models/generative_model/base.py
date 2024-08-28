"""Base class for DynamicForagingAgent with MLE fitting
"""
from typing import Type, Tuple
from pydantic import BaseModel

from aind_behavior_gym.dynamic_foraging.agent import DynamicForagingAgentBase


class DynamicForagingAgentMLEBase(DynamicForagingAgentBase):
    """Base class of "DynamicForagingAgentBase" + "MLE fitting"
    """

    def __init__(
        self,
        agent_kwargs: dict = {},
        params: dict = {},
        **kwargs,
    ):
        """Init"""
        super().__init__(**kwargs)  # Set self.rng etc.

        # Get pydantic model for the parameters and bounds
        self.ParamModel, self.ParamFitBoundModel = self._get_params_model(agent_kwargs)

        # Set and validate the model parameters. Use default parameters if some are not provided
        self.params = self.ParamModel(**params)

        # Add model fitting related attributes
        self.fitting_result = None
        self.fitting_result_cross_validation = None

        # Some initializations
        self.n_actions = 2
        self.task = None

    def _get_params_model(self, agent_kwargs, params) -> Tuple[Type[BaseModel], Type[BaseModel]]:
        """Dynamically generate the Pydantic model for parameters and fitting bounds.
        
        This should be overridden by the subclass!!
        It should return ParamModel and ParamFitBoundModel here.
        """
        raise NotImplementedError("This should be overridden by the subclass!!")
