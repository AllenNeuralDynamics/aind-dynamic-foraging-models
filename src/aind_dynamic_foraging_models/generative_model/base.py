"""Base class for DynamicForagingAgent with MLE fitting
"""
from typing import Type, Tuple
from pydantic import BaseModel

import numpy as np

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
        self.n_trials = 0  # Should be set in perform or perform_closed_loop

    def _get_params_model(self, agent_kwargs, params) -> Tuple[Type[BaseModel], Type[BaseModel]]:
        """Dynamically generate the Pydantic model for parameters and fitting bounds.
        
        This should be overridden by the subclass!!
        It should return ParamModel and ParamFitBoundModel here.
        """
        raise NotImplementedError("This should be overridden by the subclass!!")
    
    def set_params(self, params):
        """Update the model parameters and validate"""
        # This is safer than model_copy(update) because it will NOT validate the input params
        _params = self.params.model_dump()
        _params.update(params)
        self.params = self.ParamModel(**_params)
        return self.get_params()

    def get_params(self):
        """Get the model parameters in a dictionary format"""
        return self.params.model_dump()

    def _reset(self):
        """Reset the agent"""
        self.trial = 0

        # MLE agent must have choice_prob
        self.choice_prob = np.full([self.n_actions, self.n_trials + 1], np.nan)
        self.choice_prob[:, 0] = 1 / self.n_actions  # To be strict (actually no use)

        # Choice and reward history have n_trials length
        self.choice_history = np.full(self.n_trials, fill_value=-1, dtype=int)  # Choice history
        # Reward history, separated for each port (Corrado Newsome 2005)
        self.reward_history = np.zeros(self.n_trials)
