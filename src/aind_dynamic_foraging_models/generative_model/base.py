"""Base class for DynamicForagingAgent with MLE fitting
"""
from typing import Type, Tuple
from pydantic import BaseModel

import numpy as np

from aind_behavior_gym.dynamic_foraging.task import DynamicForagingTaskBase
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

    def perform(
        self,
        task: DynamicForagingTaskBase,
    ):
        """Generative simulation of a task, or "open-loop" simulation

        Override the base class method to include choice_prob caching etc.
        """
        self.task = task
        self.n_trials = task.num_trials

        # --- Main task loop ---
        self._reset()  # Reset agent
        _, _ = self.task.reset()  # Reset task and get the initial observation
        task_done = False
        while not task_done:
            assert self.trial == self.task.trial  # Ensure the two timers are in sync

            # -- Agent performs an action
            choice, choice_prob = self.act(_)

            # -- Environment steps (enviromnet's timer ticks here!!!)
            _, reward, task_done, _, _ = task.step(choice)

            # -- Update agent history
            self.choice_prob[:, self.trial] = choice_prob
            self.choice_history[self.trial] = choice
            # In Sutton & Barto's convention, reward belongs to the next time step, but we put it
            # in the current time step for the sake of consistency with neuroscience convention
            self.reward_history[self.trial] = reward

            # -- Agent's timer ticks here !!!
            self.trial += 1

            # -- Update q values
            # Note that this will update the q values **after the last trial**, a final update that
            # will not be used to make the next action (because task is *done*) but may be used for
            # correlating with physiology recordings
            self.learn(_, choice, reward, _, task_done)
            
    def perform_closed_loop(self, fit_choice_history, fit_reward_history):
        """Simulates the agent over a fixed choice and reward history using its params.
        Also called "teacher forcing" or "closed-loop" simulation.

        Unlike .perform() ("generative" simulation), this is called "predictive" simulation,
        which does not need a task and is used for model fitting.
        """
        self.n_trials = len(fit_choice_history)
        self._reset()

        while self.trial <= self.n_trials - 1:
            # -- Compute and cache choice_prob (key to model fitting)
            _, choice_prob = self.act(None)
            self.choice_prob[:, self.trial] = choice_prob

            # -- Clamp history to fit_history
            clamped_choice = fit_choice_history[self.trial].astype(int)
            clamped_reward = fit_reward_history[self.trial]
            self.choice_history[self.trial] = clamped_choice
            self.reward_history[self.trial] = clamped_reward

            # -- Agent's timer ticks here !!!
            self.trial += 1

            # -- Update q values
            self.learn(None, clamped_choice, clamped_reward, None, None)

    def act(self, observation):
        """
        Chooses an action based on the current observation.
        I just copy and paste this from DynamicForagingAgentBase here for clarity.

        Args:
            observation: The current observation from the environment.

        Returns:
            action: The action chosen by the agent.
        """
        raise NotImplementedError("The 'act' method should be overridden by subclasses.")

    def learn(self, observation, action, reward, next_observation, done):
        """
        Updates the agent's knowledge or policy based on the last action and its outcome.
        I just copy and paste this from DynamicForagingAgentBase here for clarity.

        This is the core method that should be implemented by all non-trivial agents.
        It could be Q-learning, policy gradients, neural networks, etc.

        Args:
            observation: The observation before the action was taken.
            action: The action taken by the agent.
            reward: The reward received after taking the action.
            next_observation: The next observation after the action.
            done: Whether the episode has ended.
        """
        raise NotImplementedError("The 'learn' method should be overridden by subclasses.")
