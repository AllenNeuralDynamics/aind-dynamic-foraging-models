"""Compare-to-threshold foraging model implementation
"""

from typing import Literal

import numpy as np
from aind_behavior_gym.dynamic_foraging.task import L, R

from .base import DynamicForagingAgentMLEBase
from .learn_functions import learn_choice_kernel
from .params.forager_compare_threshold_params import generate_pydantic_compare_threshold_params


class ForagerCompareThreshold(DynamicForagingAgentMLEBase):
    """Compare-to-threshold foraging model.
    
    This model only tracks a single value (for exploiting the current option) and 
    makes decisions by comparing this value to a threshold.
    """

    def __init__(
        self,
        choice_kernel: Literal["none", "one_step", "full"] = "none",
        params: dict = {},
        **kwargs,
    ):
        """Initialize the compare-to-threshold foraging agent.

        Parameters
        ----------
        choice_kernel : Literal["none", "one_step", "full"], optional
            Choice kernel type, by default "none"
            If "none", no choice kernel will be included in the model.
            If "one_step", choice_kernel_step_size will be set to 1.0, i.e., only the last choice
                affects the choice kernel.
            If "full", both choice_kernel_step_size and choice_kernel_relative_weight will be included
        params : dict, optional
            Initial parameters of the model, by default {}.
        """
        # -- Pack the agent_kwargs --
        self.agent_kwargs = dict(
            choice_kernel=choice_kernel,
        )  # Note that the class and self.agent_kwargs fully define the agent

        # -- Initialize the model parameters --
        super().__init__(agent_kwargs=self.agent_kwargs, params=params, **kwargs)

    def _get_params_model(self, agent_kwargs):
        """Implement the base class method to dynamically generate Pydantic models
        for parameters and fitting bounds for the compare-to-threshold foraging model.
        """
        return generate_pydantic_compare_threshold_params(**agent_kwargs)

    def get_agent_alias(self):
        """Get the agent alias"""
        _ck = {"none": "", "one_step": "_CK1", "full": "_CKfull"}[
            self.agent_kwargs["choice_kernel"]
        ]
        return "ForagingCompareThreshold" + _ck

    def _reset(self):
        """Reset the agent"""
        # --- Call the base class reset ---
        super()._reset()

        # --- Agent family specific variables ---
        # Initialize a single value (for the exploit option) for all trials
        self.value = np.full(self.n_trials + 1, np.nan)
        self.value[0] = self.params.threshold  # Initialize to threshold instead of 0
        
        # Track the current chosen option (L or R)
        self.current_option = np.full(self.n_trials + 1, -1)
        
        # Probability of exploiting vs exploring
        self.p_exploit = np.full(self.n_trials + 1, np.nan)
        self.p_exploit[0] = 0.5  # Initial probability
        
        # Always initialize choice_kernel with nan, even if choice_kernel = "none"
        self.choice_kernel = np.full([self.n_actions, self.n_trials + 1], np.nan)
        self.choice_kernel[:, 0] = 0  # Initial choice kernel as 0

    def act(self, _):
        """Action selection based on compare-to-threshold mechanism"""
        # First, calculate p(exploit) using softmax comparison with threshold
        # p(exploit) = 1 / (1 + e^(-β(v-ρ)))
        value = self.value[self.trial]
        threshold = self.params.threshold
        beta = self.params.softmax_inverse_temperature
        
        p_exploit = 1 / (1 + np.exp(-beta * (value - threshold)))
        self.p_exploit[self.trial] = p_exploit
        
        # Decide whether to exploit or explore
        exploit = self.rng.random() < p_exploit
        
        # If this is the first trial or we're exploring, choose randomly
        if self.trial == 0 or not exploit:
            # For exploration: Reset current option and choose randomly with bias
            self.current_option[self.trial] = -1
            
            # Apply bias to the choice probabilities
            choice_prob = np.array([0.5, 0.5])
            choice_prob[L] += self.params.biasL
            choice_prob = np.clip(choice_prob, 0, 1)
            choice_prob = choice_prob / np.sum(choice_prob)  # Normalize
            
            choice = self.rng.choice([L, R], p=choice_prob)
            
            # Update current option for the next trial
            self.current_option[self.trial+1] = choice
            
        else:
            # For exploitation: Continue with current option
            choice = self.current_option[self.trial]
            # If current_option is invalid (e.g., after reset), choose randomly
            if choice < 0:
                choice_prob = np.array([0.5 + self.params.biasL, 0.5 - self.params.biasL])
                choice_prob = np.clip(choice_prob, 0, 1)
                choice_prob = choice_prob / np.sum(choice_prob)  # Normalize
                choice = self.rng.choice([L, R], p=choice_prob)
                self.current_option[self.trial+1] = choice
            
        # Handle choice kernel if used
        if self.agent_kwargs["choice_kernel"] != "none":
            # Incorporate choice kernel into final choice probability
            choice_kernel = self.choice_kernel[:, self.trial]
            choice_kernel_weight = self.params.choice_kernel_relative_weight
            
            # Adjust choice probabilities based on choice kernel
            base_prob = np.zeros(self.n_actions)
            base_prob[choice] = 1.0
            
            # Mix base probability with choice kernel influence
            final_prob = (1 - choice_kernel_weight) * base_prob + choice_kernel_weight * choice_kernel
            final_prob = final_prob / np.sum(final_prob)  # Normalize
            
            # Final choice based on adjusted probabilities
            choice = self.rng.choice([L, R], p=final_prob)
            self.current_option[self.trial+1] = choice
            
            # Return choice and final probabilities
            return choice, final_prob
        
        # If no choice kernel, return the choice and simple probabilities
        choice_prob = np.zeros(self.n_actions)
        choice_prob[choice] = 1.0
        return choice, choice_prob

    def learn(self, _observation, choice, reward, _next_observation, done):
        """Update value based on whether we're exploring or exploiting"""
        
        # Check if we were exploring in this trial
        was_exploring = self.p_exploit[self.trial-1] < self.rng.random()
        
        if was_exploring:
            # If we were exploring, reset value to threshold
            self.value[self.trial] = self.params.threshold
        else:
            # If we were exploiting, update value using delta rule
            self.value[self.trial] = self.value[self.trial-1] + self.params.learn_rate * (reward - self.value[self.trial-1])
        
        # Update choice kernel, if used
        if self.agent_kwargs["choice_kernel"] != "none":
            self.choice_kernel[:, self.trial] = learn_choice_kernel(
                choice=choice,
                choice_kernel_tminus1=self.choice_kernel[:, self.trial - 1],
                choice_kernel_step_size=self.params.choice_kernel_step_size,
            )

    def get_latent_variables(self):
        """Return latent variables for analysis"""
        return {
            "value": self.value.tolist(),
            "p_exploit": self.p_exploit.tolist(),
            "current_option": self.current_option.tolist(),
            "choice_kernel": self.choice_kernel.tolist(),
            "choice_prob": self.choice_prob.tolist(),
        }

    def plot_latent_variables(self, ax, if_fitted=False):
        """Plot latent variables"""
        if if_fitted:
            style = dict(lw=2, ls=":")
            prefix = "fitted_"
        else:
            style = dict(lw=0.5)
            prefix = ""

        x = np.arange(self.n_trials + 1) + 1  # When plotting, we start from 1
        
        # Plot value
        ax.plot(x, self.value, label=f"{prefix}value", color="purple", **style)
        
        # Plot threshold
        ax.axhline(y=self.params.threshold, color="black", linestyle="--", 
                 label=f"{prefix}threshold", **style)
        
        # Plot p(exploit)
        ax.plot(x, self.p_exploit, label=f"{prefix}p(exploit)", color="green", **style)
        
        # Add choice kernel, if used
        if self.agent_kwargs["choice_kernel"] != "none":
            ax.plot(
                x,
                self.choice_kernel[L, :],
                label=f"{prefix}choice_kernel(L)",
                color="red",
                **style,
            )
            ax.plot(
                x,
                self.choice_kernel[R, :],
                label=f"{prefix}choice_kernel(R)",
                color="blue",
                **style,
            )