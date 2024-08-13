"""Define the base model for foraging task that interacts with the gymnasium environment
"""

"""Couple block task for dynamic bandit environment
This is very close to the task used in mice training.

First coded by Han for the project in Neuromatch Academy: Deep Learning
https://github.com/hanhou/meta_rl/blob/bd9b5b1d6eb93d217563ff37608aaa2f572c08e6/han/environment/dynamic_bandit_env.py
"""

import numpy as np

class DynamicBanditTask():
    """A general task object for dynamic bandit environment 
    """
    def __init__(self):
        pass

    def reset(self, seed=None):
        """Initialization
        
        Following lines are mandatory
        
        self.trial = -1  # Index of trial number, starting from 0
        self.trial_p_reward = []  # Rwd prob per trial; list of lists that contains the 
                                    reward probabilities for each action
        self.next_trial()  # Generate next p_reward
        """
        # Seed the random number generator and pass it to the task as well
        self.rng = np.random.default_rng(seed=seed)


    def add_action(self, action):
        """Pass the agent's action to the task (optional)
        This is important when the state transition depends on the action 
        like in the Uncoupled task.
        """
        pass

    def next_trial(self):
        """Generate a new trial and increment the trial number
        I'm doing this trial-by-trial because the block switch may depend on the action.

        # Following lines are mandatory
        self.trial_p_reward.append(...)
        self.trial += 1
        """
        raise NotImplementedError("next_trial() should be overridden by subclasses")
        
        
