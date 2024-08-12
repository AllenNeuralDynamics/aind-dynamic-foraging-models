"""Define the base model for foraging task that interacts with the gymnasium environment
"""

"""Couple block task for dynamic bandit environment
This is very close to the task used in mice training.

First coded by Han for the project in Neuromatch Academy: Deep Learning
https://github.com/hanhou/meta_rl/blob/bd9b5b1d6eb93d217563ff37608aaa2f572c08e6/han/environment/dynamic_bandit_env.py
"""

class DynamicBanditTask():
    """A general task object for dynamic bandit environment
    """
    def __init__(self):
        pass

    def reset(self):
        """Initialization
        
        Following lines are mandatory
        
        self.trial = -1  # Index of trial number, starting from 0
        self.trial_p_reward = []  # Rwd prob per trial
        self.next_trial()
        """
        raise NotImplementedError("reset() should be overridden by subclasses")

    def next_trial(self):
        """Generate a new trial and increment the trial number
        I'm doing this trial-by-trial because the block switch may depend on the action.

        # Following lines are mandatory
        self.trial_p_reward.append(...)
        self.trial += 1
        """
        raise NotImplementedError("next_trial() should be overridden by subclasses")
        
        
