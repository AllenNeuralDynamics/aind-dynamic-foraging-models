"""Couple block task for dynamic bandit environment
This is very close to the task used in mice training.

First coded by Han for the project in Neuromatch Academy: Deep Learning
https://github.com/hanhou/meta_rl/blob/bd9b5b1d6eb93d217563ff37608aaa2f572c08e6/han/environment/dynamic_bandit_env.py
"""


import numpy as np
rng = np.random.default_rng()

class CoupledBlockTask():
    """
    Generate block-like reward probabilities for 2-arm non-stationary bandit environment.
    Instead of pre-generatingµ all reward probabilities, I'm generating them on the fly.
    This is because the reward probabilities may depend on the action, especially in
    real animal training.

    This default setting roughly matches what has been used in this paper:
    https://www.sciencedirect.com/science/article/pii/S089662731930529X
    """
    def __init__(
        self,
        block_min=40,  # Min block length
        block_max=80,   # Max block length
        block_beta=20,   # Time constant of the exponential distribution (the larger the flatter)
        p_reward_pairs=[
            [0.225, 0.225],  # 1:1
            [0.45/4*1, 0.45/4*3],  # 1:3
            [0.45/7*1, 0.45/7*6],  # 1:6
            [0.05, 0.40], # 1:8
        ],
    ):
        self.block_min = block_min
        self.block_max = block_max
        self.block_beta = block_beta
        self.p_reward_pairs = [sorted(ps) for ps in p_reward_pairs] # Always sort the input ps

    def reset(self):
        # Initialization
        self.trial_p_reward = []  # Rwd prob per trial
        self.block_starts = [0]  # Start of each block. The first block always starts at trial 0
        self.block_lens = []  # Lengths of each block
        self.block_p_reward = []  # Rwd prob of each block
        self.trial = -1  # Index of trial number, starting from 0

        self.next_trial()

    def next_trial(self):
        """
        Generate a new trial, return reward probability for each arm.

        I'm doing this trial-by-trial because the block switch may depend on the action.
        """
        # Start a new block if necessary
        if (self.trial == -1
            or self.trial == self.block_starts[-1]):
            self._next_block()

        # Generate reward probabilities for this trial
        self.trial_p_reward.append(self.block_p_reward[-1])
        self.trial += 1

        return self.trial_p_reward[-1]

    def _next_block(self):
        """
        Generate the next block
        """
        # Generate the block length
        self.block_lens.append(
            int(
                generate_trunc_exp(
                    self.block_min, self.block_max, self.block_beta
                    )[0]
               )
            )
        self.block_starts.append(self.block_starts[-1] + self.block_lens[-1])

        # Generate the reward probability
        self.block_p_reward.append(self._generate_block_p_reward())
        return

    def _generate_block_p_reward(self):
        """
        Generate the reward probability for the next block.
        """
        # If it is the first block, randomly choose a pair and the side
        if len(self.block_p_reward) == 0:
            p_reward = rng.choice(self.p_reward_pairs)
            p_reward = self._flip_side(p_reward, None)
            return p_reward

        # Else, generate a new p_reward based on the current p_reward
        # 1. if current p_L == p_R, randomly choose a p_reward_pair (excluding p_L == p_R)
        #    and make sure the new block is flipped compare
        #    to the one before the equal-probability block
        # 2. else, randomly choose a p_reward_pair and always flip the side
        if self.block_p_reward[-1][0] == self.block_p_reward[-1][1]:
            # Cannot be p_L == p_R again
            valid_pairs = [p for p in self.p_reward_pairs if p[0] != p[1]]
            # Randomly choose from the valid pairs
            p_reward = rng.choice(valid_pairs)
            # If there is a block before the equal-probability block, flip relative to it
            # otherwise, randomly choose
            p_reward = self._flip_side(p_reward,
                                       self.block_p_reward[-2]
                                       if len(self.block_p_reward) > 1
                                       else None
            )
        else:
            # Randomly choose from any pairs
            p_reward = rng.choice(self.p_reward_pairs)
            # Make sure the side is flipped
            p_reward = self._flip_side(p_reward, self.block_p_reward[-1])

        return p_reward

    @staticmethod
    def _flip_side(p_reward_new, p_reward_old=None):
        """
        Make sure the new block is flipped compare to the one before the equal-probability block.
        If old is None, flip it with a 0.5 probability.
        """
        should_flip = p_reward_old is None and rng.random() < 0.5
        if p_reward_old is not None:
            should_flip = (p_reward_new[0] < p_reward_new[1]) == (p_reward_old[0] < p_reward_old[1])

        return p_reward_new[::-1] if should_flip else p_reward_new


def generate_trunc_exp(lower, upper, beta, n=1):
    """
    Generate n samples from a truncated exponential distribution
    """
    x = lower + rng.exponential(beta, n)
    x[x > upper] = upper
    return x