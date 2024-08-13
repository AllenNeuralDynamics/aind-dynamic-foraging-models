import unittest
import numpy as np

from aind_dynamic_foraging_models.generative_model.gym_env.dynamic_bandit_env import (DynamicBanditEnv, L, R, IGNORE)
from aind_dynamic_foraging_models.generative_model.dynamic_foraging_tasks.coupled_block_task import CoupledBlockTask
from aind_dynamic_foraging_models.generative_model.dynamic_foraging_tasks.uncoupled_block_task import UncoupledBlockTask


class TestDynamicBanditEnv(unittest.TestCase):
    def setUp(self):
        np.random.seed(56)
        self.L, self.R, self.IGNORE = 0, 1, 2

        self.task = UncoupledBlockTask(
            rwd_prob_array=[0.1, 0.5, 0.9],
            block_min=20, block_max=35,
            persev_add=True, perseverative_limit=4,
            max_block_tally=4,  # Max number of consecutive blocks in which one side has higher rwd prob than the other
        )

        self.env = DynamicBanditEnv(
            self.task, 
            num_arms=2,
            allow_ignore=True,
            num_trials=1000
        )

    def test_bandit_env(self):
        observation, info = self.env.reset(seed=42)
        done = False
        actions = []
        rewards = []

        rng_agent = np.random.default_rng(seed=42)  # Another independent random number generator

        while not done:  # Trial loop
            # Choose an action (a random agent with left bias and ignores)
            action = [self.L, self.R, self.IGNORE][rng_agent.choice([0]*100 + [1]*20 + [2]*1)]
            
            # Can also apply block hold here (optional)
            self.task.hold_this_block = 500 < self.task.trial < 700
            
            # Take the action and observe the next observation and reward
            next_observation, reward, terminated, truncated, info = self.env.step(action)
            done = terminated or truncated        
            
            # Move to the next observation
            observation = next_observation
            
            actions.append(action)
            rewards.append(reward)

        self.task.plot_reward_schedule()

        # Assertions to verify the behavior of block ends
        self.assertEqual(self.task.block_ends[L], 
            [21, 66, 87, 109, 143, 235, 269, 306, 331, 356, 398, 
             432, 457, 701, 726, 776, 807, 840, 863, 930, 961, 985, 1006])

        self.assertEqual(self.task.block_ends[R], 
            [17, 67, 102, 129, 183, 240, 269, 321, 348, 387, 426, 451, 
             480, 701, 726, 787, 820, 854, 924, 964, 985, 1018])

        # Verify rewards
        print(rewards[-20:])
        self.assertEqual(rewards[-25:], 
                         [1, 0, 1, 1, 1, 1, 0, 1, 0, 1, 0, 0, 0,
                          0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0])
        self.assertEqual(np.array(rewards)[np.array(actions) == self.IGNORE].sum(), 0)
        

if __name__ == '__main__':
    unittest.main(verbosity=2)
