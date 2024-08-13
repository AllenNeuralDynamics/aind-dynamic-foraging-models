import unittest
import numpy as np
from aind_dynamic_foraging_models.generative_model.dynamic_foraging_tasks.uncoupled_block_task import (
    UncoupledBlockTask, L, R, IGNORE
    )


class TestUncoupledBlockTask(unittest.TestCase):
    """Test the UncoupledBlockTask by itself
    """
    def setUp(self):
        self.total_trial = 1000
        self.reward_schedule = UncoupledBlockTask(perseverative_limit=4)
        self.reward_schedule.reset(seed=42)  # Already includes a next_trial()

    def test_reward_schedule(self):
        
        rng = np.random.default_rng(seed=42) # Another independent random number generator
        
        while self.reward_schedule.trial < self.total_trial:
            # Replace this with the actual choice
            choice = [L, R, IGNORE][rng.choice([0]*100 + [1]*20 + [2]*1)]
            
            # Add choice
            self.reward_schedule.add_action(choice)

            # Arbitrary hold (optional)
            self.reward_schedule.hold_this_block = 500 < self.reward_schedule.trial < 700
            
            # Next trial
            self.reward_schedule.next_trial()

        # Call plot function and check it runs without error
        fig = self.reward_schedule.plot_reward_schedule()
        fig.savefig("test_uncoupled_block_task.png")
        self.assertIsNotNone(fig)  # Ensure the figure is created

        # Assertions to verify the behavior of block ends
        self.assertEqual(self.reward_schedule.block_ends[L], 
            [21, 66, 87, 109, 143, 235, 269, 306, 331, 356, 398, 
             432, 457, 701, 726, 776, 807, 840, 863, 930, 961, 985, 1006])

        self.assertEqual(self.reward_schedule.block_ends[R], 
            [17, 67, 102, 129, 183, 240, 269, 321, 348, 387, 426, 451, 
             480, 701, 726, 787, 820, 854, 924, 964, 985, 1018])


if __name__ == '__main__':
    unittest.main(verbosity=2)
