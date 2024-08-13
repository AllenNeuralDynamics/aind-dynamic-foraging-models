import unittest
import numpy as np
from aind_dynamic_foraging_models.generative_model.dynamic_foraging_tasks.uncoupled_block_task import (
    UncoupledBlockTask, L, R, IGNORE
    )


class TestUncoupledBlockTask(unittest.TestCase):
    """Test the UncoupledBlockTask by itself
    """
    def setUp(self):
        np.random.seed(56)
        self.total_trial = 1000
        self.reward_schedule = UncoupledBlockTask(perseverative_limit=4)
        self.reward_schedule.reset()  # Already includes a next_trial()

    def test_reward_schedule(self):

        while self.reward_schedule.trial < self.total_trial:
            # Replace this with the actual choice
            choice = [L, R, IGNORE][np.random.choice([0]*100 + [1]*20 + [2]*1)]
            
            # Add choice
            self.reward_schedule.add_action(choice)

            # Arbitrary hold (optional)
            self.reward_schedule.hold_this_block = 500 < self.reward_schedule.trial < 700
            
            # Next trial
            self.reward_schedule.next_trial()

        # Assertions to verify the behavior of block ends
        self.assertEqual(self.reward_schedule.block_ends[L], 
            [12, 34, 35, 55, 94, 126, 146, 173, 252, 286, 313, 
            379, 406, 431, 461, 482, 701, 721, 755, 775, 807,
            831, 863, 883, 910, 975, 1001])

        self.assertEqual(self.reward_schedule.block_ends[R], 
            [35, 46, 94, 114, 147, 171, 253, 285, 307, 380, 404,
            406, 444, 476, 701, 721, 772, 793, 822, 831, 872, 
            894, 910, 983, 1011])

    def test_plot_reward_schedule(self):
        # Call plot function and check it runs without error
        fig = self.reward_schedule.plot_reward_schedule()
        self.assertIsNotNone(fig)  # Ensure the figure is created

if __name__ == '__main__':
    unittest.main(verbosity=2)
