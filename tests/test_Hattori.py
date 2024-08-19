import unittest
import numpy as np

from aind_dynamic_foraging_models.generative_model.agent_q_learning import (
    forager_Hattori2019
)
from aind_behavior_gym.dynamic_foraging.task.coupled_block_task import CoupledBlockTask
from aind_dynamic_foraging_basic_analysis import plot_foraging_session


# Start a new test case
class TestHattoriGenerative(unittest.TestCase):
    
    def test_Hattori(self):
        # Create task and forager
        forager = forager_Hattori2019(seed=42)
        task = CoupledBlockTask(reward_baiting=True, seed=42)
        
        forager.perform(task)
        fig = forager.plot_session()
        
        fig.savefig("tests/results/test_Hattori.png")
        
        # Make sure histories match between agent and env
        np.testing.assert_array_equal(forager.choice_history[0], forager.task.get_choice_history())
        np.testing.assert_array_equal(np.sum(forager.reward_history, axis=0), 
                                      forager.task.get_reward_history())
        
        
if __name__ == '__main__':
    unittest.main(verbosity=2)

