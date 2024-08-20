import unittest
import numpy as np

from aind_dynamic_foraging_models.generative_model.agent_q_learning import (
    forager_Hattori2019
)
from aind_behavior_gym.dynamic_foraging.task.coupled_block_task import CoupledBlockTask
from aind_dynamic_foraging_basic_analysis import plot_foraging_session


# Start a new test case
class TestHattori(unittest.TestCase):

    def test_Hattori(self):
        # -- Create task and forager --
        forager = forager_Hattori2019(
            dict(
                softmax_inverse_temperature=5,
                biasL=0.5,
                ),
            seed=42,
            )
        task = CoupledBlockTask(reward_baiting=True, seed=42)

        # -- 1. Generative run --
        forager.perform(task)
        ground_truth_params = forager.params
        ground_truth_choice_prob = forager.choice_prob

        # --    1.1 test figure --
        fig = forager.plot_session()
        fig.savefig("tests/results/test_Hattori.png")
        self.assertIsNotNone(fig)

        # --    1.2 make sure histories match between agent and env --
        np.testing.assert_array_equal(forager.choice_history, forager.task.get_choice_history())
        np.testing.assert_array_equal(forager.reward_history, 
                                      forager.task.get_reward_history())

        # -- 2. Parameter recovery --
        choice_history = forager.get_choice_history()
        reward_history = forager.get_reward_history()
        p_reward = forager.get_p_reward()

        # --    2.1 check predictive_perform --
        # It should recover the ground truth choice_prob because the params are exactly the same
        forager.predictive_perform(choice_history, reward_history)
        np.testing.assert_array_almost_equal(forager.choice_prob, ground_truth_choice_prob)
        
        

if __name__ == '__main__':
    unittest.main(verbosity=2)
