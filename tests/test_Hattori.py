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
                softmax_inverse_temperature=3,
                biasL=0.3,
                ),
            seed=42,
            )
        task = CoupledBlockTask(
            reward_baiting=True, 
            num_trials=1000,
            seed=42)

        # -- 1. Generative run --
        forager.perform(task)
        ground_truth_params = forager.params.model_dump()
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
        
        # --    2.2 model fitting --
        forager = forager_Hattori2019()  # Start a new agent
        forager.fit(choice_history, reward_history, 
                    fit_bounds_override={
                        'softmax_inverse_temperature': [0, 100]
                        },
                    clamp_params={},
                    DE_workers=1)
        
        fitting_result = forager.fitting_result
        ground_truth_params = [ground_truth_params[key] for key in fitting_result.fit_names]
        relative_error = np.abs((fitting_result.x - ground_truth_params) / ground_truth_params)
        
        assert fitting_result.success
        print(f'\n\nNum_trials: {len(choice_history)}')
        print(f'Fitting names: {fitting_result.fit_names}')
        print(f'Fitting result: {[f"{num:.3f}" for num in fitting_result.x]}')
        print(f'Ground truth:   {[f"{num:.3f}" for num in ground_truth_params]}')
        print(f'Relative error: {relative_error}')
        
        np.testint.assert_array_almost_equal(
            actual=fitting_result.x, 
            desired=[0.582, 0.114, 0.183, 2.867, 0.273],
            decimal=2,
        )
        
        # np.testing.assert_allclose(
        #     actual=fitting_result.x, 
        #     desired=ground_truth_params,
        #     rtol=0.15,  # 2000 trials typically has 10% error
        # )
        

if __name__ == '__main__':
    unittest.main(verbosity=2)
