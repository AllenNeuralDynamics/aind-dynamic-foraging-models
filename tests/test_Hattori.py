import unittest
import numpy as np
import multiprocessing as mp

from aind_dynamic_foraging_models.generative_model.agent_q_learning import forager_Hattori2019
from aind_behavior_gym.dynamic_foraging.task.coupled_block_task import CoupledBlockTask
from aind_dynamic_foraging_basic_analysis import plot_foraging_session


# Start a new test case
class TestHattori(unittest.TestCase):

    def test_Hattori(self):
        # -- Create task and forager --
        forager = forager_Hattori2019(
            dict(
                softmax_inverse_temperature=5,
                biasL=0,
            ),
            seed=42,
        )
        task = CoupledBlockTask(reward_baiting=True, num_trials=1000, seed=42)

        # -- 1. Generative run --
        forager.perform(task)
        ground_truth_params = forager.params.model_dump()
        ground_truth_choice_prob = forager.choice_prob

        # --    1.1 test figure --
        fig, axes = forager.plot_session()
        fig.savefig("tests/results/test_Hattori.png")
        self.assertIsNotNone(fig)

        # --    1.2 make sure histories match between agent and env --
        np.testing.assert_array_equal(forager.choice_history, forager.task.get_choice_history())
        np.testing.assert_array_equal(forager.reward_history, forager.task.get_reward_history())

        # -- 2. Parameter recovery --
        choice_history = forager.get_choice_history()
        reward_history = forager.get_reward_history()
        p_reward = forager.get_p_reward()

        # --    2.1 check predictive_perform --
        # It should recover the ground truth choice_prob because the params are exactly the same
        forager.predictive_perform(choice_history, reward_history)
        np.testing.assert_array_almost_equal(forager.choice_prob, ground_truth_choice_prob)

        # --    2.2 model fitting --
        forager = forager_Hattori2019()  # To fit a model, just create a new forager
        forager.fit(
            choice_history,
            reward_history,
            fit_bounds_override={"softmax_inverse_temperature": [0, 100]},
            clamp_params={"biasL": 0},
            DE_workers=8,
        )

        fitting_result = forager.fitting_result
        assert fitting_result.success

        # Check fitted parameters
        fit_names = fitting_result.fit_settings["fit_names"]
        ground_truth = [num for name, num in ground_truth_params.items() if name in fit_names]
        print(f"Fitted parameters: {fit_names}")
        print(f'Ground truth: {[f"{num:.4f}" for num in ground_truth]}')
        print(f'Fitted:       {[f"{num:.4f}" for num in fitting_result.x]}')

        np.testing.assert_array_almost_equal(
            fitting_result.x, [0.6010, 0.1087, 0.1544, 4.8908], decimal=2
        )

        # Plot fitted Q and choice_prob
        forager.predictive_perform(
            choice_history, reward_history
        )  # Note that forager's params are already set to fitted params
        # Plot fitted Q values
        axes[0].plot(forager.q_estimation[0], lw=1, color="red", ls=":", label="fitted_Q(L)")
        axes[0].plot(forager.q_estimation[1], lw=1, color="blue", ls=":", label="fitted_Q(R)")
        # Plot fitted choice_prob
        axes[0].plot(
            forager.choice_prob[1] / forager.choice_prob.sum(axis=0),
            lw=2,
            color="green",
            ls=":",
            label="fitted_choice_prob(R/R+L)",
        )
        axes[0].legend(fontsize=6, loc="upper left", bbox_to_anchor=(0.6, 1.3), ncol=4)
        fig.savefig("tests/results/test_Hattori_fitted.png")


if __name__ == "__main__":
    unittest.main(verbosity=2)
