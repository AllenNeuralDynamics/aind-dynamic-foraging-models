"""Testing loss-counting model"""

import multiprocessing as mp
import sys
import unittest

import numpy as np
from aind_behavior_gym.dynamic_foraging.task import CoupledBlockTask

from aind_dynamic_foraging_models.generative_model import ForagerLossCounting


# Start a new test case
class TestLossCounting(unittest.TestCase):
    """Testing LossCounting model"""

    def test_LossCounting(self):
        """Test LossCounting model"""
        # -- Create task and forager --
        forager = ForagerLossCounting(
            choice_kernel="full",  # No choice kernel
            params=dict(
                loss_count_threshold_mean=5.0,
                loss_count_threshold_std=2.0,
                biasL=-0.2,
                choice_kernel_step_size=1.0,
                choice_kernel_relative_weight=0.2,
            ),
            seed=42,
        )

        n_trials = 100
        task = CoupledBlockTask(reward_baiting=True, num_trials=n_trials, seed=42)

        # -- 1. Generative run --
        forager.perform(task)
        ground_truth_params = forager.params.model_dump()
        ground_truth_choice_prob = forager.choice_prob

        # --    1.1 test figure --
        fig, axes = forager.plot_session(if_plot_latent=True)
        fig.savefig("tests/results/test_LossCounting.png")
        self.assertIsNotNone(fig)

        # --    1.2 make sure histories match between agent and env --
        np.testing.assert_array_equal(forager.choice_history, forager.task.get_choice_history())
        np.testing.assert_array_equal(forager.reward_history, forager.task.get_reward_history())

        # -- 2. Parameter recovery --
        choice_history = forager.get_choice_history()
        reward_history = forager.get_reward_history()

        # --    2.1 check predictive_perform --
        # It should recover the ground truth choice_prob because the params are exactly the same
        forager.perform_closed_loop(choice_history, reward_history)
        np.testing.assert_array_almost_equal(forager.choice_prob, ground_truth_choice_prob)

        # --    2.2 model fitting with cross-validation --
        forager = ForagerLossCounting(
            choice_kernel="full",
            seed=42,
        )  # To fit a model, just create a new forager
        forager.fit(
            choice_history,
            reward_history,
            DE_kwargs=dict(workers=mp.cpu_count(), disp=False, seed=np.random.default_rng(42)),
            k_fold_cross_validation=None,
        )

        fitting_result = forager.fitting_result
        assert fitting_result.success

        # Check get_fitting_result_dict
        forager.get_fitting_result_dict()

        # Check fitted parameters
        fit_names = fitting_result.fit_settings["fit_names"]
        ground_truth = [num for name, num in ground_truth_params.items() if name in fit_names]
        print(f"Loss counting, num of trials: {len(choice_history)}")
        print(f"Fitted parameters: {fit_names}")
        print(f'Ground truth: {[f"{num:.4f}" for num in ground_truth]}')
        print(f'Fitted:       {[f"{num:.4f}" for num in fitting_result.x]}')
        print(f"Likelihood-Per-Trial: {fitting_result.LPT}")
        print(f"Prediction accuracy full dataset: {fitting_result.prediction_accuracy}\n")

        # Plot fitted latent variables
        fig_fitting, axes = forager.plot_fitted_session(if_plot_latent=True)
        # Add groundtruth
        x = np.arange(forager.n_trials) + 1  # When plotting, we start from 1
        axes[0].plot(
            x,
            ground_truth_choice_prob[1] / ground_truth_choice_prob.sum(axis=0),
            lw=1,
            color="green",
            ls="-",
            label="actual_choice_probability(R/(R+L))",
        )

        axes[0].legend(fontsize=6, loc="upper left", bbox_to_anchor=(0.6, 1.3), ncol=4)
        fig_fitting.savefig("tests/results/test_LossCounting_fitted.png")

        if sys.version_info[:2] == (3, 9) and n_trials == 100:
            """For unknown reasons the DE's rng will change behavior across python versions"""
            np.testing.assert_array_almost_equal(
                fitting_result.x, [3.9196, 1.3266, -0.1700, 0.4781, 0.3401], decimal=2
            )
            print("Fitting result tested")
        else:
            print("Not python 3.9. Fitting result not tested")


if __name__ == "__main__":
    unittest.main(verbosity=2)
