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
        """Test LossCounting model (biased win-stay-lose-shift)"""
        # -- Create task and forager --
        forager = ForagerLossCounting(
            choice_kernel="full",  # No choice kernel
            params=dict(
                loss_count_threshold_mean=1.0,  # Win-stay-lose-shift
                loss_count_threshold_std=0.0,
                biasL=-0.2,
                choice_kernel_step_size=0.1,
                choice_kernel_relative_weight=0.1,
            ),
            seed=42,
        )

        n_trials = 500
        task = CoupledBlockTask(reward_baiting=True, num_trials=n_trials, seed=42)

        # -- 1. Generative run --
        forager.perform(task)
        ground_truth_params = forager.params.model_dump()
        ground_truth_choice_prob = forager.choice_prob
        ground_truth_loss_count = forager.loss_count

        # --    1.1 test figure --
        fig, axes = forager.plot_session()
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
            choice_kernel="full",  # No choice kernel
            seed=42,
            )   # To fit a model, just create a new forager
        forager.fit(
            choice_history,
            reward_history,
            DE_kwargs=dict(workers=mp.cpu_count(), disp=False, seed=np.random.default_rng(42)),
            k_fold_cross_validation=2,
        )

        fitting_result = forager.fitting_result
        fitting_result_cross_validation = forager.fitting_result_cross_validation
        assert fitting_result.success

        # Check fitted parameters
        fit_names = fitting_result.fit_settings["fit_names"]
        ground_truth = [num for name, num in ground_truth_params.items() if name in fit_names]
        print(f"Num of trials: {len(choice_history)}")
        print(f"Fitted parameters: {fit_names}")
        print(f'Ground truth: {[f"{num:.4f}" for num in ground_truth]}')
        print(f'Fitted:       {[f"{num:.4f}" for num in fitting_result.x]}')
        print(f"Likelihood-Per-Trial: {fitting_result.LPT}")
        print(f"Prediction accuracy full dataset: {fitting_result.prediction_accuracy}\n")
        print(
            f"Prediction accuracy cross-validation (training): "
            f'{np.mean(fitting_result_cross_validation["prediction_accuracy_fit"])}'
        )
        print(
            f"Prediction accuracy cross-validation (test): "
            f'{np.mean(fitting_result_cross_validation["prediction_accuracy_test"])}'
        )
        print(
            f"Prediction accuracy cross-validation (test, bias only): "
            f'{np.mean(fitting_result_cross_validation["prediction_accuracy_test_bias_only"])}'
        )

        # Plot fitted latent variables
        fig_fitting, axes = forager.plot_fitted_session()
        # Add groundtruth
        x = np.arange(forager.n_trials) + 1  # When plotting, we start from 1
        axes[0].plot(
            x, 
            ground_truth_choice_prob[1] / ground_truth_choice_prob.sum(axis=0),
            lw=1, color="green", ls="-", label="actual_choice_probability(R/(R+L))"
        )

        axes[0].legend(fontsize=6, loc="upper left", bbox_to_anchor=(0.6, 1.3), ncol=4)
        fig_fitting.savefig("tests/results/test_LossCounting_fitted.png")

        if sys.version_info[:2] == (3, 9) and n_trials == 100:
            """For unknown reasons the DE's rng will change behavior across python versions"""
            np.testing.assert_array_almost_equal(
                fitting_result.x, [0.5074, 0.3762, -0.0952], decimal=2
            )
            print("Fitting result tested")
        else:
            print("Not python 3.9. Fitting result not tested")


if __name__ == "__main__":
    unittest.main(verbosity=2)