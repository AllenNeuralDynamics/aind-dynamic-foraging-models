"""Testing Bari2019 model"""

import multiprocessing as mp
import unittest
import sys

import numpy as np
from aind_behavior_gym.dynamic_foraging.task import CoupledBlockTask

from aind_dynamic_foraging_models.generative_model.agent_q_learning import ForagerSimpleQ


# Start a new test case
class TestBari(unittest.TestCase):
    """Testing Bari model"""

    def test_Bari(self):
        """Test Bari model"""
        # -- Create task and forager --
        forager = ForagerSimpleQ(
            number_of_learning_rate=1,
            number_of_forget_rate=1,
            choice_kernel="one_step",
            action_selection="softmax",
            seed=42,
        )
        forager.set_params(
            dict(
                learn_rate=0.3,
                forget_rate_unchosen=0.1,
                choice_kernel_relative_weight=0.1,
                softmax_inverse_temperature=10,
                biasL=0,
            )
        )

        n_trials = 100
        task = CoupledBlockTask(reward_baiting=True, num_trials=n_trials, seed=42)

        # -- 1. Generative run --
        forager.perform(task)
        ground_truth_params = forager.params.model_dump()
        ground_truth_choice_prob = forager.choice_prob
        ground_truth_q_estimation = forager.q_estimation
        ground_truth_choice_kernel = forager.choice_kernel

        # --    1.1 test figure --
        fig, axes = forager.plot_session()
        fig.savefig("tests/results/test_Bari.png")
        self.assertIsNotNone(fig)

        # --    1.2 make sure histories match between agent and env --
        np.testing.assert_array_equal(forager.choice_history, forager.task.get_choice_history())
        np.testing.assert_array_equal(forager.reward_history, forager.task.get_reward_history())

        # -- 2. Parameter recovery --
        choice_history = forager.get_choice_history()
        reward_history = forager.get_reward_history()

        # --    2.1 check predictive_perform --
        # It should recover the ground truth choice_prob because the params are exactly the same
        forager.predictive_perform(choice_history, reward_history)
        np.testing.assert_array_almost_equal(forager.choice_prob, ground_truth_choice_prob)

        # --    2.2 model fitting with cross-validation --
        forager = ForagerSimpleQ(
            number_of_learning_rate=1,
            number_of_forget_rate=1,
            choice_kernel="one_step",
            action_selection="softmax",
            seed=42,
        )  # To fit a model, just create a new forager
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
        x = np.arange(forager.n_trials + 1) + 1  # When plotting, we start from 1
        axes[0].plot(
            x, ground_truth_q_estimation[0], lw=1, color="red", ls="-", label="actual_Q(L)"
        )
        axes[0].plot(
            x, ground_truth_q_estimation[1], lw=1, color="blue", ls="-", label="actual_Q(R)"
        )
        axes[0].plot(
            x,
            ground_truth_choice_kernel[0],
            lw=1,
            color="purple",
            ls="-",
            label="actual_choice_kernel(L)",
        )
        axes[0].plot(
            x,
            ground_truth_choice_kernel[1],
            lw=1,
            color="cyan",
            ls="-",
            label="actual_choice_kernel(R)",
        )
        axes[0].legend(fontsize=6, loc="upper left", bbox_to_anchor=(0.6, 1.3), ncol=4)
        fig_fitting.savefig("tests/results/test_Bari_fitted.png")

        if sys.version_info[:2] == (3, 9) and n_trials == 100:
            """For unknown reasons the DE's rng will change behavior across python versions"""
            np.testing.assert_array_almost_equal(
                fitting_result.x, [0.7810, 0.0000, 0.0127, 1.0000, -0.2543, 94.9749], decimal=2
            )
            print("Fitting result tested")
        else:
            print("Not python 3.9. Fitting result not tested")


if __name__ == "__main__":
    unittest.main(verbosity=2)
