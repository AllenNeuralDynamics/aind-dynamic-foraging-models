"""Testing Bari2019 model"""

import multiprocessing as mp
import sys
import unittest

import numpy as np
from aind_behavior_gym.dynamic_foraging.task import CoupledBlockTask

from aind_dynamic_foraging_models.generative_model import ForagerCollection


# Start a new test case
class TestBari(unittest.TestCase):
    """Testing Bari model"""

    def test_Bari(self):
        """Test Bari model"""
        # -- Create task and forager --
        forager = ForagerCollection().get_preset_forager("Bari2019", seed=42)
        forager.set_params(
            learn_rate=0.3,
            forget_rate_unchosen=0.1,
            choice_kernel_relative_weight=0.1,
            softmax_inverse_temperature=10,
            biasL=0,
        )

        n_trials = 100
        task = CoupledBlockTask(reward_baiting=True, num_trials=n_trials, seed=42)

        # -- 1. Generative run --
        forager.perform(task)
        ground_truth_params = forager.params.model_dump()
        ground_truth_choice_prob = forager.choice_prob
        ground_truth_q_value = forager.q_value
        ground_truth_choice_kernel = forager.choice_kernel

        # --    1.1 test figure --
        fig, axes = forager.plot_session(if_plot_latent=True)
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
        forager.perform_closed_loop(choice_history, reward_history)
        np.testing.assert_array_almost_equal(forager.choice_prob, ground_truth_choice_prob)

        # # --    2.2 model fitting with cross-validation --
        # To fit a model, just create a new forager
        forager = ForagerCollection().get_preset_forager("Bari2019", seed=42)
        forager.fit(
            choice_history,
            reward_history,
            DE_kwargs=dict(workers=mp.cpu_count(), disp=False, seed=np.random.default_rng(42)),
            k_fold_cross_validation=None,
        )

        fitting_result = forager.fitting_result
        assert fitting_result.success

        # Check fitted parameters
        fit_names = fitting_result.fit_settings["fit_names"]
        ground_truth = [num for name, num in ground_truth_params.items() if name in fit_names]
        print(f"Bari2019, num of trials: {len(choice_history)}")
        print(f"Fitted parameters: {fit_names}")
        print(f'Ground truth: {[f"{num:.4f}" for num in ground_truth]}')
        print(f'Fitted:       {[f"{num:.4f}" for num in fitting_result.x]}')
        print(f"Likelihood-Per-Trial: {fitting_result.LPT}")
        print(f"Prediction accuracy full dataset: {fitting_result.prediction_accuracy}\n")

        # Check get_fitting_result_dict
        forager.get_fitting_result_dict()

        # Plot fitted latent variables
        fig_fitting, axes = forager.plot_fitted_session(if_plot_latent=True)
        # Add groundtruth
        x = np.arange(forager.n_trials + 1) + 1  # When plotting, we start from 1
        axes[0].plot(x, ground_truth_q_value[0], lw=1, color="red", ls="-", label="actual_Q(L)")
        axes[0].plot(x, ground_truth_q_value[1], lw=1, color="blue", ls="-", label="actual_Q(R)")
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
                fitting_result.x, [0.7727, 0.0000, 0.0139, -0.2480, 86.4626], decimal=2
            )
            print("Fitting result tested")
        else:
            print("Not python 3.9. Fitting result not tested")


if __name__ == "__main__":
    unittest.main(verbosity=2)
