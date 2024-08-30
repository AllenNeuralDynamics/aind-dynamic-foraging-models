"""Testing Hattori2019 model"""

import multiprocessing as mp
import unittest

import numpy as np
from aind_behavior_gym.dynamic_foraging.task import CoupledBlockTask

from aind_dynamic_foraging_models.generative_model import ForagerCollection


# Start a new test case
class TestHattori(unittest.TestCase):
    """Testing Hattori model"""

    def test_Hattori(self):
        """Test Hattori model"""
        # -- Create task and forager --
        forager = ForagerCollection().get_preset_forager("Hattori2019", seed=42)
        forager.set_params(
            softmax_inverse_temperature=5,
            biasL=0,
        )
        task = CoupledBlockTask(reward_baiting=True, num_trials=100, seed=42)

        # -- 1. Generative run --
        forager.perform(task)
        ground_truth_params = forager.params.model_dump()
        ground_truth_choice_prob = forager.choice_prob
        ground_truth_q_value = forager.q_value

        # --    1.1 test figure --
        fig, axes = forager.plot_session(if_plot_latent=True)
        fig.savefig("tests/results/test_Hattori.png")
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
        # To fit a model, just create a new forager
        forager = ForagerCollection().get_preset_forager("Hattori2019", seed=42)
        forager.fit(
            choice_history,
            reward_history,
            fit_bounds_override={"softmax_inverse_temperature": [0, 100]},
            clamp_params={"biasL": 0},
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
        fig_fitting, axes = forager.plot_fitted_session(if_plot_latent=True)
        # Add groundtruth
        axes[0].plot(ground_truth_q_value[0], lw=1, color="red", ls="-", label="actual_Q(L)")
        axes[0].plot(ground_truth_q_value[1], lw=1, color="blue", ls="-", label="actual_Q(R)")
        axes[0].legend(fontsize=6, loc="upper left", bbox_to_anchor=(0.6, 1.3), ncol=4)
        fig_fitting.savefig("tests/results/test_Hattori_fitted.png")

        np.testing.assert_array_almost_equal(
            fitting_result.x, [0.6033, 0.1988, 0.2559, 5.3599], decimal=2
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
