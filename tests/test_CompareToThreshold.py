"""Testing CompareToThreshold model"""

import multiprocessing as mp
import os
import unittest

import numpy as np
from aind_behavior_gym.dynamic_foraging.task import CoupledBlockTask

from aind_dynamic_foraging_models.generative_model import ForagerCollection


# Start a new test case
class TestCompareToThreshold(unittest.TestCase):
    """Testing CompareToThreshold model"""

    def test_CompareToThreshold(self):
        """Test CompareToThreshold model"""
        # Create results directory if it doesn't exist
        os.makedirs("tests/results", exist_ok=True)

        # -- Create task and forager --
        forager = ForagerCollection().get_preset_forager("CompareToThreshold", seed=42)
        forager.set_params(
            learn_rate=0.3,
            threshold=0.5,
            softmax_inverse_temperature=5,
            biasL=0,
        )
        task = CoupledBlockTask(reward_baiting=True, num_trials=100, seed=42)

        # -- 1. Generative run --
        forager.perform(task)
        ground_truth_params = forager.params.model_dump()
        ground_truth_choice_prob = forager.choice_prob
        ground_truth_value = forager.value
        ground_truth_exploiting = forager.exploiting

        # Get histories for later use
        choice_history = forager.get_choice_history()
        reward_history = forager.get_reward_history()

        # --    1.1 test figure --
        fig, axes = forager.plot_session(if_plot_latent=True)
        fig.savefig("tests/results/test_CompareToThreshold.png")
        self.assertIsNotNone(fig)

        # --    1.2 make sure histories match between agent and env --
        np.testing.assert_array_equal(forager.choice_history, forager.task.get_choice_history())
        np.testing.assert_array_equal(forager.reward_history, forager.task.get_reward_history())

        # --    1.3 test specific compare-to-threshold model properties --
        # Check that value and exploiting arrays have expected shapes
        self.assertEqual(len(ground_truth_value), len(choice_history) + 1)
        self.assertEqual(len(ground_truth_exploiting), len(choice_history))

        # Check that value starts at threshold
        self.assertEqual(ground_truth_value[0], forager.params.threshold)

        # -- 2. Parameter recovery --

        # --    2.1 check predictive_perform --
        # It should recover the ground truth choice_prob because the params are exactly the same
        forager.perform_closed_loop(choice_history, reward_history)
        np.testing.assert_array_almost_equal(forager.choice_prob, ground_truth_choice_prob)

        # --    2.2 model fitting with cross-validation --
        # To fit a model, just create a new forager
        forager = ForagerCollection().get_preset_forager("CompareToThreshold", seed=42)
        forager.fit(
            choice_history,
            reward_history,
            fit_bounds_override={"softmax_inverse_temperature": [0, 100]},
            clamp_params={"biasL": 0},
            DE_kwargs=dict(
                workers=mp.cpu_count(),
                disp=False,
                seed=np.random.default_rng(42),
                polish=True,
            ),
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

        # Check get_fitting_result_dict
        forager.get_fitting_result_dict()

        # Plot fitted latent variables
        fig_fitting, axes = forager.plot_fitted_session(if_plot_latent=True)
        # Add groundtruth
        axes[0].plot(ground_truth_value, lw=1, color="red", ls="-", label="actual_value")
        axes[0].axhline(
            y=ground_truth_params["threshold"],
            color="black",
            linestyle="--",
            lw=1,
            label="actual_threshold",
        )
        axes[0].legend(fontsize=6, loc="upper left", bbox_to_anchor=(0.6, 1.3), ncol=4)
        os.makedirs("tests/results", exist_ok=True)
        fig_fitting.savefig("tests/results/test_CompareToThreshold_fitted.png")

        # Test that fitted parameters are reasonable (allowing some tolerance due to optimization)
        self.assertAlmostEqual(fitting_result.x[0], 0.675, delta=0.1)  # learn_rate
        self.assertAlmostEqual(fitting_result.x[1], 0.4634, delta=0.1)  # threshold
        self.assertAlmostEqual(fitting_result.x[2], 3.83, delta=0.1)  # softmax_inverse_temperature

    def test_CompareToThreshold_with_choice_kernel(self):
        """Test CompareToThreshold model with choice kernel enabled"""
        # -- Create forager with choice kernel --
        forager_collection = ForagerCollection()
        forager = forager_collection.get_forager(
            agent_class_name="ForagerCompareThreshold",
            agent_kwargs={"choice_kernel": "one_step"},
            seed=42,
        )
        forager.set_params(
            learn_rate=0.3,
            threshold=0.5,
            softmax_inverse_temperature=5,
            biasL=0,
            choice_kernel_step_size=1.0,
            choice_kernel_relative_weight=0.2,
        )
        task = CoupledBlockTask(reward_baiting=True, num_trials=100, seed=42)

        # -- Generative run --
        forager.perform(task)

        # --    test figure --
        os.makedirs("tests/results", exist_ok=True)
        fig, axes = forager.plot_session(if_plot_latent=True)
        fig.savefig("tests/results/test_CompareToThreshold_choice_kernel.png")
        self.assertIsNotNone(fig)

        # Test that choice kernel is being used
        self.assertFalse(np.all(np.isnan(forager.choice_kernel)))

        # Test latent variables
        latent_vars = forager.get_latent_variables()
        self.assertIn("value", latent_vars)
        self.assertIn("threshold", latent_vars)
        self.assertIn("exploiting", latent_vars)
        self.assertIn("choice_kernel", latent_vars)
        self.assertIn("choice_prob", latent_vars)
        self.assertIn("p_exploit", latent_vars)


if __name__ == "__main__":
    unittest.main(verbosity=2)
