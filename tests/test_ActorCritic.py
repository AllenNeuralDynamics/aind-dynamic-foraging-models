"""Testing the actor-critic model"""

import multiprocessing as mp
import os
import unittest

import numpy as np
from aind_behavior_gym.dynamic_foraging.task import CoupledBlockTask

from aind_dynamic_foraging_models.generative_model import ForagerActorCritic


class TestActorCritic(unittest.TestCase):
    """Testing the actor-critic model"""

    def test_actor_critic(self):
        """Test generative simulation and MLE parameter recovery of the actor-critic model."""
        # Create results directory if it doesn't exist
        os.makedirs("tests/results", exist_ok=True)

        # -- Create task and forager --
        forager = ForagerActorCritic(choice_kernel="none", seed=42)
        forager.set_params(
            learn_rate_actor=0.5,
            learn_rate_critic=0.3,
            softmax_inverse_temperature=1.0,  # Clamped during fitting (see below)
            biasL=0.0,
        )
        task = CoupledBlockTask(reward_baiting=True, num_trials=300, seed=42)

        # -- 1. Generative run --
        forager.perform(task)
        ground_truth_params = forager.params.model_dump()
        ground_truth_choice_prob = forager.choice_prob

        # --    1.1 test figure --
        fig, axes = forager.plot_session(if_plot_latent=True)
        fig.savefig("tests/results/test_ActorCritic.png")
        self.assertIsNotNone(fig)

        # --    1.2 latent variables should have no NaNs after the first trial --
        self.assertFalse(np.isnan(forager.actor_preference[:, 1:]).any())
        self.assertFalse(np.isnan(forager.value[1:]).any())

        # --    1.3 make sure histories match between agent and env --
        np.testing.assert_array_equal(forager.choice_history, forager.task.get_choice_history())
        np.testing.assert_array_equal(forager.reward_history, forager.task.get_reward_history())

        # -- 2. Parameter recovery --
        choice_history = forager.get_choice_history()
        reward_history = forager.get_reward_history()

        # --    2.1 closed-loop should recover choice_prob exactly with the same params --
        forager.perform_closed_loop(choice_history, reward_history)
        np.testing.assert_array_almost_equal(forager.choice_prob, ground_truth_choice_prob)

        # --    2.2 model fitting --
        # NOTE: learn_rate_actor and beta are not jointly identifiable, so we clamp beta.
        forager = ForagerActorCritic(choice_kernel="none", seed=42)
        fitting_result, _ = forager.fit(
            choice_history,
            reward_history,
            clamp_params={"softmax_inverse_temperature": 1.0, "biasL": 0.0},
            DE_kwargs=dict(
                workers=mp.cpu_count(),
                disp=False,
                seed=np.random.default_rng(42),
                polish=True,
            ),
        )

        assert fitting_result.success

        fit_names = fitting_result.fit_settings["fit_names"]
        ground_truth = [ground_truth_params[name] for name in fit_names]
        print(f"\nNum of trials: {len(choice_history)}")
        print(f"Fitted parameters: {fit_names}")
        print(f'Ground truth: {[f"{num:.4f}" for num in ground_truth]}')
        print(f'Fitted:       {[f"{num:.4f}" for num in fitting_result.x]}')
        print(f"Likelihood-Per-Trial: {fitting_result.LPT}")
        print(f"Prediction accuracy full dataset: {fitting_result.prediction_accuracy}\n")

        # The two learning rates should be recovered reasonably well
        np.testing.assert_array_almost_equal(fitting_result.x, ground_truth, decimal=1)


if __name__ == "__main__":
    unittest.main(verbosity=2)
