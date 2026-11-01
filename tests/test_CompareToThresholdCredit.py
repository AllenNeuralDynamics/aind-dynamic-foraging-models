"""Testing the stay-credit form of the compare-to-threshold model"""

import unittest

import numpy as np
from aind_behavior_gym.dynamic_foraging.task import CoupledBlockTask

from aind_dynamic_foraging_models.generative_model import (
    ForagerCompareThreshold,
    ForagerCompareThresholdCredit,
)


def _histories(seed, n_sessions=3, n_trials=300):
    rng = np.random.default_rng(seed)
    choices = [rng.integers(0, 2, n_trials).astype(float) for _ in range(n_sessions)]
    rewards = [rng.integers(0, 2, n_trials).astype(float) for _ in range(n_sessions)]
    return choices, rewards


class TestCompareToThresholdCredit(unittest.TestCase):
    """The credit form is CompareToThreshold with stay credits as its parameters"""

    def test_likelihood_equals_compare_to_threshold(self):
        """Matched parameters give the same choice probabilities, trial by trial"""
        choices, rewards = _histories(0)
        for learn_rate, threshold, beta, bias in [
            (0.17, -0.64, 2.69, -0.12),
            (0.5, 0.4, 10.0, 0.0),
            (0.05, -1.0, 4.0, 1.3),
        ]:
            ctt = ForagerCompareThreshold(seed=0)
            ctt.set_params(learn_rate=learn_rate, threshold=threshold,
                           softmax_inverse_temperature=beta, biasL=bias)
            credit = ForagerCompareThresholdCredit(seed=0)
            credit.set_params(learn_rate=learn_rate, stay_credit_win=beta * (1 - threshold),
                              stay_credit_loss=-beta * threshold, biasL=bias)
            for a, b in zip(ctt.perform_closed_loop_multi_session(choices, rewards),
                            credit.perform_closed_loop_multi_session(choices, rewards)):
                np.testing.assert_allclose(np.asarray(a), np.asarray(b), rtol=0, atol=1e-12)

    def test_likelihood_equals_compare_to_threshold_for_graded_rewards(self):
        """Reward magnitudes between 0 and 1 also match (affine target, as CTT's value update)"""
        choices, _ = _histories(2)
        rng = np.random.default_rng(3)
        rewards = [rng.choice([0.0, 0.5, 1.0], size=len(c)) for c in choices]
        ctt = ForagerCompareThreshold(seed=0)
        ctt.set_params(learn_rate=0.2, threshold=0.4, softmax_inverse_temperature=10.0, biasL=0.0)
        credit = ForagerCompareThresholdCredit(seed=0)
        credit.set_params(learn_rate=0.2, stay_credit_win=6.0, stay_credit_loss=-4.0, biasL=0.0)
        for a, b in zip(ctt.perform_closed_loop_multi_session(choices, rewards),
                        credit.perform_closed_loop_multi_session(choices, rewards)):
            np.testing.assert_allclose(np.asarray(a), np.asarray(b), rtol=0, atol=1e-12)

    def test_reaches_beyond_the_threshold_bound(self):
        """A loss credit above half the win credit (threshold < -1) is inside its bounds"""
        credit = ForagerCompareThresholdCredit(seed=0)
        bounds = credit.ParamFitBoundModel().model_dump()
        self.assertEqual(list(bounds["stay_credit_win"]), [-20.0, 20.0])
        self.assertEqual(list(bounds["stay_credit_loss"]), [-20.0, 20.0])
        # threshold = -3 is c_win = 4 beta, c_loss = 3 beta; with beta = 2: (8, 6)
        ctt = ForagerCompareThreshold(seed=0)
        ctt.set_params(learn_rate=0.2, threshold=-3.0, softmax_inverse_temperature=2.0, biasL=0.0)
        credit.set_params(learn_rate=0.2, stay_credit_win=8.0, stay_credit_loss=6.0, biasL=0.0)
        choices, rewards = _histories(1, n_sessions=1)
        np.testing.assert_allclose(
            np.asarray(ctt.perform_closed_loop_multi_session(choices, rewards)[0]),
            np.asarray(credit.perform_closed_loop_multi_session(choices, rewards)[0]),
            rtol=0, atol=1e-12)

    def test_generative_run_and_fit(self):
        """It runs a task, starts with no credit, and its fit recovers four parameters"""
        forager = ForagerCompareThresholdCredit(seed=42)
        forager.set_params(learn_rate=0.3, stay_credit_win=4.0, stay_credit_loss=1.5, biasL=0.0)
        forager.perform(CoupledBlockTask(reward_baiting=True, num_trials=200, seed=42))
        self.assertEqual(forager.credit[0], 0.0)
        np.testing.assert_array_equal(forager.choice_history, forager.task.get_choice_history())
        fitting_result, _ = ForagerCompareThresholdCredit(seed=0).fit(
            fit_choice_history=forager.get_choice_history(),
            fit_reward_history=forager.get_reward_history(),
            DE_kwargs=dict(workers=1, maxiter=50),
        )
        self.assertEqual(set(fitting_result.params),
                         {"learn_rate", "stay_credit_win", "stay_credit_loss", "biasL"})


if __name__ == "__main__":
    unittest.main(verbosity=2)
