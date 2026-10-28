"""The CompareToThreshold (CTT) forager in the hierarchical Bayesian subpackage.

Parity of the JAX likelihood with the numpy ``ForagerCompareThreshold`` (the
``CompareToThreshold`` preset), the CTT path through the held-out machinery, and a frozen
Hattori2019 score that the model-agnostic refactor must leave unchanged.
"""

import unittest

from tests._hb_deps import assert_deps_present

import numpy as np
from aind_behavior_gym.dynamic_foraging.task import CoupledBlockTask

from aind_dynamic_foraging_models.generative_model import ForagerCollection

try:
    import jax

    from aind_dynamic_foraging_models.hierarchical_bayes import (
        ctt_choice_prob,
        ctt_log_likelihood,
    )
    from aind_dynamic_foraging_models.hierarchical_bayes.heldout import (
        FIXED_HYPERPRIOR,
        batched_heldout_scores,
        fit_heldout_two_level,
        fixed_hyperprior,
        importance_sampled_session_marginal,
    )
    from aind_dynamic_foraging_models.hierarchical_bayes.model import (
        CTT_PARAMS,
        ctt_session_params,
        get_model,
    )

    HAS_JAX = True
except ImportError:  # pragma: no cover - exercised only without the bayes extra
    HAS_JAX = False

assert_deps_present(HAS_JAX)


def _simulate_ctt(params, n_trials=300, seed=0):
    """Run the numpy CTT forager, then replay it teacher-forced; return history and probs."""
    forager = ForagerCollection().get_preset_forager("CompareToThreshold", seed=seed)
    forager.set_params(
        learn_rate=params["learn_rate"],
        threshold=params["threshold"],
        softmax_inverse_temperature=params["softmax_inverse_temperature"],
        biasL=params["bias_l"],
    )
    task = CoupledBlockTask(reward_baiting=True, num_trials=n_trials, seed=seed)
    forager.perform(task)
    choices = forager.get_choice_history()
    rewards = forager.get_reward_history()
    forager.perform_closed_loop(choices, rewards)
    return choices, rewards, forager.choice_prob


def _sessions(shape, seed=0):
    """Random choice and reward histories of the given shape."""
    rng = np.random.default_rng(seed)
    choices = rng.integers(0, 2, size=shape)
    rewards = (rng.random(shape) < 0.4).astype(float)
    return choices, rewards


@unittest.skipUnless(HAS_JAX, "requires the 'bayes' extra (jax)")
class TestCTTParity(unittest.TestCase):
    """The JAX CTT likelihood reproduces the numpy forager."""

    def test_choice_prob_matches_numpy_forager(self):
        """Per-trial choice probabilities agree across random and edge parameter sets."""
        rng = np.random.default_rng(0)
        cases = [
            {
                "learn_rate": float(rng.uniform(0.0, 1.0)),
                "threshold": float(rng.uniform(-1.0, 1.0)),
                "softmax_inverse_temperature": float(rng.uniform(0.1, 15.0)),
                "bias_l": float(rng.uniform(-2.0, 2.0)),
            }
            for _ in range(8)
        ]
        cases += [
            {"learn_rate": 0.0, "threshold": 0.4, "softmax_inverse_temperature": 10.0,
             "bias_l": 0.0},
            {"learn_rate": 1.0, "threshold": -0.5, "softmax_inverse_temperature": 3.0,
             "bias_l": 3.0},
            {"learn_rate": 0.3, "threshold": 0.2, "softmax_inverse_temperature": 0.0,
             "bias_l": -3.0},
        ]
        for index, params in enumerate(cases):
            with self.subTest(case=index, **params):
                choices, rewards, expected = _simulate_ctt(params, seed=index)
                actual = ctt_choice_prob(choices, rewards, **params)
                np.testing.assert_allclose(np.asarray(actual), expected, rtol=1e-5, atol=1e-6)

    def test_log_likelihood_matches_numpy(self):
        """The summed log likelihood matches the numpy forager's own probabilities."""
        params = {"learn_rate": 0.4, "threshold": 0.3, "softmax_inverse_temperature": 6.0,
                  "bias_l": 0.5}
        choices, rewards, prob = _simulate_ctt(params)
        expected = np.sum(np.log(prob[choices.astype(int), np.arange(len(choices))]))
        actual = ctt_log_likelihood(choices, rewards, **params)
        np.testing.assert_allclose(float(actual), expected, rtol=1e-5)

    def test_runs_under_x64(self):
        """Float64 parameters do not break the scan carry's dtype."""
        from jax.experimental import enable_x64

        choices, rewards = _sessions((30,))
        with enable_x64():
            params = ctt_session_params(np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float64))
            value = ctt_log_likelihood(choices, rewards, **params)
        self.assertTrue(np.isfinite(float(value)))

    def test_session_params_bounds(self):
        """The transforms land in the documented ranges, at the documented midpoints."""
        out = ctt_session_params(np.zeros(len(CTT_PARAMS)), beta_max=10.0)
        self.assertAlmostEqual(float(out["learn_rate"]), 0.5, places=6)
        self.assertAlmostEqual(float(out["threshold"]), 0.0, places=6)
        self.assertAlmostEqual(float(out["softmax_inverse_temperature"]), 5.0, places=5)
        extreme = ctt_session_params(np.array([-9.0, 9.0, 9.0, 2.0]))
        self.assertLessEqual(float(extreme["threshold"]), 1.0)
        self.assertGreaterEqual(float(extreme["learn_rate"]), 0.0)
        self.assertEqual(float(extreme["bias_l"]), 2.0)

    def test_unknown_model_is_rejected(self):
        """A misspelt model name fails loudly."""
        with self.assertRaises(ValueError):
            get_model("hattori")


@unittest.skipUnless(HAS_JAX, "requires the 'bayes' extra (jax, numpyro)")
class TestCTTHeldout(unittest.TestCase):
    """CTT runs through the fit and both held-out scorers."""

    @classmethod
    def setUpClass(cls):
        """One short two-subject CTT fit shared by the scoring tests."""
        cls.n_params = len(CTT_PARAMS)
        choices, rewards = _sessions((2, 3, 40))
        cls.samples, cls.diagnostics = fit_heldout_two_level(
            choices, rewards, rng_key=jax.random.PRNGKey(0), num_warmup=50,
            num_samples=30, num_chains=2, model="ctt",
        )

    def test_fixed_hyperprior_is_sized_for_the_model(self):
        """CTT gets a four-parameter hyperprior; Hattori keeps the module constant."""
        self.assertEqual(fixed_hyperprior("ctt")["population_mean"].shape, (self.n_params,))
        self.assertIs(fixed_hyperprior(), FIXED_HYPERPRIOR)

    def test_fit_shapes_and_diagnostics(self):
        """Draws are (chains * samples, subjects, 4) and the diagnostics are finite."""
        self.assertEqual(self.samples["mu_p"].shape, (60, 2, self.n_params))
        self.assertEqual(self.samples["log_sigma"].shape, (60, 2, self.n_params))
        self.assertTrue(np.all(np.isfinite(self.diagnostics["max_r_hat"])))
        self.assertTrue(np.all(np.isfinite(self.diagnostics["min_ess_bulk"])))

    def test_batched_scores(self):
        """Both marginals are finite, negative log likelihoods."""
        choices, rewards = _sessions((3, 40), seed=1)
        out = batched_heldout_scores(
            self.samples, np.array([0, 1, 1]), choices, rewards,
            rng_key=jax.random.PRNGKey(1), model="ctt",
        )
        for name in ("trial_marginal", "session_marginal"):
            self.assertTrue(np.all(np.isfinite(out[name])), name)
            self.assertTrue(np.all(out[name] < 0.0), name)

    def test_importance_sampled_marginal(self):
        """The importance-sampled marginal runs and is finite for a CTT subject."""
        choices, rewards = _sessions((2, 40), seed=2)
        out = importance_sampled_session_marginal(
            self.samples["mu_p"][:, 0], self.samples["log_sigma"][:, 0], choices, rewards,
            rng_key=jax.random.PRNGKey(2), n_proposal=400, chunk=200, model="ctt",
        )
        self.assertTrue(np.all(np.isfinite(out["session_marginal"])))
        self.assertTrue(np.all(np.isfinite(out["session_marginal_se"])))


@unittest.skipUnless(HAS_JAX, "requires the 'bayes' extra (jax, numpyro)")
class TestHattoriScoreUnchanged(unittest.TestCase):
    """The ``model=`` refactor leaves Hattori2019 scores exactly as before."""

    # Computed on origin/main at cf38f45, before ``model=`` existed.
    FROZEN_TRIAL = [-36.22972106933594, -36.192440032958984]
    FROZEN_SESSION = [-31.763639450073242, -31.288909912109375]

    def test_explicit_model_matches_frozen_and_default(self):
        """Default and explicit ``model="hattori2019"`` agree with each other and main."""
        rng = np.random.default_rng(3)
        choices = rng.integers(0, 2, size=(2, 40))
        rewards = (rng.random((2, 40)) < 0.4).astype(float)
        samples = {"mu_p": rng.normal(0, 0.5, (16, 1, 5)),
                   "log_sigma": rng.normal(-1, 0.3, (16, 1, 5))}
        args = (samples, np.zeros(2, int), choices, rewards)
        default = batched_heldout_scores(*args, rng_key=jax.random.PRNGKey(7))
        explicit = batched_heldout_scores(*args, rng_key=jax.random.PRNGKey(7),
                                          model="hattori2019")
        for name in ("trial_marginal", "session_marginal"):
            np.testing.assert_array_equal(default[name], explicit[name])
        np.testing.assert_allclose(explicit["trial_marginal"], self.FROZEN_TRIAL, rtol=1e-6)
        np.testing.assert_allclose(explicit["session_marginal"], self.FROZEN_SESSION, rtol=1e-6)


if __name__ == "__main__":
    unittest.main()
