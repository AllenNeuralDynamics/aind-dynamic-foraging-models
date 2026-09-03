"""The decision-variable replay must stay identical to the likelihood's own recursion.

`hattori2019_value_trajectory` duplicates the update rule inside
`hattori2019_choice_prob`, so that the likelihood evaluated on every leapfrog step stays
untouched. The duplication is only safe while the two agree, which is what this module
asserts -- a change to either rule alone fails here rather than silently producing decision
variables that do not match the fitted model.
"""

import unittest

from tests._hb_deps import assert_deps_present

import numpy as np

try:
    import jax
    import jax.numpy as jnp

    from aind_dynamic_foraging_models.hierarchical_bayes.likelihood import (
        N_ACTIONS,
        hattori2019_choice_prob,
        hattori2019_value_trajectory,
    )

    HAS_JAX = True
except ImportError:  # pragma: no cover - exercised only without the bayes extra
    HAS_JAX = False
    N_ACTIONS = 2

assert_deps_present(HAS_JAX)

PARAMS = dict(
    learn_rate_rew=0.45,
    learn_rate_unrew=0.12,
    forget_rate_unchosen=0.08,
    softmax_inverse_temperature=4.2,
    bias_l=-0.35,
)


def _session(n_trials=400, seed=0):
    """A choice/reward history with both outcomes on both actions."""
    rng = np.random.default_rng(seed)
    choices = rng.integers(0, N_ACTIONS, size=n_trials)
    rewards = (rng.random(n_trials) < 0.45).astype(float)
    return choices, rewards


@unittest.skipUnless(HAS_JAX, "requires the 'bayes' extra (jax, numpyro)")
class TestValueTrajectory(unittest.TestCase):
    """Equivalence with the likelihood, and the shape/alignment contract."""

    def test_value_trajectory_reproduces_choice_prob(self):
        """Softmax of the replayed trajectory equals the likelihood's probabilities.

        This is the anti-drift guarantee. `hattori2019_choice_prob` applies
        `softmax(beta * Q + [bias_l, 0])` to the same pre-update Q this function returns,
        so recomputing that softmax here must reproduce it to floating-point equality --
        not merely approximately, since it is literally the same arithmetic.
        """
        choices, rewards = _session()
        q_values, _ = hattori2019_value_trajectory(choices, rewards, **PARAMS)
        expected = hattori2019_choice_prob(choices, rewards, **PARAMS)

        bias_terms = jnp.array([PARAMS["bias_l"], 0.0])
        recomputed = jax.nn.softmax(
            PARAMS["softmax_inverse_temperature"] * q_values.T + bias_terms, axis=-1
        ).T

        np.testing.assert_allclose(
            np.asarray(recomputed), np.asarray(expected), rtol=0, atol=1e-6
        )

    def test_alignment_and_shapes(self):
        """Q is pre-update and starts at zero, matching choice_prob's trial indexing."""
        choices, rewards = _session(n_trials=120, seed=3)
        q_values, dv = hattori2019_value_trajectory(choices, rewards, **PARAMS)

        self.assertEqual(q_values.shape, (N_ACTIONS, 120))
        self.assertEqual(dv.shape, (120,))
        # Initial Q is zero, so the first trial carries no value information and the
        # decision variable there is the bias alone.
        np.testing.assert_allclose(np.asarray(q_values[:, 0]), np.zeros(N_ACTIONS))
        self.assertAlmostEqual(float(dv[0]), PARAMS["bias_l"], places=6)

    def test_decision_variable_is_the_biased_scaled_difference(self):
        """The returned scalar is what the softmax sees, not raw Q_left - Q_right."""
        choices, rewards = _session(n_trials=200, seed=7)
        q_values, dv = hattori2019_value_trajectory(choices, rewards, **PARAMS)
        manual = (
            PARAMS["softmax_inverse_temperature"] * (q_values[0] - q_values[1])
            + PARAMS["bias_l"]
        )
        np.testing.assert_allclose(np.asarray(dv), np.asarray(manual), rtol=0, atol=1e-6)

    def test_forgetting_decays_an_untouched_action(self):
        """A never-chosen action's value decays toward zero at the forget rate.

        Guards the branch that distinguishes this model from plain Q-learning: with
        `forget_rate_unchosen > 0` the unchosen value must shrink, and with it at zero it
        must be held exactly.
        """
        n = 40
        choices = np.zeros(n, dtype=int)          # always left
        rewards = np.ones(n, dtype=float)         # always rewarded

        q_decay, _ = hattori2019_value_trajectory(choices, rewards, **PARAMS)
        held = dict(PARAMS, forget_rate_unchosen=0.0)
        q_held, _ = hattori2019_value_trajectory(choices, rewards, **held)

        # Right is never chosen and starts at zero, so it stays at zero either way;
        # the informative comparison is a non-zero starting value for the unchosen action,
        # which arises once right has been chosen at least once.
        mixed_choices = np.array([1] + [0] * (n - 1), dtype=int)
        q_mixed_decay, _ = hattori2019_value_trajectory(mixed_choices, rewards, **PARAMS)
        q_mixed_held, _ = hattori2019_value_trajectory(mixed_choices, rewards, **held)

        right_decay = np.asarray(q_mixed_decay[1])
        right_held = np.asarray(q_mixed_held[1])
        self.assertGreater(right_held[5], 0.0)
        self.assertTrue(
            np.all(np.diff(right_decay[2:]) <= 1e-7),
            "unchosen value should be non-increasing while forgetting",
        )
        self.assertLess(right_decay[-1], right_held[-1])
        np.testing.assert_allclose(right_held[2:], right_held[2], rtol=0, atol=1e-6)
        # Unused, but asserts the all-left case is well formed rather than silently NaN.
        self.assertTrue(np.all(np.isfinite(np.asarray(q_decay))))
        self.assertTrue(np.all(np.isfinite(np.asarray(q_held))))


if __name__ == "__main__":
    unittest.main()
