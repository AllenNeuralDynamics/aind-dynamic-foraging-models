"""E-step, predictive scoring and posteriors: brute force, padding invariance, dynamax parity."""

import itertools
import unittest

import numpy as np

from tests._glmhmm_deps import DEPS_PRESENT, assert_deps_present

assert_deps_present()
if DEPS_PRESENT:
    import jax.numpy as jnp
    from jax.experimental import enable_x64
    from scipy.special import logsumexp

    from aind_dynamic_foraging_models.glm_hmm import (
        GLMHMM,
        GLOBAL_PRIOR,
        dirichlet_map,
        make_buckets,
        predictive_log_probs,
        state_posteriors,
    )
    from aind_dynamic_foraging_models.glm_hmm.design import build_inputs
    from aind_dynamic_foraging_models.glm_hmm.model import _bucket_e_step

    from tests._glmhmm_sim import simulate

N_LAGS = 2


def _model_and_params(emission, seed=0):
    """A K=2 model with fixed, distinct emission weights."""
    rng = np.random.default_rng(seed)
    model = GLMHMM(2, N_LAGS, emission)
    if emission == "hurdle":
        weights = dict(side_weights=rng.normal(size=(2, 5)), engage_weights=rng.normal(size=(2, 5)))
    else:
        weights = dict(weights=rng.normal(size=(2, 2, 9)))
    with enable_x64():
        params, props = model.initialize(
            initial_probs=np.array([0.7, 0.3]),
            transition_matrix=np.array([[0.9, 0.1], [0.2, 0.8]]),
            **weights,
        )
    return model, params, props


def _brute_force_loglik(model, params, choices, rewards):
    """log p(y_1..y_T) by summing over every state path."""
    x = build_inputs(choices, rewards, N_LAGS)
    with enable_x64():
        class_lp = np.asarray(model.emission_component.class_log_probs(params.emissions, x))
    lp_y = class_lp[np.arange(len(choices)), :, choices]  # (T, K)
    pi = np.log(np.asarray(params.initial.probs))
    A = np.log(np.asarray(params.transitions.transition_matrix))
    totals = []
    for path in itertools.product(range(2), repeat=len(choices)):
        total = pi[path[0]] + lp_y[0, path[0]]
        for t in range(1, len(choices)):
            total += A[path[t - 1], path[t]] + lp_y[t, path[t]]
        totals.append(total)
    return logsumexp(totals)


@unittest.skipUnless(DEPS_PRESENT, "needs the glmhmm extra")
class TestBruteForce(unittest.TestCase):
    """Marginal likelihood and one-step predictive against exhaustive enumeration."""

    def test_marginal_and_predictive_match_enumeration(self):
        """For both emissions: E-step log likelihood and each p(y_t | y_<t)."""
        choices = np.array([1, 2, 0, 0, 1, 2, 1])
        rewards = np.array([1, 0, 0, 1, 0, 0, 1])
        for emission in ("hurdle", "reference"):
            with self.subTest(emission=emission):
                model, params, _ = _model_and_params(emission)
                buckets = make_buckets([(choices, rewards)], N_LAGS, bucket_size=16)
                with enable_x64():
                    loglik = float(_bucket_e_step(model, params, *buckets[0][:2])[0])
                expected = _brute_force_loglik(model, params, choices, rewards)
                self.assertAlmostEqual(loglik, expected, places=8)
                pred = predictive_log_probs(model, params, buckets)[0]
                for t in range(len(choices)):
                    prefix = _brute_force_loglik(model, params, choices[: t + 1], rewards[: t + 1])
                    before = (
                        _brute_force_loglik(model, params, choices[:t], rewards[:t]) if t else 0.0
                    )
                    self.assertAlmostEqual(pred[t, choices[t]], prefix - before, places=8)
                np.testing.assert_allclose(np.exp(pred).sum(1), 1.0, atol=1e-10)


@unittest.skipUnless(DEPS_PRESENT, "needs the glmhmm extra")
class TestRareState(unittest.TestCase):
    """A state with predicted probability below 1e-8 that later trials support.

    dynamax's ``hmm_smoother`` zeroes such a state (``isclose`` to 0 at 1e-8); the two-filter
    smoother does not. Switching costs 1e-9 so the rare state stays above the separate 1e-15
    floor in dynamax's normalisation, which this test is not about.
    """

    def test_smoothed_posteriors_match_enumeration(self):
        """Smoothed posteriors and E-step counts equal brute force when a switch is rare."""
        model = GLMHMM(2, N_LAGS, "hurdle")
        zeros = np.zeros(5)
        with enable_x64():
            # State 0 responds (engagement bias +7), state 1 ignores (-7).
            params, _ = model.initialize(
                initial_probs=np.array([1 - 1e-9, 1e-9]),
                transition_matrix=np.array([[1 - 1e-9, 1e-9], [1e-9, 1 - 1e-9]]),
                side_weights=np.zeros((2, 5)),
                engage_weights=np.stack([np.append(zeros[:4], 7.0), np.append(zeros[:4], -7.0)]),
            )
        choices = np.array([0, 1, 0, 2, 2, 2, 2, 2])
        rewards = np.zeros(8, dtype=int)
        x = build_inputs(choices, rewards, N_LAGS)
        with enable_x64():
            class_lp = np.asarray(model.emission_component.class_log_probs(params.emissions, x))
        lp_y = class_lp[np.arange(len(choices)), :, choices]
        log_pi = np.log(np.asarray(params.initial.probs))
        log_A = np.log(np.asarray(params.transitions.transition_matrix))
        paths = np.array(list(itertools.product(range(2), repeat=len(choices))))
        scores = log_pi[paths[:, 0]] + lp_y[np.arange(len(choices)), paths].sum(1)
        scores += log_A[paths[:, :-1], paths[:, 1:]].sum(1)
        weights = np.exp(scores - logsumexp(scores))
        expected = np.stack([(weights[:, None] * (paths == k)).sum(0) for k in range(2)], 1)
        self.assertGreater(expected[-1, 1], 0.99)  # the data do support the switch
        post = state_posteriors(model, params, make_buckets([(choices, rewards)], N_LAGS))[0]
        np.testing.assert_allclose(post["smoothed"], expected, atol=1e-9)
        with enable_x64():
            bucket = make_buckets([(choices, rewards)], N_LAGS)[0]
            _, _, transitions, _ = _bucket_e_step(model, params, bucket.labels, bucket.inputs)
        expected_switches = (weights * ((paths[:, :-1] == 0) & (paths[:, 1:] == 1)).sum(1)).sum()
        self.assertAlmostEqual(float(np.asarray(transitions)[0, 1]), expected_switches, places=8)


@unittest.skipUnless(DEPS_PRESENT, "needs the glmhmm extra")
class TestPaddingInvariance(unittest.TestCase):
    """Bucket size, trailing padding and filler rows change nothing."""

    def setUp(self):
        """A dozen simulated sessions of different lengths."""
        self.model, self.params, _ = _model_and_params("hurdle", seed=1)
        weights = {
            "side": np.asarray(self.params.emissions.side_weights),
            "engage": np.asarray(self.params.emissions.engage_weights),
        }
        self.sessions, _ = simulate(
            "hurdle", weights, np.array([[0.9, 0.1], [0.2, 0.8]]), np.array([0.7, 0.3]),
            N_LAGS, n_sessions=12, n_trials=60, seed=2,
        )

    def _stats(self, bucket_size):
        """Summed E-step statistics over all buckets."""
        buckets = make_buckets(self.sessions, N_LAGS, bucket_size=bucket_size)
        total = None
        with enable_x64():
            for bucket in buckets:
                ll, ic, tc, _ = _bucket_e_step(self.model, self.params, bucket.labels,
                                               bucket.inputs)
                stats = np.concatenate([[float(ll)], np.asarray(ic), np.asarray(tc).ravel()])
                total = stats if total is None else total + stats
        return total, buckets

    def test_statistics_and_predictive_do_not_depend_on_bucketing(self):
        """Tight buckets (8) and one wide bucket (128) give identical results."""
        tight, tight_buckets = self._stats(8)
        wide, wide_buckets = self._stats(128)
        np.testing.assert_allclose(tight, wide, rtol=1e-10)
        self.assertGreater(len(tight_buckets), 1)
        # Initial counts sum to the number of real sessions: filler rows add nothing.
        self.assertAlmostEqual(tight[1:3].sum(), len(self.sessions), places=8)
        # Transition counts sum to the number of transitions between real trials.
        n_transitions = sum(len(c) - 1 for c, _ in self.sessions)
        self.assertAlmostEqual(tight[3:].sum(), n_transitions, places=6)
        p_tight = predictive_log_probs(self.model, self.params, tight_buckets)
        p_wide = predictive_log_probs(self.model, self.params, wide_buckets)
        for index, (choices, _) in enumerate(self.sessions):
            self.assertEqual(p_tight[index].shape, (len(choices), 3))
            np.testing.assert_allclose(p_tight[index], p_wide[index], atol=1e-10)

    def test_matches_dynamax_filter_on_an_unpadded_session(self):
        """Our predictive sums to the marginal log likelihood dynamax's own filter reports."""
        choices, rewards = self.sessions[0]
        inputs = build_inputs(choices, rewards, N_LAGS)
        with enable_x64():
            post = self.model.filter(self.params, jnp.asarray(choices), jnp.asarray(inputs))
        buckets = make_buckets([(choices, rewards)], N_LAGS, bucket_size=64)
        pred = predictive_log_probs(self.model, self.params, buckets)[0]
        self.assertAlmostEqual(
            pred[np.arange(len(choices)), choices].sum(), float(post.marginal_loglik), places=8
        )

    def test_viterbi_and_posteriors_do_not_depend_on_padding(self):
        """Identity transitions into padding keep the most likely path unchanged."""
        # A one-trial session included: its bucket has no transitions at bucket_size=1.
        sessions = self.sessions + [(np.array([2]), np.array([0]))]
        tight = state_posteriors(self.model, self.params,
                                 make_buckets(sessions, N_LAGS, bucket_size=1))
        wide = state_posteriors(self.model, self.params,
                                make_buckets(sessions, N_LAGS, bucket_size=256))
        self.assertEqual(tight[len(sessions) - 1]["viterbi"].shape, (1,))
        for index in range(len(sessions)):
            np.testing.assert_array_equal(tight[index]["viterbi"], wide[index]["viterbi"])
            np.testing.assert_allclose(tight[index]["smoothed"], wide[index]["smoothed"],
                                       atol=1e-10)
            np.testing.assert_allclose(tight[index]["smoothed"].sum(1), 1.0, atol=1e-10)


@unittest.skipUnless(DEPS_PRESENT, "needs the glmhmm extra")
class TestDirichlet(unittest.TestCase):
    """The NaN-safe Dirichlet MAP update and log prior."""

    def test_map_update(self):
        """alpha = 1 is the MLE, an empty row is uniform, alpha = 2 adds one pseudo-count."""
        with enable_x64():
            counts = jnp.array([[3.0, 1.0, 0.0], [0.0, 0.0, 0.0]])
            mle = np.asarray(dirichlet_map(counts, 1.0))
            smoothed = np.asarray(dirichlet_map(counts, 2.0))
        np.testing.assert_allclose(mle[0], [0.75, 0.25, 0.0])
        np.testing.assert_allclose(mle[1], [1 / 3] * 3)
        np.testing.assert_allclose(smoothed[0], [4 / 7, 2 / 7, 1 / 7])

    def test_log_prior_is_finite_with_zero_transitions_at_alpha_one(self):
        """xlogy keeps 0 * log 0 = 0, where tfp's Dirichlet would give NaN."""
        model, params, _ = _model_and_params("reference")
        with enable_x64():
            params = params._replace(transitions=params.transitions._replace(
                transition_matrix=jnp.array([[1.0, 0.0], [0.3, 0.7]])))
            self.assertTrue(np.isfinite(float(model.log_prior(params, GLOBAL_PRIOR))))


if __name__ == "__main__":
    unittest.main()
