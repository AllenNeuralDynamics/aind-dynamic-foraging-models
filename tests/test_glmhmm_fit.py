"""EM fits: parameter recovery for both emissions, monotonicity, K=1, restarts, refits."""

import unittest

import numpy as np

from tests._glmhmm_deps import DEPS_PRESENT, assert_deps_present

assert_deps_present()
if DEPS_PRESENT:
    import jax
    import jax.numpy as jnp
    from jax.experimental import enable_x64

    from aind_dynamic_foraging_models.glm_hmm import (
        GLMHMM,
        SUBJECT_PRIOR,
        Prior,
        fit_em,
        fit_with_restarts,
        make_buckets,
        match_by_weights,
    )
    from aind_dynamic_foraging_models.glm_hmm.design import build_inputs
    from aind_dynamic_foraging_models.glm_hmm.newton import fit_logistic

    from tests._glmhmm_sim import simulate

N_LAGS = 2
TRANSITIONS = np.array([[0.97, 0.03], [0.05, 0.95]])
INITIAL = np.array([0.6, 0.4])
# Side weights: RewC_1, RewC_2, UnrC_1, UnrC_2, bias; engagement: Ignored_1..2, Rewarded_1..2, bias.
HURDLE = {
    "side": np.array([[1.2, 0.5, -0.6, -0.2, 1.5], [0.4, 0.2, 0.1, 0.0, -1.5]]),
    "engage": np.array([[-1.0, 0.0, 0.5, 0.0, 3.0], [1.5, 0.5, -0.5, 0.0, -0.5]]),
}
REFERENCE = {
    "weights": np.array([
        [[-1.0, 0.0, 0.5, 0.0, -1.0, 0.0, 0.5, 0.0, 1.5],
         [1.0, 0.5, -0.5, 0.0, -1.0, 0.0, 0.5, 0.0, 3.0]],
        [[0.5, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.5],
         [0.5, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, -1.0]],
    ])
}


def _weights_of(params, emission):
    """Fitted emission weights as one array per state, for matching and comparison."""
    if emission == "hurdle":
        return np.concatenate(
            [np.asarray(params.emissions.side_weights),
             np.asarray(params.emissions.engage_weights)],
            axis=1,
        )
    return np.asarray(params.emissions.weights).reshape(2, -1)


@unittest.skipUnless(DEPS_PRESENT, "needs the glmhmm extra")
class TestRecovery(unittest.TestCase):
    """Simulate from known parameters, refit from noisy starts, compare."""

    def _check(self, emission, true_weights, weight_atol=None):
        """Simulate, refit K=2 from noisy starts, and compare what is identifiable.

        Checks: EM is monotone and converges; the fit's log joint is at least that of the
        true parameters (the MLE property); transitions are within 0.02; class probabilities
        of each trial under the state that generated it are within 0.01 on average (0.05 at
        the 99th percentile, since rare input patterns are weakly determined). Raw weights are
        compared only when ``weight_atol`` is given: a weight on a regressor that is almost
        never non-zero in a state (e.g. an ignore two trials back, in an engaged state) is
        barely identified, while the probabilities it produces are.
        """
        sessions, paths = simulate(emission, true_weights, TRANSITIONS, INITIAL, N_LAGS,
                                   n_sessions=120, n_trials=400, seed=11)
        buckets = make_buckets(sessions, N_LAGS)
        model = GLMHMM(2, N_LAGS, emission)
        prior = Prior(sigma=100.0, alpha=1.0)
        best, results = fit_with_restarts(model, buckets, jax.random.PRNGKey(0),
                                          num_restarts=3, prior=prior)
        for result in results:
            self.assertTrue(np.all(np.diff(result.log_joint) > -1e-6), "EM decreased")
        self.assertTrue(best.converged)

        if emission == "hurdle":
            truth = np.concatenate([true_weights["side"], true_weights["engage"]], axis=1)
            init = dict(side_weights=true_weights["side"], engage_weights=true_weights["engage"])
        else:
            truth = true_weights["weights"].reshape(2, -1)
            init = dict(weights=true_weights["weights"])
        true_params, props = model.initialize(
            transition_matrix=TRANSITIONS, initial_probs=INITIAL, **init
        )
        at_truth = fit_em(model, true_params, props, buckets, prior, max_iters=1).log_joint[0]
        self.assertGreaterEqual(best.log_joint[-1], at_truth - 1e-6)

        fitted = _weights_of(best.params, emission)
        permutation, _ = match_by_weights(fitted, truth)
        A = np.asarray(best.params.transitions.transition_matrix)[np.ix_(permutation, permutation)]
        np.testing.assert_allclose(A, TRANSITIONS, atol=0.02)
        inputs = np.vstack([build_inputs(c, r, N_LAGS) for c, r in sessions])
        with enable_x64():
            component = model.emission_component
            p_fit = np.exp(component.class_log_probs(best.params.emissions, inputs))
            p_true = np.exp(component.class_log_probs(true_params.emissions, inputs))
        # Each trial under the state that generated it: a state's probabilities on trials it
        # never produced are not determined by the data.
        states = np.concatenate(paths)
        trials = np.arange(len(states))
        error = np.abs(p_fit[trials, permutation[states]] - p_true[trials, states])
        self.assertLess(error.mean(), 0.01)
        self.assertLess(np.quantile(error, 0.99), 0.05)
        if weight_atol is not None:
            np.testing.assert_allclose(fitted[permutation], truth, atol=weight_atol)

    def test_hurdle(self):
        """Hurdle: identifiable quantities, and weights within 0.35 of the truth."""
        self._check("hurdle", HURDLE, weight_atol=0.35)

    def test_reference(self):
        """Reference: identifiable quantities (class probabilities, transitions)."""
        self._check("reference", REFERENCE)


@unittest.skipUnless(DEPS_PRESENT, "needs the glmhmm extra")
class TestSpecialCasesAndRefits(unittest.TestCase):
    """K=1 reduces to plain regressions; restarts and per-subject refits behave."""

    def setUp(self):
        """A modest simulated data set from the hurdle truth."""
        self.sessions, _ = simulate("hurdle", HURDLE, TRANSITIONS, INITIAL, N_LAGS,
                                    n_sessions=20, n_trials=200, seed=5)
        self.buckets = make_buckets(self.sessions, N_LAGS)

    def test_k1_hurdle_is_two_logistic_regressions(self):
        """With one state, the side and engagement halves are ordinary logistic fits."""
        model = GLMHMM(1, N_LAGS, "hurdle")
        params, props = model.initialize()
        result = fit_em(model, params, props, self.buckets, Prior(sigma=10.0, alpha=1.0))
        choices = np.concatenate([c for c, _ in self.sessions])
        inputs = np.vstack([build_inputs(c, r, N_LAGS) for c, r in self.sessions])
        engaged = (choices != 2) * 1.0
        side_x = np.hstack([inputs[:, :4], inputs[:, 8:9]])
        engage_x = np.hstack([inputs[:, 4:8], inputs[:, 8:9]])
        with enable_x64():
            side = fit_logistic(jnp.zeros(5), side_x, (choices == 1) * 1.0, engaged, 0.01, 30)
            engage = fit_logistic(jnp.zeros(5), engage_x, engaged, np.ones(len(choices)),
                                  0.01, 30)
        np.testing.assert_allclose(result.params.emissions.side_weights[0], side, atol=1e-6)
        np.testing.assert_allclose(result.params.emissions.engage_weights[0], engage, atol=1e-6)

    def test_restarts_keep_the_best(self):
        """The returned fit has the highest final log joint of all starts."""
        best, results = fit_with_restarts(GLMHMM(2, N_LAGS, "hurdle"), self.buckets,
                                          jax.random.PRNGKey(3), num_restarts=2, max_iters=20)
        self.assertEqual(best.log_joint[-1], max(r.log_joint[-1] for r in results))

    def test_frozen_initial_and_transitions_are_kept(self):
        """Untrainable pi and A survive EM unchanged; the emissions still move."""
        model = GLMHMM(2, N_LAGS, "hurdle")
        start, props = model.initialize(jax.random.PRNGKey(5), transition_matrix=TRANSITIONS,
                                        initial_probs=INITIAL, noise_scale=0.3)
        frozen = props._replace(
            initial=props.initial._replace(
                probs=props.initial.probs.__class__(trainable=False)),
            transitions=props.transitions._replace(
                transition_matrix=props.transitions.transition_matrix.__class__(trainable=False)),
        )
        result = fit_em(model, start, frozen, self.buckets, max_iters=5)
        np.testing.assert_allclose(result.params.initial.probs, INITIAL)
        np.testing.assert_allclose(result.params.transitions.transition_matrix, TRANSITIONS)
        self.assertFalse(np.allclose(result.params.emissions.side_weights,
                                     start.emissions.side_weights))

    def test_last_log_joint_describes_the_returned_parameters(self):
        """On the iteration-limit path the final entry is the returned fit's own objective."""
        model = GLMHMM(2, N_LAGS, "hurdle")
        start, props = model.initialize(jax.random.PRNGKey(6), noise_scale=0.3)
        for max_iters in (1, 3):
            result = fit_em(model, start, props, self.buckets, max_iters=max_iters, tol=0.0)
            self.assertFalse(result.converged)
            self.assertEqual(len(result.log_joint), max_iters + 1)
            again = fit_em(model, result.params, props, self.buckets, max_iters=0)
            self.assertAlmostEqual(result.log_joint[-1], again.log_joint[0], places=6)

    def test_subject_refit_from_global_parameters(self):
        """A refit under the subject prior starts from given parameters and improves them."""
        model = GLMHMM(2, N_LAGS, "hurdle")
        start, props = model.initialize(
            jax.random.PRNGKey(4), transition_matrix=TRANSITIONS, initial_probs=INITIAL,
            side_weights=HURDLE["side"], engage_weights=HURDLE["engage"], noise_scale=0.3,
        )
        result = fit_em(model, start, props, self.buckets[:1], SUBJECT_PRIOR, max_iters=50)
        self.assertGreater(result.log_joint[-1], result.log_joint[0])
        self.assertTrue(np.all(np.diff(result.log_joint) > -1e-6))


if __name__ == "__main__":
    unittest.main()
