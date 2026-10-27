"""The Newton M-step against scipy and sklearn optimisers on the same objectives."""

import unittest

import numpy as np
from scipy.optimize import minimize

from tests._glmhmm_deps import DEPS_PRESENT, assert_deps_present

assert_deps_present()
if DEPS_PRESENT:
    import jax.numpy as jnp
    from jax.experimental import enable_x64
    from sklearn.linear_model import LogisticRegression

    from aind_dynamic_foraging_models.glm_hmm.newton import (
        fit_logistic,
        fit_multinomial,
        logistic_objective,
        multinomial_objective,
    )
    from aind_dynamic_foraging_models.logistic_regression import prepare_logistic_design_matrix


def _problem(seed, n=400, d=4):
    """A random design, positive weights, and a true parameter vector."""
    rng = np.random.default_rng(seed)
    X = np.hstack([rng.normal(size=(n, d - 1)), np.ones((n, 1))])
    weights = rng.uniform(0.0, 1.0, n)
    return rng, X, weights


@unittest.skipUnless(DEPS_PRESENT, "needs the glmhmm extra")
class TestLogisticNewton(unittest.TestCase):
    """Weighted logistic regression with a Gaussian prior."""

    def test_matches_scipy(self):
        """Newton reaches the scipy optimum of the same objective."""
        rng, X, weights = _problem(0)
        targets = (rng.random(len(X)) < 1 / (1 + np.exp(-X @ [1.0, -0.5, 0.2, 0.3]))) * 1.0
        precision = 1.0 / 2.0**2
        with enable_x64():
            ours = np.asarray(
                fit_logistic(jnp.zeros(4), X, targets, weights, precision, num_steps=20)
            )

            def neg(theta):
                """Negative objective for scipy."""
                return -float(logistic_objective(jnp.asarray(theta), X, targets, weights,
                                                 precision))

            reference = minimize(neg, np.zeros(4), method="BFGS", options={"gtol": 1e-8}).x
        np.testing.assert_allclose(ours, reference, atol=1e-4)

    def test_never_decreases_from_a_far_start(self):
        """The line search keeps the objective monotone even on near-separable data."""
        rng, X, weights = _problem(1)
        targets = (X[:, 0] > 0) * 1.0
        with enable_x64():
            start = jnp.array([-20.0, 15.0, -10.0, 5.0])
            values = [float(logistic_objective(start, X, targets, weights, 1e-4))]
            theta = start
            for _ in range(10):
                theta = fit_logistic(theta, X, targets, weights, 1e-4, num_steps=1)
                values.append(float(logistic_objective(theta, X, targets, weights, 1e-4)))
        self.assertTrue(np.all(np.diff(values) >= -1e-9), values)

    def test_k1_parity_with_unpenalised_sklearn_on_su2022(self):
        """With a negligible prior and unit weights, Newton equals sklearn's MLE."""
        rng = np.random.default_rng(2)
        choices, rewards = [], []
        for _ in range(3000):  # a simple win-stay agent, so the weights are non-trivial
            prev_choice = choices[-1] if choices else 0
            stay = 0.75 if (rewards and rewards[-1]) else 0.55
            choices.append(prev_choice if rng.random() < stay else 1 - prev_choice)
            rewards.append(int(rng.random() < 0.4))
        design = prepare_logistic_design_matrix(choices, rewards, "Su2022", n_trial_back=3)
        X = np.hstack([design.X.to_numpy(), np.ones((len(design), 1))])
        y = (design.Y.to_numpy().ravel() > 0) * 1.0
        reference = LogisticRegression(penalty=None, solver="lbfgs", tol=1e-10, max_iter=10000)
        reference.fit(X[:, :-1], y)
        expected = np.append(reference.coef_[0], reference.intercept_)
        with enable_x64():
            ours = np.asarray(
                fit_logistic(jnp.zeros(X.shape[1]), X, y, np.ones(len(y)), 1e-12, num_steps=25)
            )
        np.testing.assert_allclose(ours, expected, atol=1e-4)


@unittest.skipUnless(DEPS_PRESENT, "needs the glmhmm extra")
class TestMultinomialNewton(unittest.TestCase):
    """Weighted reference-class multinomial regression with a Gaussian prior."""

    def test_matches_scipy(self):
        """Newton reaches the scipy optimum of the same objective."""
        rng, X, weights = _problem(3)
        true = np.array([[1.0, -0.5, 0.0, 0.5], [-0.5, 1.0, 0.3, 0.2]])
        logits = np.hstack([X @ true.T, np.zeros((len(X), 1))])
        probs = np.exp(logits) / np.exp(logits).sum(1, keepdims=True)
        labels = np.array([rng.choice(3, p=p) for p in probs])
        precision = 1.0 / 3.0**2
        with enable_x64():
            ours = np.asarray(
                fit_multinomial(jnp.zeros((2, 4)), X, labels, weights, precision, num_steps=20)
            )

            def neg(theta):
                """Negative objective for scipy."""
                return -float(multinomial_objective(jnp.asarray(theta), X, labels, weights,
                                                    precision, 2))

            reference = minimize(neg, np.zeros(8), method="BFGS", options={"gtol": 1e-8}).x
        np.testing.assert_allclose(ours.reshape(-1), reference, atol=1e-4)


if __name__ == "__main__":
    unittest.main()
