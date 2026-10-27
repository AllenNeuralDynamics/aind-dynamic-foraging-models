"""Exact M-step for the GLM emissions: Newton's method on a weighted, L2-penalised GLM.

In EM the emission update for state k maximises

    sum_t gamma_t(k) * log p(y_t | x_t, theta_k)  -  |theta_k|^2 / (2 sigma^2)

over theta_k, where gamma_t(k) are the smoothed state posteriors. For a logistic or a
reference-class multinomial GLM that objective is concave, so Newton steps are exact and
converge in a handful of iterations from the previous EM iterate. dynamax's stock
``CategoricalRegressionHMM`` instead takes 50 Adam steps with no prior, which is neither exact
nor regularised; that is the reason this module exists.

Each step uses the analytic gradient and Hessian, solves for the Newton direction, and
backtracks from step 1 by halving until the Armijo condition holds (at most 8 halvings; if none
holds, the iterate is kept). The full step is almost always accepted, so a step costs one
objective evaluation. The objective therefore never decreases, which keeps EM monotone.

The data enter only through the rows of X, their outcomes and their weights, so callers pass
*unique* rows with summed weights (see ``emissions.prepare_m_step``): exact, and for short
history designs it shrinks millions of trials to a few thousand rows.

Run these in float64 (``jax.experimental.enable_x64``): the Hessians sum millions of trials
against a prior term of 1/sigma^2 ~ 1e-4 and are badly conditioned in float32.
"""

import jax
import jax.numpy as jnp

_MAX_HALVINGS = 8
_ARMIJO = 1e-4


def _newton(objective, grad_hess, theta, num_steps):
    """Maximise a concave ``objective`` by damped Newton steps on a flat parameter vector.

    Parameters
    ----------
    objective : callable
        theta (P,) -> scalar.
    grad_hess : callable
        theta (P,) -> (gradient (P,), Hessian (P, P)), Hessian negative definite.
    theta : jnp.ndarray
        Starting point, shape (P,).
    num_steps : int
        Number of Newton iterations.

    Returns
    -------
    jnp.ndarray
        The final iterate, shape (P,).
    """

    def step(_, theta):
        """One damped Newton step."""
        grad, hess = grad_hess(theta)
        direction = jnp.linalg.solve(-hess, grad)
        current = objective(theta)
        slope = _ARMIJO * (grad @ direction)

        def rejected(state):
            """Keep halving while the Armijo condition fails."""
            alpha, value, halvings = state
            ok = jnp.isfinite(value) & (value >= current + alpha * slope)
            return (~ok) & (halvings < _MAX_HALVINGS)

        def halve(state):
            """Try half the step."""
            alpha, _, halvings = state
            alpha = 0.5 * alpha
            return alpha, objective(theta + alpha * direction), halvings + 1

        alpha, value, _ = jax.lax.while_loop(
            rejected, halve, (1.0, objective(theta + direction), 0)
        )
        accepted = jnp.isfinite(value) & (value >= current + alpha * slope)
        return jnp.where(accepted, theta + alpha * direction, theta)

    return jax.lax.fori_loop(0, num_steps, step, theta)


def logistic_objective(theta, X, targets, weights, prior_precision):
    """Weighted Bernoulli log likelihood minus a zero-mean Gaussian penalty.

    Parameters
    ----------
    theta : jnp.ndarray
        Weights, shape (D,).
    X : jnp.ndarray
        Design, shape (N, D).
    targets : jnp.ndarray
        0/1 outcomes, shape (N,).
    weights : jnp.ndarray
        Non-negative per-trial weights, shape (N,).
    prior_precision : float
        1 / sigma^2.

    Returns
    -------
    jnp.ndarray
        Scalar objective.
    """
    eta = X @ theta
    loglik = targets * eta - jax.nn.softplus(eta)
    return jnp.sum(weights * loglik) - 0.5 * prior_precision * jnp.sum(theta**2)


def _logistic_grad_hess(theta, X, targets, weights, prior_precision):
    """Analytic gradient and Hessian of :func:`logistic_objective`."""
    p = jax.nn.sigmoid(X @ theta)
    grad = X.T @ (weights * (targets - p)) - prior_precision * theta
    curvature = weights * p * (1.0 - p)
    hess = -X.T @ (X * curvature[:, None]) - prior_precision * jnp.eye(theta.shape[0])
    return grad, hess


def fit_logistic(theta0, X, targets, weights, prior_precision, num_steps=5):
    """MAP weights of a weighted logistic regression, by Newton's method.

    Parameters
    ----------
    theta0 : jnp.ndarray
        Starting weights, shape (D,).
    X, targets, weights, prior_precision
        See :func:`logistic_objective`.
    num_steps : int
        Newton iterations.

    Returns
    -------
    jnp.ndarray
        Shape (D,).
    """
    return _newton(
        lambda t: logistic_objective(t, X, targets, weights, prior_precision),
        lambda t: _logistic_grad_hess(t, X, targets, weights, prior_precision),
        theta0,
        num_steps,
    )


def reference_log_probs(theta, X):
    """Class log probabilities of a multinomial GLM whose last class is the reference.

    Parameters
    ----------
    theta : jnp.ndarray
        Weights of the non-reference classes, shape (C - 1, D).
    X : jnp.ndarray
        Design, shape (N, D).

    Returns
    -------
    jnp.ndarray
        Shape (N, C); the reference class has logit 0.
    """
    logits = X @ theta.T
    logits = jnp.concatenate([logits, jnp.zeros(logits.shape[:-1] + (1,))], axis=-1)
    return jax.nn.log_softmax(logits, axis=-1)


def multinomial_objective(theta_flat, X, labels, weights, prior_precision, num_free):
    """Weighted reference-class multinomial log likelihood minus a Gaussian penalty.

    Parameters
    ----------
    theta_flat : jnp.ndarray
        Weights of the ``num_free`` non-reference classes, flattened, shape (num_free * D,).
    X : jnp.ndarray
        Design, shape (N, D).
    labels : jnp.ndarray
        Integer classes in 0..num_free (num_free is the reference), shape (N,).
    weights : jnp.ndarray
        Non-negative per-trial weights, shape (N,).
    prior_precision : float
        1 / sigma^2.
    num_free : int
        Number of non-reference classes (C - 1).

    Returns
    -------
    jnp.ndarray
        Scalar objective.
    """
    theta = theta_flat.reshape(num_free, -1)
    log_probs = reference_log_probs(theta, X)
    loglik = jnp.take_along_axis(log_probs, labels[:, None], axis=-1)[:, 0]
    return jnp.sum(weights * loglik) - 0.5 * prior_precision * jnp.sum(theta_flat**2)


def _multinomial_grad_hess(theta_flat, X, labels, weights, prior_precision, num_free):
    """Analytic gradient and Hessian of :func:`multinomial_objective`."""
    theta = theta_flat.reshape(num_free, -1)
    dim = theta.shape[1]
    probs = jnp.exp(reference_log_probs(theta, X))[:, :num_free]  # (N, C-1)
    onehot = jax.nn.one_hot(labels, num_free + 1)[:, :num_free]
    grad = ((weights[:, None] * (onehot - probs)).T @ X).reshape(-1)
    grad = grad - prior_precision * theta_flat
    # Block (c, c') = -X^T diag(w p_c (delta_cc' - p_c')) X; symmetric, so build c <= c'.
    blocks = [[None] * num_free for _ in range(num_free)]
    for c in range(num_free):
        for c2 in range(c, num_free):
            factor = weights * probs[:, c] * ((c == c2) - probs[:, c2])
            blocks[c][c2] = -X.T @ (X * factor[:, None])
            blocks[c2][c] = blocks[c][c2].T
    hess = jnp.block(blocks) - prior_precision * jnp.eye(num_free * dim)
    return grad, hess


def fit_multinomial(theta0, X, labels, weights, prior_precision, num_steps=5):
    """MAP weights of a weighted reference-class multinomial regression, by Newton's method.

    Parameters
    ----------
    theta0 : jnp.ndarray
        Starting weights of the non-reference classes, shape (C - 1, D).
    X, labels, weights, prior_precision
        See :func:`multinomial_objective`.
    num_steps : int
        Newton iterations.

    Returns
    -------
    jnp.ndarray
        Shape (C - 1, D).
    """
    num_free = theta0.shape[0]
    theta_flat = _newton(
        lambda t: multinomial_objective(t, X, labels, weights, prior_precision, num_free),
        lambda t: _multinomial_grad_hess(t, X, labels, weights, prior_precision, num_free),
        theta0.reshape(-1),
        num_steps,
    )
    return theta_flat.reshape(theta0.shape)
