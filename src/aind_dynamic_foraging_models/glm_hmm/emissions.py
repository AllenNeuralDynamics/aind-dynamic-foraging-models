"""GLM emissions over {left, right, ignore}, as dynamax ``HMMEmissions`` components.

Two parameterisations, both exactly normalised over the three outcomes:

``hurdle``
    Per state k, an engagement GLM and a side GLM::

        p(respond | k) = sigmoid(a_k . [engage lags, bias])
        p(right | respond, k) = sigmoid(w_k . [side lags, bias])
        p(ignore) = 1 - p(respond)
        p(right) = p(respond) p(right | respond),  p(left) = p(respond) (1 - p(right | respond))

    The side weights see signed regressors and the engagement weights unsigned ones, and the
    M-step splits into two independent logistic regressions per state.

``reference``
    Per state k, one multinomial GLM over every regressor, ignore as the zero-logit class
    (``ssm``'s ``InputDrivenObservations``, as in Ashwood et al. 2022 and Faeze's fit)::

        logit_left = W_k,left . x,  logit_right = W_k,right . x,  logit_ignore = 0

Both differ from dynamax's ``CategoricalRegressionHMM`` in three ways that matter here: a
Gaussian prior on the weights, an exact Newton M-step (:mod:`.newton`), and a mask. The mask
is the last input column; padded trials get log likelihood 0 for every state, which leaves the
real trials' filtered and smoothed posteriors and the session's marginal likelihood unchanged
as long as the padding is trailing.
"""

from typing import NamedTuple

import jax
import jax.numpy as jnp
from dynamax.hidden_markov_model.models.abstractions import HMMEmissions
from dynamax.parameters import ParameterProperties
from tensorflow_probability.substrates.jax import distributions as tfd

from .design import IGNORE, NUM_CLASSES, RIGHT, column_slices
from .newton import fit_logistic, fit_multinomial, reference_log_probs


class ParamsHurdleEmissions(NamedTuple):
    """Hurdle emission weights; the last column of each block is the bias."""

    side_weights: jnp.ndarray  # (K, 2L + 1)
    engage_weights: jnp.ndarray  # (K, 2L + 1)


class ParamsReferenceEmissions(NamedTuple):
    """Reference emission weights for left and right; ignore is the zero-logit class."""

    weights: jnp.ndarray  # (K, 2, 4L + 1)


class _GLMEmissions(HMMEmissions):
    """Shared plumbing: input layout, masking, Gaussian prior, single-trial distribution."""

    def __init__(self, num_states, n_lags, num_newton_steps=5):
        """Set the state count and input layout.

        Parameters
        ----------
        num_states : int
            Number of latent states K.
        n_lags : int
            Trials of history per regressor family; fixes the input layout.
        num_newton_steps : int
            Newton iterations per M-step.
        """
        super().__init__()
        self.num_states = num_states
        self.n_lags = n_lags
        self.num_newton_steps = num_newton_steps
        self.columns = column_slices(n_lags)

    @property
    def emission_shape(self):
        """Emissions are scalar class labels."""
        return ()

    @property
    def inputs_shape(self):
        """One row of regressors plus bias and mask."""
        return (4 * self.n_lags + 2,)

    def _side_design(self, inputs):
        """Side lags and bias, shape (..., 2L + 1)."""
        cols = self.columns
        return jnp.concatenate(
            [inputs[..., cols["side"]], inputs[..., cols["bias"], None]], axis=-1
        )

    def _engage_design(self, inputs):
        """Engagement lags and bias, shape (..., 2L + 1)."""
        cols = self.columns
        return jnp.concatenate(
            [inputs[..., cols["engage"]], inputs[..., cols["bias"], None]], axis=-1
        )

    def class_log_probs(self, params, inputs):
        """log p(y = c | z = k, x) for every state and class.

        Parameters
        ----------
        params : NamedTuple
            Emission parameters.
        inputs : jnp.ndarray
            Shape (..., 4L + 2).

        Returns
        -------
        jnp.ndarray
            Shape (..., K, 3), classes ordered left, right, ignore.
        """
        raise NotImplementedError

    def distribution(self, params, state, inputs=None):
        """Categorical over {left, right, ignore} for one state and one trial."""
        return tfd.Categorical(logits=self.class_log_probs(params, inputs)[state])

    def _compute_conditional_logliks(self, params, emissions, inputs=None):
        """Per-trial, per-state log likelihood, zero on masked (padded) trials.

        Parameters
        ----------
        params : NamedTuple
            Emission parameters.
        emissions : jnp.ndarray
            Integer labels, shape (T,). Values on padded trials are ignored.
        inputs : jnp.ndarray
            Shape (T, 4L + 2); the last column is the mask.

        Returns
        -------
        jnp.ndarray
            Shape (T, K).
        """
        log_probs = self.class_log_probs(params, inputs)  # (T, K, 3)
        labels = jnp.clip(emissions.astype(jnp.int32), 0, NUM_CLASSES - 1)
        loglik = jnp.take_along_axis(log_probs, labels[:, None, None], axis=-1)[..., 0]
        mask = inputs[:, self.columns["mask"]]
        return jnp.where(mask[:, None] > 0, loglik, 0.0)

    def log_prior(self, params, prior_sigma=100.0):
        """Zero-mean Gaussian log density of every weight.

        Parameters
        ----------
        params : NamedTuple
            Emission parameters.
        prior_sigma : float
            Prior standard deviation.

        Returns
        -------
        jnp.ndarray
            Scalar.
        """
        leaves = jax.tree_util.tree_leaves(params)
        return sum(jnp.sum(tfd.Normal(0.0, prior_sigma).log_prob(w)) for w in leaves)

    def initialize_m_step_state(self, params, props):
        """The Newton M-step keeps no state between iterations."""
        return None


class HurdleGLMEmissions(_GLMEmissions):
    """Hurdle emissions: an engagement GLM and a side GLM per state."""

    def initialize(self, key=None, side_weights=None, engage_weights=None, noise_scale=0.0):
        """Weights from given values (tiled over states if 1-D) plus optional Gaussian noise.

        Parameters
        ----------
        key : jax.random.PRNGKey, optional
            Needed when ``noise_scale > 0``.
        side_weights, engage_weights : jnp.ndarray, optional
            Shape (2L + 1,) or (K, 2L + 1); zeros if omitted.
        noise_scale : float
            Standard deviation of the noise added to every weight.

        Returns
        -------
        tuple
            (ParamsHurdleEmissions, matching properties).
        """
        dim = 2 * self.n_lags + 1
        side = _tile(side_weights, self.num_states, (dim,))
        engage = _tile(engage_weights, self.num_states, (dim,))
        if noise_scale > 0:
            key_side, key_engage = jax.random.split(key)
            side = side + noise_scale * jax.random.normal(key_side, side.shape)
            engage = engage + noise_scale * jax.random.normal(key_engage, engage.shape)
        params = ParamsHurdleEmissions(side_weights=side, engage_weights=engage)
        props = ParamsHurdleEmissions(ParameterProperties(), ParameterProperties())
        return params, props

    def class_log_probs(self, params, inputs):
        """log p(y | z = k, x) for left, right, ignore; see the module docstring."""
        eta_side = self._side_design(inputs) @ params.side_weights.T  # (..., K)
        eta_engage = self._engage_design(inputs) @ params.engage_weights.T
        log_respond = jax.nn.log_sigmoid(eta_engage)
        log_left = log_respond + jax.nn.log_sigmoid(-eta_side)
        log_right = log_respond + jax.nn.log_sigmoid(eta_side)
        log_ignore = jax.nn.log_sigmoid(-eta_engage)
        return jnp.stack([log_left, log_right, log_ignore], axis=-1)

    def m_step(self, params, props, batch_stats, m_step_state, prior_sigma=100.0):
        """Newton MAP update of both halves for every state.

        Parameters
        ----------
        params : ParamsHurdleEmissions
            Current weights (the Newton starting point).
        props : ParamsHurdleEmissions
            Parameter properties; untrainable blocks are kept.
        batch_stats : tuple
            ``(gamma, labels, inputs)`` flattened over real trials: smoothed posteriors
            (N, K), labels (N,), inputs (N, 4L + 2). Rows with mask 0 carry no weight.
        m_step_state : None
            Unused.
        prior_sigma : float
            Gaussian prior standard deviation.

        Returns
        -------
        tuple
            (updated ParamsHurdleEmissions, None).
        """
        gamma, labels, inputs = batch_stats
        mask = inputs[:, self.columns["mask"]]
        precision = 1.0 / prior_sigma**2
        engaged = (labels != IGNORE).astype(inputs.dtype)
        side_weights, engage_weights = params.side_weights, params.engage_weights
        if props.side_weights.trainable:
            X = self._side_design(inputs)
            targets = (labels == RIGHT).astype(inputs.dtype)
            side_weights = jax.vmap(
                lambda theta, g: fit_logistic(
                    theta, X, targets, g * mask * engaged, precision, self.num_newton_steps
                ),
                in_axes=(0, 1),
            )(side_weights, gamma)
        if props.engage_weights.trainable:
            X = self._engage_design(inputs)
            engage_weights = jax.vmap(
                lambda theta, g: fit_logistic(
                    theta, X, engaged, g * mask, precision, self.num_newton_steps
                ),
                in_axes=(0, 1),
            )(engage_weights, gamma)
        return params._replace(side_weights=side_weights, engage_weights=engage_weights), None


class ReferenceGLMEmissions(_GLMEmissions):
    """Reference emissions: one multinomial GLM per state, ignore as the zero-logit class."""

    def initialize(self, key=None, weights=None, noise_scale=0.0):
        """Weights from given values (tiled over states if 2-D) plus optional Gaussian noise.

        Parameters
        ----------
        key : jax.random.PRNGKey, optional
            Needed when ``noise_scale > 0``.
        weights : jnp.ndarray, optional
            Shape (2, 4L + 1) or (K, 2, 4L + 1); zeros if omitted.
        noise_scale : float
            Standard deviation of the noise added to every weight.

        Returns
        -------
        tuple
            (ParamsReferenceEmissions, matching properties).
        """
        shape = (NUM_CLASSES - 1, 4 * self.n_lags + 1)
        weights = _tile(weights, self.num_states, shape)
        if noise_scale > 0:
            weights = weights + noise_scale * jax.random.normal(key, weights.shape)
        params = ParamsReferenceEmissions(weights=weights)
        return params, ParamsReferenceEmissions(ParameterProperties())

    def class_log_probs(self, params, inputs):
        """log p(y | z = k, x) for left, right, ignore (the reference class)."""
        X = inputs[..., self.columns["regressors"]]
        return jax.vmap(lambda w: reference_log_probs(w, X), out_axes=-2)(params.weights)

    def m_step(self, params, props, batch_stats, m_step_state, prior_sigma=100.0):
        """Newton MAP update of the multinomial GLM for every state.

        Parameters
        ----------
        params : ParamsReferenceEmissions
            Current weights (the Newton starting point).
        props : ParamsReferenceEmissions
            Parameter properties; untrainable weights are kept.
        batch_stats : tuple
            ``(gamma, labels, inputs)`` flattened over real trials; see
            :meth:`HurdleGLMEmissions.m_step`.
        m_step_state : None
            Unused.
        prior_sigma : float
            Gaussian prior standard deviation.

        Returns
        -------
        tuple
            (updated ParamsReferenceEmissions, None).
        """
        if not props.weights.trainable:
            return params, None
        gamma, labels, inputs = batch_stats
        mask = inputs[:, self.columns["mask"]]
        X = inputs[:, self.columns["regressors"]]
        labels = labels.astype(jnp.int32)
        weights = jax.vmap(
            lambda theta, g: fit_multinomial(
                theta, X, labels, g * mask, 1.0 / prior_sigma**2, self.num_newton_steps
            ),
            in_axes=(0, 1),
        )(params.weights, gamma)
        return params._replace(weights=weights), None


def _tile(values, num_states, shape):
    """Zeros of ``(num_states, *shape)``, or ``values`` broadcast to it."""
    if values is None:
        return jnp.zeros((num_states,) + shape)
    values = jnp.asarray(values, dtype=float)
    if values.shape == shape:
        return jnp.broadcast_to(values, (num_states,) + shape)
    if values.shape != (num_states,) + shape:
        raise ValueError(f"weights of shape {values.shape}; expected {shape} or "
                         f"{(num_states,) + shape}")
    return values


EMISSIONS = {"hurdle": HurdleGLMEmissions, "reference": ReferenceGLMEmissions}
