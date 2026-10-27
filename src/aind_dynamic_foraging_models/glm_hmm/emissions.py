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

The M-step sees the data as *compressed rows*. Each regression block (hurdle: side, engagement;
reference: one) depends on the trials only through (design row, outcome, posterior weight), so
trials sharing a row and outcome are merged once per fit (:func:`compress_rows`) and their
posteriors summed every EM iteration (:func:`aggregate`). This is exact. With a few lags of
history the cohort's millions of trials collapse to thousands of rows; with long histories
little merges, and the cost is that of the uncompressed fit. States are fitted one after
another (``lax.map``), so memory does not grow with K.
"""

from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
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


class CompressedRows(NamedTuple):
    """Unique (design row, outcome) pairs of one regression block, and how trials map to them.

    ``X`` (U, D) and ``targets`` (U,) are padded with zero rows up to a size on a fixed grid,
    so fits on different subjects reuse compiled shapes; ``source`` (n,) indexes the flat
    trials the block uses and ``inverse`` (n,) maps each of them to its row.
    """

    X: np.ndarray
    targets: np.ndarray
    source: np.ndarray
    inverse: np.ndarray


def _grid_size(n, minimum=256):
    """Smallest size >= n of the form 2**k * {1, 1.25, 1.5, 1.75}, at least ``minimum``."""
    size = minimum
    while size < n:
        size *= 2
    for fraction in (0.625, 0.75, 0.875):
        if size * fraction >= max(n, minimum):
            return int(size * fraction)
    return size


def compress_rows(X, targets, source):
    """Merge the trials ``source`` that share a design row and an outcome.

    Parameters
    ----------
    X : np.ndarray
        Design rows of all flat trials, shape (N, D).
    targets : np.ndarray
        Outcome of each flat trial, shape (N,).
    source : np.ndarray
        Indices of the trials this block uses.

    Returns
    -------
    CompressedRows
    """
    keyed = np.ascontiguousarray(np.hstack([X[source], targets[source, None]]))
    unique, inverse = np.unique(keyed, axis=0, return_inverse=True)
    size = _grid_size(len(unique))
    X_unique = np.zeros((size, X.shape[1]))
    targets_unique = np.zeros(size)
    X_unique[: len(unique)] = unique[:, :-1]
    targets_unique[: len(unique)] = unique[:, -1]
    return CompressedRows(X_unique, targets_unique, np.asarray(source), inverse.ravel())


def aggregate(gamma, rows):
    """Posterior weight of every compressed row and state: a sum over its trials.

    Parameters
    ----------
    gamma : np.ndarray
        Smoothed posteriors of all flat trials, shape (N, K).
    rows : CompressedRows
        The block's compression.

    Returns
    -------
    np.ndarray
        Shape (U, K).
    """
    used = gamma[rows.source]
    return np.stack(
        [np.bincount(rows.inverse, weights=used[:, k], minlength=len(rows.X))
         for k in range(gamma.shape[1])],
        axis=1,
    )


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

    def prepare_m_step(self, labels, inputs):
        """Compress the side block (engaged trials) and the engagement block (all trials).

        Parameters
        ----------
        labels : np.ndarray
            Labels of the real trials, flattened, shape (N,).
        inputs : np.ndarray
            Their inputs, shape (N, 4L + 2).

        Returns
        -------
        tuple of CompressedRows
            (side, engagement).
        """
        engaged = labels != IGNORE
        side = compress_rows(
            np.asarray(self._side_design(inputs)), (labels == RIGHT) * 1.0, np.flatnonzero(engaged)
        )
        engage = compress_rows(
            np.asarray(self._engage_design(inputs)), engaged * 1.0, np.arange(len(labels))
        )
        return side, engage

    def m_step(self, params, props, batch_stats, m_step_state, prior_sigma=100.0):
        """Newton MAP update of both halves, one state after another.

        Parameters
        ----------
        params : ParamsHurdleEmissions
            Current weights (the Newton starting point).
        props : ParamsHurdleEmissions
            Parameter properties; untrainable blocks are kept.
        batch_stats : tuple
            ``((X, targets, weights), (X, targets, weights))`` for the side and engagement
            blocks: compressed rows (U, D), outcomes (U,) and summed posteriors (U, K).
        m_step_state : None
            Unused.
        prior_sigma : float
            Gaussian prior standard deviation.

        Returns
        -------
        tuple
            (updated ParamsHurdleEmissions, None).
        """
        (X_side, t_side, w_side), (X_engage, t_engage, w_engage) = batch_stats
        precision = 1.0 / prior_sigma**2

        def fit_block(theta, X, targets, weights):
            """Every state's logistic regression on one block."""
            return jax.lax.map(
                lambda args: fit_logistic(args[0], X, targets, args[1], precision,
                                          self.num_newton_steps),
                (theta, weights.T),
            )

        side_weights, engage_weights = params.side_weights, params.engage_weights
        if props.side_weights.trainable:
            side_weights = fit_block(side_weights, X_side, t_side, w_side)
        if props.engage_weights.trainable:
            engage_weights = fit_block(engage_weights, X_engage, t_engage, w_engage)
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

    def prepare_m_step(self, labels, inputs):
        """Compress the single block: every regressor column, every trial.

        Parameters
        ----------
        labels : np.ndarray
            Labels of the real trials, flattened, shape (N,).
        inputs : np.ndarray
            Their inputs, shape (N, 4L + 2).

        Returns
        -------
        tuple of CompressedRows
            One element.
        """
        X = np.asarray(inputs)[:, self.columns["regressors"]]
        return (compress_rows(X, labels * 1.0, np.arange(len(labels))),)

    def m_step(self, params, props, batch_stats, m_step_state, prior_sigma=100.0):
        """Newton MAP update of the multinomial GLM, one state after another.

        Parameters
        ----------
        params : ParamsReferenceEmissions
            Current weights (the Newton starting point).
        props : ParamsReferenceEmissions
            Parameter properties; untrainable weights are kept.
        batch_stats : tuple
            ``((X, labels, weights),)``: compressed rows (U, D), class labels (U,) and
            summed posteriors (U, K).
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
        ((X, labels, weights),) = batch_stats
        labels = labels.astype(jnp.int32)
        new_weights = jax.lax.map(
            lambda args: fit_multinomial(args[0], X, labels, args[1], 1.0 / prior_sigma**2,
                                         self.num_newton_steps),
            (params.weights, weights.T),
        )
        return params._replace(weights=new_weights), None


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
