"""The GLM-HMM: a dynamax ``HMM`` with GLM emissions, plus an EM driver built for many sessions.

Model: a stationary K-state Markov chain per session (initial distribution pi, transition
matrix A), restarted at every session, with the GLM emissions of :mod:`.emissions`.

Why our own EM driver rather than ``HMM.fit_em``. ``fit_em`` takes one batch of equal-length
sequences, runs a fixed number of iterations and has no restarts. Sessions vary in length from
tens to more than a thousand trials, so padding all of them to the longest wastes most of the
compute. Here sessions are bucketed by length (a multiple of ``bucket_size``) and the E-step
runs per bucket. Four details keep the result identical to an unpadded fit:

* padded trials have log likelihood 0 (the emission mask), and padding is trailing only;
* expected transition counts are summed only over transitions *into* real trials (dynamax's
  smoother would also count transitions into padding);
* filler sessions that only round a bucket's batch up to a power of two have no real trials,
  so they contribute nothing to the initial-state counts or the likelihood;
* the emission M-step runs on the real trials only, flattened across buckets and merged
  into unique design rows (exact; see :mod:`.emissions`).

Priors (the :class:`Prior` passed to :func:`fit_em`) are arguments, not model attributes, so
one model instance serves the global and every per-subject fit without re-compiling. The
Dirichlet updates are done here rather than by dynamax's components, whose
``Dirichlet(...).mode()`` is NaN whenever a concentration is <= 1 -- exactly the alpha = 1
maximum-likelihood setting of Ashwood et al.'s global fit.

Everything runs in float64 (``jax.experimental.enable_x64``), on the CPU by default (see
:func:`_compute`).
"""

import contextlib
import os
from functools import partial
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
from dynamax.hidden_markov_model.inference import (
    hmm_filter,
    hmm_posterior_mode,
    hmm_two_filter_smoother,
)
from dynamax.hidden_markov_model.models.abstractions import HMM
from dynamax.hidden_markov_model.models.initial import StandardHMMInitialState
from dynamax.hidden_markov_model.models.transitions import StandardHMMTransitions
from jax.experimental import enable_x64
from jax.scipy.special import gammaln, logsumexp, xlogy

from .design import build_inputs
from .emissions import EMISSIONS, aggregate


DEVICE_ENV = "AIND_GLMHMM_DEVICE"


@contextlib.contextmanager
def _compute():
    """float64, on the CPU unless ``AIND_GLMHMM_DEVICE=gpu``.

    The CPU is the default because of the CUDA libraries pip resolves for jax-cuda12-plugin
    0.6.1. The plugin caps cuBLAS below 12.9 but not cuSOLVER, and cuSOLVER 11.7.5 (CUDA 12.9)
    over cuBLAS 12.8 makes ``jnp.linalg.solve`` on a GPU fail with ``gpusolverDnCreate ...
    cuSolver internal error``. Pinning ``nvidia-cusolver-cu12==11.7.3.90``,
    ``nvidia-cusparse-cu12==12.5.8.93`` and ``nvidia-nvjitlink-cu12==12.8.93`` fixes it
    (verified 2026-10-27 on V100, A100 and L40S, in clean environments).

    With that pin, use ``AIND_GLMHMM_DEVICE=gpu`` for a large pooled fit. At 4.9M trials and
    K = 10 the GPU E-step is 5-6x faster than 8 CPU cores, and the L = 10 M-step 25-130x
    faster. Keep small fits, such as one mouse's refit, on the CPU, where they run about twice
    as fast.
    """
    device = os.environ.get(DEVICE_ENV, "cpu").lower()
    with enable_x64(), jax.default_device(jax.devices(device)[0]):
        yield


class Prior(NamedTuple):
    """Hyperparameters of the MAP fit."""

    sigma: float = 100.0  # standard deviation of the Gaussian prior on every GLM weight
    alpha: float = 1.0  # Dirichlet concentration of pi and of every row of A
    stickiness: float = 0.0  # extra Dirichlet concentration on the diagonal of A


# Ashwood et al. 2022: a near-flat global fit, then per-animal MAP fits.
GLOBAL_PRIOR = Prior(sigma=100.0, alpha=1.0)
SUBJECT_PRIOR = Prior(sigma=2.0, alpha=2.0)


class ParamsGLMHMM(NamedTuple):
    """Parameters of a GLM-HMM, in dynamax's (initial, transitions, emissions) layout."""

    initial: NamedTuple
    transitions: NamedTuple
    emissions: NamedTuple


class SessionBucket(NamedTuple):
    """Sessions padded to one common length.

    ``labels`` (B, T) int, ``inputs`` (B, T, 4L + 2) with the mask last, and ``session_ids``
    (B,) indices into the caller's session list, -1 for filler rows.
    """

    labels: np.ndarray
    inputs: np.ndarray
    session_ids: np.ndarray


class FitResult(NamedTuple):
    """Output of :func:`fit_em`."""

    params: ParamsGLMHMM
    log_joint: np.ndarray  # log p(y, theta) per EM iteration
    converged: bool


class GLMHMM(HMM):
    """K-state HMM with hurdle or reference GLM emissions over {left, right, ignore}."""

    def __init__(self, num_states, n_lags, emission="hurdle", num_newton_steps=5):
        """Build the model components.

        Parameters
        ----------
        num_states : int
            Number of latent states K.
        n_lags : int
            Trials of history per regressor family.
        emission : {"hurdle", "reference"}
            Emission parameterisation.
        num_newton_steps : int
            Newton iterations per emission M-step.
        """
        if emission not in EMISSIONS:
            raise KeyError(f"unknown emission {emission!r}; known: {sorted(EMISSIONS)}")
        super().__init__(
            num_states,
            StandardHMMInitialState(num_states, initial_probs_concentration=1.0),
            StandardHMMTransitions(num_states, concentration=1.0),
            EMISSIONS[emission](num_states, n_lags, num_newton_steps),
        )
        self.n_lags = n_lags
        self.emission = emission

    @property
    def inputs_shape(self):
        """One row of regressors plus bias and mask."""
        return self.emission_component.inputs_shape

    def initialize(
        self, key=None, initial_probs=None, transition_matrix=None, noise_scale=0.0,
        **emission_weights,
    ):
        """Parameters and properties, from given values or ``ssm``-style defaults.

        Parameters
        ----------
        key : jax.random.PRNGKey, optional
            Used for the default transition matrix and the emission noise.
        initial_probs : jnp.ndarray, optional
            Shape (K,); uniform if omitted.
        transition_matrix : jnp.ndarray, optional
            Shape (K, K); if omitted, 0.95 I plus uniform noise of 0.05, row-normalised
            (``ssm``'s default, which Ashwood et al. used).
        noise_scale : float
            Gaussian noise added to the emission weights (to break symmetry across restarts).
        **emission_weights
            Passed to the emission component's ``initialize`` (``side_weights`` and
            ``engage_weights`` for hurdle, ``weights`` for reference).

        Returns
        -------
        tuple
            (ParamsGLMHMM, matching properties).
        """
        key = jax.random.PRNGKey(0) if key is None else key
        key_transitions, key_emissions = jax.random.split(key)
        K = self.num_states
        if initial_probs is None:
            initial_probs = jnp.ones(K) / K
        if transition_matrix is None:
            noise = jax.random.uniform(key_transitions, (K, K))
            transition_matrix = 0.95 * jnp.eye(K) + 0.05 * noise
            transition_matrix /= transition_matrix.sum(axis=1, keepdims=True)
        initial, initial_props = self.initial_component.initialize(
            initial_probs=jnp.asarray(initial_probs)
        )
        transitions, transition_props = self.transition_component.initialize(
            transition_matrix=jnp.asarray(transition_matrix)
        )
        emissions, emission_props = self.emission_component.initialize(
            key_emissions, noise_scale=noise_scale, **emission_weights
        )
        params = ParamsGLMHMM(initial, transitions, emissions)
        props = ParamsGLMHMM(initial_props, transition_props, emission_props)
        return params, props

    def log_prior(self, params, prior=GLOBAL_PRIOR):
        """log p(theta) under ``prior``: Dirichlet on pi and rows of A, Gaussian on weights.

        Parameters
        ----------
        params : ParamsGLMHMM
            Model parameters.
        prior : Prior
            Hyperparameters.

        Returns
        -------
        jnp.ndarray
            Scalar.
        """
        K = self.num_states
        concentration = prior.alpha * jnp.ones((K, K)) + prior.stickiness * jnp.eye(K)
        lp = _dirichlet_log_prob(params.initial.probs, prior.alpha * jnp.ones(K))
        lp += jnp.sum(_dirichlet_log_prob(params.transitions.transition_matrix, concentration))
        return lp + self.emission_component.log_prior(params.emissions, prior.sigma)


def _dirichlet_log_prob(x, concentration):
    """Dirichlet log density along the last axis, finite at x = 0 when concentration = 1."""
    return (
        gammaln(concentration.sum(-1))
        - gammaln(concentration).sum(-1)
        + xlogy(concentration - 1.0, x).sum(-1)
    )


def dirichlet_map(counts, concentration):
    """Mode of Dirichlet(concentration + counts) along the last axis.

    Well defined for concentration >= 1 (alpha = 1 gives the maximum-likelihood update);
    a row with no mass at all falls back to uniform.

    Parameters
    ----------
    counts : jnp.ndarray
        Expected counts, shape (..., K).
    concentration : jnp.ndarray
        Prior concentration, broadcastable to ``counts``.

    Returns
    -------
    jnp.ndarray
        Probabilities, same shape as ``counts``.
    """
    mass = jnp.maximum(counts + concentration - 1.0, 0.0)
    total = mass.sum(-1, keepdims=True)
    uniform = jnp.ones_like(mass) / mass.shape[-1]
    return jnp.where(total > 0, mass / jnp.where(total > 0, total, 1.0), uniform)


def make_buckets(sessions, n_lags, design="default", bucket_size=64):
    """Build per-trial inputs and group sessions into length buckets.

    Parameters
    ----------
    sessions : sequence of (choices, rewards)
        One pair of per-trial arrays per session (see :func:`.design.default_design`).
    n_lags : int
        Trials of history per regressor family.
    design : str
        Key into :data:`.design.DESIGNS`.
    bucket_size : int
        Sessions are padded up to a multiple of this many trials. Each bucket's batch is
        also padded up to a power of two with filler rows, so repeated fits on different
        subjects reuse a small set of compiled shapes.

    Returns
    -------
    list of SessionBucket
    """
    grouped = {}
    for index, (choices, rewards) in enumerate(sessions):
        if len(choices) == 0:
            raise ValueError(f"session {index} has no trials")
        inputs = build_inputs(choices, rewards, n_lags, design)
        padded = -(-len(choices) // bucket_size) * bucket_size
        grouped.setdefault(padded, []).append((index, np.asarray(choices), inputs))
    buckets = []
    for padded in sorted(grouped):
        rows = grouped[padded]
        batch = 1 << (len(rows) - 1).bit_length()
        labels = np.zeros((batch, padded), dtype=np.int32)
        inputs = np.zeros((batch, padded, rows[0][2].shape[1]))
        session_ids = np.full(batch, -1)
        for row, (index, choices, session_inputs) in enumerate(rows):
            labels[row, : len(choices)] = choices
            inputs[row, : len(choices)] = session_inputs
            session_ids[row] = index
        buckets.append(SessionBucket(labels, inputs, session_ids))
    return buckets


def _smoother(initial_probs, transition_matrix, logliks):
    """Forward-backward posteriors of one session.

    dynamax's two-filter smoother (the one its own E-step uses), not ``hmm_smoother``: the
    latter zeroes any state whose predicted probability is within 1e-8 of 0 (``isclose``),
    which drops a state that later trials strongly support -- reachable here, since the
    alpha = 1 global fit can drive transitions that low. dynamax also sums transition
    probabilities, with the same cutoff and into padding too, so that output is unused;
    ``compute_trans_probs=False`` cannot be passed because it is not a static argument of
    dynamax's jitted smoother.

    Known limit: dynamax's filters raise any normalised probability below 1e-15 to 1e-15.
    That changes a one-step predictive probability by at most ~1e-15, but a state held
    below 1e-15 and then strongly supported by later trials (transitions near 1e-12) gets
    a slightly inflated smoothed posterior.
    """
    return hmm_two_filter_smoother(initial_probs, transition_matrix, logliks)


@partial(jax.jit, static_argnums=0)
def _bucket_e_step(model, params, labels, inputs):
    """Sufficient statistics of one bucket under ``params``.

    Returns the summed marginal log likelihood, summed initial-state counts (K,), summed
    transition counts (K, K) over transitions into real trials, and the smoothed posteriors
    (B, T, K), zeroed on padding.
    """

    def one(y, x):
        """Statistics of one session."""
        initial_probs, transition_matrix, logliks = model._inference_args(params, y, x)
        post = _smoother(initial_probs, transition_matrix, logliks)
        mask = x[:, -1]
        # p(z_t = i, z_{t+1} = j | y) = filtered_t(i) A_ij smoothed_{t+1}(j) / predicted_{t+1}(j)
        predicted = post.predicted_probs[1:]
        ratio = jnp.where(predicted > 0, post.smoothed_probs[1:] / predicted, 0.0)
        transitions = transition_matrix * (post.filtered_probs[:-1].T @ (ratio * mask[1:, None]))
        gamma = post.smoothed_probs * mask[:, None]
        return post.marginal_loglik, gamma[0], transitions, gamma

    loglik, initial_counts, transition_counts, gamma = jax.vmap(one)(labels, inputs)
    return loglik.sum(), initial_counts.sum(0), transition_counts.sum(0), gamma


@partial(jax.jit, static_argnums=0)
def _m_step(model, params, props, initial_counts, transition_counts, emission_stats, prior):
    """MAP update of every trainable component from summed statistics."""
    K = model.num_states
    initial, transitions = params.initial, params.transitions
    if props.initial.probs.trainable:
        initial = initial._replace(probs=dirichlet_map(initial_counts, prior.alpha))
    if props.transitions.transition_matrix.trainable:
        concentration = prior.alpha + prior.stickiness * jnp.eye(K)
        transitions = transitions._replace(
            transition_matrix=dirichlet_map(transition_counts, concentration)
        )
    emissions, _ = model.emission_component.m_step(
        params.emissions, props.emissions, emission_stats, None, prior.sigma
    )
    return ParamsGLMHMM(initial, transitions, emissions)


def _float64(params):
    """Parameters as float64 arrays (call inside ``enable_x64``); scans need one dtype."""
    return jax.tree_util.tree_map(lambda x: jnp.asarray(x, dtype=jnp.float64), params)


def _flatten_real_trials(buckets):
    """Labels and inputs of every real trial, plus the indices that select them."""
    labels, inputs, selectors = [], [], []
    for bucket in buckets:
        mask = bucket.inputs[..., -1].reshape(-1) > 0
        labels.append(bucket.labels.reshape(-1)[mask])
        inputs.append(bucket.inputs.reshape(-1, bucket.inputs.shape[-1])[mask])
        selectors.append(np.flatnonzero(mask))
    return np.concatenate(labels), np.concatenate(inputs), selectors


def fit_em(
    model, params, props, buckets, prior=GLOBAL_PRIOR, max_iters=300, tol=1e-6, verbose=False
):
    """Fit by EM from ``params`` until the log joint stops improving.

    Parameters
    ----------
    model : GLMHMM
        The model.
    params, props : ParamsGLMHMM
        Starting parameters and their properties.
    buckets : list of SessionBucket
        Data, from :func:`make_buckets`.
    prior : Prior
        MAP hyperparameters.
    max_iters : int
        Maximum number of M-steps (``0`` only evaluates the log joint at ``params``).
    tol : float
        Stop when the log joint improves by less than ``tol`` nats per real trial, so the
        criterion means the same for one subject's sessions and for a whole cohort.
    verbose : bool
        Print the log joint every 10 iterations.

    Returns
    -------
    FitResult
    """
    with _compute():
        params = _float64(params)
        prior = Prior(*(float(v) for v in prior))
        labels_flat, inputs_flat, selectors = _flatten_real_trials(buckets)
        n_trials = len(labels_flat)
        blocks = model.emission_component.prepare_m_step(labels_flat, inputs_flat)
        block_rows = [(jnp.asarray(b.X), jnp.asarray(b.targets)) for b in blocks]
        history, converged = [], False
        # max_iters M-steps; the log joint is evaluated before each and once more after the
        # last, so history[-1] always describes the returned parameters.
        for iteration in range(max_iters + 1):
            loglik, initial_counts, transition_counts, gammas = 0.0, 0.0, 0.0, []
            for bucket, selector in zip(buckets, selectors):
                ll, ic, tc, gamma = _bucket_e_step(model, params, bucket.labels, bucket.inputs)
                loglik, initial_counts, transition_counts = (
                    loglik + ll, initial_counts + ic, transition_counts + tc
                )
                gammas.append(np.asarray(gamma).reshape(-1, model.num_states)[selector])
            history.append(float(loglik + model.log_prior(params, prior)))
            if verbose and iteration % 10 == 0:
                print(f"EM iteration {iteration}: log joint {history[-1]:.4f}")
            if len(history) > 1 and abs(history[-1] - history[-2]) < tol * n_trials:
                converged = True
                break
            if iteration == max_iters:
                break
            gamma = np.concatenate(gammas)
            emission_stats = tuple(
                (X, targets, jnp.asarray(aggregate(gamma, block)))
                for (X, targets), block in zip(block_rows, blocks)
            )
            params = _m_step(
                model, params, props, initial_counts, transition_counts, emission_stats, prior
            )
        params = jax.tree_util.tree_map(np.asarray, params)
    return FitResult(params, np.asarray(history), converged)


def fit_with_restarts(
    model, buckets, key, num_restarts=1, prior=GLOBAL_PRIOR, noise_scale=0.2, max_iters=300,
    tol=1e-6, **init_kwargs,
):
    """Run :func:`fit_em` from several noisy starts and keep the best log joint.

    Parameters
    ----------
    model : GLMHMM
        The model.
    buckets : list of SessionBucket
        Data.
    key : jax.random.PRNGKey
        Seeds the starts.
    num_restarts : int
        Number of starts.
    prior : Prior
        MAP hyperparameters.
    noise_scale : float
        Emission-weight noise per start (Ashwood et al.: 0.2 around the GLM weights).
    max_iters, tol
        See :func:`fit_em`.
    **init_kwargs
        Passed to :meth:`GLMHMM.initialize` (e.g. starting emission weights).

    Returns
    -------
    tuple
        (best FitResult, list of every FitResult in start order).
    """
    results = []
    for restart_key in jax.random.split(key, num_restarts):
        params, props = model.initialize(restart_key, noise_scale=noise_scale, **init_kwargs)
        results.append(fit_em(model, params, props, buckets, prior, max_iters, tol))
    best = max(results, key=lambda r: r.log_joint[-1])
    return best, results


@partial(jax.jit, static_argnums=0)
def _bucket_predictive(model, params, labels, inputs):
    """One-step predictive log p(y_t = c | y_<t), shape (B, T, 3)."""

    def one(y, x):
        """Predictive of one session."""
        initial_probs, transition_matrix, logliks = model._inference_args(params, y, x)
        post = hmm_filter(initial_probs, transition_matrix, logliks)
        class_log_probs = model.emission_component.class_log_probs(params.emissions, x)
        log_pred = logsumexp(
            jnp.log(post.predicted_probs)[:, :, None] + class_log_probs, axis=1
        )
        return log_pred

    return jax.vmap(one)(labels, inputs)


def predictive_log_probs(model, params, buckets):
    """Per-session one-step-ahead log p(y_t = c | y_1..y_{t-1}) for every class.

    This is the forward filter only, so each trial is predicted from the session's earlier
    trials and nothing later -- the quantity a recurrent network outputs, and the one every
    held-out likelihood in these studies is computed from. Summed over a session at the
    observed labels it equals the session's marginal log likelihood.

    Parameters
    ----------
    model : GLMHMM
        The model.
    params : ParamsGLMHMM
        Fitted parameters.
    buckets : list of SessionBucket
        Data.

    Returns
    -------
    dict
        session index -> array (T, 3), classes left, right, ignore.
    """
    out = {}
    with _compute():
        params = _float64(params)
        for bucket in buckets:
            log_pred = np.asarray(
                _bucket_predictive(model, params, bucket.labels, bucket.inputs)
            )
            lengths = bucket.inputs[..., -1].sum(axis=1).astype(int)
            for row, index in enumerate(bucket.session_ids):
                if index >= 0:
                    out[int(index)] = log_pred[row, : lengths[row]]
    return out


@partial(jax.jit, static_argnums=0)
def _bucket_posteriors(model, params, labels, inputs):
    """Filtered and smoothed posteriors and the Viterbi path of every session in a bucket."""

    def one(y, x):
        """Posteriors of one session."""
        initial_probs, transition_matrix, logliks = model._inference_args(params, y, x)
        post = _smoother(initial_probs, transition_matrix, logliks)
        # Padding is likelihood-neutral for filtering and smoothing, but not for Viterbi: a
        # max-product through padded steps can change the best final real state. Identity
        # transitions into padded trials carry every path through unchanged.
        if x.shape[0] == 1:
            # A one-trial bucket has no transitions; dynamax's Viterbi still indexes one.
            matrices = transition_matrix
        else:
            into_real = x[1:, -1] > 0
            K = transition_matrix.shape[0]
            matrices = jnp.where(into_real[:, None, None], transition_matrix, jnp.eye(K))
        return (
            post.filtered_probs,
            post.smoothed_probs,
            hmm_posterior_mode(initial_probs, matrices, logliks),
        )

    return jax.vmap(one)(labels, inputs)


def state_posteriors(model, params, buckets):
    """Per-session filtered and smoothed state posteriors and most likely state paths.

    Smoothed posteriors use the whole session; use them to describe a session, never to
    score a prediction.

    Parameters
    ----------
    model : GLMHMM
        The model.
    params : ParamsGLMHMM
        Fitted parameters.
    buckets : list of SessionBucket
        Data.

    Returns
    -------
    dict
        session index -> dict with ``filtered`` (T, K), ``smoothed`` (T, K), ``viterbi`` (T,).
    """
    out = {}
    with _compute():
        params = _float64(params)
        for bucket in buckets:
            posteriors = _bucket_posteriors(model, params, bucket.labels, bucket.inputs)
            filtered, smoothed, viterbi = (np.asarray(a) for a in posteriors)
            lengths = bucket.inputs[..., -1].sum(axis=1).astype(int)
            for row, index in enumerate(bucket.session_ids):
                if index >= 0:
                    n = lengths[row]
                    out[int(index)] = {
                        "filtered": filtered[row, :n],
                        "smoothed": smoothed[row, :n],
                        "viterbi": viterbi[row, :n],
                    }
    return out
