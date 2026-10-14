"""Findling et al. Weber-imprecision Bayesian-inference model.

This is a small, NumPy implementation of the released particle filter.  The
model is not a :class:`DynamicForagingAgentMLEBase` subclass because its
likelihood marginalizes a stochastic latent belief state for every parameter
candidate.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
from scipy.special import betaln, digamma, logsumexp
from scipy.stats import qmc


@dataclass(frozen=True)
class FindlingWeberFilterResult:
    """Trial predictions and marginal log likelihood for a parameter grid."""

    probability_right: np.ndarray
    log_likelihood: np.ndarray


def findling_weber_parameter_grid() -> np.ndarray:
    """Return the released 1,000-candidate temperature/Weber-slope grid.

    The author's Python-2 pickle is the first 1,000 nonzero points of the
    standard, unscrambled two-dimensional Sobol sequence.
    """

    points = qmc.Sobol(d=2, scramble=False).random_base2(10)
    return np.asarray(points[1:1001], dtype=float)


def _symmetric_beta_kl(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    """Return half the sum of both directed KL divergences for Beta pairs."""

    def directed(first: np.ndarray, second: np.ndarray) -> np.ndarray:
        a, b = first[..., 0], first[..., 1]
        c, d = second[..., 0], second[..., 1]
        return (
            betaln(c, d)
            - betaln(a, b)
            + (a - c) * digamma(a)
            + (b - d) * digamma(b)
            + (c - a + d - b) * digamma(a + b)
        )

    return 0.5 * (directed(left, right) + directed(right, left))


def _stratified_resample(weights: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Vectorized equivalent of the release's stratified particle resampler."""

    n_particles = weights.shape[1]
    cumulative = np.cumsum(weights, axis=1)
    positions = (rng.random((len(weights), 1)) + np.arange(n_particles, dtype=float)) / n_particles
    indices = np.sum(positions[:, :, None] > cumulative[:, None, :], axis=2)
    return np.minimum(indices, n_particles - 1)


def filter_findling_weber_session(
    choices: Sequence[int] | np.ndarray,
    rewards: Sequence[int] | np.ndarray,
    parameters: np.ndarray,
    *,
    n_particles: int = 2,
    seed: int | None = 0,
) -> FindlingWeberFilterResult:
    """Filter one session for every ``[temperature, Weber slope]`` candidate."""

    choices_array = np.asarray(choices, dtype=int).reshape(-1)
    rewards_array = np.asarray(rewards, dtype=int).reshape(-1)
    parameters = np.asarray(parameters, dtype=float)
    if choices_array.shape != rewards_array.shape:
        raise ValueError("choices and rewards must have identical shapes.")
    if np.any((choices_array != 0) & (choices_array != 1)):
        raise ValueError("choices must be binary 0/1.")
    if np.any((rewards_array != 0) & (rewards_array != 1)):
        raise ValueError("rewards must be binary 0/1.")
    if parameters.ndim != 2 or parameters.shape[1] != 2:
        raise ValueError("parameters must have shape (n_candidates, 2).")
    if np.any(parameters[:, 0] <= 0.0) or np.any(parameters[:, 1] < 0.0):
        raise ValueError("temperature must be positive and Weber slope non-negative.")
    if n_particles <= 0:
        raise ValueError("n_particles must be positive.")

    rng = np.random.default_rng(seed)
    n_parameters = len(parameters)
    particles = np.ones((n_parameters, n_particles, 2), dtype=float)
    ancestors = np.broadcast_to(
        np.arange(n_particles, dtype=int), (n_parameters, n_particles)
    ).copy()
    probability_right = np.empty((n_parameters, len(choices_array)), dtype=float)
    log_likelihood = np.zeros(n_parameters, dtype=float)

    for trial_index, (choice, reward) in enumerate(zip(choices_array, rewards_array)):
        if trial_index > 0:
            previous = np.take_along_axis(particles, ancestors[:, :, None], axis=1)
            updated = previous.copy()
            previous_choice = choices_array[trial_index - 1]
            previous_reward = rewards_array[trial_index - 1]
            updated[..., 0] += float(previous_choice != previous_reward)
            updated[..., 1] += float(previous_choice == previous_reward)

            distance = _symmetric_beta_kl(updated, previous)
            noise_ceiling = distance * parameters[:, 1, None]
            mean = updated[..., 0] / np.sum(updated, axis=-1)
            variance = (
                updated[..., 0]
                * updated[..., 1]
                / (np.sum(updated, axis=-1) ** 2 * (np.sum(updated, axis=-1) + 1.0))
            )
            variance += rng.random(variance.shape) * noise_ceiling
            alpha = ((1.0 - mean) / variance - 1.0 / mean) * mean**2
            beta = alpha * (1.0 / mean - 1.0)
            particles[..., 0] = np.maximum(alpha, 1.0)
            particles[..., 1] = np.maximum(beta, 1.0)

        probability_left_rewarding = particles[..., 0] / np.sum(particles, axis=-1)
        probability_right_rewarding = 1.0 - probability_left_rewarding
        right_logit = (
            np.log(probability_right_rewarding) - np.log(probability_left_rewarding)
        ) / parameters[:, 0, None]
        particle_log_probability_right = -np.logaddexp(0.0, -right_logit)
        particle_log_probability_left = -np.logaddexp(0.0, right_logit)
        particle_probability_right = np.exp(particle_log_probability_right)
        probability_right[:, trial_index] = particle_probability_right.mean(axis=1)
        particle_log_choice_probability = (
            particle_log_probability_right if choice == 1 else particle_log_probability_left
        )
        normalizer = logsumexp(particle_log_choice_probability, axis=1, keepdims=True)
        log_likelihood += normalizer[:, 0] - np.log(n_particles)
        normalized_weights = np.exp(particle_log_choice_probability - normalizer)
        ancestors = _stratified_resample(normalized_weights, rng)

    return FindlingWeberFilterResult(
        probability_right=probability_right,
        log_likelihood=log_likelihood,
    )


def fit_findling_weber_map(
    choice_sessions: Sequence[Sequence[int] | np.ndarray],
    reward_sessions: Sequence[Sequence[int] | np.ndarray],
    *,
    parameters: np.ndarray | None = None,
    n_particles: int = 2,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Select the release-grid MAP candidate from independent real sessions."""

    if len(choice_sessions) != len(reward_sessions) or not choice_sessions:
        raise ValueError("choice_sessions and reward_sessions must align and be non-empty.")
    grid = findling_weber_parameter_grid() if parameters is None else np.asarray(parameters)
    log_likelihood = np.zeros(len(grid), dtype=float)
    seed_sequence = np.random.SeedSequence(seed).spawn(len(choice_sessions))
    for choices, rewards, session_seed in zip(choice_sessions, reward_sessions, seed_sequence):
        result = filter_findling_weber_session(
            choices,
            rewards,
            grid,
            n_particles=n_particles,
            seed=int(session_seed.generate_state(1)[0]),
        )
        log_likelihood += result.log_likelihood
    map_index = int(np.argmax(log_likelihood))
    return np.asarray(grid[map_index], dtype=float), log_likelihood, map_index
