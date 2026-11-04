"""Agents: one interface for every model the fingerprints compare, and the replay driver.

An agent follows MEvolve's program protocol, bound to its parameters:

    latents = agent.init_latents(n_trials, session)   # at the start of each session
    p = agent.predict(latents)                         # probabilities over [left, right]
    latents = agent.update(latents, choice, reward)    # after the trial

Trial t is predicted from the latent variables after trials 0 .. t-1. ``replay`` feeds an agent a
session's real choices and rewards (teacher forcing). A rollout uses the same three calls, drawing
the choice from ``predict`` and the reward from a task, so the interface serves both modes.
``n_trials`` is the session's length, which agents that preallocate per-trial arrays (the
foraging-models foragers) need; ``session`` holds optional session facts (MEvolve's ``stage``,
``coupled``, ``baited``, ``index``).

Adapters:

- :class:`ProgramAgent`: a MEvolve program (a module or object with ``init_latents(p, session)``,
  ``predict(latents, p)`` returning logits or log-probabilities, ``update(latents, choice, reward,
  p)``) and its parameters;
- :class:`ForagerAgent`: a foraging-models forager, through its own ``act`` and ``learn``;
- :class:`GLMAgent`: the Su2022 logistic regression (rewarded and unrewarded choice at lags 1 to
  ``n_lags``, and a bias), its lag buffer held as the latent variables and zero before each
  session's first trial.
"""

from __future__ import annotations

from typing import Any, Mapping, Optional, Protocol, Sequence

import numpy as np


class Agent(Protocol):
    """A model bound to its parameters, stepped one trial at a time."""

    def init_latents(self, n_trials: int, session: Optional[Mapping[str, Any]] = None) -> Any:
        """Latent variables at the start of a session of ``n_trials`` trials."""

    def predict(self, latents: Any) -> np.ndarray:
        """Choice probabilities ``[p_left, p_right]`` for the next trial."""

    def update(self, latents: Any, choice: int, reward: float) -> Any:
        """Latent variables after a trial with ``choice`` (0 left, 1 right) and ``reward``."""


def replay(
    agent: Agent,
    choices: Sequence,
    rewards: Sequence,
    session: Optional[Mapping[str, Any]] = None,
) -> np.ndarray:
    """Per-trial p(right) of ``agent`` on one session's real choices and rewards.

    ``choices`` are engaged trials only (0 left, 1 right); remove ignored trials first
    (``drop_ignores``), so the previous trial is the previous engaged one.
    """
    c = np.asarray(choices, dtype=np.int64)
    r = np.asarray(rewards, dtype=float)
    if len(c) != len(r):
        raise ValueError(f"{len(c)} choices but {len(r)} rewards")
    p_right = np.empty(len(c))
    latents = agent.init_latents(len(c), session)
    for t in range(len(c)):
        p = np.asarray(agent.predict(latents), dtype=float)
        p_right[t] = p[1] / (p[0] + p[1])  # renormalised over the engaged choices (ADR 0004)
        latents = agent.update(latents, int(c[t]), float(r[t]))
    return p_right


def _softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - np.max(x))
    return e / e.sum()


class ProgramAgent:
    """A MEvolve program and its parameters (``{name: value}``): the bridge from MEvolve's raw
    ``(p, session)`` functions to :class:`Agent`.

    ``predict`` returns the softmax of the program's output, as MEvolve's ``program.replay`` does.
    Session facts the caller does not give are 0, as in MEvolve's ``program.replay``.
    """

    SESSION_FACTS = ("stage", "coupled", "baited", "index")  # mevolve.program.SESSION_FACTS

    def __init__(self, program: Any, params: Mapping[str, float]):
        self.program = program
        self.params = dict(params)

    def init_latents(self, n_trials: int, session: Optional[Mapping[str, Any]] = None):
        facts = {k: 0 for k in self.SESSION_FACTS}
        facts.update(session or {})
        return self.program.init_latents(self.params, facts)

    def predict(self, latents) -> np.ndarray:
        return _softmax(np.asarray(self.program.predict(latents, self.params), dtype=float))

    def update(self, latents, choice: int, reward: float):
        return self.program.update(latents, choice, reward, self.params)


class ForagerAgent:
    """A foraging-models forager (``DynamicForagingAgentMLEBase``), stepped through its own
    ``act`` and ``learn`` as ``perform_closed_loop`` steps it.

    The forager is its own latent state, so one ``ForagerAgent`` runs one session at a time.
    """

    def __init__(self, forager: Any):
        self.forager = forager
        self._p = None

    def init_latents(self, n_trials: int, session: Optional[Mapping[str, Any]] = None):
        self.forager.n_trials = n_trials
        self.forager._reset()
        return self.forager

    def predict(self, latents) -> np.ndarray:
        _, choice_prob = latents.act(None)
        self._p = np.asarray(choice_prob, dtype=float)
        return self._p

    def update(self, latents, choice: int, reward: float):
        f = latents
        f.choice_prob[:, f.trial] = self._p  # some learners read the probability they chose with
        f.choice_history[f.trial] = choice
        f.reward_history[f.trial] = reward
        f.trial += 1
        f.learn(None, choice, reward, None, None)
        return f


class GLMAgent:
    """The Su2022 logistic regression: p(right) = sigmoid(sum_k w_rew[k] RewC[t-1-k] +
    w_unr[k] UnrC[t-1-k] + bias).

    RewC is +1 for a rewarded right choice, -1 for a rewarded left one and 0 otherwise; UnrC the
    same for unrewarded choices (foraging-models ``logistic_regression``'s ``Su2022`` encoding).
    Coefficients are ordered by lag, lag 1 first. The latent variables are the last ``n_lags``
    values of RewC and UnrC, most recent first, zero at the start of each session.
    """

    def __init__(self, coef_rewc: Sequence[float], coef_unrc: Sequence[float], bias: float):
        self.w_rew = np.asarray(coef_rewc, dtype=float)
        self.w_unr = np.asarray(coef_unrc, dtype=float)
        if self.w_rew.shape != self.w_unr.shape:
            raise ValueError("coef_rewc and coef_unrc must have one value per lag")
        self.bias = float(bias)

    @property
    def n_lags(self) -> int:
        return len(self.w_rew)

    def init_latents(self, n_trials: int, session: Optional[Mapping[str, Any]] = None):
        return np.zeros((2, self.n_lags))

    def predict(self, latents) -> np.ndarray:
        z = latents[0] @ self.w_rew + latents[1] @ self.w_unr + self.bias
        p_right = 1.0 / (1.0 + np.exp(-z))
        return np.array([1.0 - p_right, p_right])

    def update(self, latents, choice: int, reward: float):
        side = 2.0 * choice - 1.0
        rewarded = reward > 0
        new = np.empty_like(latents)
        new[:, 1:] = latents[:, :-1]
        new[0, 0] = side if rewarded else 0.0
        new[1, 0] = 0.0 if rewarded else side
        return new
