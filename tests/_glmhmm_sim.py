"""Closed-loop simulation of GLM-HMM sessions, for the GLM-HMM tests.

The emission formulas are re-implemented here in NumPy rather than called from the package,
so recovery tests check the package against an independent statement of the model. Rewards
are a coin flip on responded trials: recovery needs history variation, not a real task.
"""

import numpy as np

from aind_dynamic_foraging_models.glm_hmm.design import build_inputs, column_slices


def _sigmoid(x):
    """Logistic function."""
    return 1.0 / (1.0 + np.exp(-x))


def class_probs(emission, weights, x, n_lags, state):
    """p(left), p(right), p(ignore) for one trial's inputs ``x`` under one state.

    Parameters
    ----------
    emission : {"hurdle", "reference"}
        Emission parameterisation.
    weights : dict
        ``side`` and ``engage`` arrays (K, 2L + 1) for hurdle; ``weights`` (K, 2, 4L + 1)
        for reference.
    x : np.ndarray
        One inputs row, shape (4L + 2,).
    n_lags : int
        Trials of history per regressor family.
    state : int
        Latent state.

    Returns
    -------
    np.ndarray
        Shape (3,).
    """
    cols = column_slices(n_lags)
    if emission == "hurdle":
        side_x = np.append(x[cols["side"]], x[cols["bias"]])
        engage_x = np.append(x[cols["engage"]], x[cols["bias"]])
        p_respond = _sigmoid(engage_x @ weights["engage"][state])
        p_right = _sigmoid(side_x @ weights["side"][state])
        return np.array([p_respond * (1 - p_right), p_respond * p_right, 1 - p_respond])
    logits = np.append(weights["weights"][state] @ x[cols["regressors"]], 0.0)
    expl = np.exp(logits - logits.max())
    return expl / expl.sum()


def simulate(emission, weights, transition_matrix, initial_probs, n_lags, n_sessions,
             n_trials, seed, reward_prob=0.4):
    """Simulate sessions trial by trial (history regressors depend on earlier choices).

    Parameters
    ----------
    emission : {"hurdle", "reference"}
        Emission parameterisation.
    weights : dict
        See :func:`class_probs`.
    transition_matrix : np.ndarray
        Shape (K, K).
    initial_probs : np.ndarray
        Shape (K,).
    n_lags : int
        Trials of history per regressor family.
    n_sessions : int
        Number of sessions.
    n_trials : int
        Sessions have between ``n_trials // 2`` and ``n_trials`` trials.
    seed : int
        RNG seed.
    reward_prob : float
        Probability that a responded trial is rewarded.

    Returns
    -------
    tuple
        (list of (choices, rewards), list of state paths).
    """
    rng = np.random.default_rng(seed)
    K = len(initial_probs)
    sessions, states = [], []
    for _ in range(n_sessions):
        length = int(rng.integers(n_trials // 2, n_trials + 1))
        choices, rewards, path = [], [], []
        state = rng.choice(K, p=initial_probs)
        for t in range(length):
            if t > 0:
                state = rng.choice(K, p=transition_matrix[state])
            # The current row depends only on the last n_lags trials.
            recent = slice(max(0, t - n_lags), t)
            x = build_inputs(np.array(choices[recent] + [0]), np.array(rewards[recent] + [0]),
                             n_lags)[-1]
            choice = rng.choice(3, p=class_probs(emission, weights, x, n_lags, state))
            choices.append(int(choice))
            rewards.append(int(choice != 2 and rng.random() < reward_prob))
            path.append(state)
        sessions.append((np.array(choices), np.array(rewards)))
        states.append(np.array(path))
    return sessions, states
