"""Design matrices for the GLM-HMM: trial history as regressors.

Choices are coded as the wrapper codes ``animal_response``: 0 = left, 1 = right, 2 = ignore.
Rewards are 0/1 (``earned_reward``). Every regressor at trial t is built from trials
t-1 ... t-L of the same session, and lags that fall before the session start are 0, so trial 0
is predicted from the bias alone -- the same information an RNN has at a session start.

Lags count trials as the sequence presents them. An ignore is a trial whose side code is 0, so
it contributes nothing to the side regressors but is visible to the engagement regressors.

One ``inputs`` row per trial holds, in order::

    [RewC_1..RewC_L, UnrC_1..UnrC_L,           side lags, signed (+1 right, -1 left, 0 ignore)
     Ignored_1..Ignored_L, Rewarded_1..Rewarded_L,   engagement lags, unsigned
     bias, mask]

The hurdle emission reads the side block (+ bias) and the engagement block (+ bias) with
separate weights; the reference emission reads everything up to the bias with one weight
vector per non-reference class. ``mask`` is 1 on real trials and 0 on padding.
"""

import numpy as np

LEFT, RIGHT, IGNORE = 0, 1, 2
NUM_CLASSES = 3


def _lagged(values, n_lags):
    """Stack ``values`` shifted by 1..n_lags trials, zero-filled before the session start.

    Parameters
    ----------
    values : np.ndarray
        Per-trial values of one session, shape (T,).
    n_lags : int
        Number of lags.

    Returns
    -------
    np.ndarray
        Shape (T, n_lags); column ``l - 1`` holds ``values[t - l]``.
    """
    n_trials = len(values)
    out = np.zeros((n_trials, n_lags))
    for lag in range(1, n_lags + 1):
        out[lag:, lag - 1] = values[: n_trials - lag]
    return out


def default_design(choices, rewards, n_lags):
    """Side and engagement history regressors for one session.

    Parameters
    ----------
    choices : array-like of int
        Per-trial choice, 0 = left, 1 = right, 2 = ignore.
    rewards : array-like
        Per-trial reward, 0/1. Forced to 0 on ignored trials.
    n_lags : int
        Trials of history per regressor family.

    Returns
    -------
    np.ndarray
        Shape (T, 4 * n_lags + 1): the regressor blocks described in the module docstring
        plus the bias column (no mask column; :func:`build_inputs` appends it).
    """
    choices = np.asarray(choices, dtype=int)
    rewards = np.asarray(rewards, dtype=float)
    if not np.isin(choices, (LEFT, RIGHT, IGNORE)).all():
        raise ValueError("choices must be 0 (left), 1 (right) or 2 (ignore)")
    if not np.isin(rewards, (0, 1)).all():
        raise ValueError("rewards must be 0 or 1")
    ignored = choices == IGNORE
    rewards = np.where(ignored, 0.0, rewards)
    side = np.select([choices == RIGHT, choices == LEFT], [1.0, -1.0], 0.0)
    blocks = [
        _lagged(side * rewards, n_lags),  # RewC
        _lagged(side * (1.0 - rewards), n_lags),  # UnrC
        _lagged(ignored.astype(float), n_lags),  # Ignored
        _lagged(rewards, n_lags),  # Rewarded
        np.ones((len(choices), 1)),  # bias
    ]
    return np.hstack(blocks)


# Named designs, so a cross-check can reproduce another definition without touching the
# default. Each maps (choices, rewards, n_lags) -> (T, 4 * n_lags + 1) in the column layout
# above.
DESIGNS = {"default": default_design}


def build_inputs(choices, rewards, n_lags, design="default"):
    """The full per-trial ``inputs`` array for one session, mask column included.

    Parameters
    ----------
    choices, rewards : array-like
        See :func:`default_design`.
    n_lags : int
        Trials of history per regressor family.
    design : str
        Key into :data:`DESIGNS`.

    Returns
    -------
    np.ndarray
        Shape (T, 4 * n_lags + 2), last column all ones.
    """
    if design not in DESIGNS:
        raise KeyError(f"unknown design {design!r}; known: {sorted(DESIGNS)}")
    regressors = DESIGNS[design](choices, rewards, n_lags)
    return np.hstack([regressors, np.ones((len(regressors), 1))])


def column_slices(n_lags):
    """Column ranges of each regressor block inside an ``inputs`` row.

    Parameters
    ----------
    n_lags : int
        Trials of history per regressor family.

    Returns
    -------
    dict
        ``side``, ``engage`` (lag columns only), ``bias`` and ``mask`` column indices, and
        ``regressors`` (every column the reference emission reads, bias included).
    """
    return {
        "side": slice(0, 2 * n_lags),
        "engage": slice(2 * n_lags, 4 * n_lags),
        "bias": 4 * n_lags,
        "regressors": slice(0, 4 * n_lags + 1),
        "mask": 4 * n_lags + 1,
    }


def column_names(n_lags):
    """Readable names for the regressor columns (bias included, mask excluded).

    Parameters
    ----------
    n_lags : int
        Trials of history per regressor family.

    Returns
    -------
    list of str
    """
    names = []
    for family in ("RewC", "UnrC", "Ignored", "Rewarded"):
        names += [f"{family}_{lag}" for lag in range(1, n_lags + 1)]
    return names + ["bias"]
