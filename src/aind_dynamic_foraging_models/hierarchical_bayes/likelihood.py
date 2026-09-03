"""JAX likelihoods for the foraging foragers, for hierarchical Bayesian fitting.

These are independent reimplementations of the trial dynamics in
``generative_model``. That duplication is deliberate: correctness of this module is
established by a parity test asserting it reproduces the numpy foragers' per-trial
choice probabilities, and that test only means something if the two implementations
do not share code. This module must therefore never import from ``generative_model``.

Parameter names and conventions follow ``generative_model`` throughout, **not** the
reference Stan implementation, which uses a retention factor ``aF = 1 -
forget_rate_unchosen`` and the opposite sign for the side bias.
"""

import jax
import jax.numpy as jnp

# Action indices, matching ``aind_behavior_gym.dynamic_foraging.task``.
L, R = 0, 1
N_ACTIONS = 2


def hattori2019_choice_prob(
    choice_history,
    reward_history,
    learn_rate_rew,
    learn_rate_unrew,
    forget_rate_unchosen,
    softmax_inverse_temperature,
    bias_l,
):
    """Per-trial choice probabilities for the Hattori2019 forager.

    Teacher-forced ("closed loop"): the agent is replayed over an observed choice and
    reward history, and the probability it assigned to each action on each trial is
    returned. This is the quantity a likelihood is built from.

    The value of the chosen option moves toward the outcome at a reward-dependent rate;
    the unchosen option decays by ``forget_rate_unchosen``. Action selection is a softmax
    over ``softmax_inverse_temperature * Q``, with ``bias_l`` added to the left action.

    Trailing padding is safe to leave in ``choice_history``: padded trials only affect Q
    values after the real trials, so the returned probabilities for real trials are
    unchanged. Mask the padding when summing the log likelihood instead
    (see :func:`hattori2019_log_likelihood`).

    Parameters
    ----------
    choice_history : array_like of int, shape (n_trials,)
        Observed actions, 0 for left and 1 for right.
    reward_history : array_like of float, shape (n_trials,)
        Observed outcomes; any positive value counts as rewarded.
    learn_rate_rew : float
        Learning rate applied on rewarded trials, in [0, 1].
    learn_rate_unrew : float
        Learning rate applied on unrewarded trials, in [0, 1].
    forget_rate_unchosen : float
        Per-trial decay of the unchosen option's value, in [0, 1]. Zero means no
        forgetting.
    softmax_inverse_temperature : float
        Inverse temperature of the softmax action selection.
    bias_l : float
        Additive bias toward the left action, on the logit scale.

    Returns
    -------
    jnp.ndarray, shape (N_ACTIONS, n_trials)
        Probability assigned to each action on each trial, before that trial's update.
    """
    choice_history = jnp.asarray(choice_history, dtype=jnp.int32)
    reward_history = jnp.asarray(reward_history, dtype=jnp.float32)

    bias_terms = jnp.array([bias_l, 0.0])

    def _step(q_value, trial):
        """Advance one trial: emit the choice probabilities, then update Q values."""
        choice, reward = trial

        # -- Act: softmax over biased, temperature-scaled Q values --
        choice_prob = jax.nn.softmax(softmax_inverse_temperature * q_value + bias_terms)

        # -- Learn: chosen option moves toward the outcome, unchosen decays --
        chosen = jax.nn.one_hot(choice, N_ACTIONS)
        learn_rate = jnp.where(reward > 0, learn_rate_rew, learn_rate_unrew)
        q_chosen = q_value + learn_rate * (reward - q_value)
        q_unchosen = (1.0 - forget_rate_unchosen) * q_value
        q_next = chosen * q_chosen + (1.0 - chosen) * q_unchosen

        return q_next, choice_prob

    _, choice_prob = jax.lax.scan(
        _step,
        jnp.zeros(N_ACTIONS),  # Initial Q values are 0
        (choice_history, reward_history),
    )
    return choice_prob.T


def hattori2019_value_trajectory(
    choice_history,
    reward_history,
    learn_rate_rew,
    learn_rate_unrew,
    forget_rate_unchosen,
    softmax_inverse_temperature,
    bias_l,
):
    """Per-trial Q values for the Hattori2019 forager -- the model's decision variable.

    :func:`hattori2019_choice_prob` computes these inside its scan and throws them away as
    the carry, because a likelihood only needs the probabilities. They are what a decision
    variable analysis wants: the latent quantity the animal is claimed to be tracking, for
    regression against neural activity or for showing the dynamics behind a fit.

    This replays the identical recursion and emits Q instead. It is a separate function so
    that the likelihood evaluated on every leapfrog step of every chain stays untouched --
    the recursion is deterministic, so nothing is lost by recomputing it after the fact.

    Duplicating an update rule invites the two copies to drift apart. What prevents that
    here is a test, not discipline: ``test_value_trajectory_reproduces_choice_prob``
    asserts that pushing this trajectory back through the softmax reproduces
    :func:`hattori2019_choice_prob` exactly, so a change to either rule alone fails CI.

    **This conditions on the session's own choices, by design.** That is correct for asking
    what the animal's internal state was during a session it actually performed, and it is
    exactly what the held-out likelihood must never do -- there, conditioning on the target
    session's choices would be using the targets to fit the latent. The same operation is
    right in one setting and leakage in the other; see ``batched_heldout_log_lik``.

    Parameters
    ----------
    choice_history : array_like of int, shape (n_trials,)
        Observed actions, 0 for left and 1 for right.
    reward_history : array_like of float, shape (n_trials,)
        Observed outcomes; any positive value counts as rewarded.
    learn_rate_rew, learn_rate_unrew, forget_rate_unchosen : float
        Value-update parameters, as in :func:`hattori2019_choice_prob`.
    softmax_inverse_temperature, bias_l : float
        Action-selection parameters. Accepted so that one call carries the full session
        parameter set, and because the decision variable usually wanted is the biased,
        temperature-scaled difference rather than raw Q.

    Returns
    -------
    q_values : jnp.ndarray, shape (N_ACTIONS, n_trials)
        Q value of each action on each trial, **before** that trial's update -- the same
        alignment :func:`hattori2019_choice_prob` uses, so the two index the same trial.
        Initial Q is zero, so ``q_values[:, 0]`` is all zeros.
    decision_variable : jnp.ndarray, shape (n_trials,)
        ``softmax_inverse_temperature * (Q_left - Q_right) + bias_l``: the quantity the
        softmax actually sees, positive meaning left-preferring. This is the scalar to
        regress against neural data, since raw ``Q_left - Q_right`` omits the gain and
        offset the model applies to it.

    Notes
    -----
    Trailing padding is safe to leave in the inputs, exactly as for
    :func:`hattori2019_choice_prob`: padded trials only affect Q *after* the real trials.
    Mask the padding before plotting or regressing.
    """
    choice_history = jnp.asarray(choice_history, dtype=jnp.int32)
    reward_history = jnp.asarray(reward_history, dtype=jnp.float32)

    def _step(q_value, trial):
        """Advance one trial: emit the pre-update Q values, then update them."""
        choice, reward = trial

        # -- Learn: identical to hattori2019_choice_prob's update, deliberately --
        chosen = jax.nn.one_hot(choice, N_ACTIONS)
        learn_rate = jnp.where(reward > 0, learn_rate_rew, learn_rate_unrew)
        q_chosen = q_value + learn_rate * (reward - q_value)
        q_unchosen = (1.0 - forget_rate_unchosen) * q_value
        q_next = chosen * q_chosen + (1.0 - chosen) * q_unchosen

        return q_next, q_value

    _, q_values = jax.lax.scan(
        _step,
        jnp.zeros(N_ACTIONS),  # Initial Q values are 0
        (choice_history, reward_history),
    )
    q_values = q_values.T
    decision_variable = (
        softmax_inverse_temperature * (q_values[0] - q_values[1]) + bias_l
    )
    return q_values, decision_variable


def hattori2019_log_likelihood(
    choice_history,
    reward_history,
    valid_mask=None,
    **params,
):
    """Total log likelihood of an observed choice history under the Hattori2019 forager.

    Parameters
    ----------
    choice_history : array_like of int, shape (n_trials,)
        Observed actions, 0 for left and 1 for right.
    reward_history : array_like of float, shape (n_trials,)
        Observed outcomes.
    valid_mask : array_like of bool, shape (n_trials,), optional
        Trials to include. Use this to drop padding and ignored (no-response) trials.
        Defaults to all trials.
    **params
        Forager parameters, passed through to :func:`hattori2019_choice_prob`.

    Returns
    -------
    jnp.ndarray
        Scalar sum of per-trial log probabilities of the observed actions.
    """
    choice_history = jnp.asarray(choice_history, dtype=jnp.int32)
    choice_prob = hattori2019_choice_prob(choice_history, reward_history, **params)

    prob_observed = choice_prob[choice_history, jnp.arange(choice_history.shape[0])]
    log_prob = jnp.log(prob_observed)

    if valid_mask is None:
        return jnp.sum(log_prob)
    return jnp.sum(jnp.where(jnp.asarray(valid_mask, dtype=bool), log_prob, 0.0))
