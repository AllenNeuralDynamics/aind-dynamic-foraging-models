"""Functions for action selection in generative models"""

from typing import Optional

import numpy as np
from aind_behavior_gym.dynamic_foraging.task import L, R
from scipy.stats import norm


def act_softmax(
    q_value_t: np.array,
    softmax_inverse_temperature: float,
    bias_terms: np.array,
    choice_kernel_relative_weight=None,
    choice_kernel=None,
    rng=None,
):
    """Given q values and softmax_inverse_temperature, return the choice and choice probability.

    If chocie_kernel is not None, it will sum it into the softmax function like this

    1. Compute adjusted Q values by adding bias terms and choice kernel

        :math:`Q' = \\beta * (Q + w_{ck} * choice\\_kernel) + bias`

        :math:`\\beta` ~ softmax_inverse_temperature

        :math:`w_{ck}` ~ choice_kernel_relative_weight

    2. Compute choice probabilities by softmax function

        :math:`choice\\_prob = exp(Q'_i) / \\sum_i(exp(Q'_i))`

    Parameters
    ----------
    q_value_t : list or np.array
        array of q values, by default 0
    softmax_inverse_temperature : int, optional
        inverse temperature of softmax function, by default 0
    bias_terms : np.array, optional
        _description_, by default 0
    choice_kernel_relative_weight : _type_, optional
        relative strength of choice kernel relative to Q in decision, by default None.
        If not none, choice kernel will have an inverse temperature of
        softmax_inverse_temperature * choice_kernel_relative_weight
    choice_kernel : _type_, optional
        _description_, by default None
    rng : _type_, optional
        random number generator, by default None

    Returns
    -------
    _type_
        _description_
    """

    # -- Compute adjusted Q value --
    # Note that the bias term is outside the temperature term to make it comparable across
    # different softmax_inverse_temperatures.
    # Also, I switched to inverse_temperature from temperature to make
    # the fittings more numerically stable.
    adjusted_Q = softmax_inverse_temperature * q_value_t + bias_terms
    if choice_kernel is not None:
        adjusted_Q += softmax_inverse_temperature * choice_kernel_relative_weight * choice_kernel

    # -- Compute choice probabilities --
    choice_prob = softmax(adjusted_Q, rng=rng)

    # -- Choose action --
    choice = choose_ps(choice_prob, rng=rng)
    return choice, choice_prob


def act_epsilon_greedy(
    q_value_t: np.array,
    epsilon: float,
    bias_terms: np.array,
    choice_kernel=None,
    choice_kernel_relative_weight=None,
    rng=None,
):
    """Action selection by epsilon-greedy method.

    Steps:
    1. Compute adjusted Q values by adding bias terms and choice kernel
        Q' = Q + bias + choice_kernel_relative_weight * choice_kernel
    2. The espilon-greedy method is quivalent to choice probabilities:
        If Q'_L != Q'_R (for simplicity, we assume only two choices)
            choice_prob [(argmax(Q')] = 1 - epsilon / 2
            choice_prob [(argmin(Q'))] = epsilon / 2
        else
            choice_prob [:] = 0.5

    Parameters
    ----------
    q_value_t : np.array
        Current Q-values
    epsilon : float
        Probability of exploration
    bias_terms : np.array
        Bias terms
    choice_kernel : None or np.array, optional
        If not None, it will be added to Q-values, by default None
    choice_kernel_relative_weight : _type_, optional
        If not None, it controls the relative weight of choice kernel, by default None
    rng : _type_, optional
        _description_, by default None
    """
    rng = rng or np.random.default_rng()

    # -- Compute adjusted Q value --
    adjusted_Q = q_value_t + bias_terms
    if choice_kernel is not None:
        adjusted_Q += choice_kernel_relative_weight * choice_kernel

    # -- Compute choice probabilities --
    if adjusted_Q[0] == adjusted_Q[1]:
        choice_prob = np.array([0.5, 0.5])
    else:
        argmax_Q = np.argmax(adjusted_Q)
        choice_prob = np.array([epsilon / 2, epsilon / 2])
        choice_prob[argmax_Q] = 1 - epsilon / 2

    # -- Choose action --
    choice = choose_ps(choice_prob, rng=rng)
    return choice, choice_prob


def act_loss_counting(
    previous_choice: Optional[int],
    loss_count: int,
    loss_count_threshold_mean: float,
    loss_count_threshold_std: float,
    bias_terms: np.array,
    choice_kernel=None,
    choice_kernel_relative_weight=None,
    rng=None,
):
    """Action selection by loss counting method.

    Parameters
    ----------
    previous_choice : int
        Last choice
    loss_count : int
        Current loss count
    loss_count_threshold_mean : float
        Mean of the loss count threshold
    loss_count_threshold_std : float
        Standard deviation of the loss count threshold
    bias_terms: np.array
        Bias terms loss count
    choice_kernel : None or np.array, optional
        If not None, it will be added to Q-values, by default None
    choice_kernel_relative_weight : _type_, optional
        If not None, it controls the relative weight of choice kernel, by default None
    rng : _type_, optional
    """
    rng = rng or np.random.default_rng()

    # -- Return random if this is the first trial --
    if previous_choice is None:
        choice_prob = np.array([0.5, 0.5])
        return choose_ps(choice_prob, rng=rng), choice_prob

    # -- Compute probability of switching --
    # This cdf trick is equivalent to:
    #   1) sample a threshold from the normal distribution
    #   2) compare the threshold with the loss count
    prob_switch = norm.cdf(
        loss_count,
        loss_count_threshold_mean
        - 1e-10,  # To make sure this is equivalent to ">=" if the threshold is an integer
        loss_count_threshold_std + 1e-16,  # To make sure this cdf trick works for std=0
    )
    choice_prob = np.array([prob_switch, prob_switch])  # Assuming only two choices
    choice_prob[int(previous_choice)] = 1 - prob_switch

    # -- Add choice kernel --
    # For a fair comparison with other models that have choice kernel.
    # However, choice kernel of different families are not directly comparable.
    # Here, I first compute a normalized choice probability for choice kernel alone using softmax
    # with inverse temperature 1.0, compute a bias introduced by the choice kernel, and then add
    # it to the original choice probability.
    if choice_kernel is not None:
        choice_prob_choice_kernel = softmax(choice_kernel, rng=rng)
        bias_L_from_choice_kernel = (
            choice_prob_choice_kernel[L] - 0.5
        ) * choice_kernel_relative_weight  # A biasL term introduced by the choice kernel
        choice_prob[L] += bias_L_from_choice_kernel

    # -- Add global bias --
    # For a fair comparison with other models that have bias terms.
    # However, bias terms of different families are not directly comparable.
    # Here, the bias term is added to the choice probability directly, whereas in other models,
    # the bias term is added to the Q-values.
    choice_prob[L] += bias_terms[L]

    # -- Re-normalize choice probability --
    choice_prob[L] = np.clip(choice_prob[L], 0, 1)
    choice_prob[R] = 1 - choice_prob[L]

    return choose_ps(choice_prob, rng=rng), choice_prob


# --- Helper functions ---


def softmax(x, rng=None):
    """

    Parameters
    ----------
    x : _type_
        _description_
    rng : _type_, optional
        _description_, by default None

    Returns
    -------
    _type_
        _description_
    """
    rng = rng or np.random.default_rng()

    if np.max(x) > 700:  # To prevent explosion of EXP
        argmax_x = np.argmax(rng.permutation(x))  # Randomly choose one of the max values
        greedy = np.zeros(len(x))
        greedy[argmax_x] = 1
        return greedy
    else:  # Normal softmax
        return np.exp(x) / np.sum(np.exp(x))


def choose_ps(ps, rng=None):
    """
    "Poisson"-choice process
    """
    rng = rng or np.random.default_rng()

    ps = ps / np.sum(ps)
    return np.max(np.argwhere(np.hstack([-1e-16, np.cumsum(ps)]) < rng.random()))
