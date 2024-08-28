"""Functions for action selection in generative models"""

import numpy as np


def act_softmax(
    q_estimation_t: np.array,
    softmax_inverse_temperature: float,
    bias_terms: np.array,
    choice_kernel_relative_weight=None,
    choice_kernel=None,
    rng=None,
):
    """Given q values and softmax_inverse_temperature, return the choice and choice probability.
    If chocie_kernel is not None, it will sum it into the softmax function like this
    
    Steps:
    1. Compute adjusted Q values by adding bias terms and choice kernel
       Q' = softmax_inverse_temperature * (Q + choice_kernel_relative_weight * choice_kernel) + bias
    2. Compute choice probabilities by softmax function
       choice_prob ~ exp(Q') / sum(exp(Q'))

    Parameters
    ----------
    q_estimation_t : list or np.array
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
    adjusted_Q = softmax_inverse_temperature * q_estimation_t + bias_terms
    if choice_kernel is not None:
        adjusted_Q += softmax_inverse_temperature * choice_kernel_relative_weight * choice_kernel + np.array([0.1, 0])
        
    # -- Compute choice probabilities --
    choice_prob = softmax(adjusted_Q)
    
    # -- Choose action --
    choice = choose_ps(choice_prob, rng=rng)
    return choice, choice_prob


def act_epsilon_greedy(
    q_estimation_t: np.array,
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
    q_estimation_t : np.array
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
    adjusted_Q = q_estimation_t + bias_terms
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
        argmax_x = np.argmax(rng.permutation(x)) # Randomly choose one of the max values
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
