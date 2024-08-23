"""Functions for action selection in generative models"""

import numpy as np


def act_softmax(
    q_estimation_t,
    softmax_inverse_temperature=1,
    bias_terms=0,
    choice_kernel_relative_weight=None,
    choice_kernel=None,
    rng=None,
):
    """Given q values and softmax_inverse_temperature, return the choice and choice probability.
    If chocie_kernel is not None, it will sum it into the softmax function

    Parameters
    ----------
    q_estimation_t : list or np.array
        array of q values, by default 0
    softmax_inverse_temperature : int, optional
        inverse temperature of softmax function, by default 0
    bias_terms : int, optional
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
    if choice_kernel is not None:
        q_estimation_t = np.vstack(
            [q_estimation_t, choice_kernel]
        ).transpose()  # the first dimension is the choice and the second is usual
        # valu in position 0 and kernel in position 1
        softmax_inverse_temperature = np.array(
            [
                softmax_inverse_temperature,
                softmax_inverse_temperature * choice_kernel_relative_weight,
            ]
        )[np.newaxis, :]
    choice_prob = softmax(
        q_estimation_t, inverse_temperature=softmax_inverse_temperature, bias=bias_terms, rng=rng
    )
    choice = choose_ps(choice_prob, rng=rng)
    return choice, choice_prob


def softmax(x, inverse_temperature=1, bias=0, rng=None):
    """I switched to inverse_temperature from temperature to make
    the fittings more numerically stable.

    Parameters
    ----------
    x : _type_
        _description_
    inverse_temperature : int, optional
        _description_, by default 1
    bias : int, optional
        _description_, by default 0
    rng : _type_, optional
        _description_, by default None

    Returns
    -------
    _type_
        _description_
    """
    # Put the bias outside /sigma to make it comparable across
    # different softmax_inverse_temperatures.
    rng = rng or np.random.default_rng()

    if len(x.shape) == 1:
        X = x * inverse_temperature + bias  # Backward compatibility
    else:
        X = (
            np.sum(x * inverse_temperature, axis=1) + bias
        )  # Allow more than one kernels (e.g., choice kernel)

    max_temp = np.max(X)

    if max_temp > 700:  # To prevent explosion of EXP
        greedy = np.zeros(len(x))
        greedy[rng.choice(np.where(X == np.max(X))[0])] = 1
        return greedy
    else:  # Normal softmax
        return np.exp(X) / np.sum(np.exp(X))  # Accept np.


def choose_ps(ps, rng=None):
    """
    "Poisson"-choice process
    """
    rng = rng or np.random.default_rng()

    ps = ps / np.sum(ps)
    return np.max(np.argwhere(np.hstack([-1e-16, np.cumsum(ps)]) < rng.random()))
