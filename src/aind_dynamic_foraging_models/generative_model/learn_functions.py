"""Functions for update latent variables in generative models."""

import numpy as np


def learn_RWlike(choice, reward, q_estimation_tminus1, forget_rates, learn_rates):
    """Learning function for Rescorla-Wagner-like model.

    Parameters
    ----------
    choice : int
        this choice
    reward : float
        this reward
    q_estimation_tminus1 : np.ndarray
        array of old q values
    forget_rates : list
        forget rates for [unchosen, chosen] sides
    learn_rates : _type_
        learning rates for [rewarded, unrewarded] sides

    Returns
    -------
    np.ndarray
        array of new q values
    """
    # Reward-dependent step size ('Hattori2019')
    learn_rate_rew, learn_rate_unrew = learn_rates[0], learn_rates[1]
    if reward:
        learn_rate = learn_rate_rew
    else:
        learn_rate = learn_rate_unrew

    # Choice-dependent forgetting rate ('Hattori2019')
    # Chosen:   Q(n+1) = (1- forget_rate_chosen) * Q(n) + step_size * (Reward - Q(n))
    q_estimation_t = np.zeros_like(q_estimation_tminus1)
    K = q_estimation_tminus1.shape[0]
    q_estimation_t[choice] = (1 - forget_rates[1]) * q_estimation_tminus1[choice] + learn_rate * (
        reward - q_estimation_tminus1[choice]
    )
    # Unchosen: Q(n+1) = (1-forget_rate_unchosen) * Q(n)
    unchosen_idx = [cc for cc in range(K) if cc != choice]
    q_estimation_t[unchosen_idx] = (1 - forget_rates[0]) * q_estimation_tminus1[unchosen_idx]
    return q_estimation_t


def learn_choice_kernel(choice, choice_kernel_tminus1, choice_step_size):
    """Learning function for choice kernel.

    Parameters
    ----------
    choice : int
        this choice
    choice_kernel_tminus1 : np.ndarray
        array of old choice kernel values
    choice_kernel_step_size : float
        step size for choice kernel

    Returns
    -------
    np.ndarray
        array of new choice kernel values
    """

    # Choice vector
    choice_vector = np.array([0, 0])
    choice_vector[choice] = 1

    # Update choice kernel (see Model 5 of Wilson and Collins, 2019)
    # Note that if chocie_step_size = 1, degenerates to Bari 2019
    # (choice kernel = the last choice only)
    return choice_kernel_tminus1 + choice_step_size * (choice_vector - choice_kernel_tminus1)
