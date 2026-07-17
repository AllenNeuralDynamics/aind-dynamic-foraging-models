"""Functions for update latent variables in generative models."""

import numpy as np


def learn_RWlike(choice, reward, q_value_tminus1, forget_rates, learn_rates):
    """Learning function for Rescorla-Wagner-like model.

    Parameters
    ----------
    choice : int
        this choice
    reward : float
        this reward
    q_value_tminus1 : np.ndarray
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
    q_value_t = np.zeros_like(q_value_tminus1)
    K = q_value_tminus1.shape[0]
    q_value_t[choice] = (1 - forget_rates[1]) * q_value_tminus1[choice] + learn_rate * (
        reward - q_value_tminus1[choice]
    )
    # Unchosen: Q(n+1) = (1-forget_rate_unchosen) * Q(n)
    unchosen_idx = [cc for cc in range(K) if cc != choice]
    q_value_t[unchosen_idx] = (1 - forget_rates[0]) * q_value_tminus1[unchosen_idx]
    return q_value_t


def learn_choice_kernel(choice, choice_kernel_tminus1, choice_kernel_step_size):
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
    return choice_kernel_tminus1 + choice_kernel_step_size * (choice_vector - choice_kernel_tminus1)


def learn_loss_counting(choice, reward, just_switched, loss_count_tminus1) -> int:
    """Update loss counting

    Returns the new loss count
    """
    if reward:
        return 0

    # If not reward
    if just_switched:
        return 1
    else:
        return loss_count_tminus1 + 1


def learn_actor_critic(
    choice,
    reward,
    actor_preference_tminus1,
    value_tminus1,
    choice_prob_t,
    learn_rates,
):
    """One-step actor-critic update for a (state-less) two-armed foraging task.

    This implements the classic Sutton & Barto actor-critic with a softmax actor:

    1. Critic (state value V) is updated by the TD error. Because the dynamic
       foraging task is effectively state-less (a single state / bandit), there is
       no bootstrap term, so the TD error reduces to the reward-prediction error::

           delta_t = reward_t - V_{t-1}
           V_t     = V_{t-1} + alpha_critic * delta_t

       Here V acts as a running reward baseline shared across actions.

    2. Actor (per-action preference H) is updated along the softmax policy gradient,
       using the same TD error as the critic's teaching signal::

           H_t(a) = H_{t-1}(a) + alpha_actor * delta_t * (1{a == choice} - pi(a))

       where ``pi`` is the policy (choice probability) that was actually used to
       select the action on this trial.

    Parameters
    ----------
    choice : int
        This trial's choice (action index).
    reward : float
        This trial's reward.
    actor_preference_tminus1 : np.ndarray
        Array of old actor preferences H, one entry per action.
    value_tminus1 : float
        Old critic state value V (scalar).
    choice_prob_t : np.ndarray
        The policy pi (choice probability per action) that was used to make this
        trial's choice. Used for the softmax policy-gradient term.
    learn_rates : list
        Learning rates as [alpha_actor, alpha_critic].

    Returns
    -------
    tuple(np.ndarray, float)
        (new actor preferences H, new critic value V)

    Notes
    -----
    The actor's overall scale (alpha_actor) and the policy's softmax inverse
    temperature (beta) are not jointly identifiable, because both scale the
    preferences inside the softmax. When fitting, it is therefore recommended to
    clamp ``softmax_inverse_temperature`` (e.g. to 1.0) and let ``learn_rate_actor``
    absorb the scale. For that reason, beta is intentionally *not* multiplied into
    the actor gradient here.
    """
    learn_rate_actor, learn_rate_critic = learn_rates[0], learn_rates[1]

    # -- TD error (state-less reward-prediction error) --
    td_error = reward - value_tminus1

    # -- Critic update --
    value_t = value_tminus1 + learn_rate_critic * td_error

    # -- Actor update (softmax policy gradient) --
    K = actor_preference_tminus1.shape[0]
    choice_onehot = np.zeros(K)
    choice_onehot[choice] = 1.0
    actor_preference_t = actor_preference_tminus1 + learn_rate_actor * td_error * (
        choice_onehot - np.asarray(choice_prob_t)
    )

    return actor_preference_t, value_t
