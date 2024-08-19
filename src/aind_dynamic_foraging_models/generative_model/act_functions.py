import numpy as np


def act_softmax(
    q_estimation_t=0,
    softmax_temperature=0,
    bias_terms=0,
    choice_softmax_temperature=None,
    choice_kernel=None,
    rng=None,
):
    if choice_kernel is not None:
        q_estimation_t = np.vstack(
            [q_estimation_t, choice_kernel]
        ).transpose()  # the first dimension is the choice and the second is usual valu in position 0 and kernel in position 1
        softmax_temperature = np.array([softmax_temperature, choice_softmax_temperature])[
            np.newaxis, :
        ]
    choice_prob = softmax(q_estimation_t, temperature=softmax_temperature, bias=bias_terms, rng=rng)
    choice = choose_ps(choice_prob, rng=rng)
    return choice, choice_prob

def softmax(x, temperature=1, bias=0, rng=None):
    # Put the bias outside /sigma to make it comparable across different softmax_temperatures.
    rng = rng or np.random.default_rng()
    
    if len(x.shape) == 1:
        X = x / temperature + bias  # Backward compatibility
    else:
        X = np.sum(x / temperature, axis=1) + bias  # Allow more than one kernels (e.g., choice kernel)

    max_temp = np.max(X)

    if max_temp > 700:  # To prevent explosion of EXP
        greedy = np.zeros(len(x))
        greedy[rng.choice(np.where(X == np.max(X))[0])] = 1
        return greedy
    else:  # Normal softmax
        return np.exp(X) / np.sum(np.exp(X))  # Accept np.

def choose_ps(ps, rng=None):
    '''
    "Poisson"-choice process
    '''
    rng = rng or np.random.default_rng()
    
    ps = ps / np.sum(ps)
    return np.max(np.argwhere(np.hstack([-1e-16, np.cumsum(ps)]) < rng.random()))
