import numpy as np
import pandas as pd

pd.set_option('display.expand_frame_repr', False)
np.set_printoptions(linewidth=1000)
pd.set_option("display.max_columns", None)

# matplotlib.get_backend()
# matplotlib.use('module://backend_interagg')

def moving_average(a, n=3):
    ret = np.nancumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n

def softmax(x, temperature=1, bias=0):
    # Put the bias outside /sigma to make it comparable across different softmax_temperatures.
    if len(x.shape) == 1:
        X = x / temperature + bias  # Backward compatibility
    else:
        X = np.sum(x / temperature, axis=1) + bias  # Allow more than one kernels (e.g., choice kernel)

    max_temp = np.max(X)

    if max_temp > 700:  # To prevent explosion of EXP
        greedy = np.zeros(len(x))
        greedy[np.random.choice(np.where(X == np.max(X))[0])] = 1
        return greedy
    else:  # Normal softmax
        return np.exp(X) / np.sum(np.exp(X))  # Accept np.

def choose_ps(ps):
    '''
    "Poisson"-choice process
    '''
    ps = ps / np.sum(ps)
    return np.max(np.argwhere(np.hstack([-1e-16, np.cumsum(ps)]) < np.random.rand()))
