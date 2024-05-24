import os
import glob
import os
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
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

def negLL_func(fit_values, *args):
    '''
    Compute negative likelihood (Core func)
    '''
    # Arguments interpretation
    banditmodel, fit_choice_history, fit_reward_history, fit_names = args
    bandit_args = {}
    # banditmodel = forager_Hattori2019_ignore
    # fit_values = banditmodel().set_fitparams_random()
    # fit_names = banditmodel().fit_names
    for name, value in zip(fit_names, fit_values):
        bandit_args.update({name:value})

    # Run **PREDICTIVE** simulation
    bandit = banditmodel(**bandit_args)
    bandit.simulate_fit(fit_choice_history, fit_reward_history)
    negLL = bandit.negLL(fit_choice_history, fit_reward_history)
    return negLL

def plot_session_lightweight(fake_data, fitted_choice_history=None, smooth_factor=5, base_color='y'):
    sns.set(style="ticks", context="paper", font_scale=1.4)
    choice_history, reward_history, p_reward = fake_data
    
    # == Fetch data ==
    n_trials = np.shape(choice_history)[1]
    p_reward_fraction = p_reward[1, :] / (np.sum(p_reward, axis=0))

    rewarded_trials = np.any(reward_history, axis=0)
    unrewarded_trials = np.logical_not(rewarded_trials)

    # == Choice trace ==
    fig = plt.figure(figsize=(25, 4), dpi=150)

    ax = fig.add_subplot(111)
    fig.subplots_adjust(left=0.1, right=0.8)

    # Rewarded trials
    ax.plot(np.nonzero(rewarded_trials)[0], 0.5 + (choice_history[0, rewarded_trials] - 0.5) * 1.4, 'k|', color='black', markersize=20, markeredgewidth=2)

    # Unrewarded trials
    ax.plot(np.nonzero(unrewarded_trials)[0], 0.5 + (choice_history[0, unrewarded_trials] - 0.5) * 1.4, '|', color='gray', markersize=10, markeredgewidth=1)

    # Base probability
    ax.plot(np.arange(0, n_trials), p_reward_fraction, color=base_color, label='base rew. prob.', lw=1.5)

    # Smoothed choice history
    y = moving_average(choice_history, smooth_factor)
    x = np.arange(0, len(y)) + int(smooth_factor / 2)
    ax.plot(x, y, linewidth=1., color='black', label='choice (smooth = %g)' % smooth_factor)

    # For each session, if any
    if fitted_choice_history is not None:
        y = moving_average(fitted_choice_history, smooth_factor)
        x = np.arange(0, len(y)) + int(smooth_factor / 2)
        ax.plot(x, y, linewidth=1., color='blue', label='model choice (smooth = %g)' % smooth_factor)
        # ax.plot(np.arange(0, n_trials), fitted_data[1, :], linewidth=1., label='model')
    ax.legend(fontsize=10, loc=1, bbox_to_anchor=(0.985, 0.89), bbox_transform=plt.gcf().transFigure)

    ax.set_yticks([0, 1])
    ax.set_yticklabels(['Left', 'Right'])
    # ax.set_xlim(0,300)

    # fig.tight_layout()
    sns.despine(trim=True)

    return ax
