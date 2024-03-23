"""
Descriptive analysis for the foraging task

Logistic regression on choice and reward history

Two models are supported:
    1. Su and Cohen, 2022, bioRxiv
        logit (p_R) ~ Rewarded choice + Unrewarded choice + Bias
    2. Hattori 2019 https://www.sciencedirect.com/science/article/pii/S0092867419304465?via%3Dihub
        logit (p_R) ~ Rewarded choice + Unrewarded choice + Choice + Bias
  
Han Hou, Feb 2023
"""

from typing import Union, Literal, List, Tuple

import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV


def prepare_logistic_design_matrix(
    choice: Union[List, np.ndarray],
    reward: Union[List, np.ndarray],
    dependent_variable: List[Literal['reward_choice', 'unreward_choice', 'choice']] = [
        'reward_choice', 'unreward_choice'],
    trials_back: int = 15,
    selected_trial_idx: Union[List, np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Prepare logistic regression design matrix from choice and reward history.

    Args:
        choice (Union[List, np.ndarray]): choice history (0 = left choice, 1 = right choice).
        reward (Union[List, np.ndarray]): reward history (0 = unrewarded, 1 = rewarded).
        dependent_variable (List[Literal['rewarded_choice', 'unrewarded_choice', 'choice']], optional):
            The dependent variables. Defaults to ['rewarded_choice', 'unrewarded_choice'] (Sue and Cohen 2022).
        trials_back (int, optional): Number of trials back into history. Defaults to 15.
        selected_trial_idx (Union[List, np.ndarray], optional):
            If None, use all trials; else, only look at selected trials for fitting, but using the full history.

    Returns:
        Tuple[np.ndarray, np.ndarray]: The design matrix (X) and the dependent variable (Y).
    """

    n_trials = len(choice)
    trials_back = 20
    data = []

    # Encoding data
    RewC, UnrC, C = np.zeros(n_trials), np.zeros(n_trials), np.zeros(n_trials)
    RewC[(choice == 0) & (reward == 1)] = -1   # L rew = -1, R rew = 1, others = 0
    RewC[(choice == 1) & (reward == 1)] = 1
    UnrC[(choice == 0) & (reward == 0)] = -1    # L unrew = -1, R unrew = 1, others = 0
    UnrC[(choice == 1) & (reward == 0)] = 1
    C[choice == 0] = -1
    C[choice == 1] = 1

    # Select trials
    if selected_trial_idx is None:
        trials = range(trials_back, n_trials)
    else:
        trials = np.intersect1d(selected_trial_idx, range(trials_back, n_trials))
        
    for trial in trials:
        data.append(np.hstack([RewC[trial - trials_back : trial],
                            UnrC[trial - trials_back : trial], 
                            C[trial - trials_back : trial]]))
    data = np.array(data)
    
    Y = C[trials]  # Use -1/1 or 0/1?
    
    return data, Y


def prepare_logistic_no_C(choice, reward, trials_back=20, selected_trial_idx=None, **kwargs):
    '''    
    Assuming format:
    choice = np.array([0, 1, 1, 0, ...])  # 0 = L, 1 = R
    reward = np.array([0, 0, 0, 1, ...])  # 0 = Unrew, 1 = Reward
    trials_back: number of trials back into history
    selected_trial_idx = np.array([selected zero-based trial idx]): 
        if None, use all trials; 
        else, only look at selected trials, but using the full history!
        e.g., p (stay at the selected trials | win at the previous trials of the selected trials) 
        therefore, the minimum idx of selected_trials is 1 (the second trial)
    ---
    return: data, Y
    '''
    n_trials = len(choice)
    data = []

    # Encoding data
    RewC, UnrC, C = np.zeros(n_trials), np.zeros(n_trials), np.zeros(n_trials)
    RewC[(choice == 0) & (reward == 1)] = -1   # L rew = -1, R rew = 1, others = 0
    RewC[(choice == 1) & (reward == 1)] = 1
    UnrC[(choice == 0) & (reward == 0)] = -1    # L unrew = -1, R unrew = 1, others = 0
    UnrC[(choice == 1) & (reward == 0)] = 1
    C[choice == 0] = -1
    C[choice == 1] = 1


    # Select trials
    if selected_trial_idx is None:
        trials = range(trials_back, n_trials)
    else:
        trials = np.intersect1d(selected_trial_idx, range(trials_back, n_trials))
        
    for trial in trials:
        data.append(np.hstack([RewC[trial - trials_back : trial],
                            UnrC[trial - trials_back : trial], 
                            ]))
    data = np.array(data)
    
    Y = C[trials]  # Use -1/1 or 0/1?
    
    return data, Y


def logistic_regression(data, Y, solver='liblinear', penalty='l2', C=1, test_size=0.10, **kwargs):
    '''
    Run one logistic regression fit
    (Reward trials + Unreward trials + Choice + bias)
    Han 20230208
    '''
    trials_back = int(data.shape[1] / 3)
    
    # Do training
    # x_train, x_test, y_train, y_test = train_test_split(data, Y, test_size=test_size)
    logistic_reg = LogisticRegression(solver=solver, fit_intercept=True, penalty=penalty, C=C)

    # if sum(Y == 1) == 1 or sum(Y == -1) == 1:
    #     logistic_reg_cv.coef_ = np.zeros((1, data.shape[1]))
    #     logistic_reg_cv.intercept_ = 10 * np.sign(np.median(Y))   # If all left, set bias = 10 and other parameters 0
    #     logistic_reg_cv.C_ = np.nan
    # else:
    logistic_reg.fit(data, Y)

    output = np.concatenate([logistic_reg.coef_[0], logistic_reg.intercept_])
    
    (logistic_reg.b_RewC, 
    logistic_reg.b_UnrC, 
    logistic_reg.b_C, 
    logistic_reg.bias) = decode_betas(output, trials_back)
    
    return output, logistic_reg


def logistic_regression_CV(data, Y, Cs=10, cv=10, solver='liblinear', penalty='l2', n_jobs=-1):
    '''
    logistic regression with cross validation
    1. Use cv-fold cross validation to determine best penalty C
    2. Using the best C, refit the model with cv-fold again
    3. Report the mean and CI (1.96 * std) of fitted parameters in logistic_reg_refit
    
    Cs: number of Cs to grid search
    cv: number of folds
    
    -----
    return: logistic_reg_cv, logistic_reg_refit
    
    Han 20230208
    '''

    # Do cross validation, try different Cs
    logistic_reg_cv = LogisticRegressionCV(solver=solver, fit_intercept=True, penalty=penalty, Cs=Cs, cv=cv, n_jobs=n_jobs)
    
    # if sum(Y == 1) == 1 or sum(Y == -1) == 1:
    #     logistic_reg_cv.coef_ = np.zeros((1, data.shape[1]))
    #     logistic_reg_cv.intercept_ = 10 * np.sign(np.median(Y))   # If all left, set bias = 10 and other parameters 0
    #     logistic_reg_cv.C_ = np.nan
    # else:
    logistic_reg_cv.fit(data, Y)

    return logistic_reg_cv


def bootstrap(func, data, Y, n_bootstrap=1000, n_samplesize=None, **kwargs):
    # Generate bootstrap samples
    indices = np.random.choice(range(Y.shape[0]), size=(n_bootstrap, Y.shape[0] if n_samplesize is None else n_samplesize), replace=True)   # Could do subsampling
    bootstrap_Y = [Y[index] for index in indices]
    bootstrap_data = [data[index, :] for index in indices]
    
    # Fit the logistic regression model to each bootstrap sample
    outputs = np.array([func(data, Y, **kwargs)[0] for data, Y in zip(bootstrap_data, bootstrap_Y)])
    
    # Get bootstrap mean, std, and CI
    bs = {'raw': outputs,
          'mean': np.mean(outputs, axis=0),
          'std': np.std(outputs, axis=0),
          'CI_lower': np.percentile(outputs, 2.5, axis=0),
          'CI_upper': np.percentile(outputs, 97.5, axis=0)}
    
    return bs
    
    
def decode_betas(coef, trials_back=20):
    
    # Decode fitted betas
    coef = np.atleast_2d(coef)
    # trials_back = int((coef.shape[1] - 1) / 3)  # Hard-coded
    
    b_RewC = coef[:, trials_back - 1::-1]
    b_UnrC = coef[:, 2 * trials_back - 1: trials_back - 1:-1]
    
    if coef.shape[1] >= 3 * trials_back:
        b_C = coef[:, 3 * trials_back - 1:2 * trials_back - 1:-1]
    else:
        b_C = np.full_like(b_UnrC, np.nan)
    
    bias = coef[:, -1:]
    
    return b_RewC, b_UnrC, b_C, bias


def logistic_regression_bootstrap(data, Y, n_bootstrap=1000, n_samplesize=None, **kwargs):
    '''
    1. use cross-validataion to determine the best L2 penality parameter, C
    2. use bootstrap to determine the CI and std
    '''
    
    # Cross validation
    logistic_reg = logistic_regression_CV(data, Y, **kwargs)
    best_C = logistic_reg.C_
    para_mean = np.hstack([logistic_reg.coef_[0], logistic_reg.intercept_])
    
    (logistic_reg.b_RewC, 
     logistic_reg.b_UnrC, 
     logistic_reg.b_C, 
     logistic_reg.bias) = decode_betas(para_mean)
    
    # Bootstrap
    if n_bootstrap > 0:
        bs = bootstrap(logistic_regression, data, Y, n_bootstrap=n_bootstrap, n_samplesize=n_samplesize, C=best_C[0], **kwargs)
        
        logistic_reg.coefs_bootstrap = bs
        (logistic_reg.b_RewC_CI, 
        logistic_reg.b_UnrC_CI, 
        logistic_reg.b_C_CI, 
        logistic_reg.bias_CI) = decode_betas(np.vstack([bs['CI_lower'], bs['CI_upper']]))

        # # Override with bootstrap mean
        # (logistic_reg.b_RewC, 
        # logistic_reg.b_UnrC, 
        # logistic_reg.b_C, 
        # logistic_reg.bias) = decode_betas(np.vstack([bs['mean'], bs['mean']]))
    
    return logistic_reg


# ----- Plotting functions -----
def plot_logistic_regression(logistic_reg, ax=None, ls='-o'):
    if ax is None:
        fig, ax = plt.subplots(1, 1)
        
    # return 
    if_CV = hasattr(logistic_reg, 'b_RewC_CI') # If cross-validated
    x = np.arange(1, logistic_reg.b_RewC.shape[1] + 1)
    plot_spec = {'b_RewC': 'g', 'b_UnrC': 'r', 'b_C': 'b', 'bias': 'k'}    

    for name, col in plot_spec.items():
        mean = getattr(logistic_reg, name)
        if np.all(np.isnan(mean)):
            continue
        ax.plot(x if name != 'bias' else 1, np.atleast_2d(mean)[0, :], ls + col, label=name + ' $\pm$ CI')

        if if_CV:  # From cross validation
            CI = np.atleast_2d(getattr(logistic_reg, name + '_CI'))
            ax.fill_between(x=x if name != 'bias' else [1], 
                            y1=CI[0, :], 
                            y2=CI[1, :], 
                            color=col,
                            alpha=0.3)
        
    if if_CV and hasattr(logistic_reg, "scores_"):
        score_mean = np.mean(logistic_reg.scores_[1.0])
        score_std = np.std(logistic_reg.scores_[1.0])
        if hasattr(logistic_reg, 'cv'):
            ax.set(title=f'{logistic_reg.cv}-fold CV, score $\pm$ std = {score_mean:.3g} $\pm$ {score_std:.2g}\n'
                    f'best C = {logistic_reg.C_[0]:.3g}')
    else:
        pass
        # ax.set(title=f'train: {logistic_reg.train_score:.3g}, test: {logistic_reg.test_score:.3g}')
    
    ax.legend()
    ax.set(xlabel='Past trials', ylabel='Logistic regression coeffs')
    ax.axhline(y=0, color='k', linestyle=':', linewidth=0.5)
    
    return ax


def plot_logistic_compare(logistic_to_compare, 
                          past_trials_to_plot = [1, 2, 3, 4],
                          labels=['ctrl', 'photostim', 'photostim_next'], 
                          edgecolors=['None', 'deepskyblue', 'skyblue'],
                          plot_spec = {'b_RewC': 'g', 'b_UnrC': 'r', 'b_C': 'b', 'bias': 'k'},
                          ax_all=None):
    
    '''
    Compare logistic regressions. Columns for betas, Rows for past trials
    '''
    
    if ax_all is None:   # ax_all is only one axis
        fig, ax_all = plt.subplots(1, 1, figsize=(10, 3 * len(past_trials_to_plot)), layout="constrained")

    # add subaxis in ax_all
    gs = ax_all._subplotspec.subgridspec(len(past_trials_to_plot), len(plot_spec))
    axes = []
        
    for i, past_trial in enumerate(past_trials_to_plot):
        for j, (name, col) in enumerate(plot_spec.items()):
            # ax = axes[i, j]
            ax = ax_all.get_figure().add_subplot(gs[i, j])
            axes.append(ax)
            
            if name == 'bias' and i > 0: 
                ax.set_ylim(0, 0)
                ax.remove()
                continue

            for k, logistic in enumerate(logistic_to_compare):
                mean = getattr(logistic, f'{name}')[0, past_trial - 1]
                yerrs = np.abs(getattr(logistic, f'{name}_CI')[:, past_trial - 1:past_trial] - mean)
                ax.plot(k, mean, marker='o', color=col, markersize=10, markeredgecolor=edgecolors[k], markeredgewidth=2)
                ax.errorbar(x=k, 
                            y=mean,
                            yerr=yerrs,
                            fmt='o', color=col, markeredgecolor=edgecolors[k],
                            ecolor=col, lw=2, capsize=5, capthick=2)

            ax.set(xlim=[-0.5, 0.5 + k])
            ax.axhline(y=0, linestyle='--', c='k', lw=1)
            ax.spines[['right', 'top']].set_visible(False)

            if i == 0:
                ax.set_title(name)

            if i == len(past_trials_to_plot) - 1: 
                ax.set_xticks(range(len(labels)))
                ax.set_xticklabels(labels, rotation=45, ha='right')
            else:
                ax.set_xticks([])

            if j == 0: 
                ax.set_ylabel(f'past_trial = {past_trial}')
            else:
                ax.set_yticklabels([])    


    ylim_min = min([ax.get_ylim()[0] for ax in axes])
    ylim_max = max([ax.get_ylim()[1] for ax in axes])
    for ax in axes:
        ax.set_ylim(ylim_min, ylim_max)
        
    ax_all.remove()
    
    return axes


# --- Wrappers ---
def do_logistic_regression(choice, reward, **kwargs):
    data, Y = prepare_logistic_RewC_UnRC(choice, reward, **kwargs)
    logistic_reg = logistic_regression_bootstrap(data, Y, **kwargs)
    return plot_logistic_regression(logistic_reg)

def do_logistic_regression_no_C(choice, reward, **kwargs):
    data, Y = prepare_logistic_no_C(choice, reward, **kwargs)
    logistic_reg = logistic_regression_bootstrap(data, Y, **kwargs)
    return plot_logistic_regression(logistic_reg)
