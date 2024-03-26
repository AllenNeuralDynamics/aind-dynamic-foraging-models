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
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV

# See https://github.com/AllenNeuralDynamics/aind-dynamic-foraging-models/discussions/10
MODEL_MAPPER = {
    'Su2022': ['RewC', 'UnrC'],
    'Bari2019': ['RewC', 'Choice'],
    'Hattori2019': ['RewC', 'UnrC', 'Choice'],
    'Miller2021': ['Choice', 'Reward', 'Choice_x_Reward'],
}

def prepare_logistic_design_matrix(
    choice_history: Union[List, np.ndarray],
    reward_history: Union[List, np.ndarray],
    logistic_model: Literal['Su2022', 'Bari2019', 'Hattori2019', 'Miller2021'] = 'Su2022',
    trials_back: int = 15,
    selected_trial_idx: Union[List, np.ndarray] = None,
) -> pd.DataFrame:
    """Prepare logistic regression design matrix from choice and reward history.
    
    See discussion here:
        https://github.com/AllenNeuralDynamics/aind-dynamic-foraging-models/discussions/10
    
    Parameters
    ----------
    choice_history : Union[List, np.ndarray]
        Choice history (0 = left choice, 1 = right choice).
    reward_history : Union[List, np.ndarray]
        Reward history (0 = unrewarded, 1 = rewarded).
    logistic_model : Literal['Su2022', 'Bari2019', 'Hattori2019', 'Miller2021'], optional
        The logistic regression model to use. Defaults to 'Su2022'.
        Supported models: 'Su2022', 'Bari2019', 'Hattori2019', 'Miller2021'.
    trials_back : int, optional
        Number of trials back into history. Defaults to 15.
    selected_trial_idx : Union[List, np.ndarray], optional
        If None, use all trials; 
        else, only look at selected trials for fitting, but using the full history.

    Returns
    -------
    df_design: pd.DataFrame
        A dataframe with index of (trial) and hierachical columns (Y.RewC, UnrC, Choice, Bias)
    """
    
    # Remove ignore trials in choice and reward
    choice_history = np.array(choice_history)
    reward_history = np.array(reward_history)
    
    ignored = np.isnan(choice_history)
    choice_history = choice_history[~ignored]
    reward_history = reward_history[~ignored]
    
    # Sanity checks
    assert len(choice_history) == len(reward_history), "Choice and reward must have the same length."
    assert logistic_model in MODEL_MAPPER.keys(), \
        f"Invalid logistic model. Models supported: {list(MODEL_MAPPER.keys())}."
    assert all(x in [0, 1] for x in choice_history), "Choice must be 0, 1, or np.nan"
    assert all(x in [0, 1] for x in reward_history), "Reward must be 0 or 1"
    
    n_trials = len(choice_history)
    assert n_trials >= trials_back + 2, "Number of trials must be greater than trials_back + 2."
    
    # Encoding data
    encoding = {}
    encoding['Choice'] = 2 * choice_history - 1
    encoding['Reward'] = 2 * reward_history - 1
    encoding['RewC'] = encoding['Choice'] * (encoding['Reward'] == 1)
    encoding['UnrC'] = encoding['Choice'] * (encoding['Reward'] == -1)
    encoding['Choice_x_Reward'] = encoding['Choice'] * encoding['Reward']
    
    assert np.array_equal(encoding['Choice'], encoding['RewC'] + encoding['UnrC'])
    assert np.array_equal(encoding['Choice_x_Reward'], encoding['RewC'] - encoding['UnrC'])
        
    # Package independent variables
    X = []
    if selected_trial_idx is None:
        trials_to_include = range(trials_back, n_trials)
    else:
        trials_to_include = np.intersect1d(selected_trial_idx, 
                                range(trials_back, n_trials)
                                )
    for trial in trials_to_include:
        selected_indices = slice(trial - trials_back, trial)
        X.append(np.hstack(
            [encoding[var][selected_indices] 
             for var in MODEL_MAPPER[logistic_model]]
            ))

    # Prepare df_design
    var_names = [f"{var}_{n}" 
                 for var in MODEL_MAPPER[logistic_model] 
                 for n in reversed(range(1, trials_back + 1))]
    X = np.array(X)
    Y = encoding['Choice'][trials_to_include]
    df_design = pd.DataFrame(
        X, 
        columns=pd.MultiIndex.from_product([['X'], var_names]), 
        index=pd.Index(trials_to_include, name='trial')
    )
    df_design['Y', 'Choice'] = Y
    return df_design

def fit_logistic(X, Y, solver='liblinear', penalty='l2', C=1, test_size=0.10, **kwargs):
    '''
    Run one logistic regression fit
    (Reward trials + Unreward trials + Choice + bias)
    Han 20230208
    '''
    trials_back = int(X.shape[1] / 3)
    
    # Do training
    # x_train, x_test, y_train, y_test = train_test_split(data, Y, test_size=test_size)
    logistic_reg = LogisticRegression(solver=solver, fit_intercept=True, penalty=penalty, C=C)

    # if sum(Y == 1) == 1 or sum(Y == -1) == 1:
    #     logistic_reg_cv.coef_ = np.zeros((1, data.shape[1]))
    #     logistic_reg_cv.intercept_ = 10 * np.sign(np.median(Y))   # If all left, set bias = 10 and other parameters 0
    #     logistic_reg_cv.C_ = np.nan
    # else:
    logistic_reg.fit(X, Y)

    output = np.concatenate([logistic_reg.coef_[0], logistic_reg.intercept_])
    
    (logistic_reg.b_RewC, 
    logistic_reg.b_UnrC, 
    logistic_reg.b_C, 
    logistic_reg.bias) = decode_betas(output, trials_back)
    
    return output, logistic_reg


def fit_logistic_CV(data, Y, Cs=10, cv=10, solver='liblinear', penalty='l2', n_jobs=-1):
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


def _bootstrap(func, data, Y, n_bootstrap=1000, n_samplesize=None, **kwargs):
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
    logistic_reg = fit_logistic_CV(data, Y, **kwargs)
    best_C = logistic_reg.C_
    para_mean = np.hstack([logistic_reg.coef_[0], logistic_reg.intercept_])
    
    (logistic_reg.b_RewC, 
     logistic_reg.b_UnrC, 
     logistic_reg.b_C, 
     logistic_reg.bias) = decode_betas(para_mean)
    
    # Bootstrap
    if n_bootstrap > 0:
        bs = _bootstrap(fit_logistic, data, Y, n_bootstrap=n_bootstrap, n_samplesize=n_samplesize, C=best_C[0], **kwargs)
        
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





# --- Wrappers ---
def do_logistic_regression(choice, reward, **kwargs):
    data, Y = prepare_logistic_RewC_UnRC(choice, reward, **kwargs)
    logistic_reg = logistic_regression_bootstrap(data, Y, **kwargs)
    return plot_logistic_regression(logistic_reg)

def do_logistic_regression_no_C(choice, reward, **kwargs):
    data, Y = prepare_logistic_no_C(choice, reward, **kwargs)
    logistic_reg = logistic_regression_bootstrap(data, Y, **kwargs)
    return plot_logistic_regression(logistic_reg)
