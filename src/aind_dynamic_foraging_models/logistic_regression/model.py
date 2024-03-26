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
#%%
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
        A dataframe with index of (trial) and hierachical columns where
            df_design.Y: Choice
            df_design.X: each column represents an independent variable
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
    

def fit_logistic_regression(choice_history: Union[List, np.ndarray],
                            reward_history: Union[List, np.ndarray],
                            logistic_model: Literal['Su2022', 'Bari2019', 'Hattori2019', 'Miller2021'] = 'Su2022',
                            trials_back: int = 15,
                            selected_trial_idx: Union[List, np.ndarray] = None,
                            solver='liblinear', 
                            penalty='l2',
                            Cs=10,
                            cv=10,
                            n_jobs=-1,
                            n_bootstrap=1000, 
                            n_samplesize=None,
                            **kwargs
                            ):
    '''
    1. use cross-validataion to determine the best L2 penality parameter, C
    2. use bootstrap to determine the CI and std
    '''
    
    # -- Prepare design matrix --
    df_design = prepare_logistic_design_matrix(choice_history, 
                                                reward_history, 
                                                logistic_model=logistic_model, 
                                                trials_back=trials_back)
    Y = df_design.Y.to_numpy().ravel()
    X = df_design.X.to_numpy()
    
    # -- Do cross validation with all data and find the best C --
    logistic_reg_cv = LogisticRegressionCV(solver=solver, 
                                           penalty=penalty, 
                                           Cs=Cs, 
                                           cv=cv, 
                                           n_jobs=n_jobs,
                                           **kwargs)
    logistic_reg_cv.fit(X, Y)
    best_C = logistic_reg_cv.C_[0]
    beta_from_CV = np.hstack([logistic_reg_cv.coef_[0], logistic_reg_cv.intercept_])
    beta_names = df_design.X.columns.tolist() + ['bias']
    df_betas = pd.DataFrame([beta_from_CV], columns=[beta_names], index=['cross_validation'])
    
    # -- Do bootstrap with the best C to get confidence interval --
    if n_bootstrap > 0:
        # Prepare a simple function for bootstrap (fit without CV and return all coefs as an array)
        def _fit_logistic_one_sample(X, Y, **kwargs):
            logistic_reg = LogisticRegression(solver=solver, penalty=penalty, C=best_C, **kwargs)
            logistic_reg.fit(X, Y)
            return np.concatenate([logistic_reg.coef_[0], logistic_reg.intercept_])

        beta_bootstrap = _bootstrap(_fit_logistic_one_sample, X, Y, 
                                    n_bootstrap=n_bootstrap, n_samplesize=n_samplesize,
                                    **kwargs
                                    )
        # Get bootstrap mean, std, and CI
        df_betas.loc['bootstrap_mean'] = beta_bootstrap.mean(axis=0)
        df_betas.loc['bootstrap_std'] = beta_bootstrap.std(axis=0)
        df_betas.loc['bootstrap_CI_lower'] = np.percentile(beta_bootstrap, 2.5, axis=0)
        df_betas.loc['bootstrap_CI_upper'] = np.percentile(beta_bootstrap, 97.5, axis=0)
        
    # -- Fit exponential model --
    # TODO: Implement exponential model fitting
    
    return {
        'model': logistic_model,
        'model_terms': MODEL_MAPPER[logistic_model] + ['bias'],
        'trials_back': trials_back,
        'df_design': df_design, 
        'df_betas': df_betas,  # Main output
        'logistic_reg_cv': logistic_reg_cv, # raw output of the fitting with CV
        'beta_bootstrap': beta_bootstrap if n_bootstrap > 0 else None, # raw beta from all bootstrap samples
        }


# --- Helper functions ---
def _bootstrap(func, X, Y, n_bootstrap=1000, n_samplesize=None, **kwargs):
    # Generate bootstrap samples
    indices = np.random.choice(range(Y.shape[0]), 
                               size=(n_bootstrap, Y.shape[0] if n_samplesize is None else n_samplesize), 
                               replace=True)   # Could do subsampling
    bootstrap_Y = [Y[index] for index in indices]
    bootstrap_X = [X[index, :] for index in indices]
    
    # Apply func to each bootstrap sample    
    return np.array([func(X, Y, **kwargs) 
                     for X, Y in zip(bootstrap_X, bootstrap_Y)])
