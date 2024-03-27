"""
Plot functions for logistic regression
"""
import numpy as np
import matplotlib.pyplot as plt

from .model import exp_func

COLOR_MAPPER = {
    'RewC': 'g', 
    'UnrC': 'r', 
    'Choice': 'b',
    'Reward': 'c',
    'Choice_x_Reward': 'm', 
    'bias': 'k'
}

def plot_logistic_regression(dict_logistic_result, 
                             ax=None, 
                             ls='-o',
                             alpha=0.3):
    if ax is None:
        fig, ax = plt.subplots(1, 1)
    
    # -- Plot beta values and confidence intervals --
    logistic_reg = dict_logistic_result['logistic_reg_cv']
    df_beta = dict_logistic_result['df_beta']
    df_beta_exp_fit = dict_logistic_result['df_beta_exp_fit']
    
    for var in dict_logistic_result['model_terms']:
        col = COLOR_MAPPER[var]
        
        var_mean = df_beta.loc[var, 'cross_validation']
        trials_back = var_mean.index
        
        if np.all(np.isnan(var_mean)):
            continue
        ax.plot(trials_back if var != 'bias' else 1, 
                np.atleast_2d(var_mean)[0, :], 
                ls + col, label=var + ' $\pm$ CI')

        # Add confidence intervals, if available
        if 'bootstrap_CI_upper' in df_beta.columns:
            var_CI_upper = df_beta.loc[var, 'bootstrap_CI_upper']
            var_CI_lower = df_beta.loc[var, 'bootstrap_CI_lower']

            ax.fill_between(x=trials_back if var != 'bias' else [1], 
                            y1=var_CI_upper, 
                            y2=var_CI_lower, 
                            color=col,
                            alpha=alpha)
    
        # -- Add exponential fit --
        if var != 'bias':
            xx = np.linspace(1, trials_back.max(), 100)
            yy = exp_func(xx, *df_beta_exp_fit.loc[var, (slice(None), 'fitted')])
            ax.plot(xx, yy, color=col, ls='--', lw=4, 
                    label=fR"$\tau$ = {df_beta_exp_fit.loc[var, ('tau', 'fitted')]:.2f}")
        
    # -- Add title and labels --
    score_mean = np.mean(logistic_reg.scores_[1.0])
    score_std = np.std(logistic_reg.scores_[1.0])
    if hasattr(logistic_reg, 'cv'):
        ax.set(title=f'{logistic_reg.cv}-fold CV, score $\pm$ std = {score_mean:.3g} $\pm$ {score_std:.2g}\n'
                f'best C = {logistic_reg.C_[0]:.3g}')
    else:
        pass
    
    ax.legend()
    ax.set(xlabel='Past trials', 
           ylabel='Logistic regression coeffs',
           xticks=[1] + list(range(5, dict_logistic_result['n_trial_back'] + 1, 5))
    )
    ax.axhline(y=0, color='k', linestyle=':', linewidth=0.5)

    
    return ax
