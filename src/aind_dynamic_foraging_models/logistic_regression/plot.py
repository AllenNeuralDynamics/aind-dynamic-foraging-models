"""
Plot functions for logistic regression
"""
import numpy as np
import matplotlib.pyplot as plt

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
