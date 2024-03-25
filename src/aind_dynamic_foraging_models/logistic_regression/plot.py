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