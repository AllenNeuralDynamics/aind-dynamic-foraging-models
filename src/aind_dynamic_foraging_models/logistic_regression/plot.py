"""
Plot functions for logistic regression
"""

import matplotlib.pyplot as plt
import numpy as np

from .model import exp_func

__all__ = ["COLOR_MAPPER", "plot_logistic_regression"]

COLOR_MAPPER = {
    "RewC": "g",
    "UnrC": "r",
    "Choice": "b",
    "Reward": "c",
    "Choice_x_Reward": "m",
    "bias": "k",
}


def plot_logistic_regression(dict_logistic_result, ax=None, ls="-o", alpha=0.3):
    """Plot logistic regression results with the output dictionary
    from model.fit_logistic_regression

    Parameters
    ----------
    dict_logistic_result : Dict
        The dictionary output from model.fit_logistic_regression
    ax : , optional
        If None, create a new figure and axis, by default None
    ls : str, optional
        Line style for the plot, by default '-o'
    alpha : float, optional
        Transparency of the confidence interval band, by default 0.3

    Returns
    -------
    ax : matplotlib.axes.Axes
        a matplotlib axis
    """
    if ax is None:
        fig, ax = plt.subplots(1, 1)

    # -- Plot beta values and confidence intervals --
    logistic_reg = dict_logistic_result["logistic_reg_cv"]
    df_beta = dict_logistic_result["df_beta"]
    df_beta_exp_fit = dict_logistic_result["df_beta_exp_fit"]

    for var in dict_logistic_result["model_terms"]:
        col = COLOR_MAPPER[var]

        var_mean = df_beta.loc[var, "cross_validation"]
        trials_back = var_mean.index

        if np.all(np.isnan(var_mean)):
            continue

        if var == "bias":
            label = (
                f"bias = {df_beta.loc['bias', 'cross_validation'].values[0]:.3f}"
                Rf" $\pm$ {df_beta.loc['bias', 'bootstrap_std'].values[0]:.2g}"
            )
        else:
            label = var + R" $\pm$ 95% CI"

        ax.plot(
            trials_back if var != "bias" else 1,
            np.atleast_2d(var_mean)[0, :],
            ls + col,
            label=label,
        )

        # Add confidence intervals, if available
        if "bootstrap_CI_upper" in df_beta.columns:
            var_CI_upper = df_beta.loc[var, "bootstrap_CI_upper"]
            var_CI_lower = df_beta.loc[var, "bootstrap_CI_lower"]

            ax.fill_between(
                x=trials_back if var != "bias" else [1],
                y1=var_CI_upper,
                y2=var_CI_lower,
                color=col,
                alpha=alpha,
            )

        # -- Add exponential fit --
        if var != "bias":
            xx = np.linspace(1, trials_back.max(), 100)
            yy = exp_func(xx, *df_beta_exp_fit.loc[var, (slice(None), "fitted")])
            ax.plot(
                xx,
                yy,
                color=col,
                ls="--",
                lw=4,
                label=Rf"$\beta_0$ = {df_beta_exp_fit.loc[var, ('amp', 'fitted')]:.2f}, "
                Rf"$\tau$ = {df_beta_exp_fit.loc[var, ('tau', 'fitted')]:.2f}",
            )

    # -- Add title and labels --
    ind_C = np.where(logistic_reg.Cs_ == logistic_reg.C_)[0][0]  # Get the ind of best C
    score_mean = np.mean(logistic_reg.scores_[1.0][:, ind_C], axis=0)
    score_std = np.std(logistic_reg.scores_[1.0][:, ind_C], axis=0)
    penalty = dict_logistic_result["logistic_reg_cv"].penalty.upper()

    str_bs = (
        ""
        if dict_logistic_result["beta_bootstrap"] is None
        else f"; CI from {dict_logistic_result['beta_bootstrap'].shape[0]} bootstraps"
    )
    ax.set(
        title=f"{dict_logistic_result['model']}, {penalty} penalty, "
        f"best C = {logistic_reg.C_[0]:.3g} from {logistic_reg.cv}-fold CV\n"
        Rf"score $\pm$ std = {score_mean:.3g} $\pm$ {score_std:.2g}" + str_bs
    )

    ax.legend()
    ax.set(
        xlabel="Past trials",
        ylabel="Logistic regression coeffs",
        xticks=[1] + list(range(5, dict_logistic_result["n_trial_back"] + 1, 5)),
    )
    ax.axhline(y=0, color="k", linestyle=":", linewidth=0.5)

    return ax
