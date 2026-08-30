"""Figures for a fitted hierarchical model.

Three questions a fit has to answer visually: did it recover the population, does
conditioning on a subject's own sessions actually help, and is partial pooling doing
anything a per-session fit would not.

Colours use the validated categorical order (blue, orange, aqua) in fixed slots, so the
same entity keeps the same hue across every figure: blue is always this model, orange the
per-subject MLE reference, aqua the neural-model reference. Ground truth is drawn in ink
rather than a series colour, because it is a reference, not a series.
"""

import numpy as np

# Fixed categorical slots. Identity, not rank: an entity keeps its hue across figures.
HB = "#2a78d6"
MLE = "#eb6834"
GRU = "#1baf7a"
INK = "#33383F"
MUTED = "#8A919E"
GRID = "#E3E6EB"
SURFACE = "#FFFFFF"

PARAM_LABELS = {
    "learn_rate_rew": r"$\alpha_{+}$  learn rate, rewarded",
    "learn_rate_unrew": r"$\alpha_{-}$  learn rate, unrewarded",
    "forget_rate_unchosen": r"$\delta$  forget rate, unchosen",
    "softmax_inverse_temperature": r"$\beta$  inverse temperature",
    "bias_l": r"$b_L$  side bias",
}


def _style(ax):
    """Recessive axes and grid, so the marks carry the figure."""
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=8, length=3)
    ax.grid(True, color=GRID, linewidth=0.6, alpha=0.9)
    ax.set_axisbelow(True)


def plot_population_recovery(population_draws, subject_means, truth=None,
                             param_names=None, beta_max=10.0, path=None):
    """Population posterior against ground truth, with each subject's estimate beneath.

    One panel per parameter. The population posterior is a density; individual subjects
    are ticks along the axis, so the spread of subjects and the width of the population
    estimate can be read against each other.

    Parameters
    ----------
    population_draws : np.ndarray, shape (n_draws, n_params)
        Draws of the population location, on the unconstrained scale.
    subject_means : np.ndarray, shape (n_subjects, n_params)
        Posterior mean per subject, same scale.
    truth : np.ndarray, optional
        Generating values, when fitting simulated data.
    param_names : sequence of str, optional
        Names for the panels.
    beta_max : float, optional
        Unused here; kept so callers can pass one signature everywhere.
    path : str, optional
        Where to save. Returns the figure regardless.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    population_draws = np.asarray(population_draws)
    subject_means = np.asarray(subject_means)
    n_params = population_draws.shape[-1]
    names = list(param_names or list(PARAM_LABELS)[:n_params])

    fig, axes = plt.subplots(1, n_params, figsize=(3.1 * n_params, 3.0), facecolor=SURFACE)
    axes = np.atleast_1d(axes)

    for i, ax in enumerate(axes):
        draws = population_draws[:, i]
        ax.hist(draws, bins=36, density=True, color=HB, alpha=0.75,
                edgecolor=SURFACE, linewidth=0.5)

        # Subjects as rug ticks: shows the spread the population is summarising.
        lo, hi = ax.get_ylim()
        ax.plot(subject_means[:, i], np.full(subject_means.shape[0], hi * 0.045),
                marker="|", linestyle="none", color=HB, alpha=0.65,
                markersize=8, markeredgewidth=1.1)

        if truth is not None:
            ax.axvline(np.asarray(truth)[i], color=INK, linewidth=1.6,
                       linestyle=(0, (4, 2)), zorder=5)

        _style(ax)
        ax.set_yticks([])
        ax.set_title(PARAM_LABELS.get(names[i], names[i]), fontsize=9,
                     color=INK, pad=8)

    label = "population posterior  ·  ticks: individual subjects"
    if truth is not None:
        label += "  ·  dashed: ground truth"
    fig.suptitle(label, fontsize=9, color=MUTED, y=1.02)
    fig.tight_layout()
    if path:
        fig.savefig(path, dpi=170, bbox_inches="tight", facecolor=SURFACE)
    return fig


def plot_conditioning_curve(scores, references=None, path=None, title=None):
    """Held-out likelihood against the number of context sessions.

    The curve traces how much of a new subject's behaviour the cohort prior already
    explains (k=0) and how fast its own sessions improve on that.

    Parameters
    ----------
    scores : mapping
        Context-session count to held-out likelihood. A ``"matched"`` key is drawn as a
        separate marker, since it is a different protocol rather than another k.
    references : mapping, optional
        Name to (value, colour) for horizontal comparison lines.
    path : str, optional
        Where to save.
    title : str, optional
        Figure title.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ks = sorted(k for k in scores if isinstance(k, (int, float)))
    values = [scores[k] for k in ks]

    fig, ax = plt.subplots(figsize=(6.0, 3.9), facecolor=SURFACE)
    ax.plot(ks, values, color=HB, linewidth=2, marker="o", markersize=6,
            markerfacecolor=HB, markeredgecolor=SURFACE, markeredgewidth=1.6,
            zorder=4, label="HB, k context sessions")

    if "matched" in scores:
        ax.axhline(scores["matched"], color=HB, linewidth=1.4,
                   linestyle=(0, (1, 2)), alpha=0.85, zorder=3)
        ax.annotate(f"matched  {scores['matched']:.4f}",
                    xy=(ks[-1], scores["matched"]), xytext=(-4, 6),
                    textcoords="offset points", ha="right", fontsize=8, color=HB)

    for name, (value, colour) in (references or {}).items():
        ax.axhline(value, color=colour, linewidth=1.5, linestyle="-", alpha=0.9, zorder=2)
        ax.annotate(f"{name}  {value:.4f}", xy=(ks[0], value), xytext=(2, 5),
                    textcoords="offset points", fontsize=8, color=colour)

    _style(ax)
    ax.set_xlabel("context sessions (k)", fontsize=9, color=INK)
    ax.set_ylabel("held-out likelihood per trial", fontsize=9, color=INK)
    ax.set_xticks(ks)
    if title:
        ax.set_title(title, fontsize=10, color=INK, pad=10, loc="left")
    fig.tight_layout()
    if path:
        fig.savefig(path, dpi=170, bbox_inches="tight", facecolor=SURFACE)
    return fig


def plot_shrinkage(subject_means, population_mean, unpooled=None,
                   param_index=0, param_name=None, path=None):
    """Each subject's pooled estimate against its unpooled one.

    Partial pooling pulls subjects toward the cohort; how far each moves is the whole
    argument for the hierarchy, and it should be largest for subjects with least data.

    Parameters
    ----------
    subject_means : np.ndarray, shape (n_subjects, n_params)
        Pooled posterior means.
    population_mean : float or np.ndarray
        The cohort location the subjects shrink toward.
    unpooled : np.ndarray, optional
        Per-subject estimates with no pooling, same shape.
    param_index : int, optional
        Which parameter to draw.
    param_name : str, optional
        Label for it.
    path : str, optional
        Where to save.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    pooled = np.asarray(subject_means)[:, param_index]
    centre = float(np.atleast_1d(population_mean)[param_index])
    order = np.argsort(pooled)

    fig, ax = plt.subplots(figsize=(6.0, 3.9), facecolor=SURFACE)
    y = np.arange(len(order))

    if unpooled is not None:
        raw = np.asarray(unpooled)[:, param_index][order]
        for i, (a, b) in enumerate(zip(raw, pooled[order])):
            ax.plot([a, b], [i, i], color=MUTED, linewidth=0.9, alpha=0.7, zorder=1)
        ax.plot(raw, y, marker="o", linestyle="none", markersize=4.5, color=MLE,
                markeredgecolor=SURFACE, markeredgewidth=1, zorder=3, label="no pooling")

    ax.plot(pooled[order], y, marker="o", linestyle="none", markersize=4.5, color=HB,
            markeredgecolor=SURFACE, markeredgewidth=1, zorder=4, label="partially pooled")
    ax.axvline(centre, color=INK, linewidth=1.4, linestyle=(0, (4, 2)), zorder=2)
    ax.annotate("cohort", xy=(centre, len(order) * 0.98), xytext=(4, 0),
                textcoords="offset points", fontsize=8, color=INK)

    _style(ax)
    ax.set_yticks([])
    ax.set_ylabel("subjects", fontsize=9, color=INK)
    ax.set_xlabel(param_name or PARAM_LABELS.get(
        list(PARAM_LABELS)[param_index], "parameter"), fontsize=9, color=INK)
    if unpooled is not None:
        ax.legend(frameon=False, fontsize=8, loc="lower right", labelcolor=INK)
    fig.tight_layout()
    if path:
        fig.savefig(path, dpi=170, bbox_inches="tight", facecolor=SURFACE)
    return fig
