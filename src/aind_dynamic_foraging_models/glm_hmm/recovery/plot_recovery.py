"""Figures and tables for ``parameter_recovery.py`` (reads ``results/*.json``).

Writes ``fig_recovery_weights.svg``, ``fig_recovery_vs_size.svg``,
``fig_reference_split.svg``, ``fig_model_selection.svg`` and ``fig_per_mouse.svg`` next to this
file, and the tables into ``RESULTS.md`` between its ``BEGIN tables`` / ``END tables`` markers.
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

EMISSIONS = ("hurdle", "reference")
STATE_COLORS = ("#0072B2", "#D55E00", "#009E73")
EMISSION_COLORS = {"hurdle": "#0072B2", "reference": "#D55E00"}
MM = 1 / 25.4


def _style():
    """Small, editable-text figure style."""
    plt.rcParams.update({
        "font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 6, "axes.titlesize": 6.5, "axes.labelsize": 6, "xtick.labelsize": 5.5,
        "ytick.labelsize": 5.5, "legend.fontsize": 5.5, "axes.linewidth": 0.5,
        "xtick.major.width": 0.5, "ytick.major.width": 0.5, "lines.linewidth": 0.9,
        "axes.spines.top": False, "axes.spines.right": False, "svg.fonttype": "none",
        # Deterministic SVGs: fixed element ids, so regenerating identical data changes no bytes.
        "svg.hashsalt": "glmhmm-recovery",
    })


def _load(results, experiment):
    """{emission: result} for the experiments that exist."""
    out = {}
    for emission in EMISSIONS:
        path = results / f"{experiment}_{emission}.json"
        if path.exists():
            out[emission] = json.loads(path.read_text())
    return out


def _weight_scatter(ax, row, emission):
    """True against recovered weights, one colour per state."""
    fit, true = np.asarray(row["fit_weights"]), np.asarray(row["true_weights"])
    occupancy = np.asarray(row["true_occupancy"]) / np.sum(row["true_occupancy"])
    lim = max(np.abs(true).max(), 1.0) * 1.15
    ax.plot([-lim, lim], [-lim, lim], color="0.7", lw=0.5, zorder=0)
    for k in range(fit.shape[0]):
        rmse = np.sqrt(np.mean((fit[k] - true[k]) ** 2))
        ax.scatter(true[k], np.clip(fit[k], -lim, lim), s=5, color=STATE_COLORS[k], lw=0,
                   label=f"state {k}: occupancy {occupancy[k]:.2f}, RMSE {rmse:.2f}")
    ax.set(xlim=(-lim, lim), ylim=(-lim, lim), xlabel="true weight", ylabel="recovered weight",
           title=f"{emission}: {row['n_sessions']} sessions, {row['n_trials']:,} trials")
    ax.legend(frameon=False, loc="upper left", handletextpad=0.2, borderaxespad=0.2)
    ax.set_aspect("equal")


def _transitions(ax, true_A, fit_A):
    """Recovered transition matrix, true values in brackets."""
    fit_A, true_A = np.asarray(fit_A), np.asarray(true_A)
    ax.imshow(np.log10(np.clip(fit_A, 1e-4, 1)), cmap="Greys", vmin=-4, vmax=0)
    for i in range(fit_A.shape[0]):
        for j in range(fit_A.shape[1]):
            ax.text(j, i, f"{fit_A[i, j]:.4f}\n({true_A[i, j]:.4f})", ha="center", va="center",
                    fontsize=4.8, color="white" if fit_A[i, j] > 0.3 else "black")
    ax.set(xticks=range(3), yticks=range(3), xlabel="to state", ylabel="from state",
           title="transitions: fit (true)")
    for spine in ax.spines.values():
        spine.set_visible(False)


def figure_weights(size, out):
    """Weights and transitions at the largest data size, seed 0, both emissions."""
    fig, axes = plt.subplots(len(size), 2, figsize=(120 * MM, 58 * MM * len(size)),
                             gridspec_kw={"width_ratios": [1.15, 1]})
    axes = np.atleast_2d(axes)
    for row_axes, (emission, result) in zip(axes, size.items()):
        largest = max(r["n_sessions"] for r in result["rows"])
        row = next(r for r in result["rows"] if r["n_sessions"] == largest and r["seed"] == 0)
        _weight_scatter(row_axes[0], row, emission)
        _transitions(row_axes[1], result["true_transitions"], row["fit_transitions"])
    fig.tight_layout()
    fig.savefig(out / "fig_recovery_weights.svg", metadata={"Date": None})
    plt.close(fig)


def _median_abs_weight_error(row):
    """Median over coefficients of |recovered - true| (robust to a few diverging weights)."""
    return float(np.median(np.abs(np.asarray(row["fit_weights"])
                                  - np.asarray(row["true_weights"]))))


def figure_size(size, out):
    """Errors and decoding against the number of simulated trials."""
    panels = [
        ("class_prob_mae", "class-probability error\n(mean |p fit - p true|)", True),
        ("median_weight_err", "weight error\n(median |w fit - w true|)", True),
        ("transition_max_abs_err", "transition error\n(max |A fit - A true|)", True),
        ("decoding_gap", "state decoding\n(oracle - fit accuracy)", False),
    ]
    fig, axes = plt.subplots(1, len(panels), figsize=(175 * MM, 42 * MM))
    for ax, (key, label, log) in zip(axes, panels):
        for emission, result in size.items():
            rows = result["rows"]
            for r in rows:
                r["median_weight_err"] = _median_abs_weight_error(r)
                r["decoding_gap"] = r["oracle_decoding_accuracy"] - r["decoding_accuracy"]
            sizes = sorted({r["n_trials"] for r in rows})
            n_sessions = sorted({r["n_sessions"] for r in rows})
            for r in rows:
                ax.scatter(r["n_trials"], max(r[key], 1e-6) if log else r[key], s=4, lw=0,
                           color=EMISSION_COLORS[emission], alpha=0.6)
            means = [np.mean([r[key] for r in rows if r["n_sessions"] == n]) for n in n_sessions]
            xs = [np.mean([r["n_trials"] for r in rows if r["n_sessions"] == n])
                  for n in n_sessions]
            ax.plot(xs, means, color=EMISSION_COLORS[emission], label=emission)
            if log and emission == "hurdle":
                guide = means[0] * np.sqrt(xs[0] / np.asarray(xs))
                ax.plot(xs, guide, color="0.6", lw=0.5, ls="--", label=r"$\propto 1/\sqrt{N}$")
            del sizes
        ax.set_xscale("log")
        if log:
            ax.set_yscale("log")
        else:
            ax.axhline(0, color="0.7", lw=0.5)
        ax.set(xlabel="simulated trials", ylabel=label)
    axes[0].legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out / "fig_recovery_vs_size.svg", metadata={"Date": None})
    plt.close(fig)


def _reference_split(weights):
    """Reference weights (K, 2 * D) as side contrast (right - left) and common mode."""
    w = np.asarray(weights).reshape(len(weights), 2, -1)
    return w[:, 1] - w[:, 0], 0.5 * (w[:, 1] + w[:, 0])


def figure_reference_split(size, out):
    """Reference emission: which combination of the left and right weights is recovered."""
    if "reference" not in size:
        return
    result = size["reference"]
    largest = max(r["n_sessions"] for r in result["rows"])
    row = next(r for r in result["rows"] if r["n_sessions"] == largest and r["seed"] == 0)
    fit, true = _reference_split(row["fit_weights"]), _reference_split(row["true_weights"])
    fig, axes = plt.subplots(1, 2, figsize=(110 * MM, 52 * MM))
    titles = ("side contrast: right - left", "common mode: (right + left) / 2")
    for ax, f, t, title in zip(axes, fit, true, titles):
        lim = max(np.abs(t).max(), 1.0) * 1.15
        ax.plot([-lim, lim], [-lim, lim], color="0.7", lw=0.5, zorder=0)
        for k in range(f.shape[0]):
            rmse = np.sqrt(np.mean((f[k] - t[k]) ** 2))
            ax.scatter(t[k], np.clip(f[k], -lim, lim), s=5, lw=0, color=STATE_COLORS[k],
                       label=f"state {k}: RMSE {rmse:.2f}")
        ax.set(xlim=(-lim, lim), ylim=(-lim, lim), xlabel="true", ylabel="recovered",
               title=title)
        ax.set_aspect("equal")
        ax.legend(frameon=False, loc="upper left", handletextpad=0.2)
    fig.suptitle(f"reference emission, {row['n_sessions']} sessions", fontsize=6.5)
    fig.tight_layout()
    fig.savefig(out / "fig_reference_split.svg", metadata={"Date": None})
    plt.close(fig)


def figure_selection(selection, out):
    """Held-out log likelihood per trial against K, relative to the true model."""
    fig, ax = plt.subplots(figsize=(55 * MM, 42 * MM))
    for emission, result in selection.items():
        ks = [r["num_states"] for r in result["rows"]]
        gap = [np.log(r["heldout_likelihood"]) - np.log(result["oracle_likelihood"])
               for r in result["rows"]]
        ax.plot(ks, np.asarray(gap) * 1e3, marker="o", ms=2.5, color=EMISSION_COLORS[emission],
                label=emission)
    ax.axhline(0, color="0.6", lw=0.5, ls="--")
    ax.axvline(3, color="0.85", lw=3, zorder=0)
    ax.set(xlabel="fitted K (true K = 3)",
           ylabel="held-out log lik. vs true model\n(millinats / trial)", xticks=range(1, 6))
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out / "fig_model_selection.svg", metadata={"Date": None})
    plt.close(fig)


def figure_per_mouse(per_mouse, out):
    """Per-mouse refits: weight error to each mouse's own truth, and held-out likelihood."""
    fig, axes = plt.subplots(1, 2, figsize=(110 * MM, 45 * MM))
    for offset, (emission, result) in enumerate(per_mouse.items()):
        rows, color = result["rows"], EMISSION_COLORS[emission]
        x = np.array([0, 1]) + 2.6 * offset
        for r in rows:
            axes[0].plot(x, [r["global_weight_median_err"], r["refit_weight_median_err"]],
                         color=color, lw=0.4, alpha=0.6)
            axes[1].plot(x[:1].tolist() + [x[1], x[1] + 1],
                         [r["zero_shot_likelihood"] - r["oracle_likelihood"],
                          r["refit_likelihood"] - r["oracle_likelihood"], 0.0],
                         color=color, lw=0.4, alpha=0.6, marker="o", ms=1.5)
        axes[0].text(x.mean(), axes[0].get_ylim()[1], emission, ha="center", va="bottom",
                     color=color)
    axes[0].set_xticks([0, 1, 2.6, 3.6], ["global", "refit"] * 2)
    axes[0].set_ylabel("median |weight - mouse's own truth|")
    axes[1].set_xticks([0, 1, 2, 2.6, 3.6, 4.6], ["zero-shot", "refit", "oracle"] * 2,
                       rotation=30)
    axes[1].axhline(0, color="0.7", lw=0.5)
    axes[1].set_ylabel("held-out likelihood - mouse's oracle")
    fig.tight_layout()
    fig.savefig(out / "fig_per_mouse.svg", metadata={"Date": None})
    plt.close(fig)


def _size_table(size):
    """Recovery against data size."""
    lines = ["### Recovery against data size (mean over 3 seeds)", "",
             "| emission | sessions | trials | class-prob MAE | median weight err | weight RMSE "
             "| transition max err | decoding (oracle) | log joint - truth's | converged |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    for emission, result in size.items():
        for n in sorted({r["n_sessions"] for r in result["rows"]}):
            rows = [r for r in result["rows"] if r["n_sessions"] == n]
            mean = lambda key: np.mean([r[key] for r in rows])  # noqa: E731
            lines.append(
                f"| {emission} | {n} | {mean('n_trials'):,.0f} | {mean('class_prob_mae'):.4f} | "
                f"{np.mean([_median_abs_weight_error(r) for r in rows]):.3f} | "
                f"{mean('weight_rmse'):.2f} | {mean('transition_max_abs_err'):.4f} | "
                f"{mean('decoding_accuracy'):.3f} ({mean('oracle_decoding_accuracy'):.3f}) | "
                f"{mean('log_joint_minus_truth'):+.1f} | "
                f"{sum(r['converged'] for r in rows)}/{len(rows)} |")
    return lines


def _reference_split_table(size):
    """Reference weight RMSE split into side contrast and common mode."""
    if "reference" not in size:
        return []
    lines = ["", "### Reference emission: weight RMSE split by combination (mean over seeds)",
             "", "| sessions | side contrast (R - L) | common mode ((R + L) / 2) |",
             "|---|---|---|"]
    for n in sorted({r["n_sessions"] for r in size["reference"]["rows"]}):
        rows = [r for r in size["reference"]["rows"] if r["n_sessions"] == n]
        errs = []
        for r in rows:
            f, t = _reference_split(r["fit_weights"]), _reference_split(r["true_weights"])
            errs.append([np.sqrt(np.mean((f[i] - t[i]) ** 2)) for i in range(2)])
        side, common = np.mean(errs, axis=0)
        lines.append(f"| {n} | {side:.2f} | {common:.2f} |")
    return lines


def _selection_table(selection):
    """Held-out score against K."""
    lines = ["", "### Held-out score against K (truth K = 3)", "",
             "| emission | " + " | ".join(f"K={k}" for k in range(1, 6)) + " | true model |",
             "|---|" + "---|" * 6]
    for emission, result in selection.items():
        cells = " | ".join(f"{r['heldout_likelihood']:.5f}" for r in result["rows"])
        lines.append(f"| {emission} | {cells} | {result['oracle_likelihood']:.5f} |")
    return lines


def _per_mouse_table(per_mouse):
    """Per-mouse protocol: weights to each mouse's own truth, and held-out likelihood."""
    first = next(iter(per_mouse.values()), {})
    lines = ["", f"### Per-mouse protocol ({len(first.get('rows', []))} mice, "
             f"{first.get('n_sessions', '?')} sessions each, half context; own weights = truth "
             f"+ N(0, {first.get('spread', '?')}²))", "",
             "| emission | median weight err global → refit | mice improved (median err) "
             "| weight RMSE global → refit | mice improved (RMSE) | held-out zero-shot / refit "
             "/ oracle (mean) | mice with refit > zero-shot |", "|---|---|---|---|---|---|---|"]
    for emission, result in per_mouse.items():
        rows = result["rows"]
        med = [np.median([r[f"{w}_weight_median_err"] for r in rows]) for w in ("global", "refit")]
        rms = [np.median([r[f"{w}_weight_rmse"] for r in rows]) for w in ("global", "refit")]
        better_med = sum(r["refit_weight_median_err"] < r["global_weight_median_err"]
                         for r in rows)
        better_rms = sum(r["refit_weight_rmse"] < r["global_weight_rmse"] for r in rows)
        z, rf, o = (np.mean([r[k] for r in rows]) for k in
                    ("zero_shot_likelihood", "refit_likelihood", "oracle_likelihood"))
        gain = sum(r["refit_likelihood"] > r["zero_shot_likelihood"] for r in rows)
        lines.append(f"| {emission} | {med[0]:.3f} → {med[1]:.3f} | {better_med}/{len(rows)} | "
                     f"{rms[0]:.3f} → {rms[1]:.3f} | {better_rms}/{len(rows)} | "
                     f"{z:.4f} / {rf:.4f} / {o:.4f} | {gain}/{len(rows)} |")
    return lines + _per_state_table(per_mouse)


def _per_state_table(per_mouse):
    """Per-mouse weight RMSE broken down by state."""
    lines = ["", "#### Per-mouse weight RMSE by state (median over mice; state occupancy in "
             "context trials)", "", "| emission | state | occupancy | global | refit | mice "
             "improved |", "|---|---|---|---|---|---|"]
    for emission, result in per_mouse.items():
        rows = result["rows"]
        if "refit_weight_rmse_by_state" not in rows[0]:
            continue
        occupancy = np.sum([r["context_occupancy"] for r in rows], 0)
        occupancy = occupancy / occupancy.sum()
        for k in range(len(occupancy)):
            g = np.median([r["global_weight_rmse_by_state"][k] for r in rows])
            f = np.median([r["refit_weight_rmse_by_state"][k] for r in rows])
            better = sum(r["refit_weight_rmse_by_state"][k] < r["global_weight_rmse_by_state"][k]
                         for r in rows)
            lines.append(f"| {emission} | {k} | {occupancy[k]:.2f} | {g:.3f} | {f:.3f} | "
                         f"{better}/{len(rows)} |")
    return lines


def _ssm_table(results):
    """Cross-check against ssm (``ssm_parity.py``), if it has been run."""
    path = results / "ssm_parity.json"
    if not path.exists():
        return []
    r = json.loads(path.read_text())
    lines = ["", f"### Cross-check against ssm (reference emission, {r['n_sessions']} sessions, "
             f"{r['n_trials']:,} trials, same data, start and priors)", "",
             "| parameters | log likelihood, this package | log likelihood, ssm |", "|---|---|---|"]
    for name in ("truth", "start", "ours"):
        lines.append(f"| {name} | {r[f'ours_ll_at_{name}']:.6f} | {r[f'ssm_ll_at_{name}']:.6f} |")
    lines.append(f"| ssm's own fit | {r['ours_ll_at_ssm']:.6f} | (same parameters) |")
    lines += ["", "| fit | EM iterations | seconds | predictive MAE vs truth | side-contrast RMSE "
              "to truth by state | transition max err |", "|---|---|---|---|---|---|"]
    for name in ("ours", "ssm"):
        side = ", ".join(f"{v:.3f}" for v in r[f"{name}_side_rmse_to_truth_by_state"])
        lines.append(f"| {'this package' if name == 'ours' else 'ssm'} | {r[f'{name}_iters']} | "
                     f"{r[f'{name}_seconds']:.1f} | {r[f'predictive_mae_{name}_vs_truth']:.4f} | "
                     f"{side} | {r[f'{name}_transition_max_err']:.4f} |")
    side = ", ".join(f"{v:.3f}" for v in r["ssm_vs_ours_side_rmse_by_state"])
    lines += ["", f"Between the two fits: predictive MAE {r['predictive_mae_ssm_vs_ours']:.4f}; "
              f"side-contrast RMSE by state {side}."]
    return lines


def tables(size, selection, per_mouse, results=None):
    """Markdown tables of every number the figures show."""
    lines = (_size_table(size) + _reference_split_table(size) + _selection_table(selection)
             + _per_mouse_table(per_mouse) + (_ssm_table(results) if results else []))
    return "\n".join(lines) + "\n"


def plot_all(results, out):
    """Every figure and the tables, from whatever results exist."""
    _style()
    size, selection, per_mouse = (_load(results, e) for e in ("size", "selection", "per_mouse"))
    if size:
        figure_weights(size, out)
        figure_size(size, out)
        figure_reference_split(size, out)
    if selection:
        figure_selection(selection, out)
    if per_mouse:
        figure_per_mouse(per_mouse, out)
    results_md = out / "RESULTS.md"
    begin, end = "<!-- BEGIN tables (plot_recovery.py) -->", "<!-- END tables -->"
    text = results_md.read_text()
    head, rest = text.split(begin)
    _, tail = rest.split(end)
    results_md.write_text(head + begin + "\n\n" + tables(size, selection, per_mouse, results)
                          + "\n"
                          + end + tail)
    print(f"wrote figures and the RESULTS.md tables to {out}")


if __name__ == "__main__":
    here = Path(__file__).resolve().parent
    plot_all(here / "results", here)
