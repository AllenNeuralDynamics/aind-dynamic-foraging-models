"""Persist a fit so the expensive part never has to be repeated.

A cohort fit costs hours; scoring, diagnostics and figures cost minutes. Keeping only
summary statistics means any new question -- a different conditioning rung, a paired
per-subject test, a calibration check -- forces a refit.

Posterior draws and sampler diagnostics go to netCDF via ArviZ, which is the standard
container for them and is readable by the wider Bayesian tooling. Scalars and provenance go
to a small JSON alongside.
"""

import json
import subprocess
from pathlib import Path

import numpy as np

# Sites worth keeping from a three-level fit. Session-level parameters are large, so they
# are opt-in rather than saved by default.
POPULATION_SITES = (
    "population_mean", "population_scale", "log_sigma_mean", "log_sigma_spread",
)
SUBJECT_SITES = ("mu_p", "log_sigma")
SESSION_SITES = (
    "learn_rate_rew", "learn_rate_unrew", "forget_rate_unchosen",
    "softmax_inverse_temperature", "bias_l", "session_log_lik",
)


def _git_sha(repo_path):
    """Return the current commit of a repository, or None outside one."""
    try:
        result = subprocess.run(
            ["git", "-C", str(repo_path), "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=10, check=True,
        )
        return result.stdout.strip()
    except (subprocess.SubprocessError, OSError):
        return None


def _as_dataset(group):
    """Return an InferenceData group as an xarray Dataset.

    ArviZ >= 1.0 builds an ``xarray.DataTree``, whose nodes reject both the list-of-names
    indexing used to select sites and the ``in`` test used to look for a diagnostic.
    Earlier versions hand back a Dataset already, so this normalises the two.
    """
    to_dataset = getattr(group, "to_dataset", None)
    return to_dataset() if callable(to_dataset) else group


def summarise_diagnostics(idata):
    """Reduce an InferenceData to the diagnostics worth checking before trusting a fit.

    Parameters
    ----------
    idata : arviz.InferenceData
        Output of :func:`save_fit`.

    Returns
    -------
    dict
        Worst-case r_hat, minimum bulk ESS, divergence count and mean tree depth.
    """
    import arviz as az

    out = {}
    try:
        summary = az.summary(idata, var_names=list(POPULATION_SITES), round_to=None)
        out["max_r_hat"] = float(np.nanmax(summary["r_hat"].values))
        out["min_ess_bulk"] = float(np.nanmin(summary["ess_bulk"].values))
    except (KeyError, ValueError):  # pragma: no cover - depends on which sites exist
        pass

    stats = getattr(idata, "sample_stats", None)
    if stats is not None:
        stats = _as_dataset(stats)
        if "diverging" in stats:
            out["divergences"] = int(np.sum(np.asarray(stats["diverging"])))
        if "tree_depth" in stats:
            out["mean_tree_depth"] = float(np.mean(np.asarray(stats["tree_depth"])))
        if "acceptance_rate" in stats:
            out["mean_accept_rate"] = float(np.mean(np.asarray(stats["acceptance_rate"])))
    return out


def save_fit(mcmc, output_dir, *, name="fit", include_session_sites=False, meta=None):
    """Write posterior draws, diagnostics and provenance for one fit.

    Parameters
    ----------
    mcmc : numpyro.infer.MCMC
        A completed sampler.
    output_dir : str or Path
        Directory to write into; created if absent.
    name : str, optional
        Basename for the artifacts.
    include_session_sites : bool, optional
        Also keep session-level parameters and per-session log likelihoods. These are the
        bulk of the data, and are what WAIC, PSIS-LOO and per-session comparisons need.
    meta : mapping, optional
        Provenance to record alongside: config, seed, cohort selection, data snapshot.

    Returns
    -------
    dict
        Paths written and the diagnostics summary.
    """
    import arviz as az

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    keep = list(POPULATION_SITES) + list(SUBJECT_SITES)
    if include_session_sites:
        keep += list(SESSION_SITES)
    available = set(mcmc.get_samples().keys())
    idata = az.from_numpyro(mcmc)
    present = [site for site in keep if site in available]

    netcdf_path = output_dir / f"{name}.nc"
    _as_dataset(idata.posterior)[present].to_netcdf(str(netcdf_path))
    stats_path = output_dir / f"{name}_sample_stats.nc"
    if getattr(idata, "sample_stats", None) is not None:
        _as_dataset(idata.sample_stats).to_netcdf(str(stats_path))

    diagnostics = summarise_diagnostics(idata)
    record = {
        "name": name,
        "sites_saved": present,
        "n_draws": int(np.asarray(mcmc.get_samples()[present[0]]).shape[0]) if present else 0,
        "diagnostics": diagnostics,
        "_meta": dict(meta or {}),
    }
    record["_meta"].setdefault(
        "models_git_sha", _git_sha(Path(__file__).resolve().parents[3])
    )

    json_path = output_dir / f"{name}.json"
    with json_path.open("w") as handle:
        json.dump(record, handle, indent=2, default=str)

    return {
        "netcdf": str(netcdf_path),
        "sample_stats": str(stats_path),
        "json": str(json_path),
        "diagnostics": diagnostics,
    }


def load_population(netcdf_path):
    """Read population draws back from a saved fit.

    Returns
    -------
    dict of str to np.ndarray
        Draws for each population site, shaped ``(n_draws, n_params)``.
    """
    import xarray as xr

    # save_fit writes a plain xarray Dataset (the posterior group), so read it as one
    # rather than through az.from_netcdf, which expects a full InferenceData file.
    with xr.open_dataset(str(netcdf_path)) as posterior:
        out = {}
        for site in POPULATION_SITES:
            if site in posterior:
                values = np.asarray(posterior[site])
                # collapse the (chain, draw) axes, keep the parameter axis
                out[site] = values.reshape(-1, values.shape[-1])
    return out
