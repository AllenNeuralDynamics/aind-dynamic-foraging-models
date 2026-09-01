"""Two-stage empirical Bayes: per-subject fits, then a population fit over them.

Stage one fits each subject independently with :func:`hattori2019_two_level`. Stage two
fits a population distribution to the resulting subject-level posteriors, giving the third
level of the hierarchy without the cost of a single joint fit over the whole cohort.

The population stage consumes each subject's posterior **mean and standard error**, not its
mean alone. Fitting to point estimates would conflate each subject's posterior uncertainty
with genuine between-subject variance and inflate the population scale, producing a
too-diffuse prior for held-out subjects. This is a random-effects (measurement-error) model
of the kind used in meta-analysis, and it separates the two sources of spread.
"""

import jax
import jax.numpy as jnp
import numpy as np
import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS

from .model import HATTORI2019_PARAMS, hattori2019_two_level

# Subject-level coordinates pooled by the population stage: the location of each parameter
# and the log of its session-level spread.
POPULATION_COORDS = tuple(f"mu_p_{name}" for name in HATTORI2019_PARAMS) + tuple(
    f"log_sigma_{name}" for name in HATTORI2019_PARAMS
)


def population_model(subject_estimates, subject_standard_errors):
    """Random-effects population model over subject-level posterior summaries.

    Each subject's true coordinate is drawn from the population distribution, and the
    subject's posterior mean is a noisy observation of it with known standard error. The
    population scale therefore measures between-subject spread net of within-subject
    posterior uncertainty.

    Parameters
    ----------
    subject_estimates : array_like, shape (n_subjects, n_coords)
        Posterior means of each subject's coordinates, ordered as
        :data:`POPULATION_COORDS`.
    subject_standard_errors : array_like, shape (n_subjects, n_coords)
        Posterior standard deviations of the same coordinates.
    """
    subject_estimates = jnp.asarray(subject_estimates)
    subject_standard_errors = jnp.asarray(subject_standard_errors)
    n_subjects, n_coords = subject_estimates.shape

    population_mean = numpyro.sample(
        "population_mean", dist.Normal(0.0, 2.0).expand([n_coords]).to_event(1)
    )
    population_scale = numpyro.sample(
        "population_scale", dist.HalfNormal(1.0).expand([n_coords]).to_event(1)
    )

    subject_true = numpyro.sample(
        "subject_true",
        dist.Normal(population_mean, population_scale).expand([n_subjects, n_coords]).to_event(2),
    )
    numpyro.sample(
        "subject_estimate",
        dist.Normal(subject_true, subject_standard_errors).to_event(2),
        obs=subject_estimates,
    )


def summarise_subject_posterior(samples):
    """Reduce one subject's posterior draws to the coordinates the population stage needs.

    Parameters
    ----------
    samples : dict of str to array_like
        Posterior draws from :func:`hattori2019_two_level`, containing ``mu_p`` and
        ``log_sigma``.

    Returns
    -------
    tuple of np.ndarray
        Posterior mean and standard deviation, each of shape ``(len(POPULATION_COORDS),)``
        and ordered as :data:`POPULATION_COORDS`.
    """
    stacked = np.concatenate(
        [np.asarray(samples["mu_p"]), np.asarray(samples["log_sigma"])], axis=-1
    )
    return stacked.mean(axis=0), stacked.std(axis=0)


def fit_subject(
    choice_history,
    reward_history,
    valid_mask=None,
    *,
    rng_key,
    num_warmup=500,
    num_samples=500,
    num_chains=1,
    progress_bar=False,
    **model_kwargs,
):
    """Fit the two-level model to one subject's sessions.

    Parameters
    ----------
    choice_history, reward_history : array_like, shape (n_sessions, n_trials)
        That subject's observed sessions, padded to a common length.
    valid_mask : array_like of bool, shape (n_sessions, n_trials), optional
        Trials to include.
    rng_key : jax.Array
        PRNG key for the sampler.
    num_warmup, num_samples, num_chains : int, optional
        NUTS settings.
    progress_bar : bool, optional
        Whether to display the sampler's progress bar.
    **model_kwargs
        Passed through to :func:`hattori2019_two_level`.

    Returns
    -------
    numpyro.infer.MCMC
        The completed sampler, from which draws and diagnostics can be read.
    """
    mcmc = MCMC(
        NUTS(hattori2019_two_level),
        num_warmup=num_warmup,
        num_samples=num_samples,
        num_chains=num_chains,
        progress_bar=progress_bar,
    )
    mcmc.run(rng_key, choice_history, reward_history, valid_mask, **model_kwargs)
    return mcmc


def fit_population(
    subject_estimates,
    subject_standard_errors,
    *,
    rng_key,
    num_warmup=500,
    num_samples=500,
    num_chains=1,
    progress_bar=False,
):
    """Fit the population stage to summarised subject posteriors.

    Parameters
    ----------
    subject_estimates, subject_standard_errors : array_like, shape (n_subjects, n_coords)
        Stacked output of :func:`summarise_subject_posterior`.
    rng_key : jax.Array
        PRNG key for the sampler.
    num_warmup, num_samples, num_chains : int, optional
        NUTS settings.
    progress_bar : bool, optional
        Whether to display the sampler's progress bar.

    Returns
    -------
    numpyro.infer.MCMC
        The completed sampler.
    """
    mcmc = MCMC(
        NUTS(population_model),
        num_warmup=num_warmup,
        num_samples=num_samples,
        num_chains=num_chains,
        progress_bar=progress_bar,
    )
    mcmc.run(rng_key, subject_estimates, subject_standard_errors)
    return mcmc


def fit_two_stage(
    subjects,
    *,
    rng_key,
    subject_kwargs=None,
    population_kwargs=None,
):
    """Run both stages: fit every subject, then fit the population over them.

    Parameters
    ----------
    subjects : sequence of tuple
        One ``(choice_history, reward_history)`` or ``(choice_history, reward_history,
        valid_mask)`` per subject.
    rng_key : jax.Array
        PRNG key, split across the subject fits and the population fit.
    subject_kwargs : dict, optional
        Extra keyword arguments for :func:`fit_subject`.
    population_kwargs : dict, optional
        Extra keyword arguments for :func:`fit_population`.

    Returns
    -------
    dict
        ``subject_mcmcs`` (one per subject), ``subject_estimates`` and
        ``subject_standard_errors`` (both ``(n_subjects, n_coords)``), and
        ``population_mcmc``.
    """
    subject_kwargs = dict(subject_kwargs or {})
    population_kwargs = dict(population_kwargs or {})

    keys = jax.random.split(rng_key, len(subjects) + 1)

    subject_mcmcs, estimates, standard_errors = [], [], []
    for subject_key, subject in zip(keys[:-1], subjects):
        choice_history, reward_history = subject[0], subject[1]
        valid_mask = subject[2] if len(subject) > 2 else None
        mcmc = fit_subject(
            choice_history,
            reward_history,
            valid_mask,
            rng_key=subject_key,
            **subject_kwargs,
        )
        mean, standard_error = summarise_subject_posterior(mcmc.get_samples())
        subject_mcmcs.append(mcmc)
        estimates.append(mean)
        standard_errors.append(standard_error)

    population_mcmc = fit_population(
        np.stack(estimates),
        np.stack(standard_errors),
        rng_key=keys[-1],
        **population_kwargs,
    )

    return {
        "subject_mcmcs": subject_mcmcs,
        "subject_estimates": np.stack(estimates),
        "subject_standard_errors": np.stack(standard_errors),
        "population_mcmc": population_mcmc,
    }
