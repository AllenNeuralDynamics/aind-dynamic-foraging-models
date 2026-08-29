"""Held-out evaluation: condition a new subject on context, score disjoint sessions.

A held-out subject's parameters are inferred from *context* sessions only, with the
population level frozen at its training-set estimate. Its test sessions are then scored by
drawing fresh session-level parameters from the adapted subject distribution. Two properties
make this valid, and both are easy to lose by accident:

* The population level is never updated on held-out data. Refitting it would leak the
  held-out subject into the prior it is being scored against.
* A test session's own parameters are never inferred. They were not observed during
  adaptation, so they are drawn fresh; inferring them from the test choices would be fitting
  the targets.

Scores are log pointwise predictive densities: posterior draws collapse to one probability
per trial *before* the log. Averaging in log space instead is a smaller quantity by Jensen's
inequality and understates the model; plugging in a point estimate overstates it.
"""

import jax
import jax.numpy as jnp
import numpy as np
import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS

from .likelihood import hattori2019_choice_prob
from .model import HATTORI2019_PARAMS, hattori2019_session_params

POPULATION_SITES = (
    "population_mean",
    "population_scale",
    "log_sigma_mean",
    "log_sigma_spread",
)


def population_point_estimate(samples):
    """Reduce a fitted population posterior to the point estimate used for adaptation.

    Parameters
    ----------
    samples : dict of str to array_like
        Posterior draws containing :data:`POPULATION_SITES`.

    Returns
    -------
    dict of str to np.ndarray
        Posterior means, one vector per population site.
    """
    return {name: np.asarray(samples[name]).mean(axis=0) for name in POPULATION_SITES}


def adapt_subject(
    choice_history,
    reward_history,
    population,
    valid_mask=None,
    beta_max=10.0,
):
    """Condition one held-out subject on its context sessions, population frozen.

    With no context sessions the model carries no likelihood term, so the posterior is
    exactly the population predictive prior. That is the zero-shot case, and it needs no
    special handling.

    Parameters
    ----------
    choice_history, reward_history : array_like, shape (k, n_trials)
        The subject's ``k`` context sessions. ``k`` may be zero.
    population : mapping
        Point estimates for :data:`POPULATION_SITES`, from
        :func:`population_point_estimate`.
    valid_mask : array_like of bool, shape (k, n_trials), optional
        Trials to include.
    beta_max : float, optional
        Upper bound of ``softmax_inverse_temperature``.
    """
    n_params = len(HATTORI2019_PARAMS)
    # Zero context is represented by an empty leading axis, shape (0, n_trials).
    choice_history = jnp.asarray(choice_history, dtype=jnp.int32)
    reward_history = jnp.asarray(reward_history, dtype=jnp.float32)
    n_context = int(choice_history.shape[0])

    mu_raw = numpyro.sample(
        "mu_raw", dist.Normal(0.0, 1.0).expand([n_params]).to_event(1)
    )
    mu_p = numpyro.deterministic(
        "mu_p", population["population_mean"] + population["population_scale"] * mu_raw
    )

    log_sigma_raw = numpyro.sample(
        "log_sigma_raw", dist.Normal(0.0, 1.0).expand([n_params]).to_event(1)
    )
    log_sigma = numpyro.deterministic(
        "log_sigma",
        population["log_sigma_mean"] + population["log_sigma_spread"] * log_sigma_raw,
    )
    sigma = jnp.exp(log_sigma)

    if n_context == 0:  # zero-shot: posterior is the population predictive prior
        return

    if valid_mask is None:
        valid_mask = jnp.ones_like(choice_history, dtype=bool)
    valid_mask = jnp.asarray(valid_mask, dtype=bool)

    theta_raw = numpyro.sample(
        "theta_raw", dist.Normal(0.0, 1.0).expand([n_context, n_params]).to_event(2)
    )
    params = hattori2019_session_params(mu_p + sigma * theta_raw, beta_max=beta_max)

    from .model import _session_log_likelihoods

    log_lik = _session_log_likelihoods(choice_history, reward_history, valid_mask, params)
    numpyro.factor("context", jnp.sum(log_lik))


def fit_adaptation(
    context_choices,
    context_rewards,
    population,
    *,
    rng_key,
    num_warmup=500,
    num_samples=500,
    beta_max=10.0,
    progress_bar=False,
):
    """Sample the adapted subject posterior given context sessions.

    Returns
    -------
    dict of str to jnp.ndarray
        Posterior draws, including ``mu_p`` and ``log_sigma``.
    """
    mcmc = MCMC(
        NUTS(adapt_subject),
        num_warmup=num_warmup,
        num_samples=num_samples,
        num_chains=1,
        progress_bar=progress_bar,
    )
    mcmc.run(rng_key, context_choices, context_rewards, population, beta_max=beta_max)
    return mcmc.get_samples()


def posterior_predictive_choice_prob(
    adapted_samples,
    test_choices,
    test_rewards,
    *,
    rng_key,
    beta_max=10.0,
    n_draws=None,
):
    """Choice probabilities for a held-out session, marginalising the session latent.

    For each posterior draw of the subject's ``(mu_p, sigma)``, a **fresh** session
    parameter vector is drawn and the test session replayed. The per-trial probabilities are
    then averaged across draws in probability space.

    Parameters
    ----------
    adapted_samples : mapping
        Draws from :func:`fit_adaptation`.
    test_choices, test_rewards : array_like, shape (n_trials,)
        One held-out session.
    rng_key : jax.Array
        Key for the fresh session-level draws.
    beta_max : float, optional
        Upper bound of ``softmax_inverse_temperature``.
    n_draws : int, optional
        Subsample this many posterior draws. Defaults to all of them.

    Returns
    -------
    np.ndarray, shape (2, n_trials)
        Probability assigned to each action on each trial, averaged over draws.
    """
    mu_p = jnp.asarray(adapted_samples["mu_p"])
    sigma = jnp.exp(jnp.asarray(adapted_samples["log_sigma"]))
    if n_draws is not None and n_draws < mu_p.shape[0]:
        mu_p, sigma = mu_p[:n_draws], sigma[:n_draws]

    noise = jax.random.normal(rng_key, mu_p.shape)
    theta = mu_p + sigma * noise  # fresh session latent per draw
    params = hattori2019_session_params(theta, beta_max=beta_max)

    def _one(learn_rew, learn_unrew, forget, beta, bias):
        """Replay the test session under one posterior draw."""
        return hattori2019_choice_prob(
            test_choices, test_rewards,
            learn_rate_rew=learn_rew, learn_rate_unrew=learn_unrew,
            forget_rate_unchosen=forget, softmax_inverse_temperature=beta, bias_l=bias,
        )

    per_draw = jax.vmap(_one)(
        params["learn_rate_rew"], params["learn_rate_unrew"],
        params["forget_rate_unchosen"], params["softmax_inverse_temperature"],
        params["bias_l"],
    )
    return np.asarray(jnp.mean(per_draw, axis=0))  # probability space, before the log


def pointwise_log_predictive_density(choice_prob, choices, valid_mask=None):
    """Log pointwise predictive density of the observed choices.

    Parameters
    ----------
    choice_prob : array_like, shape (2, n_trials)
        Draw-averaged probabilities from :func:`posterior_predictive_choice_prob`.
    choices : array_like of int, shape (n_trials,)
        Observed actions.
    valid_mask : array_like of bool, shape (n_trials,), optional
        Trials to include.

    Returns
    -------
    tuple of (float, int)
        Summed log probability of the observed choices, and the number of trials scored.
    """
    choice_prob = np.asarray(choice_prob)
    choices = np.asarray(choices).astype(int)
    observed = choice_prob[choices, np.arange(len(choices))]
    observed = np.clip(observed, 1e-10, 1.0)
    if valid_mask is None:
        return float(np.sum(np.log(observed))), int(len(choices))
    valid_mask = np.asarray(valid_mask, dtype=bool)
    return float(np.sum(np.log(observed[valid_mask]))), int(valid_mask.sum())
