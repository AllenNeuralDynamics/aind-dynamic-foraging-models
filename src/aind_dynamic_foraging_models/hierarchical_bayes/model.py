"""NumPyro hierarchical models for the foraging foragers.

The hierarchy follows the published cognitive model: subject-level hyperparameters govern
session-level parameters, which govern trials. Parameters are sampled on an unconstrained
scale and mapped into their bounded ranges by the standard normal CDF, so a standard normal
on the unconstrained scale is exactly a uniform prior on the bounded one.

Parameter names follow ``generative_model``; see the module docstring of
:mod:`~aind_dynamic_foraging_models.hierarchical_bayes.likelihood`.
"""

import jax
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
from jax.scipy.stats import norm

from .likelihood import hattori2019_log_likelihood

# Session-level parameters of the Hattori2019 forager, in the order they occupy the
# unconstrained parameter vector.
HATTORI2019_PARAMS = (
    "learn_rate_rew",
    "learn_rate_unrew",
    "forget_rate_unchosen",
    "softmax_inverse_temperature",
    "bias_l",
)


def _phi(x):
    """Standard normal CDF, mapping the unconstrained scale onto (0, 1)."""
    return norm.cdf(x)


def hattori2019_session_params(theta_unconstrained, beta_max=10.0):
    """Map unconstrained parameters onto the forager's bounded parameter ranges.

    A standard normal on the unconstrained scale becomes a uniform prior on the bounded
    range, which is how the published model expresses its "non-informative" priors.
    ``bias_l`` is unbounded and passes through untransformed.

    Parameters
    ----------
    theta_unconstrained : jnp.ndarray, shape (..., 5)
        Unconstrained parameters, ordered as :data:`HATTORI2019_PARAMS`.
    beta_max : float, optional
        Upper bound of ``softmax_inverse_temperature``. The published model uses 10.

    Returns
    -------
    dict of str to jnp.ndarray
        Bounded parameters, keyed by :data:`HATTORI2019_PARAMS`.
    """
    return {
        "learn_rate_rew": _phi(theta_unconstrained[..., 0]),
        "learn_rate_unrew": _phi(theta_unconstrained[..., 1]),
        "forget_rate_unchosen": _phi(theta_unconstrained[..., 2]),
        "softmax_inverse_temperature": _phi(theta_unconstrained[..., 3]) * beta_max,
        "bias_l": theta_unconstrained[..., 4],
    }


def _session_log_likelihoods(choice_history, reward_history, valid_mask, params):
    """Log likelihood of every session, vectorised over the session axis."""

    def _one(choices, rewards, mask, learn_rew, learn_unrew, forget, beta, bias):
        """Log likelihood of a single session."""
        return hattori2019_log_likelihood(
            choices,
            rewards,
            valid_mask=mask,
            learn_rate_rew=learn_rew,
            learn_rate_unrew=learn_unrew,
            forget_rate_unchosen=forget,
            softmax_inverse_temperature=beta,
            bias_l=bias,
        )

    return jax.vmap(_one)(
        choice_history,
        reward_history,
        valid_mask,
        params["learn_rate_rew"],
        params["learn_rate_unrew"],
        params["forget_rate_unchosen"],
        params["softmax_inverse_temperature"],
        params["bias_l"],
    )


def hattori2019_two_level(
    choice_history,
    reward_history,
    valid_mask=None,
    beta_max=10.0,
    log_sigma_loc=-1.0,
    log_sigma_scale=1.0,
    centered=False,
):
    """Two-level model for one subject: subject hyperparameters over session parameters.

    This is the published model's structure. Each session's parameters are drawn from that
    subject's distribution, non-centred so the sampler sees a well-conditioned geometry
    rather than the funnel the centred form produces.

    Parameters
    ----------
    choice_history : array_like of int, shape (n_sessions, n_trials)
        Observed actions per session, 0 for left and 1 for right. Sessions shorter than
        ``n_trials`` should be padded and masked out via ``valid_mask``.
    reward_history : array_like of float, shape (n_sessions, n_trials)
        Observed outcomes per session.
    valid_mask : array_like of bool, shape (n_sessions, n_trials), optional
        Trials to include. Defaults to all trials.
    beta_max : float, optional
        Upper bound of ``softmax_inverse_temperature``.
    log_sigma_loc, log_sigma_scale : float, optional
        Location and scale of the log-normal prior on the subject-level spread. The prior
        is placed on ``log sigma`` because that is the quantity the population level pools
        across subjects in the two-stage fit.
    centered : bool, optional
        Whether to sample session parameters directly from their subject distribution
        rather than as standardised offsets. The non-centred form (the default) avoids the
        funnel geometry that appears when the data constrain a session weakly; the centred
        form is often better sampled when each session carries many trials, so which one
        wins is an empirical question for the data at hand.
    """
    choice_history = jnp.asarray(choice_history, dtype=jnp.int32)
    reward_history = jnp.asarray(reward_history, dtype=jnp.float32)
    n_sessions = choice_history.shape[0]
    n_params = len(HATTORI2019_PARAMS)

    if valid_mask is None:
        valid_mask = jnp.ones_like(choice_history, dtype=bool)
    valid_mask = jnp.asarray(valid_mask, dtype=bool)

    # -- Subject-level hyperparameters, on the unconstrained scale --
    mu_p = numpyro.sample("mu_p", dist.Normal(0.0, 1.0).expand([n_params]).to_event(1))
    sigma = numpyro.sample(
        "sigma",
        dist.LogNormal(log_sigma_loc, log_sigma_scale).expand([n_params]).to_event(1),
    )
    numpyro.deterministic("log_sigma", jnp.log(sigma))

    # -- Session-level parameters --
    if centered:
        theta_unconstrained = numpyro.sample(
            "theta", dist.Normal(mu_p, sigma).expand([n_sessions, n_params]).to_event(2)
        )
    else:
        theta_raw = numpyro.sample(
            "theta_raw", dist.Normal(0.0, 1.0).expand([n_sessions, n_params]).to_event(2)
        )
        theta_unconstrained = mu_p + sigma * theta_raw

    params = hattori2019_session_params(theta_unconstrained, beta_max=beta_max)
    for name in HATTORI2019_PARAMS:
        numpyro.deterministic(name, params[name])

    # Subject-level means in bounded space, the quantity the published model reports.
    subject_means = hattori2019_session_params(mu_p, beta_max=beta_max)
    for name in HATTORI2019_PARAMS:
        numpyro.deterministic(f"subject_{name}", subject_means[name])

    log_lik = _session_log_likelihoods(choice_history, reward_history, valid_mask, params)
    numpyro.deterministic("session_log_lik", log_lik)
    numpyro.factor("likelihood", jnp.sum(log_lik))


# Number of parameters the published model pools across sessions: the two learning rates,
# the forgetting rate and the inverse temperature. Its side bias is deliberately excluded.
N_PUBLISHED_POOLED_PARAMS = 4


def hattori2019_published(
    choice_history,
    reward_history,
    valid_mask=None,
    beta_max=10.0,
    sigma_prior_scale=0.2,
    bias_prior_scale=20.0,
):
    """Faithful port of the published Stan model, for validating against it.

    Differs from :func:`hattori2019_two_level` in exactly two ways, both inherited from the
    reference implementation: the subject-level spread carries a half-Cauchy prior rather
    than a log-normal, and the side bias is a per-session parameter with a broad fixed prior
    rather than a pooled one.

    **On the forgetting rate.** The reference parameterises retention,
    ``aF = Phi(mu + sigma * raw)``, where ``aF = 1`` means no forgetting. This function
    parameterises decay, as the rest of the package does. The two are the same model: since
    ``1 - Phi(x) = Phi(-x)`` and the raw draws are symmetric, decay equals retention with
    the location negated. So comparing posteriors against the reference requires flipping
    the sign of that parameter's ``mu_p``; ``sigma`` is unchanged.

    Parameters
    ----------
    choice_history, reward_history : array_like, shape (n_sessions, n_trials)
        Observed sessions for one subject.
    valid_mask : array_like of bool, shape (n_sessions, n_trials), optional
        Trials to include. Defaults to all trials.
    beta_max : float, optional
        Upper bound of ``softmax_inverse_temperature``. The published model uses 10.
    sigma_prior_scale : float, optional
        Scale of the half-Cauchy prior on subject-level spread. The published model uses 0.2.
    bias_prior_scale : float, optional
        Scale of the per-session normal prior on the side bias. The reference uses 20, which
        is effectively flat on the logit scale.
    """
    choice_history = jnp.asarray(choice_history, dtype=jnp.int32)
    reward_history = jnp.asarray(reward_history, dtype=jnp.float32)
    n_sessions = choice_history.shape[0]

    if valid_mask is None:
        valid_mask = jnp.ones_like(choice_history, dtype=bool)
    valid_mask = jnp.asarray(valid_mask, dtype=bool)

    # -- Pooled subject-level hyperparameters (bias is excluded, as in the reference) --
    mu_p = numpyro.sample(
        "mu_p", dist.Normal(0.0, 1.0).expand([N_PUBLISHED_POOLED_PARAMS]).to_event(1)
    )
    sigma = numpyro.sample(
        "sigma",
        dist.HalfCauchy(sigma_prior_scale).expand([N_PUBLISHED_POOLED_PARAMS]).to_event(1),
    )
    theta_raw = numpyro.sample(
        "theta_raw",
        dist.Normal(0.0, 1.0).expand([n_sessions, N_PUBLISHED_POOLED_PARAMS]).to_event(2),
    )
    theta_unconstrained = mu_p + sigma * theta_raw

    # -- Unpooled per-session side bias --
    bias_l = numpyro.sample(
        "bias_l_raw", dist.Normal(0.0, bias_prior_scale).expand([n_sessions]).to_event(1)
    )

    params = {
        "learn_rate_rew": _phi(theta_unconstrained[..., 0]),
        "learn_rate_unrew": _phi(theta_unconstrained[..., 1]),
        "forget_rate_unchosen": _phi(theta_unconstrained[..., 2]),
        "softmax_inverse_temperature": _phi(theta_unconstrained[..., 3]) * beta_max,
        "bias_l": bias_l,
    }
    for name in HATTORI2019_PARAMS:
        numpyro.deterministic(name, params[name])

    subject_means = {
        "learn_rate_rew": _phi(mu_p[0]),
        "learn_rate_unrew": _phi(mu_p[1]),
        "forget_rate_unchosen": _phi(mu_p[2]),
        "softmax_inverse_temperature": _phi(mu_p[3]) * beta_max,
    }
    for name, value in subject_means.items():
        numpyro.deterministic(f"subject_{name}", value)

    log_lik = _session_log_likelihoods(choice_history, reward_history, valid_mask, params)
    numpyro.deterministic("session_log_lik", log_lik)
    numpyro.factor("likelihood", jnp.sum(log_lik))
