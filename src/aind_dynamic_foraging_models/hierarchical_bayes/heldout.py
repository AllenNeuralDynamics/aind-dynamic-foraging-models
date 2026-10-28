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

from .likelihood import hattori2019_choice_prob, hattori2019_value_trajectory
from .model import HATTORI2019_PARAMS, hattori2019_session_params

POPULATION_SITES = (
    "population_mean",
    "population_scale",
    "log_sigma_mean",
    "log_sigma_spread",
)

# The fixed hyperprior that turns :func:`adapt_subjects_batched` into
# :func:`~aind_dynamic_foraging_models.hierarchical_bayes.model.hattori2019_two_level` with its
# default priors: ``mu_p ~ Normal(0, 1)`` and ``log sigma ~ Normal(-1, 1)`` on every parameter.
# Passed where a population point estimate would go, it conditions each held-out subject on
# its own context sessions alone, with no population level and no training subjects.
FIXED_HYPERPRIOR = {
    "population_mean": np.zeros(len(HATTORI2019_PARAMS)),
    "population_scale": np.ones(len(HATTORI2019_PARAMS)),
    "log_sigma_mean": np.full(len(HATTORI2019_PARAMS), -1.0),
    "log_sigma_spread": np.ones(len(HATTORI2019_PARAMS)),
}


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


def adapt_subjects_batched(
    choice_history,
    reward_history,
    population,
    session_mask=None,
    valid_mask=None,
    beta_max=10.0,
    centered=False,
):
    """Condition many held-out subjects at once, as one batched model.

    Held-out subjects are independent given the frozen population, so scoring them one at a
    time pays the scan-depth cost once per subject on a device where extra lanes are nearly
    free. Batching pays it once for all of them.

    The approximation this makes is that a single sampler adapts one step size across every
    subject's block, rather than each subject getting its own. The blocks here are small and
    similarly shaped, so the cost should be modest -- but it is a real difference from
    sequential fitting and is worth measuring rather than assuming
    (see ``tests/test_hb_heldout_batched.py``).

    Parameters
    ----------
    choice_history, reward_history : array_like, shape (n_subjects, n_context, n_trials)
        Context sessions, padded on the session and trial axes.
    population : mapping
        Point estimates for :data:`POPULATION_SITES`.
    session_mask : array_like of bool, shape (n_subjects, n_context), optional
        Which context slots are real. Subjects with fewer context sessions pad here.
    valid_mask : array_like of bool, same shape as ``choice_history``, optional
        Trials to include.
    beta_max : float, optional
        Upper bound of ``softmax_inverse_temperature``.
    centered : bool, optional
        Sample the session-level parameters directly, ``theta ~ Normal(mu_p, sigma)``, rather
        than as standardised offsets (the default). The non-centred form avoids the funnel
        when the data constrain a session weakly; the centred form avoids the opposite
        geometry when they constrain it strongly, which is the refit for subjects whose
        non-centred fit diverged. ``mu_p`` and ``log_sigma`` keep their names either way.
    """
    from .model import _session_log_likelihoods

    choice_history = jnp.asarray(choice_history, dtype=jnp.int32)
    reward_history = jnp.asarray(reward_history, dtype=jnp.float32)
    n_subjects, n_context, n_trials = choice_history.shape
    n_params = len(HATTORI2019_PARAMS)

    if session_mask is None:
        session_mask = jnp.ones((n_subjects, n_context), dtype=bool)
    session_mask = jnp.asarray(session_mask, dtype=bool)
    if valid_mask is None:
        valid_mask = jnp.ones_like(choice_history, dtype=bool)
    valid_mask = jnp.asarray(valid_mask, dtype=bool)

    mu_raw = numpyro.sample(
        "mu_raw", dist.Normal(0.0, 1.0).expand([n_subjects, n_params]).to_event(2)
    )
    mu_p = numpyro.deterministic(
        "mu_p", population["population_mean"] + population["population_scale"] * mu_raw
    )
    log_sigma_raw = numpyro.sample(
        "log_sigma_raw", dist.Normal(0.0, 1.0).expand([n_subjects, n_params]).to_event(2)
    )
    log_sigma = numpyro.deterministic(
        "log_sigma",
        population["log_sigma_mean"] + population["log_sigma_spread"] * log_sigma_raw,
    )
    sigma = jnp.exp(log_sigma)

    if n_context == 0:  # zero-shot for every subject at once
        return

    if centered:
        theta = numpyro.sample(
            "theta",
            dist.Normal(mu_p[:, None, :], sigma[:, None, :])
            .expand([n_subjects, n_context, n_params]).to_event(3),
        )
    else:
        theta_raw = numpyro.sample(
            "theta_raw",
            dist.Normal(0.0, 1.0).expand([n_subjects, n_context, n_params]).to_event(3),
        )
        theta = mu_p[:, None, :] + sigma[:, None, :] * theta_raw
    params = hattori2019_session_params(theta, beta_max=beta_max)

    n_units = n_subjects * n_context
    flat = {name: value.reshape(n_units) for name, value in params.items()}
    log_lik = _session_log_likelihoods(
        choice_history.reshape(n_units, n_trials),
        reward_history.reshape(n_units, n_trials),
        valid_mask.reshape(n_units, n_trials),
        flat,
    ).reshape(n_subjects, n_context)

    numpyro.factor("context", jnp.sum(jnp.where(session_mask, log_lik, 0.0)))


def fit_adaptation_batched(
    context_choices,
    context_rewards,
    population,
    *,
    rng_key,
    session_mask=None,
    valid_mask=None,
    num_warmup=500,
    num_samples=500,
    beta_max=10.0,
    progress_bar=False,
):
    """Sample adapted posteriors for many held-out subjects in one run.

    Returns
    -------
    dict of str to jnp.ndarray
        Draws with a leading subject axis: ``mu_p`` and ``log_sigma`` are
        ``(n_draws, n_subjects, n_params)``.
    """
    mcmc = _run_adaptation(
        context_choices, context_rewards, population, rng_key=rng_key,
        session_mask=session_mask, valid_mask=valid_mask, num_warmup=num_warmup,
        num_samples=num_samples, num_chains=1, beta_max=beta_max, progress_bar=progress_bar,
    )
    return mcmc.get_samples()


def fit_heldout_two_level(
    context_choices,
    context_rewards,
    *,
    rng_key,
    session_mask=None,
    valid_mask=None,
    num_warmup=500,
    num_samples=500,
    num_chains=4,
    beta_max=10.0,
    progress_bar=False,
    centered=False,
    target_accept_prob=0.8,
    max_tree_depth=10,
):
    """Fit a two-level model to each held-out subject's context sessions, with diagnostics.

    Each subject gets subject-level hyperparameters over session-level parameters under
    :data:`FIXED_HYPERPRIOR`, with no population level: the model is
    :func:`~aind_dynamic_foraging_models.hierarchical_bayes.model.hattori2019_two_level`,
    batched across subjects. Subjects are independent, so one batched sampler fits them all.

    Several chains are run so that each subject can be checked on its own. The sampler is
    shared, so a divergence belongs to the joint batch and cannot be attributed to one
    subject; R-hat and ESS can, and are returned per subject.

    Parameters
    ----------
    context_choices, context_rewards : array_like, shape (n_subjects, n_context, n_trials)
        Context sessions, padded on the session and trial axes.
    rng_key : jax.Array
        Sampler key.
    session_mask, valid_mask : array_like of bool, optional
        As for :func:`adapt_subjects_batched`.
    num_warmup, num_samples : int, optional
        Per chain.
    num_chains : int, optional
        Chains, run vectorised on one device.
    beta_max : float, optional
        Upper bound of ``softmax_inverse_temperature``.
    progress_bar : bool, optional
        Show NumPyro's progress bar.
    centered : bool, optional
        Centred session level; see :func:`adapt_subjects_batched`.
    target_accept_prob, max_tree_depth : optional
        NUTS settings. A higher target acceptance takes smaller steps, the usual remedy for
        divergences; the defaults are NumPyro's.

    Returns
    -------
    samples : dict of str to jnp.ndarray
        Draws pooled over chains: ``mu_p`` and ``log_sigma`` are
        ``(num_chains * num_samples, n_subjects, n_params)``, as
        :func:`batched_heldout_log_lik` takes them.
    diagnostics : dict
        ``max_r_hat`` and ``min_ess_bulk``, each shape ``(n_subjects,)``, the worst value over
        that subject's ``mu_p`` and ``log_sigma``; and ``divergences``, the joint count.
    """
    mcmc = _run_adaptation(
        context_choices, context_rewards, FIXED_HYPERPRIOR, rng_key=rng_key,
        session_mask=session_mask, valid_mask=valid_mask, num_warmup=num_warmup,
        num_samples=num_samples, num_chains=num_chains, beta_max=beta_max,
        progress_bar=progress_bar, centered=centered, target_accept_prob=target_accept_prob,
        max_tree_depth=max_tree_depth,
    )
    return mcmc.get_samples(), subject_diagnostics(mcmc)


def subject_diagnostics(mcmc):
    """Per-subject worst-case R-hat and bulk ESS over ``mu_p`` and ``log_sigma``.

    R-hat is ArviZ's rank-normalised split R-hat, which takes the worse of the bulk value and
    the folded (tail) value, so chains that agree in location but not in scale are flagged
    too. ESS is ArviZ's bulk ESS. Both match what ``artifacts.summarise_diagnostics`` reports
    for the population sites and what study 08 gated on.

    Parameters
    ----------
    mcmc : numpyro.infer.MCMC
        A finished run of :func:`adapt_subjects_batched` with at least two chains.

    Returns
    -------
    dict
        ``max_r_hat`` and ``min_ess_bulk`` of shape ``(n_subjects,)``, and ``divergences``.
    """
    import arviz as az

    grouped = mcmc.get_samples(group_by_chain=True)
    r_hat, ess = [], []
    for site in ("mu_p", "log_sigma"):
        draws = np.asarray(grouped[site])            # (chains, draws, subjects, params)
        r_hat.append(np.asarray(az.rhat(draws, method="rank", chain_axis=0, draw_axis=1)))
        ess.append(np.asarray(az.ess(draws, method="bulk", chain_axis=0, draw_axis=1)))
    diverging = mcmc.get_extra_fields().get("diverging")
    return {
        "max_r_hat": np.max(np.concatenate(r_hat, axis=-1), axis=-1),
        "min_ess_bulk": np.min(np.concatenate(ess, axis=-1), axis=-1),
        "divergences": int(np.sum(np.asarray(diverging))) if diverging is not None else None,
    }


def _run_adaptation(
    context_choices, context_rewards, population, *, rng_key, session_mask, valid_mask,
    num_warmup, num_samples, num_chains, beta_max, progress_bar, centered=False,
    target_accept_prob=0.8, max_tree_depth=10,
):
    """Run NUTS on :func:`adapt_subjects_batched`; chains are vectorised on one device."""
    mcmc = MCMC(
        NUTS(adapt_subjects_batched, target_accept_prob=target_accept_prob,
             max_tree_depth=max_tree_depth),
        num_warmup=num_warmup,
        num_samples=num_samples,
        num_chains=num_chains,
        chain_method="vectorized",
        progress_bar=progress_bar,
    )
    mcmc.run(
        rng_key, context_choices, context_rewards, population,
        session_mask, valid_mask, beta_max=beta_max, centered=centered,
        extra_fields=("diverging",),
    )
    return mcmc


def batched_choice_prob(
    batched_samples,
    subject_index,
    test_choices,
    test_rewards,
    *,
    rng_key,
    beta_max=10.0,
):
    """Posterior-predictive choice probabilities for one subject out of a batched fit.

    Parameters
    ----------
    batched_samples : mapping
        Draws from :func:`fit_adaptation_batched`.
    subject_index : int
        Which subject's slice to score.
    test_choices, test_rewards : array_like, shape (n_trials,)
        One held-out session for that subject.
    rng_key : jax.Array
        Key for the fresh session-level draws.
    beta_max : float, optional
        Upper bound of ``softmax_inverse_temperature``.

    Returns
    -------
    np.ndarray, shape (2, n_trials)
        Draw-averaged probabilities, as :func:`posterior_predictive_choice_prob`.
    """
    single = {
        "mu_p": jnp.asarray(batched_samples["mu_p"])[:, subject_index, :],
        "log_sigma": jnp.asarray(batched_samples["log_sigma"])[:, subject_index, :],
    }
    return posterior_predictive_choice_prob(
        single, test_choices, test_rewards, rng_key=rng_key, beta_max=beta_max
    )


def auto_session_chunk(n_trials, n_draws, memory_fraction=0.25, floor=8, ceiling=4096):
    """Choose how many sessions to score per pass, from the device's actual memory.

    A fixed chunk size is wrong in both directions: it wastes a large GPU and can exhaust a
    small one, since the working set scales with draws and trial count as well as sessions.

    On CPU this deliberately returns a small chunk. Widening the batch is a win only while
    the device is latency-bound; on CPU the same sweep that is flat on an A100 measured
    *worse* than linear (32x the lanes for 102x the time), so batching there is
    counterproductive and the chunk stays small.

    Parameters
    ----------
    n_trials : int
        Padded trial count per session.
    n_draws : int
        Posterior draws used for scoring.
    memory_fraction : float, optional
        Share of the device's memory limit to spend on one pass.
    floor, ceiling : int, optional
        Bounds on the returned chunk size.

    Returns
    -------
    int
        Sessions per vmapped pass.
    """
    import jax

    device = jax.devices()[0]
    stats = None
    try:
        stats = device.memory_stats()
    except Exception:  # pragma: no cover - platform dependent
        stats = None

    if not stats or not stats.get("bytes_limit"):
        # CPU or an unknown device: stay small, see the docstring. Still respect the
        # caller's bounds -- an explicit ceiling means they know something we do not.
        return int(min(ceiling, max(floor, floor * 4)))

    # Per session: the per-draw choice probabilities plus the intermediates the vmapped
    # scan holds alive. Six float32 copies is deliberately conservative.
    per_session = max(1, n_draws * n_trials * 4 * 6)
    budget = float(stats["bytes_limit"]) * memory_fraction
    return int(min(ceiling, max(floor, budget // per_session)))


def batched_heldout_log_lik(
    batched_samples,
    subject_indices,
    choices,
    rewards,
    valid_mask=None,
    *,
    rng_key,
    beta_max=10.0,
    n_draws=None,
    session_chunk=None,
):
    """Score many held-out sessions at once, one vmapped pass per chunk.

    Batching the adaptation fits removed one bottleneck and exposed the next: replaying each
    held-out session in its own call means thousands of small dispatches, which on a
    latency-bound device costs far more than the arithmetic does. Session replays are
    independent, so they vectorise the same way everything else here does.

    Draws are reduced inside each chunk rather than materialised, since keeping every
    per-trial probability for every draw and session at once would run to gigabytes.

    Parameters
    ----------
    batched_samples : mapping
        Draws from :func:`fit_adaptation_batched`, with a subject axis.
    subject_indices : array_like of int, shape (n_sessions,)
        Which subject each session belongs to, indexing into the batched fit.
    choices, rewards : array_like, shape (n_sessions, n_trials)
        Sessions to score, padded on the trial axis.
    valid_mask : array_like of bool, same shape, optional
        Trials to include. Defaults to all trials.
    rng_key : jax.Array
        Key for the fresh session-level draws.
    beta_max : float, optional
        Upper bound of ``softmax_inverse_temperature``.
    n_draws : int, optional
        Subsample this many posterior draws. The posterior-predictive average converges
        quickly, so fewer draws here trade a little noise for a lot of time.
    session_chunk : int, optional
        Sessions per vmapped pass, which caps peak memory. Defaults to a size derived from
        the device's own memory limit, so the same code neither wastes a large GPU nor
        exhausts a small one.

    Returns
    -------
    tuple of np.ndarray
        Per-session summed log likelihood and trial count, both shape ``(n_sessions,)``.
    """
    choices = jnp.asarray(choices, dtype=jnp.int32)
    rewards = jnp.asarray(rewards, dtype=jnp.float32)
    subject_indices = jnp.asarray(subject_indices, dtype=jnp.int32)
    n_sessions, n_trials = choices.shape
    if valid_mask is None:
        valid_mask = jnp.ones_like(choices, dtype=bool)
    valid_mask = jnp.asarray(valid_mask, dtype=bool)

    mu_p = jnp.asarray(batched_samples["mu_p"])
    sigma = jnp.exp(jnp.asarray(batched_samples["log_sigma"]))
    if n_draws is not None and n_draws < mu_p.shape[0]:
        mu_p, sigma = mu_p[:n_draws], sigma[:n_draws]

    if session_chunk is None:
        session_chunk = auto_session_chunk(n_trials, int(mu_p.shape[0]))

    if session_chunk is None:
        session_chunk = auto_session_chunk(n_trials, int(mu_p.shape[0]))

    trial_index = jnp.arange(n_trials)

    def _observed_prob(theta, session_choices, session_rewards):
        """Probability the model assigned to the observed action, per trial."""
        params = hattori2019_session_params(theta, beta_max=beta_max)
        prob = hattori2019_choice_prob(
            session_choices, session_rewards,
            learn_rate_rew=params["learn_rate_rew"],
            learn_rate_unrew=params["learn_rate_unrew"],
            forget_rate_unchosen=params["forget_rate_unchosen"],
            softmax_inverse_temperature=params["softmax_inverse_temperature"],
            bias_l=params["bias_l"],
        )
        return prob[session_choices, trial_index]

    over_sessions = jax.vmap(_observed_prob, in_axes=(0, 0, 0))
    over_draws = jax.jit(jax.vmap(over_sessions, in_axes=(0, None, None)))

    log_lik = np.zeros(n_sessions)
    counts = np.zeros(n_sessions, dtype=int)

    for start in range(0, n_sessions, session_chunk):
        stop = min(start + session_chunk, n_sessions)
        idx = subject_indices[start:stop]
        chunk_key, rng_key = jax.random.split(rng_key)

        mu_chunk = mu_p[:, idx, :]          # (draws, chunk, params)
        sigma_chunk = sigma[:, idx, :]
        theta = mu_chunk + sigma_chunk * jax.random.normal(chunk_key, mu_chunk.shape)

        probs = over_draws(theta, choices[start:stop], rewards[start:stop])
        averaged = jnp.mean(probs, axis=0)   # probability space, before the log
        averaged = jnp.clip(averaged, 1e-10, 1.0)

        mask = valid_mask[start:stop]
        log_lik[start:stop] = np.asarray(
            jnp.sum(jnp.where(mask, jnp.log(averaged), 0.0), axis=-1)
        )
        counts[start:stop] = np.asarray(jnp.sum(mask, axis=-1))

    return log_lik, counts


def batched_heldout_scores(
    batched_samples,
    subject_indices,
    choices,
    rewards,
    valid_mask=None,
    *,
    rng_key,
    beta_max=10.0,
    n_draws=None,
    n_theta_per_draw=1,
    session_chunk=None,
):
    """Score held-out sessions by both marginals, from one shared set of session draws.

    A held-out session's session-level parameters are never fitted, so each score averages
    over them: for every posterior draw of the subject's ``(mu_p, sigma)``, fresh session
    parameters are drawn and the session replayed. The two scores differ only in where the
    average is taken.

    - **Trial-marginal**: average each trial's choice probability over draws, then sum the
      logs. Every trial may in effect use a different session's parameters, so this is not
      the predictive probability of any session the model could generate.
    - **Session-marginal**: average the whole session's likelihood over draws
      (``logsumexp`` minus ``log`` of the draw count). This is the posterior predictive
      probability of the session, and equals predicting trial by trial while updating belief
      about the session's parameters as its trials arrive.

    The session-marginal is a Monte Carlo estimate dominated, on long sessions, by the few
    draws that fit best; it is biased low when too few draws land there. Check
    ``weight_ess`` before trusting it: an ESS near 1 means the estimate rests on one draw and
    ``session_marginal_se`` is itself unreliable. Even at a moderate ESS the delta-method
    error runs about 20% low, and it accounts for the posterior draws' autocorrelation but
    not for a posterior that has not converged.

    Parameters
    ----------
    batched_samples : mapping
        Draws from :func:`fit_adaptation_batched`, with a subject axis.
    subject_indices : array_like of int, shape (n_sessions,)
        Which subject each session belongs to, indexing into the batched fit.
    choices, rewards : array_like, shape (n_sessions, n_trials)
        Sessions to score, padded on the trial axis.
    valid_mask : array_like of bool, same shape, optional
        Trials to include. Defaults to all trials.
    rng_key : jax.Array
        Key for the fresh session-level draws.
    beta_max : float, optional
        Upper bound of ``softmax_inverse_temperature``.
    n_draws : int, optional
        Subsample this many posterior draws. Defaults to all of them.
    n_theta_per_draw : int, optional
        Session-parameter draws per posterior draw. Most of the session-marginal's variance
        is in this draw, so raising it buys accuracy at linear cost without refitting. With
        the default of 1 the draws, and so both scores, match :func:`batched_heldout_log_lik`.
    session_chunk : int, optional
        Sessions per vmapped pass, which caps peak memory. Defaults to a size derived from
        the device's own memory limit.

    Returns
    -------
    dict of np.ndarray, each shape ``(n_sessions,)``
        ``trial_marginal`` and ``session_marginal`` (summed log likelihoods),
        ``session_marginal_se`` (delta-method standard error of the latter), ``weight_ess``
        (effective number of draws behind it), and ``n_trials`` (trials scored); and
        ``relative_weights``, shape ``(n_posterior_draws, n_sessions)``: each posterior
        draw's session likelihood relative to the session's mean, which
        :func:`pooled_likelihood` needs to pool errors that share posterior draws; and
        ``pareto_k``, the PSIS Pareto k-hat of each session's draw weights, a reliability
        diagnostic only (the scores are not smoothed). See :func:`pareto_k_threshold`.
    """
    choices = jnp.asarray(choices, dtype=jnp.int32)
    rewards = jnp.asarray(rewards, dtype=jnp.float32)
    subject_indices = jnp.asarray(subject_indices, dtype=jnp.int32)
    n_sessions, n_trials = choices.shape
    if valid_mask is None:
        valid_mask = jnp.ones_like(choices, dtype=bool)
    valid_mask = jnp.asarray(valid_mask, dtype=bool)

    mu_p = jnp.asarray(batched_samples["mu_p"])
    sigma = jnp.exp(jnp.asarray(batched_samples["log_sigma"]))
    if n_draws is not None and n_draws < mu_p.shape[0]:
        mu_p, sigma = mu_p[:n_draws], sigma[:n_draws]
    if session_chunk is None:
        session_chunk = auto_session_chunk(n_trials, int(mu_p.shape[0]) * n_theta_per_draw)

    over_draws = _observed_prob_over_draws(n_trials, beta_max)
    names = ("trial_marginal", "session_marginal", "session_marginal_se", "weight_ess",
             "pareto_k")
    out = {name: np.zeros(n_sessions) for name in names}
    out["n_trials"] = np.zeros(n_sessions, dtype=int)
    out["relative_weights"] = np.zeros((int(mu_p.shape[0]), n_sessions))

    for start in range(0, n_sessions, session_chunk):
        stop = min(start + session_chunk, n_sessions)
        idx = subject_indices[start:stop]
        chunk_key, rng_key = jax.random.split(rng_key)
        theta = _draw_session_theta(chunk_key, mu_p[:, idx, :], sigma[:, idx, :],
                                    n_theta_per_draw)
        # Both (draws, chunk, trials): the legacy probability and the exact log probability.
        probs, log_probs = over_draws(theta, choices[start:stop], rewards[start:stop])
        mask = valid_mask[start:stop]

        # Average, then clip, exactly as the trial-marginal scorer always has.
        averaged = jnp.clip(jnp.mean(probs, axis=0), 1e-10, 1.0)
        trial = jnp.sum(jnp.where(mask, jnp.log(averaged), 0.0), axis=-1)
        # Unclipped: a confidently wrong draw must count at its true, very low likelihood.
        per_draw = jnp.sum(jnp.where(mask, log_probs, 0.0), axis=-1)
        session, se, ess, relative = _session_marginal(per_draw, n_theta_per_draw)

        out["trial_marginal"][start:stop] = np.asarray(trial)
        out["session_marginal"][start:stop] = np.asarray(session)
        out["session_marginal_se"][start:stop] = np.asarray(se)
        out["weight_ess"][start:stop] = np.asarray(ess)
        out["pareto_k"][start:stop] = _pareto_k(np.asarray(per_draw))
        out["n_trials"][start:stop] = np.asarray(jnp.sum(mask, axis=-1))
        out["relative_weights"][:, start:stop] = np.asarray(relative)

    return out


def _observed_prob_over_draws(n_trials, beta_max):
    """Jitted replay of the observed action's probability, per draw, session and trial.

    One replay (:func:`hattori2019_value_trajectory`) gives the pre-update Q values, and both
    returned arrays come from the same logits, ``beta * Q + [bias_l, 0]``: the probability
    by softmax, as :func:`hattori2019_choice_prob` computes it, and the log probability by
    log-softmax, which stays exact where the probability underflows. The session-marginal
    sums the latter; a floor on the probability would cap how badly a confidently wrong draw
    can score.
    """
    trial_index = jnp.arange(n_trials)

    def _observed_prob(theta, session_choices, session_rewards):
        """Probability, and exact log probability, of the observed action per trial."""
        params = hattori2019_session_params(theta, beta_max=beta_max)
        q_values, _ = hattori2019_value_trajectory(
            session_choices, session_rewards,
            learn_rate_rew=params["learn_rate_rew"],
            learn_rate_unrew=params["learn_rate_unrew"],
            forget_rate_unchosen=params["forget_rate_unchosen"],
            softmax_inverse_temperature=params["softmax_inverse_temperature"],
            bias_l=params["bias_l"],
        )
        logits = (params["softmax_inverse_temperature"] * q_values.T
                  + jnp.array([params["bias_l"], 0.0]))          # (n_trials, actions)
        observed = (trial_index, session_choices)
        return (jax.nn.softmax(logits, axis=-1)[observed],
                jax.nn.log_softmax(logits, axis=-1)[observed])

    over_sessions = jax.vmap(_observed_prob, in_axes=(0, 0, 0))
    return jax.jit(jax.vmap(over_sessions, in_axes=(0, None, None)))


def _draw_session_theta(key, mu_p, sigma, n_theta_per_draw):
    """Fresh session parameters, ``n_theta_per_draw`` per posterior draw, flattened.

    With one per draw the noise has exactly the posterior's shape, so the stream matches the
    trial-marginal scorer's from before both scores were computed together.
    """
    if n_theta_per_draw == 1:
        return mu_p + sigma * jax.random.normal(key, mu_p.shape)
    noise = jax.random.normal(key, (n_theta_per_draw,) + mu_p.shape)
    theta = mu_p[None] + sigma[None] * noise
    return theta.reshape((-1,) + mu_p.shape[1:])


def _session_marginal(per_draw, n_theta_per_draw=1):
    """Log-mean-exp over draws, its delta-method standard error, the weights' ESS, and each
    posterior draw's weight relative to the mean.

    ``per_draw`` holds each draw's summed session log likelihood, shape
    ``(n_theta_per_draw * n_posterior, sessions)``, ordered as :func:`_draw_session_theta`
    flattens them. Two dependencies shape the standard error. Replicates of one posterior
    draw share its ``(mu_p, sigma)``, so each posterior draw's replicates are averaged first.
    Posterior draws come from MCMC and are autocorrelated, so the error divides by their
    autocorrelation-aware ESS rather than their count. ``weight_ess`` is a different number:
    it counts how many draws the estimate rests on, by the spread of their weights.
    """
    top = jnp.max(per_draw, axis=0)
    weights = jnp.exp(per_draw - top)          # in (0, 1], the best draw at 1
    mean_w = jnp.mean(weights, axis=0)
    session = top + jnp.log(mean_w)
    by_posterior = jnp.mean(weights.reshape((n_theta_per_draw, -1) + weights.shape[1:]), axis=0)
    relative = by_posterior / mean_w          # (posterior draws, sessions), mean 1
    se = _autocorrelated_se(relative)
    ess = jnp.sum(weights, axis=0) ** 2 / jnp.sum(weights ** 2, axis=0)
    return session, se, ess, relative


def _autocorrelated_se(series):
    """Standard error of the mean of an autocorrelated series, ``sd / sqrt(ESS)``.

    ``series`` has the draw axis first, in sampling order. The ESS is NumPyro's
    autocorrelation estimate for a single chain, so a slowly mixing posterior widens the
    error instead of hiding behind its draw count. With fewer than four draws there is
    nothing to estimate from, and the error is reported as infinite rather than invented.
    """
    from numpyro.diagnostics import effective_sample_size

    n = series.shape[0]
    if n < 4:
        return jnp.full(series.shape[1:], jnp.inf)
    ess = np.asarray(effective_sample_size(np.asarray(series)[None]))
    ess = np.clip(np.nan_to_num(ess, nan=1.0), 1.0, n)
    return jnp.std(series, axis=0, ddof=1) / jnp.sqrt(ess)


def _pareto_k(per_draw):
    """PSIS Pareto k-hat of each session's draw weights; ``per_draw`` is (draws, sessions).

    Uses ArviZ's ``pareto_khat`` on the weights scaled by their maximum, one session at a
    time. Checked on log-normal weights (arviz-stats 1.3.1): on the weights k-hat rises with
    the log-sd (0.06, 0.30, 0.50, 0.99, 1.68 at sd 0.5 to 5) as ESS falls from 15657 to 4 of
    20000, while ``log_weights=True`` stayed near zero throughout and ``psislw(axis=0)`` gave
    ~1 for bounded weights, so neither of those is used.
    Identical weights cannot be heavy-tailed and get ``-inf``. Fewer than 25 draws, or a
    tail ArviZ cannot fit, get ``nan``: unjudged, which a gate must treat as failing.
    """
    from arviz_stats.base.array import array_stats

    khat = np.full(per_draw.shape[1], np.nan)
    for session in range(per_draw.shape[1]):
        log_w = per_draw[:, session].astype(np.float64)   # float32 exp underflows
        if log_w.shape[0] < 25:
            continue                                      # too few to judge: nan
        if np.ptp(log_w) < 1e-12:
            khat[session] = -np.inf
        else:
            try:
                khat[session] = float(array_stats.pareto_khat(
                    np.exp(log_w - log_w.max()), chain_axis=None, draw_axis=-1,
                    tail="right"))   # only large weights matter; tiny ones underflow
            except ValueError:  # a tail ArviZ cannot fit is unjudged, not safe
                pass
    return khat


def pareto_k_threshold(n_draws):
    """Largest Pareto k-hat at which a session-marginal estimate is trusted.

    ``min(1 - 1 / log10(n_draws), 0.7)`` (Vehtari et al. 2024, "Pareto smoothed importance
    sampling", JMLR). Above it the estimate rests on too few draws to be reliable, however
    small its reported error. ``n_draws`` counts every session draw behind one estimate.
    """
    return float(min(1.0 - 1.0 / np.log10(n_draws), 0.7))


def pooled_likelihood(log_lik, n_trials, relative_weights=None):
    """Trial-weighted pooled likelihood, ``exp(total log likelihood / total trials)``.

    Sessions scored from the same fit share its posterior draws, so their Monte Carlo errors
    are correlated, and per-session errors must not be added as if independent. The pooled
    error is instead taken jointly: by the delta method, the total log likelihood moves with
    ``sum over sessions of relative_weights[d, s]`` for each posterior draw ``d``, and the
    error of that per-draw sum's mean accounts for every correlation across sessions and,
    by its autocorrelation-aware ESS, along the chain.

    Parameters
    ----------
    log_lik, n_trials : array_like, shape (n_sessions,)
        Per-session summed log likelihood and trial count.
    relative_weights : array_like, shape (n_posterior_draws, n_sessions), optional
        From :func:`batched_heldout_scores`, for the session-marginal. Leave out for the
        trial-marginal, whose Monte Carlo error is not estimated.

    Returns
    -------
    tuple of (float, float or None)
        The pooled likelihood and its standard error (``None`` without ``relative_weights``).
    """
    total, count = float(np.sum(log_lik)), int(np.sum(n_trials))
    pooled = float(np.exp(total / count))
    if relative_weights is None:
        return pooled, None
    per_draw_total = np.sum(np.asarray(relative_weights), axis=1, keepdims=True)
    total_se = float(np.asarray(_autocorrelated_se(per_draw_total))[0])
    return pooled, pooled * total_se / count


def importance_sampled_session_marginal(
    mu_p,
    log_sigma,
    choices,
    rewards,
    valid_mask=None,
    *,
    rng_key,
    beta_max=10.0,
    n_proposal=24000,
    prior_fraction=0.2,
    df=4.0,
    scale_inflation=1.5,
    chunk=2000,
):
    """Session-marginal likelihood of one subject's sessions by importance sampling.

    Estimates the same quantity as :func:`batched_heldout_scores`' ``session_marginal``,

    ``p(y | context) = integral of p(y | theta) p(theta | context) d theta``, where
    ``p(theta | context) = (1 / D) sum_d Normal(theta; mu_p[d], sigma[d])``

    is the predictive distribution of a fresh session's parameters, a mixture over the ``D``
    posterior draws whose density is exact. The plain scorer samples ``theta`` from that
    mixture, so on a long session a few draws carry the weight (high Pareto k-hat). Here
    ``theta`` is drawn from a proposal centred on the session itself: a multivariate t
    (``df`` degrees of freedom) at the session's mode under likelihood times mixture,
    with the inverse Hessian there, inflated by ``scale_inflation``, as its scale. It is
    mixed defensively with the mixture itself (at least ``prior_fraction`` of the draws), so
    every weight is at most ``p(y | theta) / prior_fraction``: bounded, whatever the mode
    search finds. The two components get fixed draw counts (a stratified sample), and the
    standard error is computed within each. Both estimators are unbiased for the same integral; this one has lower variance
    where the session pins ``theta`` down.

    Parameters
    ----------
    mu_p, log_sigma : array_like, shape (n_posterior_draws, n_params)
        One subject's posterior draws, unconstrained scale.
    choices, rewards : array_like, shape (n_sessions, n_trials)
        That subject's sessions to score, padded on the trial axis.
    valid_mask : array_like of bool, same shape, optional
        Trials to include. Defaults to all trials.
    rng_key : jax.Array
        Key for the proposal draws.
    beta_max : float, optional
        Upper bound of ``softmax_inverse_temperature``.
    n_proposal : int, optional
        Proposal draws per session.
    prior_fraction, df, scale_inflation : float, optional
        Proposal shape, as above.
    chunk : int, optional
        Proposal draws per vectorised pass, which caps memory at ``chunk * n_posterior_draws``.

    Returns
    -------
    dict of np.ndarray, each shape ``(n_sessions,)``
        ``session_marginal`` (summed log likelihood), ``session_marginal_se`` (delta-method
        standard error of the stratified estimate, conditional on the posterior draws), ``weight_ess``, ``pareto_k`` (of the
        importance weights, judged as in :func:`batched_heldout_scores`), ``n_trials``, and
        ``mode_found`` (whether the Hessian at the mode was positive definite; if not, the
        proposal falls back to the mixture's own mean and covariance).
    """
    from jax.scipy.optimize import minimize
    from jax.scipy.special import gammaln, logsumexp

    mu_p = jnp.asarray(mu_p, dtype=jnp.float32)
    sigma = jnp.exp(jnp.asarray(log_sigma, dtype=jnp.float32))
    choices = jnp.asarray(choices, dtype=jnp.int32)
    rewards = jnp.asarray(rewards, dtype=jnp.float32)
    n_sessions, n_trials = choices.shape
    if valid_mask is None:
        valid_mask = jnp.ones_like(choices, dtype=bool)
    valid_mask = jnp.asarray(valid_mask, dtype=bool)
    n_draws, n_params = mu_p.shape
    over_draws = _observed_prob_over_draws(n_trials, beta_max)

    @jax.jit
    def log_mixture(theta):                                   # (m, p) -> (m,)
        z = (theta[:, None, :] - mu_p[None]) / sigma[None]
        log_n = -0.5 * jnp.sum(z ** 2 + jnp.log(2 * jnp.pi), -1) - jnp.sum(jnp.log(sigma), -1)
        return logsumexp(log_n, axis=1) - jnp.log(n_draws)

    def log_lik(theta, c, r, m):                              # (k, p) -> (k,)
        _, log_probs = over_draws(theta[:, None, :], c[None], r[None])
        return jnp.sum(jnp.where(m[None], log_probs[:, 0, :], 0.0), axis=-1)

    def chunked(fn, theta):
        return jnp.concatenate([fn(theta[i:i + chunk]) for i in range(0, theta.shape[0], chunk)])

    def mixture_sample(key, n):
        k_pick, k_z = jax.random.split(key)
        pick = jax.random.randint(k_pick, (n,), 0, n_draws)
        return mu_p[pick] + sigma[pick] * jax.random.normal(k_z, (n, n_params))

    # The mixture's own moments: the fallback scale when the session has no clean mode.
    mix_mean = jnp.mean(mu_p, axis=0)
    mix_cov = jnp.cov(mu_p, rowvar=False) + jnp.diag(jnp.mean(sigma ** 2, axis=0))

    names = ("session_marginal", "session_marginal_se", "weight_ess", "pareto_k")
    out = {name: np.zeros(n_sessions) for name in names}
    out["n_trials"] = np.asarray(jnp.sum(valid_mask, axis=-1)).astype(int)
    out["mode_found"] = np.zeros(n_sessions, dtype=bool)
    # A fixed allocation (stratified), at least the requested defensive share; the proposal
    # density uses the realised share, so it matches the draws exactly.
    n_t = int(np.floor(n_proposal * (1.0 - prior_fraction)))
    n_prior = n_proposal - n_t
    realised_prior = n_prior / n_proposal
    if n_t < 2 or n_prior < 2:
        raise ValueError(
            f"n_proposal={n_proposal} with prior_fraction={prior_fraction} gives {n_t} t and "
            f"{n_prior} mixture draws; each needs at least 2 to estimate its variance.")

    def neg_log_post(theta, c, r, m):
        return -(log_lik(theta[None], c, r, m)[0] + log_mixture(theta[None])[0])

    @jax.jit
    def find_mode(start, c, r, m):
        """BFGS from ``start`` and the Hessian there; compiled once for every session."""
        mode = minimize(neg_log_post, start, args=(c, r, m), method="BFGS").x
        return mode, jax.hessian(neg_log_post)(mode, c, r, m)

    for s in range(n_sessions):
        c, r, m = choices[s], rewards[s], valid_mask[s]
        rng_key, k_start, k_t, k_g, k_mix = jax.random.split(rng_key, 5)

        # Start from the best of a few hundred mixture draws, then polish.
        starts = mixture_sample(k_start, 512)
        start = starts[jnp.argmax(log_lik(starts, c, r, m) + log_mixture(starts))]
        mode, hessian = find_mode(start, c, r, m)
        eig = jnp.linalg.eigvalsh(hessian)
        found = bool(jnp.all(jnp.isfinite(mode)) and jnp.all(jnp.isfinite(eig))
                     and jnp.min(eig) > 0)
        center, cov = (mode, jnp.linalg.inv(hessian)) if found else (mix_mean, mix_cov)
        shape = scale_inflation ** 2 * 0.5 * (cov + cov.T)
        chol = jnp.linalg.cholesky(shape)

        # Multivariate t draws, then the defensive mixture draws.
        g = jax.random.gamma(k_g, df / 2.0, (n_t,)) * 2.0 / df
        t_draws = center + (jax.random.normal(k_t, (n_t, n_params)) @ chol.T) / jnp.sqrt(g)[:, None]
        theta = jnp.concatenate([t_draws, mixture_sample(k_mix, n_prior)])

        dev = jax.scipy.linalg.solve_triangular(chol, (theta - center).T, lower=True).T
        log_t = (gammaln((df + n_params) / 2.0) - gammaln(df / 2.0)
                 - 0.5 * n_params * jnp.log(df * jnp.pi) - jnp.sum(jnp.log(jnp.diag(chol)))
                 - 0.5 * (df + n_params) * jnp.log1p(jnp.sum(dev ** 2, -1) / df))
        log_prior = chunked(log_mixture, theta)
        log_q = jnp.logaddexp(jnp.log1p(-realised_prior) + log_t,
                              jnp.log(realised_prior) + log_prior)
        log_w = np.asarray(
            chunked(lambda th: log_lik(th, c, r, m), theta) + log_prior - log_q, dtype=np.float64)

        top = log_w.max()
        w = np.exp(log_w - top)
        out["session_marginal"][s] = top + np.log(w.mean())
        # Stratified: the variance is within each component's draws, weighted by its count.
        variance = sum(part.size * np.var(part, ddof=1)
                       for part in (w[:n_t], w[n_t:])) / w.size ** 2
        out["session_marginal_se"][s] = np.sqrt(variance) / w.mean()
        out["weight_ess"][s] = w.sum() ** 2 / np.sum(w ** 2)
        out["pareto_k"][s] = _pareto_k(log_w[:, None])[0]
        out["mode_found"][s] = found
    return out
