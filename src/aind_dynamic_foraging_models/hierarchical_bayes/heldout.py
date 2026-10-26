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

    theta_raw = numpyro.sample(
        "theta_raw",
        dist.Normal(0.0, 1.0).expand([n_subjects, n_context, n_params]).to_event(3),
    )
    params = hattori2019_session_params(
        mu_p[:, None, :] + sigma[:, None, :] * theta_raw, beta_max=beta_max
    )

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
    mcmc = MCMC(
        NUTS(adapt_subjects_batched),
        num_warmup=num_warmup,
        num_samples=num_samples,
        num_chains=1,
        progress_bar=progress_bar,
    )
    mcmc.run(
        rng_key, context_choices, context_rewards, population,
        session_mask, valid_mask, beta_max=beta_max,
    )
    return mcmc.get_samples()


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
    """Score many held-out sessions at once by the trial-marginal, one vmapped pass per chunk.

    Batching the adaptation fits removed one bottleneck and exposed the next: replaying each
    held-out session in its own call means thousands of small dispatches, which on a
    latency-bound device costs far more than the arithmetic does. Session replays are
    independent, so they vectorise the same way everything else here does.

    This returns the trial-marginal only; :func:`batched_heldout_scores` returns it together
    with the session-marginal from the same draws.

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
        Subsample this many posterior draws. The trial-marginal average converges quickly,
        so fewer draws here trade a little noise for a lot of time.
    session_chunk : int, optional
        Sessions per vmapped pass, which caps peak memory. Defaults to a size derived from
        the device's own memory limit, so the same code neither wastes a large GPU nor
        exhausts a small one.

    Returns
    -------
    tuple of np.ndarray
        Per-session summed log likelihood and trial count, both shape ``(n_sessions,)``.
    """
    scores = batched_heldout_scores(
        batched_samples, subject_indices, choices, rewards, valid_mask,
        rng_key=rng_key, beta_max=beta_max, n_draws=n_draws, session_chunk=session_chunk,
    )
    return scores["trial_marginal"], scores["n_trials"]


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
        (effective number of draws behind it), and ``n_trials`` (trials scored).
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
    names = ("trial_marginal", "session_marginal", "session_marginal_se", "weight_ess")
    out = {name: np.zeros(n_sessions) for name in names}
    out["n_trials"] = np.zeros(n_sessions, dtype=int)

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
        session, se, ess = _session_marginal(per_draw, n_theta_per_draw)

        out["trial_marginal"][start:stop] = np.asarray(trial)
        out["session_marginal"][start:stop] = np.asarray(session)
        out["session_marginal_se"][start:stop] = np.asarray(se)
        out["weight_ess"][start:stop] = np.asarray(ess)
        out["n_trials"][start:stop] = np.asarray(jnp.sum(mask, axis=-1))

    return out


def _observed_prob_over_draws(n_trials, beta_max):
    """Jitted replay of the observed action's probability, per draw, session and trial.

    Returns two arrays. The probability from :func:`hattori2019_choice_prob` keeps the
    trial-marginal identical to what it always was. The log probability is computed from the
    decision variable (the left-minus-right logit) with ``log_sigmoid``, so it stays exact
    where the probability itself underflows; the session-marginal sums these, and a floor
    on the probability would put a ceiling on how badly a confidently wrong draw can score.
    """
    trial_index = jnp.arange(n_trials)

    def _observed_prob(theta, session_choices, session_rewards):
        """Probability, and exact log probability, of the observed action per trial."""
        params = hattori2019_session_params(theta, beta_max=beta_max)
        kwargs = {
            "learn_rate_rew": params["learn_rate_rew"],
            "learn_rate_unrew": params["learn_rate_unrew"],
            "forget_rate_unchosen": params["forget_rate_unchosen"],
            "softmax_inverse_temperature": params["softmax_inverse_temperature"],
            "bias_l": params["bias_l"],
        }
        prob = hattori2019_choice_prob(session_choices, session_rewards, **kwargs)
        _, left_minus_right = hattori2019_value_trajectory(
            session_choices, session_rewards, **kwargs
        )
        sign = jnp.where(session_choices == 0, 1.0, -1.0)  # 0 is left
        return prob[session_choices, trial_index], jax.nn.log_sigmoid(sign * left_minus_right)

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
    """Log-mean-exp over draws, its delta-method standard error, and the weights' ESS.

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
    se = _autocorrelated_se(by_posterior) / mean_w
    ess = jnp.sum(weights, axis=0) ** 2 / jnp.sum(weights ** 2, axis=0)
    return session, se, ess


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


def pooled_likelihood(log_lik, n_trials, log_lik_se=None):
    """Trial-weighted pooled likelihood, ``exp(total log likelihood / total trials)``.

    Parameters
    ----------
    log_lik, n_trials : array_like, shape (n_sessions,)
        Per-session summed log likelihood and trial count.
    log_lik_se : array_like, shape (n_sessions,), optional
        Per-session standard errors, independent across sessions, propagated by the delta
        method.

    Returns
    -------
    tuple of (float, float or None)
        The pooled likelihood and its standard error (``None`` without ``log_lik_se``).
    """
    total, count = float(np.sum(log_lik)), int(np.sum(n_trials))
    pooled = float(np.exp(total / count))
    if log_lik_se is None:
        return pooled, None
    return pooled, pooled * float(np.sqrt(np.sum(np.square(log_lik_se)))) / count
