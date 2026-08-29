"""Two-stage against one-stage on a synthetic cohort with known ground truth.

Answers the question the real-data subset run cannot: two-stage empirical Bayes is an
approximation to the one-stage joint fit, and this measures what the approximation costs,
on data where the true population is known.

The comparison is made in the currency that matters downstream -- held-out predictive
likelihood for subjects the fit never saw -- and additionally against the truth, which real
data cannot provide.

Run from the repository root::

    python recovery/estimator_comparison.py --n-subjects 30
"""

import argparse
import json
import time
import warnings

import numpy as np

warnings.filterwarnings("ignore")

BETA_MAX = 10.0
N_PARAMS = 5
FEW_SHOT_K = (0, 1, 2, 4, 8)


def simulate_cohort(n_subjects, n_sessions, n_trials, population_mean, population_scale,
                    sigma, seed=0):
    """Draw a cohort from the full three-level structure and simulate its behaviour."""
    from aind_behavior_gym.dynamic_foraging.task import CoupledBlockTask
    from scipy.stats import norm

    from aind_dynamic_foraging_models.generative_model import ForagerCollection

    rng = np.random.default_rng(seed)
    mu_p = population_mean + population_scale * rng.standard_normal((n_subjects, N_PARAMS))

    choices = np.zeros((n_subjects, n_sessions, n_trials), dtype=int)
    rewards = np.zeros((n_subjects, n_sessions, n_trials), dtype=float)
    for subject in range(n_subjects):
        theta = mu_p[subject] + sigma * rng.standard_normal((n_sessions, N_PARAMS))
        for session in range(n_sessions):
            tag = subject * 1000 + session
            forager = ForagerCollection().get_preset_forager("Hattori2019", seed=tag)
            forager.set_params(
                learn_rate_rew=float(norm.cdf(theta[session, 0])),
                learn_rate_unrew=float(norm.cdf(theta[session, 1])),
                forget_rate_unchosen=float(norm.cdf(theta[session, 2])),
                softmax_inverse_temperature=float(norm.cdf(theta[session, 3]) * BETA_MAX),
                biasL=float(theta[session, 4]),
            )
            forager.perform(
                CoupledBlockTask(reward_baiting=True, num_trials=n_trials, seed=tag)
            )
            choices[subject, session] = forager.get_choice_history()
            rewards[subject, session] = forager.get_reward_history()
    return choices, rewards, mu_p


def fit_one_stage(choices, rewards, rng_key, num_warmup, num_samples, num_chains):
    """Fit the joint three-level model and return its population posterior."""
    from numpyro.infer import MCMC, NUTS

    from aind_dynamic_foraging_models.hierarchical_bayes.model import (
        hattori2019_three_level,
    )

    mcmc = MCMC(
        NUTS(hattori2019_three_level),
        num_warmup=num_warmup, num_samples=num_samples, num_chains=num_chains,
        chain_method="vectorized", progress_bar=False,
    )
    mcmc.run(rng_key, choices, rewards)
    return mcmc


def fit_two_stage_cohort(choices, rewards, rng_key, num_warmup, num_samples):
    """Fit every subject independently, then a population over their posteriors."""
    from aind_dynamic_foraging_models.hierarchical_bayes.two_stage import fit_two_stage

    subjects = [(choices[i], rewards[i]) for i in range(choices.shape[0])]
    return fit_two_stage(
        subjects,
        rng_key=rng_key,
        subject_kwargs=dict(num_warmup=num_warmup, num_samples=num_samples),
        population_kwargs=dict(num_warmup=num_warmup, num_samples=num_samples),
    )


def population_from_two_stage(result):
    """Convert the two-stage population fit into the adaptation prior's format."""
    samples = result["population_mcmc"].get_samples()
    mean = np.asarray(samples["population_mean"]).mean(axis=0)
    scale = np.asarray(samples["population_scale"]).mean(axis=0)
    return {
        "population_mean": mean[:N_PARAMS],
        "population_scale": scale[:N_PARAMS],
        "log_sigma_mean": mean[N_PARAMS:],
        "log_sigma_spread": scale[N_PARAMS:],
    }


def score_heldout(population, choices, rewards, k_values, rng_key, num_warmup, num_samples):
    """Adapt each held-out subject on k context sessions and score the rest."""
    import jax

    from aind_dynamic_foraging_models.hierarchical_bayes.heldout import (
        fit_adaptation,
        pointwise_log_predictive_density,
        posterior_predictive_choice_prob,
    )

    n_subjects, n_sessions, _ = choices.shape
    out = {}
    for k in k_values:
        total_log_lik, total_trials = 0.0, 0
        for subject in range(n_subjects):
            key_fit, key_draw, rng_key = jax.random.split(rng_key, 3)
            samples = fit_adaptation(
                choices[subject, :k], rewards[subject, :k], population,
                rng_key=key_fit, num_warmup=num_warmup, num_samples=num_samples,
            )
            for session in range(max(k_values), n_sessions):  # disjoint from every k
                prob = posterior_predictive_choice_prob(
                    samples, choices[subject, session], rewards[subject, session],
                    rng_key=key_draw,
                )
                log_lik, n = pointwise_log_predictive_density(
                    prob, choices[subject, session]
                )
                total_log_lik += log_lik
                total_trials += n
        out[k] = float(np.exp(total_log_lik / total_trials))
    return out


def main():
    """Fit both estimators on one cohort and compare their held-out scores."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-subjects", type=int, default=30)
    parser.add_argument("--n-heldout", type=int, default=10)
    parser.add_argument("--n-sessions", type=int, default=25)
    parser.add_argument("--n-trials", type=int, default=500)
    parser.add_argument("--num-warmup", type=int, default=500)
    parser.add_argument("--num-samples", type=int, default=500)
    parser.add_argument("--num-chains", type=int, default=4)
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    import jax

    print(f"device={jax.devices()[0]}", flush=True)
    population_mean = np.array([0.3, -0.6, -0.8, 0.2, 0.0])
    population_scale = np.array([0.4] * N_PARAMS)
    sigma = np.array([0.25] * N_PARAMS)

    total_subjects = args.n_subjects + args.n_heldout
    print(f"simulating {total_subjects} subjects x {args.n_sessions} x {args.n_trials}",
          flush=True)
    choices, rewards, true_mu_p = simulate_cohort(
        total_subjects, args.n_sessions, args.n_trials,
        population_mean, population_scale, sigma,
    )
    train = slice(0, args.n_subjects)
    heldout = slice(args.n_subjects, total_subjects)

    summary = {"truth": {"population_mean": population_mean.tolist()}, "estimators": {}}

    for name in ("one_stage", "two_stage"):
        started = time.time()
        if name == "one_stage":
            mcmc = fit_one_stage(
                choices[train], rewards[train], jax.random.PRNGKey(0),
                args.num_warmup, args.num_samples, args.num_chains,
            )
            samples = mcmc.get_samples()
            population = {
                "population_mean": np.asarray(samples["population_mean"]).mean(axis=0),
                "population_scale": np.asarray(samples["population_scale"]).mean(axis=0),
                "log_sigma_mean": np.asarray(samples["log_sigma_mean"]).mean(axis=0),
                "log_sigma_spread": np.asarray(samples["log_sigma_spread"]).mean(axis=0),
            }
            divergences = int(np.sum(np.asarray(mcmc.get_extra_fields()["diverging"])))
        else:
            result = fit_two_stage_cohort(
                choices[train], rewards[train], jax.random.PRNGKey(0),
                args.num_warmup, args.num_samples,
            )
            population = population_from_two_stage(result)
            divergences = None
        fit_seconds = time.time() - started

        scores = score_heldout(
            population, choices[heldout], rewards[heldout], FEW_SHOT_K,
            jax.random.PRNGKey(1), args.num_warmup, args.num_samples,
        )
        summary["estimators"][name] = {
            "fit_seconds": fit_seconds,
            "divergences": divergences,
            "population_mean": np.asarray(population["population_mean"]).tolist(),
            "population_scale": np.asarray(population["population_scale"]).tolist(),
            "heldout_likelihood": scores,
        }
        print(f"\n{name}: fit {fit_seconds:.0f}s  divergences={divergences}", flush=True)
        print(f"  population_mean {np.round(population['population_mean'], 3)}", flush=True)
        for k, value in scores.items():
            print(f"  k={k}: heldout likelihood {value:.5f}", flush=True)

    one, two = summary["estimators"]["one_stage"], summary["estimators"]["two_stage"]
    print(f"\n{'k':>4}{'one_stage':>12}{'two_stage':>12}{'difference':>13}")
    for k in FEW_SHOT_K:
        a, b = one["heldout_likelihood"][k], two["heldout_likelihood"][k]
        print(f"{k:>4}{a:>12.5f}{b:>12.5f}{b - a:>13.5f}")
    summary["max_abs_difference"] = max(
        abs(two["heldout_likelihood"][k] - one["heldout_likelihood"][k]) for k in FEW_SHOT_K
    )
    print(f"\nlargest gap across k: {summary['max_abs_difference']:.5f}")

    if args.output:
        with open(args.output, "w") as handle:
            json.dump(summary, handle, indent=2)
        print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
