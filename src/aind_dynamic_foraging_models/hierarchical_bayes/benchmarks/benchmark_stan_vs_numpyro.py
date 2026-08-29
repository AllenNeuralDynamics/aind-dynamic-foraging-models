"""Benchmark the NumPyro reimplementation against the reference PyStan model.

Fits the same synthetic sessions with both implementations and reports posterior agreement
alongside sampling cost. Run from the repository root::

    cd src/aind_dynamic_foraging_models/hierarchical_bayes/benchmarks
    python benchmark_stan_vs_numpyro.py --n-sessions 8 --n-trials 300

Wall-clock alone is a misleading comparison here. Stan runs its chains as separate
processes, so its throughput scales with available cores, while NumPyro vectorises chains
onto one device. Effective samples per second is therefore reported as the primary figure,
with wall-clock and the core count alongside it.
"""

import argparse
import json
import os
import time
import warnings

import numpy as np

warnings.filterwarnings("ignore")

# Parameters shared by both implementations, as (reference name, this package's name).
# The reference reports retention (``mu_aF``) where this package reports decay, so that
# entry is compared after the 1 - x flip; see benchmarks/reference_stan/README.md.
SHARED_PARAMS = (
    ("mu_aP", "subject_learn_rate_rew", False),
    ("mu_aN", "subject_learn_rate_unrew", False),
    ("mu_aF", "subject_forget_rate_unchosen", True),
    ("mu_beta", "subject_softmax_inverse_temperature", False),
)


def simulate_subject(n_sessions, n_trials, seed=0):
    """Simulate one subject's sessions with the numpy forager.

    Parameters
    ----------
    n_sessions, n_trials : int
        Size of the simulated subject.
    seed : int, optional
        Seed for the subject's parameter draws and the task.

    Returns
    -------
    tuple
        ``(choices, rewards, true_subject_params)``.
    """
    from scipy.stats import norm

    from aind_behavior_gym.dynamic_foraging.task import CoupledBlockTask
    from aind_dynamic_foraging_models.generative_model import ForagerCollection

    rng = np.random.default_rng(seed)
    true_mu_p = np.array([0.3, -0.6, -0.8, 0.2, 0.0])
    theta = true_mu_p + 0.25 * rng.standard_normal((n_sessions, 5))

    choices, rewards = [], []
    for session in range(n_sessions):
        forager = ForagerCollection().get_preset_forager("Hattori2019", seed=session)
        forager.set_params(
            learn_rate_rew=float(norm.cdf(theta[session, 0])),
            learn_rate_unrew=float(norm.cdf(theta[session, 1])),
            forget_rate_unchosen=float(norm.cdf(theta[session, 2])),
            softmax_inverse_temperature=float(norm.cdf(theta[session, 3]) * 10.0),
            biasL=float(theta[session, 4]),
        )
        forager.perform(CoupledBlockTask(reward_baiting=True, num_trials=n_trials, seed=session))
        choices.append(forager.get_choice_history())
        rewards.append(forager.get_reward_history())

    truth = {
        "learn_rate_rew": float(norm.cdf(true_mu_p[0])),
        "learn_rate_unrew": float(norm.cdf(true_mu_p[1])),
        "forget_rate_unchosen": float(norm.cdf(true_mu_p[2])),
        "softmax_inverse_temperature": float(norm.cdf(true_mu_p[3]) * 10.0),
    }
    return np.stack(choices), np.stack(rewards), truth


def run_stan(choices, rewards, num_chains, num_samples, num_warmup, seed=1):
    """Fit the reference Stan model with PyStan.

    Returns
    -------
    dict
        Posterior draws by name, plus ``compile_seconds`` and ``sample_seconds``.
    """
    import stan

    here = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(here, "reference_stan", "stan_qLearning_5params.stan")) as handle:
        model_code = handle.read()

    n_sessions, n_trials = choices.shape
    data = {
        "N": int(n_sessions),
        "T": int(n_trials),
        "Tsesh": [int(n_trials)] * int(n_sessions),
        "choice": choices.astype(int).tolist(),
        "outcome": rewards.astype(int).tolist(),
    }

    started = time.time()
    posterior = stan.build(model_code, data=data, random_seed=seed)
    compile_seconds = time.time() - started

    started = time.time()
    fit = posterior.sample(
        num_chains=num_chains, num_samples=num_samples, num_warmup=num_warmup
    )
    sample_seconds = time.time() - started
    # No warm re-run here: httpstan caches fits by model, data and sampling arguments, so a
    # second identical call returns the cached draws in milliseconds rather than resampling.
    # Compilation is already excluded, having happened in stan.build above.

    draws = {name: np.asarray(fit[name]) for name, _, _ in SHARED_PARAMS}
    draws["compile_seconds"] = compile_seconds
    draws["sample_seconds"] = sample_seconds
    draws["num_chains"] = num_chains
    draws["num_samples"] = num_samples
    return draws


def run_numpyro(choices, rewards, num_chains, num_samples, num_warmup, seed=0):
    """Fit the published-priors NumPyro model.

    Returns
    -------
    dict
        Posterior draws by name, plus ``sample_seconds``.
    """
    import jax
    import numpyro
    from numpyro.infer import MCMC, NUTS

    from aind_dynamic_foraging_models.hierarchical_bayes.model import hattori2019_published

    numpyro.set_host_device_count(num_chains)

    mcmc = MCMC(
        NUTS(hattori2019_published),
        num_warmup=num_warmup,
        num_samples=num_samples,
        num_chains=num_chains,
        chain_method="vectorized",
        progress_bar=False,
    )
    # JAX fuses tracing, compilation and execution, so the first run carries the JIT cost.
    # Timing only that would compare NumPyro's compile against Stan's cached binary, so
    # both a cold and a warm run are timed and reported.
    started = time.time()
    mcmc.run(jax.random.PRNGKey(seed), choices, rewards)
    mcmc.get_samples()["mu_p"].block_until_ready()
    cold_seconds = time.time() - started

    started = time.time()
    mcmc.run(jax.random.PRNGKey(seed + 1), choices, rewards)
    mcmc.get_samples()["mu_p"].block_until_ready()
    total_seconds = time.time() - started

    samples = mcmc.get_samples()
    draws = {
        this_name: np.asarray(samples[this_name])
        for _, this_name, _ in SHARED_PARAMS
    }
    draws["cold_seconds"] = cold_seconds
    draws["sample_seconds"] = total_seconds
    draws["num_chains"] = num_chains
    draws["num_samples"] = num_samples
    draws["device"] = str(jax.devices()[0])
    return draws


def effective_sample_size(draws):
    """Bulk effective sample size of a flat draw vector, via ArviZ."""
    import arviz as az

    return float(az.ess(np.asarray(draws).reshape(1, -1), method="bulk"))


def main():
    """Run both implementations on identical data and print the comparison."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-sessions", type=int, default=8)
    parser.add_argument("--n-trials", type=int, default=300)
    parser.add_argument("--num-chains", type=int, default=4)
    parser.add_argument("--num-samples", type=int, default=500)
    parser.add_argument("--num-warmup", type=int, default=500)
    parser.add_argument("--skip-stan", action="store_true")
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    n_cores = int(os.environ.get("SLURM_CPUS_ON_NODE", os.cpu_count() or 1))
    print(f"cores={n_cores}  sessions={args.n_sessions}  trials={args.n_trials}")
    print(f"chains={args.num_chains}  warmup={args.num_warmup}  samples={args.num_samples}\n")

    choices, rewards, truth = simulate_subject(args.n_sessions, args.n_trials)

    numpyro_draws = run_numpyro(
        choices, rewards, args.num_chains, args.num_samples, args.num_warmup
    )
    print(
        f"numpyro: {numpyro_draws['sample_seconds']:.1f}s warm "
        f"({numpyro_draws['cold_seconds']:.1f}s cold, incl. JIT) "
        f"on {numpyro_draws['device']}"
    )

    stan_draws = None
    if not args.skip_stan:
        stan_draws = run_stan(
            choices, rewards, args.num_chains, args.num_samples, args.num_warmup
        )
        print(
            f"stan   : {stan_draws['sample_seconds']:.1f}s sampling "
            f"(+{stan_draws['compile_seconds']:.1f}s compile, cached across runs)\n"
        )

    header = f"{'parameter':<32}{'truth':>9}{'numpyro':>10}"
    if stan_draws is not None:
        header += f"{'stan':>10}"
    print(header)

    summary = {"truth": truth, "cores": n_cores, "params": {}}
    for reference_name, this_name, flip in SHARED_PARAMS:
        key = this_name.replace("subject_", "")
        ours = float(np.mean(numpyro_draws[this_name]))
        row = f"{key:<32}{truth[key]:>9.3f}{ours:>10.3f}"
        entry = {"truth": truth[key], "numpyro": ours}
        if stan_draws is not None:
            theirs = float(np.mean(stan_draws[reference_name]))
            theirs = 1.0 - theirs if flip else theirs
            row += f"{theirs:>10.3f}"
            entry["stan"] = theirs
        summary["params"][key] = entry
        print(row)

    print(f"\n{'':<32}{'ESS/s':>10}")
    total_draws = args.num_chains * args.num_samples
    ours_ess = np.mean(
        [effective_sample_size(numpyro_draws[n]) for _, n, _ in SHARED_PARAMS]
    )
    ours_rate = ours_ess / numpyro_draws["sample_seconds"]
    print(f"{'numpyro':<32}{ours_rate:>10.1f}   ({ours_ess:.0f} ESS of {total_draws} draws)")
    summary["numpyro"] = {
        "seconds": numpyro_draws["sample_seconds"],
        "cold_seconds": numpyro_draws["cold_seconds"],
        "ess": float(ours_ess),
        "ess_per_second": float(ours_rate),
        "device": numpyro_draws["device"],
    }
    if stan_draws is not None:
        their_ess = np.mean(
            [effective_sample_size(stan_draws[n]) for n, _, _ in SHARED_PARAMS]
        )
        their_rate = their_ess / stan_draws["sample_seconds"]
        print(f"{'stan':<32}{their_rate:>10.1f}   ({their_ess:.0f} ESS of {total_draws} draws)")
        summary["stan"] = {
            "seconds": stan_draws["sample_seconds"],
            "compile_seconds": stan_draws["compile_seconds"],
            "ess": float(their_ess),
            "ess_per_second": float(their_rate),
        }
        print(f"\nspeedup (ESS/s, numpyro / stan): {ours_rate / their_rate:.1f}x")
        summary["speedup_ess_per_second"] = float(ours_rate / their_rate)

    if args.output:
        with open(args.output, "w") as handle:
            json.dump(summary, handle, indent=2)
        print(f"\nwrote {args.output}")


if __name__ == "__main__":
    main()
