"""Compare sampler geometries for the two-level model.

The reference Stan implementation extracts noticeably more effective samples per draw than
our NumPyro model did in the first benchmark, on the same data with the same algorithm
class. That is a property of the posterior geometry and the sampler's adaptation, not of the
hardware, so it is worth fixing before reaching for a different inference method.

This sweeps the three knobs that plausibly explain the gap:

* **centred vs non-centred** session parameters. Non-centred is the standard defence against
  funnel geometry, but that funnel appears when the data constrain a unit weakly. With
  hundreds of trials per session the likelihood is strong, and in that regime the centred
  form is often better conditioned.
* **diagonal vs dense mass matrix**. The four pooled parameters are correlated in the
  posterior; a diagonal mass matrix cannot represent that.
* **target acceptance probability**. The published configuration used 0.85-0.9 to force
  smaller steps.

Run from the repository root::

    python benchmarks/geometry_experiment.py --n-sessions 8 --n-trials 300
"""

import argparse
import itertools
import json
import time
import warnings

import numpy as np

warnings.filterwarnings("ignore")

SUBJECT_PARAMS = (
    "subject_learn_rate_rew",
    "subject_learn_rate_unrew",
    "subject_forget_rate_unchosen",
    "subject_softmax_inverse_temperature",
)


def run_one(choices, rewards, centered, dense_mass, target_accept_prob, num_chains,
            num_samples, num_warmup, seed=0):
    """Fit one geometry configuration and return its sampling diagnostics.

    Returns
    -------
    dict
        Wall-clock seconds for a warm run, mean bulk ESS over the subject-level
        parameters, ESS per draw, ESS per second, and the divergence count.
    """
    import arviz as az
    import jax
    from numpyro.infer import MCMC, NUTS

    from aind_dynamic_foraging_models.hierarchical_bayes.model import hattori2019_two_level

    kernel = NUTS(
        hattori2019_two_level,
        dense_mass=dense_mass,
        target_accept_prob=target_accept_prob,
    )
    mcmc = MCMC(
        kernel,
        num_warmup=num_warmup,
        num_samples=num_samples,
        num_chains=num_chains,
        chain_method="vectorized",
        progress_bar=False,
    )

    # Warm the JIT cache first so the reported time is sampling, not compilation.
    mcmc.run(jax.random.PRNGKey(seed), choices, rewards, centered=centered)
    mcmc.get_samples()["mu_p"].block_until_ready()

    started = time.time()
    mcmc.run(jax.random.PRNGKey(seed + 1), choices, rewards, centered=centered)
    mcmc.get_samples()["mu_p"].block_until_ready()
    seconds = time.time() - started

    samples = mcmc.get_samples()
    ess = float(
        np.mean(
            [
                az.ess(np.asarray(samples[name]).reshape(1, -1), method="bulk")
                for name in SUBJECT_PARAMS
            ]
        )
    )
    divergences = int(np.sum(np.asarray(mcmc.get_extra_fields()["diverging"])))
    total_draws = num_chains * num_samples

    return {
        "seconds": seconds,
        "ess": ess,
        "ess_per_draw": ess / total_draws,
        "ess_per_second": ess / seconds,
        "divergences": divergences,
    }


def main():
    """Sweep the geometry configurations and print a ranked table."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-sessions", type=int, default=8)
    parser.add_argument("--n-trials", type=int, default=300)
    parser.add_argument("--num-chains", type=int, default=2)
    parser.add_argument("--num-samples", type=int, default=400)
    parser.add_argument("--num-warmup", type=int, default=400)
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    from benchmark_stan_vs_numpyro import simulate_subject

    choices, rewards, _ = simulate_subject(args.n_sessions, args.n_trials)
    total_draws = args.num_chains * args.num_samples
    print(f"sessions={args.n_sessions} trials={args.n_trials} draws={total_draws}\n")

    header = f"{'centred':>8}{'dense':>7}{'accept':>8}{'sec':>8}{'ESS':>8}"
    header += f"{'ESS/draw':>10}{'ESS/s':>8}{'div':>6}"
    print(header)

    results = []
    for centered, dense_mass, target in itertools.product(
        (False, True), (False, True), (0.8, 0.9)
    ):
        result = run_one(
            choices, rewards, centered, dense_mass, target,
            args.num_chains, args.num_samples, args.num_warmup,
        )
        result.update(centered=centered, dense_mass=dense_mass, target_accept_prob=target)
        results.append(result)
        print(
            f"{str(centered):>8}{str(dense_mass):>7}{target:>8.2f}"
            f"{result['seconds']:>8.1f}{result['ess']:>8.0f}"
            f"{result['ess_per_draw']:>10.3f}{result['ess_per_second']:>8.1f}"
            f"{result['divergences']:>6d}",
            flush=True,
        )

    best = max(results, key=lambda r: r["ess_per_second"])
    print(
        f"\nbest ESS/s: centred={best['centered']} dense={best['dense_mass']} "
        f"accept={best['target_accept_prob']} -> {best['ess_per_second']:.1f} ESS/s, "
        f"{best['ess_per_draw']:.3f} ESS/draw"
    )

    if args.output:
        with open(args.output, "w") as handle:
            json.dump(results, handle, indent=2)
        print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
