"""Fit the joint three-level model in Stan, and compare it with NumPyro on identical data.

Why this exists
---------------
`RESULTS.md` prices NumPyro against Stan **per subject**, where Stan wins on every measured
axis, and then argues NumPyro wins at cohort scale -- but that row is labelled "inferred, not
measured". This script exists to replace the inference with a measurement, so the framework
choice rests on numbers rather than on an extrapolation.

It fits `reference_stan/hb_three_level.stan`, which is a port of
``model.hattori2019_three_level`` and NOT of the two-level reference model beside it. The
comparison is only meaningful if both frameworks target the same posterior, so the port
follows the NumPyro model's structure and parameterisation, including its three documented
traps (``aF`` inversion, ``bias`` sign, reward-vs-PE-sign branching). See the header of the
`.stan` file.

pystan
------
`import stan` (pystan 3), matching `benchmark_stan_vs_numpyro.py` next door so the two
benchmarks in this directory are comparable and there is one Stan toolchain to keep working.
pystan builds the model through httpstan's own bundled toolchain, which is why nothing here
depends on a CmdStan installation.

One consequence worth knowing before reading the timings: httpstan does not define
``STAN_THREADS``, so the ``reduce_sum`` in the model runs **serially**. That is valid Stan --
``reduce_sum`` degrades to a plain sum when threading is off -- and it means the Stan wall
times below use one core per chain. Chains still run in parallel. If a threaded number is
ever wanted, that is a CmdStan build with ``STAN_THREADS=true``, not a model change.

Usage
-----
Synthetic validation -- fits both frameworks on the same simulated cohort and reports
per-parameter agreement plus the speed/ESS comparison::

    python fit_three_level_stan.py --synthetic --n-subjects 6 --n-sessions 8 --n-trials 300

Real cohort -- reads the arrays exported from the wrapper's loader::

    python fit_three_level_stan.py --npz cohort_d29.npz --num-warmup 2000 --num-samples 2000

Never run either mode in the sandbox or on a login node; both are sbatch work.
"""

import argparse
import json
import os
import time

import numpy as np

# Parameter order, matching HATTORI2019_PARAMS and the .stan file's theta indices.
PARAM_NAMES = (
    "learn_rate_rew",
    "learn_rate_unrew",
    "forget_rate_unchosen",
    "softmax_inverse_temperature",
    "bias_l",
)


def simulate_cohort(n_subjects, n_sessions, n_trials, beta_max=10.0, seed=0):
    """Draw a cohort from the model's own generative process and simulate choices.

    Sampling from the prior the model assumes -- rather than from hand-picked parameters --
    is what makes agreement between the two fits informative: both are then estimating a
    quantity that genuinely came from this hierarchy.

    Rewards follow a coupled block schedule (probabilities swap at block boundaries), which
    is what makes the value terms identifiable at all; a constant reward rate would leave the
    learning rates only weakly informed by the data.

    Returns
    -------
    dict
        ``choice``/``reward`` int arrays of shape ``(subjects, sessions, trials)``, the
        ragged-length bookkeeping, and the ground-truth population/session parameters.
    """
    from scipy.stats import norm

    rng = np.random.default_rng(seed)
    n_params = len(PARAM_NAMES)

    population_mean = rng.normal(0.0, 1.0, size=n_params)
    population_scale = np.abs(rng.normal(0.0, 1.0, size=n_params))
    log_sigma_mean = rng.normal(-1.0, 1.0, size=n_params)
    log_sigma_spread = np.abs(rng.normal(0.0, 1.0, size=n_params))

    mu_p = population_mean + population_scale * rng.normal(size=(n_subjects, n_params))
    log_sigma = log_sigma_mean + log_sigma_spread * rng.normal(size=(n_subjects, n_params))
    sigma = np.exp(log_sigma)
    theta = mu_p[:, None, :] + sigma[:, None, :] * rng.normal(
        size=(n_subjects, n_sessions, n_params)
    )

    bounded = {
        "learn_rate_rew": norm.cdf(theta[..., 0]),
        "learn_rate_unrew": norm.cdf(theta[..., 1]),
        "forget_rate_unchosen": norm.cdf(theta[..., 2]),
        "softmax_inverse_temperature": norm.cdf(theta[..., 3]) * beta_max,
        "bias_l": theta[..., 4],
    }

    choice = np.zeros((n_subjects, n_sessions, n_trials), dtype=np.int32)
    reward = np.zeros((n_subjects, n_sessions, n_trials), dtype=np.int32)
    block = 60
    for s in range(n_subjects):
        for m in range(n_sessions):
            p = {k: float(v[s, m]) for k, v in bounded.items()}
            q = np.zeros(2)
            rich = 0
            for t in range(n_trials):
                if t % block == 0 and t > 0:
                    rich = 1 - rich
                probs = np.array([0.1, 0.1])
                probs[rich] = 0.7
                # Same act rule as the model: bias on the LEFT option.
                logits = p["softmax_inverse_temperature"] * q + np.array([p["bias_l"], 0.0])
                logits -= logits.max()
                pr = np.exp(logits) / np.exp(logits).sum()
                c = int(rng.random() > pr[0])          # 0 = left, 1 = right
                r = int(rng.random() < probs[c])
                choice[s, m, t] = c
                reward[s, m, t] = r
                lr = p["learn_rate_rew"] if r > 0 else p["learn_rate_unrew"]
                q_new = q * (1.0 - p["forget_rate_unchosen"])
                q_new[c] = q[c] + lr * (r - q[c])
                q = q_new

    return {
        "choice": choice,
        "reward": reward,
        "n_sessions": np.full(n_subjects, n_sessions, dtype=np.int32),
        "n_trials": np.full((n_subjects, n_sessions), n_trials, dtype=np.int32),
        "truth_population_mean": population_mean,
        "truth_session_params": bounded,
    }


def stan_data(arrays, beta_max=10.0, log_sigma_loc=-1.0, log_sigma_scale=1.0, grainsize=1):
    """Shape the arrays into the `.stan` file's `data` block."""
    S, M, T = arrays["choice"].shape
    return {
        "S": S,
        "M": M,
        "T": T,
        "n_sessions": arrays["n_sessions"].astype(int).tolist(),
        "n_trials": arrays["n_trials"].astype(int).tolist(),
        "choice": arrays["choice"].astype(int).tolist(),
        "reward": arrays["reward"].astype(int).tolist(),
        "beta_max": float(beta_max),
        "log_sigma_loc": float(log_sigma_loc),
        "log_sigma_scale": float(log_sigma_scale),
        "grainsize": int(grainsize),
    }


def run_stan(arrays, num_chains, num_samples, num_warmup, seed=0, beta_max=10.0):
    """Build and fit the three-level Stan model with pystan. Returns draws plus timing."""
    import stan

    here = os.path.dirname(os.path.abspath(__file__))
    stan_file = os.path.join(here, "reference_stan", "hb_three_level.stan")
    with open(stan_file) as handle:
        program = handle.read()

    data = stan_data(arrays, beta_max=beta_max, grainsize=1)

    # Build is timed separately from sampling. Stan pays a C++ compile once and then samples
    # from a binary; JAX pays JIT inside its first sampling call. Folding the two together
    # would flatter whichever framework happened to be warm, so they are reported apart.
    started = time.time()
    posterior = stan.build(program, data=data, random_seed=seed)
    compile_seconds = time.time() - started

    started = time.time()
    fit = posterior.sample(
        num_chains=num_chains, num_samples=num_samples, num_warmup=num_warmup
    )
    sample_seconds = time.time() - started

    return {
        "fit": fit,
        "compile_seconds": compile_seconds,
        "sample_seconds": sample_seconds,
        "num_chains": num_chains,
        "num_samples": num_samples,
    }


def run_numpyro(arrays, num_chains, num_samples, num_warmup, seed=0, beta_max=10.0):
    """Fit the same data with the NumPyro three-level model."""
    import jax
    import numpyro
    from numpyro.infer import MCMC, NUTS

    from aind_dynamic_foraging_models.hierarchical_bayes.model import hattori2019_three_level

    numpyro.set_host_device_count(num_chains)

    S, M, T = arrays["choice"].shape
    valid = np.zeros((S, M, T), dtype=bool)
    for s in range(S):
        for m in range(int(arrays["n_sessions"][s])):
            valid[s, m, : int(arrays["n_trials"][s, m])] = True
    session_mask = np.arange(M)[None, :] < arrays["n_sessions"][:, None]

    mcmc = MCMC(
        NUTS(hattori2019_three_level),
        num_warmup=num_warmup,
        num_samples=num_samples,
        num_chains=num_chains,
        chain_method="vectorized",
        progress_bar=False,
    )
    started = time.time()
    mcmc.run(
        jax.random.PRNGKey(seed),
        arrays["choice"], arrays["reward"],
        valid_mask=valid, session_mask=session_mask, beta_max=beta_max,
        extra_fields=("num_steps", "energy", "accept_prob", "diverging"),
    )
    mcmc.get_samples()["population_mean"].block_until_ready()
    sample_seconds = time.time() - started

    return {
        "mcmc": mcmc,
        "sample_seconds": sample_seconds,
        "device": str(jax.devices()[0]),
        "num_chains": num_chains,
        "num_samples": num_samples,
    }


def summarise(name, draws_by_param, seconds, divergences, n_draws):
    """Common efficiency table for either framework: ESS/draw and ESS/s are the comparison."""
    import arviz as az

    rows = []
    for param, draws in draws_by_param.items():
        arr = np.asarray(draws)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        ess = float(az.ess(arr, method="bulk"))
        rhat = float(az.rhat(arr)) if arr.shape[0] > 1 else float("nan")
        rows.append({"param": param, "ess_bulk": ess, "rhat": rhat,
                     "ess_per_draw": ess / n_draws, "ess_per_second": ess / seconds})
    worst = min(rows, key=lambda r: r["ess_bulk"])
    return {
        "framework": name,
        "seconds": seconds,
        "divergences": divergences,
        "n_draws": n_draws,
        "min_ess_bulk": worst["ess_bulk"],
        "min_ess_per_draw": worst["ess_per_draw"],
        "min_ess_per_second": worst["ess_per_second"],
        "max_rhat": max(r["rhat"] for r in rows),
        "rows": rows,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--synthetic", action="store_true",
                     help="simulate a cohort from the model's own prior")
    src.add_argument("--npz", type=str,
                     help="cohort exported from the wrapper loader")
    parser.add_argument("--n-subjects", type=int, default=6)
    parser.add_argument("--n-sessions", type=int, default=8)
    parser.add_argument("--n-trials", type=int, default=300)
    parser.add_argument("--num-chains", type=int, default=4)
    parser.add_argument("--num-warmup", type=int, default=1000)
    parser.add_argument("--num-samples", type=int, default=1000)
    parser.add_argument("--beta-max", type=float, default=10.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--skip-numpyro", action="store_true")
    parser.add_argument("--skip-stan", action="store_true")
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    if args.synthetic:
        arrays = simulate_cohort(args.n_subjects, args.n_sessions, args.n_trials,
                                 beta_max=args.beta_max, seed=args.seed)
    else:
        loaded = np.load(args.npz)
        arrays = {k: loaded[k] for k in loaded.files}

    S, M, T = arrays["choice"].shape
    real_trials = int(arrays["n_trials"].sum())
    n_cores = int(os.environ.get("SLURM_CPUS_ON_NODE", os.cpu_count() or 1))
    print(f"cohort: S={S} subjects, M<={M} sessions, T<={T} trials")
    print(f"  real trials {real_trials:,} vs padded {S * M * T:,} "
          f"({100 * (1 - real_trials / (S * M * T)):.1f}% padding -- Stan skips it, JAX does not)")
    print(f"  cores={n_cores} chains={args.num_chains} "
          f"warmup={args.num_warmup} samples={args.num_samples}\n")

    report = {"cohort": {"S": S, "M": M, "T": T, "real_trials": real_trials,
                         "padded_trials": S * M * T, "cores": n_cores},
              "settings": vars(args)}
    stan_draws_by_param = {}
    numpyro_draws_by_param = {}

    if not args.skip_stan:
        out = run_stan(arrays, args.num_chains, args.num_samples, args.num_warmup,
                       seed=args.seed, beta_max=args.beta_max)
        fit = out["fit"]
        # pystan returns (*param_dims, chains * samples) with draws ordered chain-major, so
        # reshaping to (chain, draw) is what lets r_hat see between-chain variation at all --
        # ESS and r_hat computed on the flattened vector would silently treat four chains as
        # one long one.
        population_mean = np.asarray(fit["population_mean"])
        draws = {
            f"population_mean[{name}]":
                population_mean[i].reshape(args.num_chains, args.num_samples)
            for i, name in enumerate(PARAM_NAMES)
        }
        try:
            div = int(np.sum(np.asarray(fit["divergent__"])))
        except (KeyError, AttributeError):
            div = -1          # -1 = not reported, distinct from a genuine zero
        n_draws = args.num_chains * args.num_samples
        stan_draws_by_param = draws
        report["stan"] = summarise("stan", draws, out["sample_seconds"], div, n_draws)
        report["stan"]["compile_seconds"] = out["compile_seconds"]
        print(f"stan: {out['sample_seconds']:.1f}s sampling "
              f"({out['compile_seconds']:.1f}s build), {div} divergences, "
              f"min ESS/draw {report['stan']['min_ess_per_draw']:.3f}")

    if not args.skip_numpyro:
        out = run_numpyro(arrays, args.num_chains, args.num_samples, args.num_warmup,
                          seed=args.seed, beta_max=args.beta_max)
        samples = out["mcmc"].get_samples(group_by_chain=True)
        draws = {}
        for i, name in enumerate(PARAM_NAMES):
            draws[f"population_mean[{name}]"] = np.asarray(samples["population_mean"])[..., i]
        extra = out["mcmc"].get_extra_fields()
        div = int(np.sum(np.asarray(extra["diverging"]))) if "diverging" in extra else -1
        n_draws = args.num_chains * args.num_samples
        numpyro_draws_by_param = draws
        report["numpyro"] = summarise("numpyro", draws, out["sample_seconds"], div, n_draws)
        report["numpyro"]["device"] = out["device"]
        print(f"numpyro: {out['sample_seconds']:.1f}s on {out['device']}, "
              f"{div} divergences, "
              f"min ESS/draw {report['numpyro']['min_ess_per_draw']:.3f}")

    if "stan" in report and "numpyro" in report:
        # Agreement is the correctness check and comes FIRST: a speed comparison between two
        # implementations of different posteriors would be meaningless. Compared against each
        # parameter's own posterior SD rather than an absolute tolerance, because the
        # population means live on very different scales and a fixed epsilon would be
        # simultaneously too strict on one and vacuous on another.
        print("\n=== agreement on population_mean ===")
        print(f"  {'parameter':30s} {'stan':>9s} {'numpyro':>9s} {'diff':>9s} {'diff/sd':>8s}")
        agreement = []
        for name in PARAM_NAMES:
            key = f"population_mean[{name}]"
            s = np.asarray(stan_draws_by_param[key]).ravel()
            n = np.asarray(numpyro_draws_by_param[key]).ravel()
            diff = float(s.mean() - n.mean())
            pooled_sd = float(np.sqrt(0.5 * (s.var() + n.var()))) or float("nan")
            agreement.append({"param": name, "stan_mean": float(s.mean()),
                              "numpyro_mean": float(n.mean()), "diff": diff,
                              "diff_in_sd": diff / pooled_sd})
            print(f"  {name:30s} {s.mean():9.4f} {n.mean():9.4f} "
                  f"{diff:9.4f} {diff / pooled_sd:8.3f}")
        report["agreement"] = agreement
        worst = max(abs(a["diff_in_sd"]) for a in agreement)
        print(f"  worst |diff| = {worst:.3f} posterior SD "
              f"({'consistent' if worst < 0.25 else 'INVESTIGATE -- may be different models'})")

        r = report
        r["speed"] = {
            "stan_seconds": r["stan"]["seconds"],
            "numpyro_seconds": r["numpyro"]["seconds"],
            "stan_faster_by": r["numpyro"]["seconds"] / r["stan"]["seconds"],
            "stan_ess_per_second": r["stan"]["min_ess_per_second"],
            "numpyro_ess_per_second": r["numpyro"]["min_ess_per_second"],
        }
        print(f"\n  stan {r['stan']['seconds']:.1f}s vs numpyro {r['numpyro']['seconds']:.1f}s "
              f"-> stan {r['speed']['stan_faster_by']:.2f}x")
        print(f"  ESS/s (worst param): stan {r['stan']['min_ess_per_second']:.2f} vs "
              f"numpyro {r['numpyro']['min_ess_per_second']:.2f}")

    if args.output:
        with open(args.output, "w") as handle:
            json.dump(report, handle, indent=2, default=str)
        print(f"\nwrote {args.output}")


if __name__ == "__main__":
    main()
