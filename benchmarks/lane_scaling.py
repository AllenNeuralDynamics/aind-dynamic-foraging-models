"""Is the likelihood latency-bound or throughput-bound?

The case for NumPyro rests on one claim: because two-stage subject fits are independent,
every subject can share a single vmapped computation, and widening that batch is nearly
free. That holds only while wall time is dominated by per-step dispatch overhead rather
than arithmetic.

This measures one gradient evaluation at a fixed scan depth while widening the batch. Flat
time means latency-bound, so batching the whole cohort costs little and the NumPyro case
holds. Time growing linearly with lanes means throughput-bound, and batching buys nothing.

Run from the repository root::

    python benchmarks/lane_scaling.py
"""

import argparse
import json
import time
import warnings

import numpy as np

warnings.filterwarnings("ignore")


def measure(n_lanes, depth, repeats=3):
    """Time one gradient of the summed log likelihood over ``n_lanes`` sessions."""
    import jax
    import jax.numpy as jnp

    from aind_dynamic_foraging_models.hierarchical_bayes.likelihood import (
        hattori2019_log_likelihood,
    )

    key = jax.random.PRNGKey(0)
    choices = jax.random.bernoulli(key, 0.5, (n_lanes, depth)).astype(jnp.int32)
    rewards = jax.random.bernoulli(key, 0.4, (n_lanes, depth)).astype(jnp.float32)

    def total_log_lik(params):
        """Summed log likelihood across every lane, as a NUTS gradient would see it."""
        per_lane = jax.vmap(
            lambda c, r: hattori2019_log_likelihood(
                c, r,
                learn_rate_rew=params[0], learn_rate_unrew=params[1],
                forget_rate_unchosen=params[2], softmax_inverse_temperature=params[3],
                bias_l=params[4],
            )
        )(choices, rewards)
        return jnp.sum(per_lane)

    grad_fn = jax.jit(jax.grad(total_log_lik))
    params = jnp.array([0.5, 0.3, 0.2, 5.0, 0.0])

    grad_fn(params).block_until_ready()  # compile

    times = []
    for _ in range(repeats):
        started = time.time()
        grad_fn(params).block_until_ready()
        times.append(time.time() - started)
    return float(np.median(times))


def main():
    """Sweep batch width at fixed depth and report scaling."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--depth", type=int, default=650)
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    import jax

    print(f"device={jax.devices()[0]}  depth={args.depth}\n")
    print(f"{'lanes':>8}{'sec/grad':>12}{'vs 640 lanes':>15}{'sec/1k lanes':>15}")

    lane_counts = [640, 1280, 2560, 5120, 10240, 20480]
    results, baseline = [], None
    for n_lanes in lane_counts:
        seconds = measure(n_lanes, args.depth)
        if baseline is None:
            baseline = seconds
        results.append({"lanes": n_lanes, "seconds": seconds, "ratio": seconds / baseline})
        print(
            f"{n_lanes:>8}{seconds:>12.4f}{seconds / baseline:>15.2f}x"
            f"{1000 * seconds / n_lanes:>15.4f}",
            flush=True,
        )

    growth = results[-1]["ratio"] / (lane_counts[-1] / lane_counts[0])
    print(
        f"\n32x the lanes cost {results[-1]['ratio']:.1f}x the time "
        f"({growth:.2f} of linear).\n"
        "Near 0 means latency-bound and batching is nearly free; "
        "near 1 means throughput-bound and it is not."
    )

    if args.output:
        with open(args.output, "w") as handle:
            json.dump(results, handle, indent=2)


if __name__ == "__main__":
    main()
