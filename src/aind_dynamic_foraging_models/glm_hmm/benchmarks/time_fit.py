"""Time GLM-HMM EM on a real cohort, on whichever device ``AIND_GLMHMM_DEVICE`` selects.

Three modes, all on the ``train`` sessions of a cohort file written by ``dump_cohort.py``:

    python time_fit.py COHORT.npz parts    hurdle 10 5   # one iteration, split by step
    python time_fit.py COHORT.npz converge hurdle 10 5   # one restart to convergence
    python time_fit.py COHORT.npz refits   hurdle 10 5   # 30 per-mouse refits

``parts`` compiles with one EM call, then times three iterations. It splits them into the
E-step, the host-side aggregation of the posteriors and the M-step, and reports the one-time
row compression separately. ``converge`` fits K = 1 and then one K-state restart from its
weights plus noise, as the trainer does, up to ``--max-iters``, and saves the log-joint
trajectory next to the cohort file. ``refits`` refits 30 training mice under the subject
prior from that restart's parameters and scores their eval sessions, as the trainer's
K-selection step does. Each mode prints one line starting with its name in capitals.
"""

import argparse
import os
import time
from pathlib import Path

import jax
import numpy as np

import aind_dynamic_foraging_models.glm_hmm.model as glm_model
from aind_dynamic_foraging_models.glm_hmm.model import (
    GLMHMM,
    GLOBAL_PRIOR,
    SUBJECT_PRIOR,
    fit_em,
    DEVICE_ENV,
    make_buckets,
    predictive_log_probs,
)

TIMES = {"e_step": 0.0, "aggregate": 0.0, "m_step": 0.0, "compress": 0.0}


def _timed(name, fn):
    """Wrap ``fn`` so its wall time, after the result is ready, adds to ``TIMES[name]``."""

    def wrapper(*args, **kwargs):
        """Call ``fn`` and wait for its result."""
        started = time.time()
        result = jax.block_until_ready(fn(*args, **kwargs))
        TIMES[name] += time.time() - started
        return result

    return wrapper


def _instrument(model):
    """Time the E-step, aggregation, M-step and row compression that ``fit_em`` calls."""
    glm_model._bucket_e_step = _timed("e_step", glm_model._bucket_e_step)
    glm_model.aggregate = _timed("aggregate", glm_model.aggregate)
    glm_model._m_step = _timed("m_step", glm_model._m_step)
    component = type(model.emission_component)
    prepare = component.prepare_m_step
    component.prepare_m_step = _timed("compress", lambda self, *a: prepare(self, *a))


def sessions(cohort, split, subjects=None):
    """(choices, rewards) per session of one split, optionally for some subjects only."""
    splits = np.cumsum(cohort[f"{split}_lengths"])[:-1]
    choices = np.split(cohort[f"{split}_choices"].astype(int), splits)
    rewards = np.split(cohort[f"{split}_rewards"].astype(int), splits)
    keep = cohort[f"{split}_subjects"]
    return [(c, r) for c, r, s in zip(choices, rewards, keep)
            if subjects is None or s in subjects]


def time_parts(model, buckets):
    """Compile with one EM iteration, then time three and split them by step."""
    params, props = model.initialize(jax.random.PRNGKey(0), noise_scale=0.2)
    first = fit_em(model, params, props, buckets, GLOBAL_PRIOR, max_iters=1, tol=0.0)
    for key in TIMES:
        TIMES[key] = 0.0
    started = time.time()
    fit = fit_em(model, first.params, props, buckets, GLOBAL_PRIOR, max_iters=3, tol=0.0)
    seconds = time.time() - started
    # max_iters=3 runs 4 E-steps, 3 aggregations and 3 M-steps, after one row compression.
    return (f"steady {(seconds - TIMES['compress']) / 3:.2f}s per iteration + row compression "
            f"{TIMES['compress']:.1f}s per fit_em call [E-step {TIMES['e_step'] / 4:.2f}s, "
            f"aggregate {TIMES['aggregate'] / 3:.2f}s, M-step {TIMES['m_step'] / 3:.2f}s]; "
            f"log joints {np.round(fit.log_joint, 3).tolist()}")


def fit_restart(model, buckets, max_iters):
    """K = 1, then one K-state restart from its weights plus N(0, 0.2) noise."""
    single = GLMHMM(1, model.n_lags, model.emission)
    base = fit_em(single, *single.initialize(), buckets, GLOBAL_PRIOR)
    weights = {name: np.asarray(value)[0]
               for name, value in base.params.emissions._asdict().items()}
    start, props = model.initialize(jax.random.PRNGKey(1), noise_scale=0.2, **weights)
    started = time.time()
    fit = fit_em(model, start, props, buckets, GLOBAL_PRIOR, max_iters=max_iters)
    return fit, props, time.time() - started


def time_converge(model, buckets, n_trials, max_iters, out):
    """One restart to convergence; save its trajectory and report where the gain falls off."""
    fit, _, seconds = fit_restart(model, buckets, max_iters)
    np.save(out, fit.log_joint)
    iters = len(fit.log_joint) - 1
    gains = np.diff(fit.log_joint)

    def first_below(per_trial):
        """First iteration whose gain is below ``per_trial`` nats per trial."""
        below = np.flatnonzero(gains < per_trial * n_trials)
        return int(below[0]) + 1 if below.size else None

    return (f"{iters} iterations in {seconds:.0f}s ({seconds / max(iters, 1):.2f}s each), "
            f"converged={fit.converged}; gain first below 1e-5/trial at {first_below(1e-5)}, "
            f"below 1e-6/trial at {first_below(1e-6)}; trajectory in {out.name}")


def time_refits(model, cohort, buckets, n_mice=30):
    """Per-mouse MAP refits from one global restart, as the trainer's K selection runs them."""
    fit, props, _ = fit_restart(model, buckets, max_iters=300)
    mice = np.random.default_rng(0).choice(np.unique(cohort["train_subjects"]), n_mice,
                                           replace=False)
    seconds = []
    for mouse in mice:
        started = time.time()
        own = make_buckets(sessions(cohort, "train", {mouse}), model.n_lags)
        refit = fit_em(model, fit.params, props, own, SUBJECT_PRIOR)
        predictive_log_probs(model, refit.params,
                             make_buckets(sessions(cohort, "eval", {mouse}), model.n_lags))
        seconds.append(time.time() - started)
    return (f"{n_mice} training mice, per mouse median {np.median(seconds):.1f}s, "
            f"mean {np.mean(seconds):.1f}s, max {np.max(seconds):.1f}s")


def main():
    """Parse the arguments and run one mode."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("cohort", type=Path)
    parser.add_argument("mode", choices=["parts", "converge", "refits"])
    parser.add_argument("emission", choices=["hurdle", "reference"])
    parser.add_argument("num_states", type=int)
    parser.add_argument("n_lags", type=int)
    parser.add_argument("--max-iters", type=int, default=1000)
    args = parser.parse_args()

    device = os.environ.get(DEVICE_ENV, "cpu")
    cohort = np.load(args.cohort)
    train = sessions(cohort, "train")
    n_trials = sum(len(c) for c, _ in train)
    buckets = make_buckets(train, args.n_lags)
    model = GLMHMM(args.num_states, args.n_lags, args.emission)
    _instrument(model)
    if args.mode == "parts":
        result = time_parts(model, buckets)
    elif args.mode == "converge":
        out = args.cohort.with_name(
            f"{args.cohort.stem}_{args.emission}_K{args.num_states}_L{args.n_lags}_"
            f"{device}.npy")
        result = time_converge(model, buckets, n_trials, args.max_iters, out)
    else:
        result = time_refits(model, cohort, buckets)
    print(f"{args.mode.upper()} {args.emission} K={args.num_states} L={args.n_lags} "
          f"N={n_trials} sessions={len(train)} device={device}: {result}",
          flush=True)


if __name__ == "__main__":
    main()
