"""Throughput of concurrent EM fits on the dumped production cohort.

    bench_parallel.py NPZ EMISSION K L MODE R

MODE
  seq      one global-fit restart alone (the sequential baseline)
  threads  R global-fit restarts at once, in a thread pool in one process
  procs    R global-fit restarts at once, one spawned process each, cores split between them
  refits   per-mouse refits of 3*R training mice: one at a time, then R threads

Each restart runs A, then B EM iterations (tol 0) after a warm-up that compiles every bucket
shape. Row compression is computed once and reused, as ``prepare_data`` now does (this script
predates it and patches ``prepare_m_step`` instead). Per-iteration time is
(T_B - T_A) / (B - A), with every worker busy for the whole window. Results: RESULTS.md.
"""
import multiprocessing as mp
import os
import resource
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np

A, B = 3, 9
STATE = {}


def setup(npz, emission, K, L, cores=None):
    """Load the cohort, build the model, compile, and cache the row compression in STATE."""
    if cores is not None:
        os.sched_setaffinity(0, cores)  # before jax starts its CPU backend
    import jax
    import aind_dynamic_foraging_models.glm_hmm.model as M

    d = np.load(npz)
    s = np.cumsum(d["train_lengths"])[:-1]
    sessions = list(zip(np.split(d["train_choices"].astype(int), s),
                        np.split(d["train_rewards"].astype(int), s)))
    subjects = d["train_subjects"]
    model = M.GLMHMM(K, L, emission)
    cls = type(model.emission_component)
    prep, cache, lock = cls.prepare_m_step, {}, threading.Lock()

    def cached(self, labels, inputs):
        """Compress once per data shape, then reuse."""
        key = labels.shape
        with lock:
            if key not in cache:
                cache[key] = prep(self, labels, inputs)
        return cache[key]

    cls.prepare_m_step = cached
    buckets = M.make_buckets(sessions, L)
    _, props = model.initialize()
    p0, _ = model.initialize(jax.random.PRNGKey(99), noise_scale=0.2)
    t = time.time()
    M.fit_em(model, p0, props, buckets, M.GLOBAL_PRIOR, max_iters=1, tol=0.0)
    STATE.update(M=M, jax=jax, model=model, props=props, buckets=buckets, sessions=sessions,
                 subjects=subjects, L=L, warmup=time.time() - t)


def restart(seed):
    """One start: seconds per EM iteration, its log-joint trace, and its dtype."""
    M, jax, model = STATE["M"], STATE["jax"], STATE["model"]
    params, _ = model.initialize(jax.random.PRNGKey(seed), noise_scale=0.2)
    out = {}
    for n in (A, B):
        t = time.time()
        r = M.fit_em(model, params, STATE["props"], STATE["buckets"], M.GLOBAL_PRIOR,
                     max_iters=n, tol=0.0)
        out[n] = time.time() - t
    lj = np.asarray(r.log_joint)
    dtype = str(np.asarray(r.params.emissions[0]).dtype)
    return (out[B] - out[A]) / (B - A), lj, dtype


def proc_main(args, index, R, barrier, results):
    """One spawned worker on its share of the cores; puts its result on ``results``."""
    npz, emission, K, L = args
    allowed = sorted(os.sched_getaffinity(0))
    share = max(1, len(allowed) // R)  # fewer cores than workers: they share
    mine = {allowed[(index * share + j) % len(allowed)] for j in range(share)}
    setup(npz, emission, K, L, cores=mine)
    barrier.wait()
    per_iter, lj, dtype = restart(index)
    results.put((index, per_iter, lj[-1], dtype,
                 resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6))


def maxrss_gb():
    """Peak resident memory of this process, in GB."""
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6


def main():
    """Run one MODE and print a RESULT line."""
    npz, emission, K, L, mode, R = sys.argv[1:7]
    K, L, R = int(K), int(L), int(R)
    cores = len(os.sched_getaffinity(0))
    tag = f"{mode} {emission} K={K} L={L} R={R} cores={cores}"
    if mode == "procs":
        ctx = mp.get_context("spawn")
        barrier, results = ctx.Barrier(R + 1), ctx.Queue()
        procs = [ctx.Process(target=proc_main, args=((npz, emission, K, L), i, R, barrier, results))
                 for i in range(R)]
        for p in procs:
            p.start()
        barrier.wait()
        t = time.time()
        got = sorted(results.get() for _ in range(R))
        wall = time.time() - t
        for p in procs:
            p.join()
        per = [g[1] for g in got]
        print(f"RESULT {tag}: per-iteration per restart median {np.median(per):.2f}s "
              f"(min {min(per):.2f}, max {max(per):.2f}); wall for {A}+{B} iterations "
              f"{wall:.0f}s; per-process maxrss {max(g[4] for g in got):.1f} GB; "
              f"dtypes {set(g[3] for g in got)}; "
              f"final log joints {[round(g[2], 2) for g in got]}", flush=True)
        return
    t = time.time()
    setup(npz, emission, K, L)
    print(f"SETUP {tag}: {time.time() - t:.0f}s (warm-up fit {STATE['warmup']:.0f}s)", flush=True)
    if mode == "seq":
        per, lj, dtype = restart(0)
        print(f"RESULT {tag}: per-iteration {per:.2f}s; dtype {dtype}; "
              f"maxrss {maxrss_gb():.1f} GB; final log joint {lj[-1]:.4f}", flush=True)
    elif mode == "threads":
        barrier = threading.Barrier(R)

        def run(seed):
            """Start with the others, then fit."""
            barrier.wait()
            return restart(seed)

        t = time.time()
        with ThreadPoolExecutor(R) as pool:
            got = list(pool.map(run, range(R)))
        wall = time.time() - t
        per = [g[0] for g in got]
        print(f"RESULT {tag}: per-iteration per restart median {np.median(per):.2f}s "
              f"(min {min(per):.2f}, max {max(per):.2f}); wall for {A}+{B} iterations {wall:.0f}s; "
              f"maxrss {maxrss_gb():.1f} GB; dtypes {set(g[2] for g in got)}; "
              f"seed 0 final log joint {got[0][1][-1]:.4f}", flush=True)
    elif mode == "refits":
        M, model = STATE["M"], STATE["model"]
        rng = np.random.default_rng(0)
        mice = rng.choice(np.unique(STATE["subjects"]), 3 * R, replace=False)
        start = STATE["jax"].random.PRNGKey(5)
        gparams, _ = model.initialize(start, noise_scale=0.2)

        def refit(mouse, iters=20):
            """One mouse's subject-prior fit; its final log joint."""
            sub = [s for s, sj in zip(STATE["sessions"], STATE["subjects"]) if sj == mouse]
            f = M.fit_em(model, gparams, STATE["props"], M.make_buckets(sub, STATE["L"]),
                         M.SUBJECT_PRIOR, max_iters=iters, tol=0.0)
            return f.log_joint[-1]

        for m in mice:  # compile every bucket shape first, so neither timing pays for it
            refit(m, iters=1)
        t = time.time()
        seq = [refit(m) for m in mice]
        t_seq = time.time() - t
        t = time.time()
        with ThreadPoolExecutor(R) as pool:
            par = list(pool.map(refit, mice))
        t_par = time.time() - t
        print(f"RESULT {tag}: {len(mice)} refits sequential {t_seq:.0f}s, {R} threads {t_par:.0f}s "
              f"(x{t_seq / t_par:.1f}); max |dLJ| {np.max(np.abs(np.array(seq) - par)):.2e}",
              flush=True)


if __name__ == "__main__":
    main()
