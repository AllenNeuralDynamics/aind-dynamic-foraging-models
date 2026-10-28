# GLM-HMM benchmark: CPU vs GPU

Measured 2026-10-27 on the Allen HPC. Discussion and follow-ups:
- the GPU decision and the dedicated image:
  [AllenNeuralDynamics/aind-dynamic-foraging-bfm-wrapper#111](https://github.com/AllenNeuralDynamics/aind-dynamic-foraging-bfm-wrapper/issues/111);
- the EM speed-ups:
  [AllenNeuralDynamics/aind-dynamic-foraging-models#82](https://github.com/AllenNeuralDynamics/aind-dynamic-foraging-models/issues/82).

## Setup

- **Data:** study 14's production training cohort. 614 mice, 9,227 train sessions,
  4,887,818 trials (`mice_snapshot_scaling`, snapshot 20260603, `ignore_policy=include`).
  Written by `dump_cohort.py`.
- **Model:** K = 10, one fixed set of starting parameters per configuration, this package at
  models#81.
- **CPU:** 8 cores of an HPC node, the core count of study 14's Beaker template.
- **GPUs:** HPC V100, A100 and L40S nodes, in a clean environment with the CUDA solver
  libraries pinned (see "Why the GPU needed a pin").
- **Runs:** single, not repeated. ≈ marks a number added up from measured parts, not timed
  end to end.

## Verdict

- **Same answers on both devices.** At 4.9M trials, GPU and CPU log joints agree to within
  0.005. A 13-mouse real-data run through the wrapper's trainer gives identical scores on both
  (log joint −29643.79, K-selection 0.62246, zero-shot 0.49801, matched 0.52822).
- **Run the pooled global fit on a GPU.** One restart to convergence is 2.7–5.6× faster at
  L = 5. A steady iteration is 3–5× faster at L = 5 and 5–25× at L = 10: the E-step is 6–8×
  faster, and the L = 10 M-step 20–160×.
- **Run per-mouse refits on the CPU**, where they are about 2× faster. One mouse has a few
  thousand trials, so launch latency and recompiles for new bucket shapes dominate on a GPU.
- **Remaining GPU bottlenecks** (models#82):
  - the host-side aggregation of the posteriors, about 1 s per iteration;
  - a row compression of 37–107 s, once per `fit_em` call when this was measured, and now
    once per run (`prepare_data`, shared by `fit_with_restarts`).

## Steady cost of one EM iteration, in seconds (`time_fit.py parts`)

Measured as the difference between calls of 6 and 3 iterations, so setup, the final scoring
E-step and row compression drop out (jobs 27512167–70: CPU node n254, V100, A100, L40S).

| whole iteration | CPU (8 cores) | V100 | A100 | L40S |
|---|---|---|---|---|
| L = 5 hurdle | 10.07 | 3.43 | 2.16 | 2.23 |
| L = 5 reference | 8.72 | 2.47 | 1.63 | 2.05 |
| L = 10 hurdle | 28.02 | 5.33 | 2.53 | 3.19 |
| L = 10 reference | 47.99 | 4.16 | 1.88 | 4.26 |

| one step, per iteration | CPU (8 cores) | V100 | A100 | L40S |
|---|---|---|---|---|
| E-step, L = 5 hurdle | 8.30 | 1.02 | 1.18 | 1.06 |
| E-step, L = 10 reference | 8.77 | 1.05 | 1.22 | 1.48 |
| M-step, L = 5 hurdle | 0.06 | 0.01 | 0.01 | 0.02 |
| M-step, L = 10 hurdle | 16.20 | 0.42 | 0.16 | 0.41 |
| M-step, L = 10 reference | 40.86 | 0.69 | 0.26 | 2.09 (weak float64) |
| host aggregation, L = 5 hurdle | 1.53 | 0.50 | 0.93 | 0.96 |

- **Per `fit_em` call**, a further fixed 61–121 s on CPU and 32–87 s on GPU. Most of it is
  the row compression (30–107 s), which is run 11 times per production run (models#82).
- **What the whole-iteration figure also includes:** moving the posteriors to the host and
  the Python loop over buckets, which is why the steps do not add up to it.
- **Noise:** single runs. On the shared CPU node a step can come out larger than the whole
  iteration (reference L = 5: E-step 9.42 s against 8.72 s).

## One K = 10 restart to convergence, L = 5, tol 1e-6 nats/trial (`converge`, `refits`)

| | CPU | V100 | A100 | speed-up |
|---|---|---|---|---|
| hurdle | 2,470 s (stopped at the 300-iteration cap) | 914 s (300, capped) | 929 s (317, converged) | ≈ 2.7× |
| reference | 2,763 s (220, converged) | — | 495 s (220, converged) | 5.6× |
| per-mouse refit, median of 30 | 8.3 s (hurdle), 8.0 s (reference) | 16.5 s | — | GPU 2× slower |

- **hurdle:** the gain first falls below 1e-5 nats/trial at iteration 116. Iterations 300 to
  317 add 206 nats, or 4e-5 nats/trial.
- **reference:** the gain falls below 1e-5 nats/trial at iteration 165.

## One K = 10 production run: 10 restarts plus 763 per-mouse refits

The refits cover 614 training mice for K selection and 149 held-out mice.
- **Global fit:** 10 × (iterations × steady cost + fixed cost per call), with hurdle 317
  iterations and reference 220. L = 10 assumes the L = 5 iteration counts.
- **Refits:** 763 × 8.5 s on CPU, or × 17.6 s on GPU.
- Ranges span the emissions and the GPU types.

| setup | L = 5 | L = 10 (≈) |
|---|---|---|
| all CPU | about 9–11 h | about 27–31 h |
| global fit on GPU, refits on CPU | about 3–5 h | about 3–7 h |
| all GPU (the trainer as of models#81) | about 5–7 h | about 5–9 h |

## Restarts and refits at once (`bench_parallel.py`, `max_workers` / the trainer's `num_workers`)

K = 10, hurdle, L = 5, the same cohort; seconds per EM iteration of each restart:

| cores (node CPU) | one restart alone | 10 restarts at once, each | restart phase |
|---|---|---|---|
| 8 (Xeon Gold 6238) | 8.9 | 29.1 (threads) | 3.1× faster |
| 16 (Xeon Gold 6238R, busy 112-core node) | — | 28.0 (threads) | — |
| 32 (Xeon E5-2630 v4, 40-core node) | 6.1 | 12.0 (threads), 12.3 (processes) | 5.1× faster |

- **Speed-up** is 10 × (alone) / (each, at once): the restart phase's wall clock when the
  restarts need equal iteration counts. The slowest restart sets it.
- **Threads match processes**, so the GIL is not what limits them. Ten restarts in threads use
  31 GB; ten processes use 5.4 GB each.
- **Not 10×: memory bandwidth.** One restart alone keeps about 2 of 8 cores busy, yet the
  16-core run, on a node shared with other jobs, was no faster than 8 cores.
- **Same fits.** Seed 0's log joint is identical alone and in threads on the same node, and
  every result is float64. Across nodes it differs in the fourth decimal (−2397424.1955 on
  32 cores, −2397424.1953 on 8): XLA's thread count changes the order of sums.
- **Refits:** 30 refits of 20 iterations took 24 s one at a time and 8 s in 10 threads
  (3.2×), with identical log joints.
- **≈ production run** (K = 10, L = 5, hurdle, 32 cores, `num_workers` 10): restarts
  317 × 12 s ≈ 65 min, refits 763 × 8.5 s / 3.2 ≈ 35 min, so ≈ 2 h against 9–11 h on 8 cores
  one fit at a time.

## Caveats

- **Host-side variation:** host steps depend on the node's CPU.
- **Shared node:** the two CPU runs to convergence shared one node, which may explain
  reference's 12.6 s per iteration against hurdle's 8.2 s.
- **L = 10 never run to convergence:** those rows combine measured per-step times with the
  L = 5 iteration counts.
- **No H200 number:** the HPC H200 jobs never left the queue. The GLM-HMM image's Beaker
  smoke test (wrapper `beaker/glmhmm/smoke.yaml`) runs on an H200.

## Why the GPU needed a pin (`gpu_check.py`)

pip resolves jax-cuda12-plugin 0.6.1 to cuSOLVER 11.7.5.82 (CUDA 12.9) over cuBLAS
12.8.5.5. The plugin caps cuBLAS below 12.9 but not cuSOLVER. Results on a V100 unless noted,
each in a clean venv:

| CUDA libraries | float64 `jnp.linalg.solve` on the GPU |
|---|---|
| the shared HPC env's set, which is pip's resolution | fails: `gpusolverDnCreate ... cuSolver internal error` |
| the same, with only `nvidia-cusolver-cu12==11.7.3.90`, `nvidia-cusparse-cu12==12.5.8.93` and `nvidia-nvjitlink-cu12==12.8.93` | works (V100, A100; this is the image's pin) |
| a consistent CUDA 12.8 set | works on V100, A100 and L40S |
| pip's resolution with cuBLAS forced to 12.9.1.4 | works, but breaks the plugin's pin |

An earlier "cuSOLVER 11.7.3 does not help" result came from a venv layered over the shared
env. That venv never loaded its own 11.7.3, which is why `gpu_check.py` prints the library
paths it loaded.

## Reproduce

1. Write the cohort with `dump_cohort.py`. It needs wrapper and dispatcher checkouts and a
   wrapper environment, and the file holds real mouse data, so keep it out of the repository.
2. Build a GPU environment. Create a clean Python 3.12 venv and install with the image's
   constraints, so that its GPU solves work:

       pip install -c <wrapper>/beaker/glmhmm/cuda-constraints.txt \
           "jax[cuda12]==0.6.1" "aind-dynamic-foraging-models[glmhmm]"

3. From this directory:

       sbatch --gres=gpu:v100:1 slurm/gpu_check.sbatch PYTHON
       sbatch slurm/time_fit.sbatch PYTHON cohort.npz parts hurdle 10 5
       sbatch --gres=gpu:a100:1 --export=ALL,AIND_GLMHMM_DEVICE=gpu \
           slurm/time_fit.sbatch PYTHON cohort.npz converge hurdle 10 5

4. Concurrency (CPU only):

       sbatch -p aind --ntasks=1 -c 32 --mem=180G --wrap \
           "PYTHON bench_parallel.py cohort.npz hurdle 10 5 threads 10"

   `MODE` is `seq`, `threads`, `procs` or `refits`; jobs 27514561 and 27514563 (32 and 8
   cores).

## Raw result lines

Steady-cost lines from this `time_fit.py` (the 6-minus-3-iteration difference), with log
joints trimmed:

    CPU  27512167  PARTS hurdle K=10 L=5     steady 10.07s [E-step 8.30s, aggregate 1.53s, M-step 0.06s]; fixed 77.5s per call (compression 66.6s)
    CPU  27512167  PARTS reference K=10 L=5  steady 8.72s [E-step 9.42s, aggregate 0.90s, M-step 0.25s]; fixed 61.0s per call (compression 48.1s)
    CPU  27512167  PARTS hurdle K=10 L=10    steady 28.02s [E-step 8.92s, aggregate 2.07s, M-step 16.20s]; fixed 121.1s per call (compression 107.0s)
    CPU  27512167  PARTS reference K=10 L=10 steady 47.99s [E-step 8.77s, aggregate 1.14s, M-step 40.86s]; fixed 106.4s per call (compression 64.7s)
    V100 27512168  PARTS hurdle K=10 L=5     steady 3.43s [E-step 1.02s, aggregate 0.50s, M-step 0.01s]; fixed 52.8s per call (compression 50.3s)
    V100 27512168  PARTS reference K=10 L=5  steady 2.47s [E-step 0.93s, aggregate 1.06s, M-step 0.02s]; fixed 40.6s per call (compression 37.8s)
    V100 27512168  PARTS hurdle K=10 L=10    steady 5.33s [E-step 1.47s, aggregate 2.51s, M-step 0.42s]; fixed 87.0s per call (compression 83.2s)
    V100 27512168  PARTS reference K=10 L=10 steady 4.16s [E-step 1.05s, aggregate 1.47s, M-step 0.69s]; fixed 58.3s per call (compression 53.0s)
    A100 27512169  PARTS hurdle K=10 L=5     steady 2.16s [E-step 1.18s, aggregate 0.93s, M-step 0.01s]; fixed 42.1s per call (compression 39.1s)
    A100 27512169  PARTS reference K=10 L=5  steady 1.63s [E-step 1.16s, aggregate 0.49s, M-step 0.01s]; fixed 34.8s per call (compression 31.9s)
    A100 27512169  PARTS hurdle K=10 L=10    steady 2.53s [E-step 1.29s, aggregate 1.08s, M-step 0.16s]; fixed 65.6s per call (compression 61.9s)
    A100 27512169  PARTS reference K=10 L=10 steady 1.88s [E-step 1.22s, aggregate 0.63s, M-step 0.26s]; fixed 45.1s per call (compression 39.9s)
    L40S 27512170  PARTS hurdle K=10 L=5     steady 2.23s [E-step 1.06s, aggregate 0.96s, M-step 0.02s]; fixed 45.1s per call (compression 41.7s)
    L40S 27512170  PARTS reference K=10 L=5  steady 2.05s [E-step 1.18s, aggregate 0.49s, M-step 0.02s]; fixed 31.8s per call (compression 29.7s)
    L40S 27512170  PARTS hurdle K=10 L=10    steady 3.19s [E-step 1.28s, aggregate 1.07s, M-step 0.41s]; fixed 66.0s per call (compression 63.3s)
    L40S 27512170  PARTS reference K=10 L=10 steady 4.26s [E-step 1.48s, aggregate 0.62s, M-step 2.09s]; fixed 42.3s per call (compression 38.3s)

Earlier lines, from a first version of the script. Its "per EM iteration" divided a
3-iteration call by 3, so it includes one row compression and an extra scoring E-step; its
"steady" figures subtract only the compression. These lines are kept as the source of the
end-to-end and refit numbers (FULL, REFITS, LONG), which are whole-fit wall times and are
not affected:

    CPU  27510209  ITERS hurdle K=10 L=5    per EM iteration 35.52s [E-step 6.37s, host aggregate 2.86s, M-step 0.05s]
    CPU  27510209  ITERS reference K=10 L=5 per EM iteration 30.48s [E-step 7.34s, host aggregate 1.50s, M-step 0.17s]
    CPU  27510209  ITERS hurdle K=10 L=10   per EM iteration 61.59s [E-step 7.57s, host aggregate 3.28s, M-step 11.14s]
    CPU  27510209  ITERS reference K=10 L=10 per EM iteration 73.37s [E-step 8.33s, host aggregate 1.93s, M-step 34.18s]
    V100 27510205  ITERS hurdle K=10 L=5    per EM iteration 19.03s [E-step 1.05s, host aggregate 1.19s, M-step 0.01s]
    V100 27510205  ITERS reference K=10 L=5 per EM iteration 16.44s [E-step 1.04s, host aggregate 0.63s, M-step 0.02s]
    V100 27510205  ITERS hurdle K=10 L=10   per EM iteration 29.07s [E-step 1.35s, host aggregate 1.48s, M-step 0.41s]
    V100 27510205  ITERS reference K=10 L=10 per EM iteration 23.38s [E-step 1.34s, host aggregate 1.50s, M-step 0.70s]
    A100 27510206  ITERS hurdle K=10 L=5    per EM iteration 16.58s [E-step 1.12s, host aggregate 0.91s, M-step 0.01s]
    A100 27510206  ITERS reference K=10 L=5 per EM iteration 12.47s [E-step 1.16s, host aggregate 0.49s, M-step 0.01s]
    A100 27510206  ITERS hurdle K=10 L=10   per EM iteration 23.40s [E-step 1.29s, host aggregate 1.08s, M-step 0.16s]
    A100 27510206  ITERS reference K=10 L=10 per EM iteration 15.86s [E-step 1.26s, host aggregate 0.64s, M-step 0.26s]
    L40S 27510208  ITERS hurdle K=10 L=5    per EM iteration 15.96s [E-step 1.17s, host aggregate 0.91s, M-step 0.02s]
    L40S 27510208  ITERS reference K=10 L=5 per EM iteration 12.23s [E-step 1.19s, host aggregate 0.49s, M-step 0.02s]
    L40S 27510208  ITERS hurdle K=10 L=10   steady 4.06s + row compression 59.5s per fit_em call [E-step 1.31s, host aggregate 1.10s, M-step 0.42s]
    L40S 27510208  ITERS reference K=10 L=10 steady 5.34s + row compression 37.5s per fit_em call [E-step 1.34s, host aggregate 0.65s, M-step 2.09s]
    V100 27510245  ITERS hurdle K=10 L=5    steady 3.71s + row compression 48.4s per fit_em call [E-step 1.04s, host aggregate 1.22s, M-step 0.01s]
    V100 27510245  ITERS hurdle K=10 L=10   steady 4.93s + row compression 71.2s per fit_em call [E-step 1.31s, host aggregate 1.45s, M-step 0.41s]
    A100 27510565  ITERS hurdle K=10 L=5    steady 3.11s + row compression 39.1s per fit_em call [E-step 1.14s, host aggregate 0.93s, M-step 0.01s]
    V100 27510566  ITERS hurdle K=10 L=5    steady 3.29s + row compression 47.9s per fit_em call [E-step 0.99s, host aggregate 1.07s, M-step 0.01s]
    CPU  27510210  FULL hurdle K=10 L=5     K=1 fit 95s (2 iters); one K=10 restart 2470s, 300 iters (8.23s/iter), converged=False
    CPU  27510210  REFITS hurdle            30 training mice, per mouse median 8.29s, mean 8.54s, max 18.66s
    CPU  27510211  FULL reference K=10 L=5  K=1 fit 107s (2 iters); one K=10 restart 2763s, 220 iters (12.56s/iter), converged=True
    CPU  27510211  REFITS reference         30 training mice, per mouse median 7.96s, mean 9.12s, max 23.63s
    V100 27510245  FULL hurdle K=10 L=5     K=1 fit 77s (2 iters); one K=10 restart 914s, 300 iters (3.05s/iter), converged=False
    V100 27510245  REFITS hurdle            30 training mice, per mouse median 16.46s, mean 17.56s, max 40.42s
    A100 27510296  LONG hurdle K=10 L=5     317 iters in 929s, converged=True; gain < 1e-5/trial at 116, < 1e-6/trial at 317
    A100 27510297  LONG reference K=10 L=5  220 iters in 495s, converged=True; gain < 1e-5/trial at 165, < 1e-6/trial at 220
