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
- **Run the pooled global fit on a GPU.** It is 2.7–5.6× faster at L = 5 and ≈ 5–9× at
  L = 10: the E-step is about 6× faster, and the L = 10 M-step 27–130×.
- **Run per-mouse refits on the CPU**, where they are about 2× faster. One mouse has a few
  thousand trials, so launch latency and recompiles for new bucket shapes dominate on a GPU.
- **Remaining GPU bottlenecks** (models#82):
  - the host-side aggregation of the posteriors, about 1 s per iteration;
  - a one-time row compression of 37–107 s per `fit_em` call.

## Each part of one EM iteration, in seconds (`time_fit.py parts`)

| | CPU (8 cores) | V100 | A100 | L40S |
|---|---|---|---|---|
| E-step, L = 5 hurdle | 6.37 | 1.05 | 1.12 | 1.17 |
| E-step, L = 10 reference | 8.33 | 1.34 | 1.26 | 1.34 |
| M-step, L = 5 hurdle | 0.05 | 0.01 | 0.01 | 0.02 |
| M-step, L = 10 hurdle | 11.14 | 0.41 | 0.16 | 0.42 |
| M-step, L = 10 reference | 34.18 | 0.70 | 0.26 | 2.09 (weak float64) |
| host aggregation, L = 5 hurdle | 2.86 | 1.19 | 0.91 | 0.91 |

## A whole iteration in steady state (row compression excluded), in seconds

| | CPU | V100 | A100 | L40S |
|---|---|---|---|---|
| L = 5 hurdle | ≈ 8 | 3.0–3.7 | 2.9–3.1 | — |
| L = 10 hurdle | ≈ 24.5 | 4.93 | ≈ 3.5 | 4.06 |
| L = 10 reference | ≈ 47 | ≈ 4.5–5 | ≈ 3.2 | 5.34 |

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

| setup | L = 5 | L = 10 (≈) |
|---|---|---|
| all CPU | about 9–9.5 h | about 23–30 h |
| global fit on GPU, refits on CPU | about 3.2–4.4 h | about 5–6 h |
| all GPU (the trainer as of models#81) | about 5–6 h | about 7–8 h |

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

## Raw result lines

These are from the timing jobs, which ran an earlier copy of `time_fit.py` (field names
differ slightly). Before the row compression was timed on its own, "per EM iteration" counted
one compression spread over 3 iterations. Some lines below carry an explicit "steady" figure
instead. Log joints are trimmed.

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
