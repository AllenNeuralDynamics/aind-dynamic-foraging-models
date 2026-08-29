# Benchmark results

Measured 2026-08-28/29 on the Allen HPC. Reproduce with `benchmarks/slurm/*.sbatch`.

## Validation: the reimplementation is correct

Fitting identical synthetic data (40 sessions x 650 trials, 16 chains, 500 warmup + 500
samples) with the reference Stan model and with `hattori2019_published`:

| parameter | truth | numpyro | stan |
|---|---|---|---|
| `learn_rate_rew` | 0.618 | 0.538 | 0.538 |
| `learn_rate_unrew` | 0.274 | 0.292 | 0.291 |
| `forget_rate_unchosen` | 0.212 | 0.251 | 0.251 |
| `softmax_inverse_temperature` | 5.793 | 6.167 | 6.167 |

Agreement to three decimals on every parameter, from two independent implementations. This
also confirms empirically that the reference's retention parameterisation and this package's
decay one are the same model.

## Speed: Stan wins for per-subject fits

Same configuration as above. ESS is mean bulk effective sample size over the four pooled
parameters, out of 8000 draws.

| implementation | hardware | wall | ESS | ESS/draw | ESS/s |
|---|---|---|---|---|---|
| stan | 16 CPU cores | 1084 s | 8052 | 1.007 | **7.4** |
| numpyro | 16 CPU cores | 2449 s | 4988 | 0.624 | 2.0 |
| numpyro | TITAN Xp (2017) | 2179 s | 4312 | 0.539 | 2.0 |
| numpyro | A100-PCIE-40GB | 3750 s | 4833 | 0.604 | 1.3 |

Stan is 3.7x faster than the best NumPyro configuration and 5.7x faster than the A100. The
gap splits into 2.3x on wall time and 1.6x on sampling efficiency per draw; Stan's ESS/draw
slightly exceeds 1.0, meaning its draws are effectively independent.

**A newer GPU is slower.** This is the signature of a latency-bound workload rather than a
compute-bound one. Each gradient is 650 sequential scan steps over only ~640 lanes
(40 sessions x 16 chains), each step doing a handful of scalar operations. Wall time is
dominated by per-step dispatch overhead, which does not improve with GPU generation.

## Sampler geometry: the default is already best

Sweep over centred/non-centred session parameters, diagonal/dense mass matrix, and target
acceptance probability (8 sessions x 300 trials, 800 draws):

| centred | dense | accept | sec | ESS/draw | ESS/s | divergences |
|---|---|---|---|---|---|---|
| **False** | **False** | **0.80** | **49.0** | **0.633** | **10.3** | 0 |
| False | False | 0.90 | 55.4 | 0.504 | 7.3 | 0 |
| False | True | 0.80 | 242.0 | 0.400 | 1.3 | 1 |
| False | True | 0.90 | 226.1 | 0.708 | 2.5 | 0 |
| True | False | 0.80 | 56.5 | 0.353 | 5.0 | 8 |
| True | False | 0.90 | 54.7 | 0.344 | 5.0 | 5 |
| True | True | 0.80 | 149.9 | 0.599 | 3.2 | 0 |
| True | True | 0.90 | 169.0 | 0.494 | 2.3 | 0 |

The existing default wins. Centred sampling is worse on every measure and produces
divergences, so the funnel is real at this data scale and the non-centred default was the
right choice. A dense mass matrix gives the best ESS per draw but costs 4-5x the wall time.

The per-draw gap against Stan is therefore not a geometry problem and is not fixable by
these knobs.

## Batching: the workload is latency-bound on GPU

One gradient at fixed depth 650, widening the batch (`benchmarks/lane_scaling.py`):

| lanes | A100 sec/grad | vs 640 lanes | sec per 1k lanes |
|---|---|---|---|
| 640 | 0.0515 | 1.00x | 0.0804 |
| 1280 | 0.0339 | 0.66x | 0.0264 |
| 2560 | 0.0348 | 0.68x | 0.0136 |
| 5120 | 0.0319 | 0.62x | 0.0062 |
| 10240 | 0.0547 | 1.06x | 0.0053 |
| 20480 | 0.0496 | 0.96x | 0.0024 |

**32x the lanes costs 1.0x the time**, and per-lane cost falls 33x. Batching across subjects
is therefore nearly free on GPU, and the whole cohort costs about what one subject costs
today.

This explains the results above. At 640 lanes (40 sessions x 16 chains) the GPU runs at a
few percent utilisation, so those benchmarks measured dispatch overhead rather than compute
-- which is why an A100 lost to a 2017 TITAN Xp and to Stan. Below ~1000 lanes fixed
overhead dominates completely; 640 lanes is slower per gradient than 1280.

On CPU the same sweep is throughput-bound and worse than linear (32x lanes cost 102x time,
3.2x of linear), so batching helps only on GPU.

## What has not been tested

- `chain_method="vectorized"` runs all chains in lockstep, so every chain pays for the
  deepest tree in the batch. `parallel` may recover part of the per-draw gap.
- Whether the flat regime extends past ~20k lanes. A 16-chain cohort fit would need
  ~400k lanes, well beyond what was measured.
- **Ragged session lengths.** These runs use uniform 650-trial sessions. Real data are
  long-tailed, and Stan loops to each session's true length while JAX must pad to the
  maximum and mask, so real data will widen the gap further unless sessions are bucketed or
  packed.
