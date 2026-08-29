# Benchmark results

Measured 2026-08-28/29 on the Allen HPC. Reproduce with `slurm/*.sbatch`.

## Validation: the reimplementation is correct

Fitting identical synthetic data (40 sessions x 650 trials, 16 chains, 500 warmup + 500
samples) with the reference Stan model and with `hattori2019_stan_reference`:

| parameter | truth | numpyro | stan |
|---|---|---|---|
| `learn_rate_rew` | 0.618 | 0.538 | 0.538 |
| `learn_rate_unrew` | 0.274 | 0.292 | 0.291 |
| `forget_rate_unchosen` | 0.212 | 0.251 | 0.251 |
| `softmax_inverse_temperature` | 5.793 | 6.167 | 6.167 |

Agreement to three decimals on every parameter, from two independent implementations. This
also confirms empirically that the reference's retention parameterisation and this package's
decay one are the same model.

## Speed: Stan wins for per-subject fits (but see the section after it)

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

## Where NumPyro wins: cost is flat in cohort size, Stan's is not

The section above is the honest per-subject result and it favours Stan. It is also the
configuration this package should never run in production, so it needs its complement.

**Measured.** Widening one gradient from 640 to 20,480 lanes costs 1.0x the time on an A100
(see "Batching" below). Adding subjects to a fit is therefore close to free here. The same
holds for held-out scoring: 16x the subjects for 1.64x the time.

**Inferred, not measured.** Stan has no equivalent path for this likelihood. Its per-gradient
work is a sequential loop over every trial of every session, so cohort cost grows with total
trials; `reduce_sum` can split that across threads but the work itself does not shrink, and
Stan's GPU support does not cover a user-written recurrent scan. We have not benchmarked Stan
at several cohort sizes, so treat the linearity as a property of the implementation rather
than a measurement.

**The consequence.** Per subject, Stan is 3.7x faster. Across a cohort, NumPyro's cost barely
moves while Stan's accumulates:

| workload | Stan | NumPyro (batched, GPU) |
|---|---|---|
| 1 subject, 40 sessions x 650 trials, 16 chains | **1084 s measured** | 3.7x slower, **measured** |
| 30 subjects as 30 separate per-subject fits | ~9 h, *projected* as 30 x the row above | not the mode this package runs in |
| 30 subjects, one joint three-level fit | **not possible with the reference model** -- it is two-level per subject, so this would need a Stan model that does not exist and has never been written or run | 79 min **measured** (synthetic, 25 sessions x 500 trials); 3 h 32 m on real data with sessions up to 1238 trials, that figure including the k=0 scoring pass |
| 614 subjects as separate per-subject fits, 128 cores | ~23 h, *projected* from measured core-seconds | ~1 subject's cost, *projected* from flat lane scaling |
| 153 held-out subjects, one conditioning rung | not applicable | ~20 s **measured**, from ~4 h sequential |

**Read the middle row carefully.** The comparison there is not Stan-slower-than-NumPyro; it
is that Stan cannot express the estimator at all without a new model. Extending it to three
levels is possible in principle -- it is an ordinary hierarchical model -- but would also
need `reduce_sum` to use more than 16 cores, and would carry a trajectory-length penalty from
the higher dimension. None of that has been attempted, so no Stan joint-fit number exists
anywhere in this document.

So the framework choice is not "NumPyro is faster" -- it is not, per subject. It is that
**the cohort is the unit of work, and only one of the two can treat it that way.** A build
that fits subjects one at a time has chosen the slower framework for no benefit.

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

One gradient at fixed depth 650, widening the batch (`lane_scaling.py`):

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

## Depth, not total work, sets the wall clock

The lane-scaling probe above is a microbenchmark. The same behaviour appears on a real
one-stage cohort fit, which is stronger evidence:

| cohort | subjects | sessions | trials | wall |
|---|---|---|---|---|
| synthetic | 30 | 25 | 500 | 4745 s |
| real (study 01 D~30) | 29 | up to 49 | up to 1238 | 12706 s |

Total work grew 4.85x (1.96x sessions x 2.48x depth); wall time grew 2.68x, tracking the
depth ratio of 2.48x. The extra sessions were nearly free.

## Two-stage is the expensive estimator on GPU

The same fact read the other way. A one-stage joint fit places every subject in a single
batched gradient and pays the depth cost **once**. Two-stage runs one sequential NUTS fit
per subject and pays it **once per subject**. On a latency-bound device that is the worst
possible layout.

Measured on the study 01 D~30 cohort, 29 subjects:

| estimator | fit wall |
|---|---|
| one-stage (batched) | 3 h 32 m |
| two-stage (sequential) | > 5 h 45 m |

Two-stage was adopted as the cheap approximation to a joint fit assumed unaffordable. On GPU
it is both more expensive and statistically inferior. The qualification is that this compares
a batched implementation against a sequential one: a two-stage fit that vmapped the sampler
across subjects would be competitive. That work was deferred on an estimate of ~25 minutes
for D~30, which the run above overshot by more than tenfold.

## Batched adaptation for held-out scoring

Held-out subjects are independent given a frozen population, so the same batching argument
applies to scoring as to fitting. One batched adaptation fit, widening the subject count:

| subjects | seconds | seconds per subject |
|---|---|---|
| 6 | 10.4 | 1.73 |
| 24 | 17.8 | 0.74 |
| 96 | 17.1 | 0.18 |

16x the subjects costs 1.64x the time, and going from 24 to 96 is free. Measured on CPU;
per-subject cost falls tenfold.

Sequential adaptation over the real 153-subject held-out cohort measured **about four hours
per conditioning rung**, which had forced production runs down from five k rungs to one.
Batched, a rung is tens of seconds and the full sweep is affordable again.

The approximation is that one sampler adapts a single step size across every subject's
block rather than one per subject. That is measured rather than assumed: per-subject
posterior means agree with sequential fitting, and so do the held-out likelihoods that
actually get reported (within 0.01).

## What has not been tested

- `chain_method="vectorized"` runs all chains in lockstep, so every chain pays for the
  deepest tree in the batch. `parallel` may recover part of the per-draw gap.
- Whether the flat regime extends past ~20k lanes. A 16-chain cohort fit would need
  ~400k lanes, well beyond what was measured.
- **Ragged session lengths.** These runs use uniform 650-trial sessions. Real data are
  long-tailed, and Stan loops to each session's true length while JAX must pad to the
  maximum and mask, so real data will widen the gap further unless sessions are bucketed or
  packed.
