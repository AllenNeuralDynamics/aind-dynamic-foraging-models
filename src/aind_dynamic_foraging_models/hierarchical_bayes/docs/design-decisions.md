# Hierarchical Bayesian fitting — design decisions

Why `hierarchical_bayes` is built the way it is. Measurements behind these live in
[`../benchmarks/RESULTS.md`](../benchmarks/RESULTS.md).

This package is standalone: everything needed to understand, run and validate the
hierarchical Bayesian models is here, with no dependency on the disRNN stack that consumes
it downstream.

## 1. NumPyro, not Stan

The published model (`AllenNeuralDynamics/aind_stan_fit_sim`) is written in Stan, and for a
**single subject at a time Stan is 3.7x faster than this implementation**. That is measured,
not conceded in the abstract.

The reason to be here anyway is that the workload is *latency-bound*: wall time is set by
the sequential scan over trials, and widening the batch is nearly free. Fitting subjects one
at a time wastes that entirely; fitting the cohort in one batched gradient does not. So
**batching is not an optimisation of this package, it is the reason it exists.** Any code
path that fits subjects sequentially in production has given up the whole argument.

Secondary reasons: the three-level cohort model and SVI are both native here and are not
what Stan is good at.

## 2. Parameter conventions follow `generative_model`, not the reference

Where the reference Stan model and this package disagree, the package wins, because the
point is to sit on one axis with the per-session MLE fits it already produces.

| reference | this package | |
|---|---|---|
| `aP` / `aN` | `learn_rate_rew` / `learn_rate_unrew` | equivalent for binary reward with Q in [0,1] |
| `aF` | `1 - forget_rate_unchosen` | **inverted** — `aF` is retention, not decay |
| `bias` | `-biasL` | **sign flipped** |

Both traps are live: a translation layer between the two parameterisations is where an
inversion silently corrupts a cross-model comparison.

Note that retention and decay are the *same model*, not rival ones: since
`1 - Phi(x) = Phi(-x)` and the raw draws are symmetric, decay equals retention with the
location negated and the spread unchanged.

## 3. The JAX likelihood never imports `generative_model` at runtime

Only tests may import both. Having given up the published Stan implementation as an oracle,
the only thing establishing that the JAX likelihood is correct is that it reproduces the
numpy forager's per-trial choice probabilities from an **independent** implementation. If
the JAX code calls into the numpy dynamics for convenience, that test degrades into
asserting a function equals itself.

**The trial dynamics are therefore written twice on purpose.** A future contributor who
de-duplicates them removes the guarantee.

## 4. Held-out scores are pointwise lppd, averaged in probability space

Per trial, average the observed choice's probability across posterior draws, *then* take the
log. Averaging in log space is smaller by Jensen's inequality and understates the model;
plugging in a point estimate overstates it.

The whole-session alternative — marginalising the session at once — is stricter but
unusable: over hundreds of trials the per-draw session likelihoods span hundreds of nats, so
the log-sum-exp collapses onto the single best draw.

This is easy to "fix" incorrectly and doing so silently changes every reported number.

## 5. Non-centred, and the population pools scale as well as location

Non-centred was verified rather than assumed: a sweep over centred/non-centred, diagonal/dense
mass and target acceptance found the non-centred default best on every measure, with centred
sampling producing divergences. The funnel is real at this data scale.

The population pools both `mu_p` and `log sigma`, so a held-out subject inherits a
cohort-informed prior for *how variable* its sessions are likely to be, not merely where they
sit. Pooling only the location degrades exactly the zero-shot and low-context cells the model
exists to measure.

## 6. Two-stage is an approximation, and on GPU it is also the expensive one

Two-stage empirical Bayes was built as the cheap, scalable alternative to a joint fit assumed
unaffordable. Measurement reversed that: a batched one-stage fit finished in 3 h 32 m while
the sequential two-stage fit on the same cohort was still running past 5 h 45 m, for the same
reason as everything else — a joint fit pays the scan depth once for the cohort, sequential
fitting pays it once per subject.

Two-stage therefore has to justify itself on statistics alone, and only at scales where the
joint fit will not converge.

## 7. Held-out subjects are adapted in one batched fit

Independent given a frozen population, so the same argument applies: sequential adaptation
measured about four hours per conditioning rung over a 153-subject cohort; batched adaptation
scales flat (16x the subjects for 1.64x the time) and puts a rung in tens of seconds.

The approximation is one step size adapted across all subject blocks rather than one each.
That is measured, not assumed: per-subject posterior means and the held-out likelihoods that
actually get reported both match sequential fitting.

## 8. Batch sizes come from the device, and batching is a GPU-only win

Scoring sizes its own batches from the device's memory limit rather than a fixed constant.
A constant is wrong in both directions -- it wastes an H200 and can exhaust a 12 GB card --
because the working set scales with draws and trial count as well as sessions. At production
draw counts that lands near 200 sessions per pass on a 12 GB card, ~670 on a 40 GB A100 and
~2400 on an H200.

**Batching is a GPU strategy, not a universal one.** The lane sweep that is flat on an A100
measured *worse than linear* on CPU: 32x the lanes for 102x the time. So on CPU the auto-sized
chunk stays deliberately small, and a CPU-only run should be expected to behave like the
sequential code it replaced rather than better.

**Known limit.** The adaptation fit itself is not chunked: every held-out subject enters one
sampler, and its memory grows with subjects x context sessions x trials, with an autodiff tape
on top. That has been exercised at 153 subjects on a 40 GB A100. A smaller card, or a
substantially larger cohort, may need the subjects split across several fits -- which is safe,
since they are independent, and only widens the step-size sharing already discussed in section 7.

## 9. Fits are persisted in full

A cohort fit costs hours; scoring and figures cost minutes. Keeping only summary statistics
forces a refit for every new question, so `artifacts.save_fit` writes posterior draws,
sampler diagnostics and provenance rather than posterior means alone.

## 10. NumPyro sits behind an optional extra

The package core is numpy/scipy at `requires-python >= 3.9`; JAX needs `>= 3.10`. NumPyro is
behind the `bayes` extra with its own CI job on 3.11, so the core matrix is unaffected.
