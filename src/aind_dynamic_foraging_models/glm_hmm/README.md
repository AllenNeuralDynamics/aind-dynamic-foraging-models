# `glm_hmm`: a GLM-HMM over left / right / ignore

An input-driven hidden Markov model with GLM emissions (Ashwood et al. 2022, *Nat Neurosci*
25:201), built on [dynamax](https://github.com/probml/dynamax).
- **States.** Each session is a sequence of hidden states z_t ∈ {1..K}, with a stationary
  transition matrix A and an initial distribution π.
- **Emissions.** In state k, the choice y_t ∈ {left, right, ignore} follows a GLM of the
  trial history.
- **Choice of emission.** The package has two of them, `hurdle` and `reference`. Both are
  exactly normalised over the three outcomes, use the same regressors, and are scored by the
  same code; they differ in how the three outcomes are tied to the regressors.

| module | contents |
|---|---|
| `design.py` | the regressors |
| `emissions.py` | the two emissions |
| `newton.py` | the exact Newton M-steps |
| `model.py` | EM over length-bucketed sessions, scoring and posteriors |
| `align.py` | state matching |
| `recovery/` | parameter recovery and the `ssm` cross-check |

## Why this re-implementation, and why on dynamax

Faeze's earlier GLM-HMM was an `ssm` fit with the reference emission, on about 400 mice and
mature sessions only. It had no held-out mice and could not be compared directly with our
RNN models of behaviour (the GRU and disRNN). This package re-implements the model for five
reasons.

- **One shared framework.** The GLM-HMM runs through the same stack as the GRU
  ([the dynamic-foraging BFM framework](https://allenneuraldynamics.github.io/aind-dynamic-foraging-bfm-atlas/#/about)):
  - the wrapper's `GLMHMMTrainer` and the dispatcher's `model=glmhmm`, via `run_hpc`;
  - the same data loader, train / eval split and held-out cohort;
  - the same metric code and W&B keys as study 02's 3-way GRU.

  So Faeze's model, and the new hurdle variant, are scored on exactly the GRU's trials
  (study 14).
- **Faster fitting.** The M-step is an exact Newton solve on compressed rows, and E-steps are
  vectorised over length buckets of sessions. On the same data, from the same start, this EM
  reaches a higher likelihood than `ssm`'s, about 6× faster (8.8 s vs 55.3 s on 45,700
  trials; `recovery/RESULTS.md`, finding 5). A GPU makes the production fit a further
  2.7–5.6× faster (see "How well it works").
- **`ssm` is unmaintained.** Its last commits were two small patches in May 2025, after
  nothing since October 2022. It has no tagged release, and it needs numpy < 2 and a C
  compiler, so it cannot be installed next to the rest of our stack. dynamax is the JAX
  state-space-model library from the same lab, written with Kevin Murphy's group, and it is
  actively maintained.
- **A hurdle emission.** `ssm` ships only the multinomial with a reference class; a hurdle
  emission would need a custom observation class there. The hurdle (engagement × side)
  emission is new here, and it is the one that recovers better.
- **What the study protocol needs:**
  - variable-length sessions with exact masking (`ssm` loops over sessions in Python);
  - explicit priors for Ashwood et al.'s global → per-mouse MAP recipe;
  - forward-filter, one-step-ahead scoring for the held-out likelihood;
  - state alignment across mice;
  - posteriors and Viterbi paths for later studies of the ophys/ephys sessions.

**Why dynamax as a framework rather than copied code.** It provides the HMM machinery: the
filter, the two-filter smoother, Viterbi, sampling, and the Dirichlet transition and initial
components. The model-specific parts plug in as `HMMEmissions` subclasses: the emissions, the
Newton M-steps, masking and a length-bucketed EM driver. The versions dynamax needs
(`dynamax==1.0.3`, one `tfp-nightly` build) are pinned in the `[glmhmm]` extra.

**Comparability with `ssm`.** For the reference emission, both packages compute the same
likelihood to about 1e-11 nats, so results stay comparable with Faeze's fits. Ashwood et al.
used `ssm`'s binary form (C = 2) on left / right choices only.

## Regressors (both emissions)

For lags l = 1..L, with s = +1 for right, −1 for left and 0 for ignore, and r the reward:

    side lags, signed:          RewC_l = s_{t-l} r_{t-l}        UnrC_l = s_{t-l} (1 - r_{t-l})
    engagement lags, unsigned:  Ignored_l = [y_{t-l} = ignore]  Rewarded_l = r_{t-l}

- Call them h_t (side lags plus a bias, 2L + 1 values) and e_t (engagement lags plus a bias,
  2L + 1 values).
- x_t is all four families plus one bias, 4L + 1 values.
- Lags before the session start are 0, so trial 0 is predicted from the bias alone.

## `hurdle`: respond or not, then which side

Two separate logistic regressions per state:

    p(respond | k)         = σ(a_k · e_t)
    p(right | respond, k)  = σ(w_k · h_t)

    p(ignore) = 1 - p(respond)
    p(right)  = p(respond) · p(right | respond)
    p(left)   = p(respond) · (1 - p(right | respond))

- **Separation.** Whether the mouse engages depends only on unsigned history; which side it
  picks depends only on signed history. The two halves are independent, given the state.
- **Fitting.** The M-step is two independent weighted logistic regressions per state: side on
  the responded trials, engagement on all trials.
- **Parameters:** 2 (2L + 1) per state, 22 at L = 5.
- **Reading the weights.** w_k is a choice kernel on responded trials, like a logistic
  regression of the Su 2022 kind. a_k is an engagement kernel.
- **Link to the wrapper's metrics.** The two halves line up with the wrapper's engagement
  score and its `LR_engaged` score.

## `reference`: one multinomial, ignore as the zero logit

One multinomial regression per state: `ssm`'s `InputDrivenObservations` with C = 3, as in
Faeze's fits. (Ashwood et al. 2022 used the binary form, C = 2, on left / right choices
only; the 3-class form with ignore as the zero-logit reference is Faeze's.)

    logit_left = W_k,left · x_t     logit_right = W_k,right · x_t     logit_ignore = 0

- **Coupling.** Each class logit sees every regressor, so side history can move p(ignore),
  and engagement history can move the side.
- **Fitting.** The M-step is one coupled Newton solve per state.
- **Parameters:** 2 (4L + 1) per state, 42 at L = 5. The Hessian is 42 × 42, or 82 × 82 at
  L = 10, which makes this the slowest M-step on a CPU.

It is easiest to read in two combinations, with c = W_right − W_left (the side contrast) and
m = (W_right + W_left) / 2 (the common mode):

    p(right | respond)    = σ(c · x_t)
    logit p(respond)      = m · x_t + log(2 cosh(c · x_t / 2))

- **The side contrast c** plays the part of the hurdle's w, except that it may also load on
  engagement regressors.
- **Built-in coupling.** For a fixed common mode m, a stronger side preference |c · x| raises
  p(respond). The hurdle has no such coupling.

## How they differ

| | `hurdle` | `reference` |
|---|---|---|
| structure | engagement × side, independent given the state | one 3-way softmax |
| what each history family affects | side ← signed lags; engagement ← unsigned lags | every class ← every lag |
| side preference and engagement | separate | a stronger preference raises p(respond) |
| parameters per state (L = 5) | 22 | 42 |
| M-step per state | two logistic regressions | one multinomial (42 × 42 Hessian) |
| comparable with | the wrapper's engage / LR_engaged split, logistic choice kernels | Faeze's 3-class `ssm` fits |

**Neither model contains the other.** The hurdle's log-odds of left against ignore are not
linear in x, and the reference cannot keep side history out of p(ignore). So which fits
better is an empirical question: study 14 fits both and compares their held-out likelihoods.

**Identifiability.** In a state that almost never ignores, the weights that set p(ignore) are
bounded but not pinned: shifting them changes p(ignore) only slightly, which the data cannot
resolve. For `reference` these are the common mode m, and in the parameter recovery this is
the only part of a fit that was not recovered (`recovery/RESULTS.md`, finding 2, Fig. 3). The
side contrast c and every class probability were recovered. The same applies in principle to
a hurdle state's engagement weights a_k.
- Compare `reference` fits through c and the class probabilities, never through raw W.
- For both emissions, report a state's p(ignore) rather than its engagement or common-mode
  weights.

## How well it works

**Recovery: both emissions work, and the hurdle recovers better.** The full report is
[`recovery/RESULTS.md`](recovery/RESULTS.md). Parameters fitted to real mice were used to
simulate closed-loop sessions, which were then refitted with the production protocol.
- **Both emissions** recover the class probabilities, transitions and state decoding.
  - At 400 simulated sessions, decoding matches the oracle (the same decoding under the true
    parameters).
  - Probability errors shrink at about the 1/√N rate.
  - K selection finds the true K for both.
  - The reference likelihood matches `ssm` to about 1e-11 nats; `ssm` has no hurdle
    emission to compare.
- **The hurdle is recovered more accurately, and fully.** At 400 sessions:

  | | `hurdle` | `reference` |
  |---|---|---|
  | class-probability error | 0.0034 | 0.0048 |
  | median weight error | 0.020 | 0.053 |
  | weight RMSE | 0.04 | 1.80 |
  | transition error | 0.0008 | 0.0026 |
  | gap to the true model, K selection (millinats/trial) | 1 | 4 |

  The reference's large weight RMSE comes from the common mode it cannot pin down (see
  "How they differ"). In per-mouse refits, the hurdle's median weight error improves for
  12/12 mice and the reference's for 6/12.
- These are simulations. Which emission fits real mice better is study 14's held-out
  comparison.

**Speed.** On study 14's cohort (4.9M trials, K = 10), a GPU runs one EM restart to
convergence 2.7× (hurdle) to 5.6× (reference) faster than 8 CPU cores at L = 5. Per-mouse
refits are about 2× faster on the CPU. A GPU needs the CUDA solver pin described in
`model._compute`. The benchmark is in
[AllenNeuralDynamics/aind-dynamic-foraging-bfm-wrapper#111](https://github.com/AllenNeuralDynamics/aind-dynamic-foraging-bfm-wrapper/issues/111),
and its scripts and tables are added under `benchmarks/` by
[models#83](https://github.com/AllenNeuralDynamics/aind-dynamic-foraging-models/pull/83).

## Choosing

- **`hurdle`** is the default (`emission: hurdle` in the dispatcher's `model/glmhmm.yaml`).
  It has fewer parameters and is recovered better, every weight included. Its halves map
  onto the wrapper's engagement and `LR_engaged` metrics, and its side weights read
  directly as a choice kernel.
- **`reference`** is for comparing with Faeze's 3-class `ssm` fits, or when side history may
  drive disengagement.

Pick the emission with `GLMHMM(num_states, n_lags, emission="hurdle" | "reference")`.
