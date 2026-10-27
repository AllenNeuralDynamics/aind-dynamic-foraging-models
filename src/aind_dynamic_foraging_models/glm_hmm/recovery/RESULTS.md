# GLM-HMM parameter recovery

Measured 2026-10-27 on the Allen HPC (CPU) with this branch. These numbers come from simulated
data generated from parameters fitted to real mice; they are not results about mice.

Reproduce from this directory (about an hour of CPU for the six runs). The figures and the
tables below are regenerated from `results/*.json`:

    for e in hurdle reference; do for x in size selection per_mouse; do
      python parameter_recovery.py --experiment $x --emission $e; done; done
    python parameter_recovery.py --plot

## Setup

- **Ground truth from real mice** (`truth_K3_L5.json`):
  - K = 3, L = 5 global fits of each emission on 127 real sessions (58,757 trials, 10
    training mice, snapshot 20260603, `ignore_policy=include`; subject ids in the file);
  - this gives realistic regressor weights, sticky transitions, and uneven state occupancy,
    with one rarely used state per emission.
- **Closed-loop simulation** on a coupled baiting task:
  - the GLM-HMM chooses left / right / ignore from its own history;
  - reward probabilities sum to 0.45, with left:right ratios of 8, 6, 3 or 1;
  - blocks last 20–60 trials, and bait persists until collected;
  - session lengths are drawn from the real sessions;
  - the emission formulas are restated in NumPy, so the simulator shares no code with the fit.
- **Refit with the production protocol**: K = 1 first, then K = 3 from its weights plus
  N(0, 0.2) noise, 5 restarts, under the global prior (σ = 100, α = 1). Fitted states are
  matched to the true states by posterior overlap with the true state paths.
- **What is checked**: the class probabilities each trial would have under its generating
  state, which are what the data determine; the transitions; and state decoding. Decoding is
  the share of trials whose most likely smoothed state is the true one, beside the same share
  under the true parameters (the oracle). Raw weights are reported too.

## Figures

| figure | shows |
|---|---|
| `fig_recovery_weights.svg` | true vs recovered weights and transitions, 400 sessions |
| `fig_recovery_vs_size.svg` | class-probability, weight and transition error, and the decoding gap to the oracle, vs simulated trials |
| `fig_reference_split.svg` | reference emission: side contrast (right − left) vs common mode ((right + left) / 2) |
| `fig_model_selection.svg` | held-out log likelihood vs fitted K, relative to the true model |
| `fig_per_mouse.svg` | per-mouse refits: weight error to each mouse's own truth, and held-out likelihood |

## Findings

1. **The hurdle model is recovered, and the errors shrink at the √N rate of a consistent
   estimator.** At 400 sessions (about 190k trials):
   - median weight error 0.02 and RMSE 0.04;
   - transitions within about 1e-3;
   - class probabilities within 0.003;
   - decoding equal to the oracle;
   - every fit converged and reached a log joint above the truth's (as a maximum should).
2. **The reference model recovers its probabilities, transitions and decoding just as well, but
   not every raw weight.** The error sits in one combination: the common mode of the left and
   right logits of state 0, the state that almost never ignores (p(ignore | no history) ≈ 0).
   - Adding the same amount to both of its logits changes only p(ignore), which the data
     bound but do not pin.
   - Its side contrast is recovered (RMSE 0.35 at 400 sessions), and both combinations of the
     other two states are recovered to within 0.07.
   - Consequence: compare reference fits through their side contrasts and class
     probabilities, never through raw logits; this matters for the Faeze cross-check. Report
     a state's p(ignore), not its engagement or common-mode weights.
3. **K selection finds the true K for both emissions.** The held-out score rises to K = 3 and
   is flat or slightly lower after it. At K = 3 the fitted model is within 1 (hurdle) and 4
   (reference) millinats per trial of the true model.
4. **The per-mouse protocol** was run on 12 mice with 14 context sessions each, the real
   held-out cohort's median, and an *assumed* between-mouse spread of N(0, 0.3²) on every
   weight:
   - The refit improves held-out prediction for every mouse, closing most of the gap from
     zero-shot to the mouse's own oracle.
   - For hurdle, the typical weight moves toward the mouse's truth: median error improves for
     12/12 mice.
   - Weakly constrained coefficients get noisier under the subject prior (σ = 2, centred at
     0), so RMSE is mixed. Hurdle state 0 improves for 12/12 mice, but the rarely used state
     1 improves for 2/12.
   - For reference, the median weight error does not improve: 6/12 mice.
   - Per-mouse *prediction* is therefore sound. Per-mouse *weights* are trustworthy only for
     well-constrained coefficients.
   - The standard remedy is a subject prior centred on the global weights (empirical-Bayes
     shrinkage: penalise θ − θ_global instead of θ). It is listed in study 14's plan as a
     sensitivity variant and not implemented here.

<!-- BEGIN tables (plot_recovery.py) -->

### Recovery against data size (mean over 3 seeds)

| emission | sessions | trials | class-prob MAE | median weight err | weight RMSE | transition max err | decoding (oracle) | log joint - truth's | converged |
|---|---|---|---|---|---|---|---|---|---|
| hurdle | 25 | 12,204 | 0.0146 | 0.093 | 0.25 | 0.0034 | 0.922 (0.938) | +37.8 | 3/3 |
| hurdle | 100 | 46,659 | 0.0072 | 0.050 | 0.09 | 0.0014 | 0.944 (0.946) | +35.3 | 3/3 |
| hurdle | 400 | 188,065 | 0.0034 | 0.020 | 0.04 | 0.0008 | 0.944 (0.945) | +31.2 | 3/3 |
| reference | 25 | 12,204 | 0.0270 | 0.250 | 9.74 | 0.0233 | 0.836 (0.895) | +64.9 | 3/3 |
| reference | 100 | 46,659 | 0.0090 | 0.101 | 9.63 | 0.0057 | 0.884 (0.887) | +71.7 | 3/3 |
| reference | 400 | 188,065 | 0.0048 | 0.053 | 1.80 | 0.0026 | 0.891 (0.891) | +74.7 | 3/3 |

### Reference emission: weight RMSE split by combination (mean over seeds)

| sessions | side contrast (R - L) | common mode ((R + L) / 2) |
|---|---|---|
| 25 | 9.81 | 7.94 |
| 100 | 0.71 | 9.62 |
| 400 | 0.35 | 1.79 |

### Held-out score against K (truth K = 3)

| emission | K=1 | K=2 | K=3 | K=4 | K=5 | true model |
|---|---|---|---|---|---|---|
| hurdle | 0.55680 | 0.56365 | 0.56949 | 0.56921 | 0.56875 | 0.57011 |
| reference | 0.55021 | 0.55932 | 0.56226 | 0.56171 | 0.56182 | 0.56469 |

### Per-mouse protocol (12 mice, 28 sessions each, half context; own weights = truth + N(0, 0.3²))

| emission | median weight err global → refit | mice improved (median err) | weight RMSE global → refit | mice improved (RMSE) | held-out zero-shot / refit / oracle (mean) | mice with refit > zero-shot |
|---|---|---|---|---|---|---|
| hurdle | 0.205 → 0.151 | 12/12 | 0.291 → 0.364 | 4/12 | 0.5550 / 0.5642 / 0.5681 | 12/12 |
| reference | 0.343 → 0.357 | 6/12 | 4.964 → 4.901 | 7/12 | 0.5692 / 0.5891 / 0.5973 | 12/12 |

#### Per-mouse weight RMSE by state (median over mice; state occupancy in context trials)

| emission | state | occupancy | global | refit | mice improved |
|---|---|---|---|---|---|
| hurdle | 0 | 0.30 | 0.279 | 0.176 | 12/12 |
| hurdle | 1 | 0.12 | 0.286 | 0.433 | 2/12 |
| hurdle | 2 | 0.57 | 0.283 | 0.324 | 5/12 |
| reference | 0 | 0.16 | 8.580 | 8.471 | 7/12 |
| reference | 1 | 0.52 | 0.416 | 0.554 | 4/12 |
| reference | 2 | 0.32 | 0.295 | 0.193 | 12/12 |

<!-- END tables -->
