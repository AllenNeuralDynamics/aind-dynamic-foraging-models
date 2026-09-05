// Three-level Hattori2019 Q-learning: population -> subject -> session.
//
// This is a port of `model.hattori2019_three_level`, NOT of the reference two-level model
// next to it. The point of the port is a framework comparison, and that is only meaningful
// if both frameworks fit the SAME posterior -- so this file follows the NumPyro model's
// structure and parameterisation, not `stan_qLearning_5params.stan`'s.
//
// Differences from the reference model in this directory, all deliberate:
//
//   * a subject level between the population and the sessions, so session parameters are
//     drawn from their own subject's mean rather than from one cohort-wide mean;
//   * the population pools BOTH the location (`mu_p`) and the log of the session-level
//     spread, so a subject inherits a cohort-informed prior for how variable its sessions
//     are, not merely where they sit;
//   * `bias_l` is pooled like every other parameter. The reference leaves `bias` outside the
//     hierarchy entirely (`vector[N] bias ~ normal(0, 20)`, flat, per session), and it
//     carries four hierarchical parameters where we carry five.
//
// PARAMETERISATION TRAPS (docs/design-hb-baseline.md §2). This file is written in the
// package's sense throughout, so anyone comparing it line-by-line with the reference should
// expect these three to differ on purpose:
//
//   * `aF` in the reference is a RETENTION factor despite being called a forgetting rate:
//     aF = 1 - forget_rate_unchosen. Here the unchosen value decays by
//     `(1 - forget_rate_unchosen)`, so `forget_rate_unchosen = 0` means no forgetting.
//   * the reference's `bias` is `-bias_l`. Here the bias is added to the LEFT option, which
//     flips its sign relative to a right-referenced logit.
//   * the reference branches the learning rate on `sign(PE)`; we branch on whether the trial
//     was rewarded. These coincide only while Q stays in [0, 1] -- true here because Q starts
//     at 0, the chosen update is a convex combination with r in {0,1}, and the unchosen
//     update multiplies by a factor in [0, 1]. Initialising Q away from 0, or a non-binary
//     reward, breaks the equivalence.
//
// Ragged data are handled the way Stan can and JAX cannot: `n_sessions` and `n_trials` give
// each subject's and session's true length, and the loops run to it. Nothing is padded or
// masked in the likelihood. That is the mechanism this benchmark exists to price -- the JAX
// side pads every lane to the cohort maximum, and on this cohort that is ~60-75% waste.

functions {
  /* Log likelihood of one slice of subjects.
   *
   * WHAT IS SLICED, AND WHY IT MATTERS MORE THAN IT LOOKS
   *
   * reduce_sum deep-copies its SHARED arguments into every partial sum it evaluates, for any
   * argument containing parameters. The first version of this file sliced a dummy array of
   * subject indices and passed the five per-session parameter arrays as shared -- 5 x S x M
   * autodiff variables copied per partial, S partials per gradient. At D~99 that is roughly
   * 3.7M variable copies per gradient evaluation, which swamped the likelihood itself: the
   * fit sustained under 160 iterations in three hours on 92 busy cores, no better than one
   * thread. More threads only bought more copying.
   *
   * So the SLICED argument now carries the parameters -- each partial receives only its own
   * subjects' rows -- and every shared argument is data, which reduce_sum passes by
   * reference rather than copying. The likelihood computed is identical; only the amount of
   * memory traffic per gradient changes.
   *
   * The name matters too: a `_lpdf` suffix would make Stan treat this as a density and
   * demand a real variate as its first argument, which a slice is not.
   */
  real partial_sum_subjects(array[] matrix params_slice,
                            int start, int end,
                            array[,,] int choice,
                            array[,,] int reward,
                            array[] int n_sessions,
                            array[,] int n_trials) {
    real lp = 0;
    for (i in 1:size(params_slice)) {
      int s = start + i - 1;              // index back into the data arrays
      for (m in 1:n_sessions[s]) {
        int T_sm = n_trials[s, m];
        if (T_sm == 0) continue;
        // Q[1] = left, Q[2] = right, both start at 0 -- the initialisation the
        // reward-branching equivalence in the header depends on.
        vector[2] Q = rep_vector(0.0, 2);
        real a_rew = params_slice[i][m, 1];
        real a_unrew = params_slice[i][m, 2];
        real f = params_slice[i][m, 3];
        real b = params_slice[i][m, 4];
        real bl = params_slice[i][m, 5];

        for (t in 1:T_sm) {
          // choice: 0 = left, 1 = right. The NumPyro model applies the bias to the LEFT
          // option, so a right-referenced logit carries it with a minus sign.
          lp += bernoulli_logit_lpmf(choice[s, m, t] | b * (Q[2] - Q[1]) - bl);
          {
            real r = reward[s, m, t];
            real lr = r > 0 ? a_rew : a_unrew;
            if (choice[s, m, t] == 1) {          // right chosen
              Q[2] += lr * (r - Q[2]);
              Q[1] *= (1 - f);
            } else {                              // left chosen
              Q[1] += lr * (r - Q[1]);
              Q[2] *= (1 - f);
            }
          }
        }
      }
    }
    return lp;
  }
}

data {
  int<lower=1> S;                                  // subjects
  int<lower=1> M;                                  // max sessions per subject
  int<lower=1> T;                                  // max trials per session
  array[S] int<lower=0, upper=M> n_sessions;       // real sessions, per subject
  array[S, M] int<lower=0, upper=T> n_trials;      // real trials, per session
  array[S, M, T] int<lower=0, upper=1> choice;     // 0 = left, 1 = right
  array[S, M, T] int<lower=0, upper=1> reward;
  real<lower=0> beta_max;                          // 10.0, matching the published bound
  real log_sigma_loc;                              // prior mean of log session spread
  real<lower=0> log_sigma_scale;
  int<lower=1> grainsize;                          // reduce_sum partition hint
}

parameters {
  // -- Population --
  vector[5] population_mean;
  vector<lower=0>[5] population_scale;             // half-normal via the constraint
  vector[5] log_sigma_mean;
  vector<lower=0>[5] log_sigma_spread;

  // -- Subject level, non-centred --
  array[S] vector[5] mu_raw;
  array[S] vector[5] log_sigma_raw;

  // -- Session level, non-centred --
  array[S, M] vector[5] theta_raw;
}

transformed parameters {
  // Per-subject, per-session parameters, laid out so reduce_sum can slice them by subject:
  // columns are learn_rate_rew, learn_rate_unrew, forget_rate_unchosen,
  // softmax_inverse_temperature, bias_l -- the order of HATTORI2019_PARAMS.
  //
  // Phi_approx() with a standard-normal argument IS the uniform prior the published model
  // calls "non-informative"; the transform carries the prior, so no uniform statement appears
  // anywhere in this file.
  array[S] matrix[M, 5] session_params;

  for (s in 1:S) {
    vector[5] mu_p = population_mean + population_scale .* mu_raw[s];
    vector[5] sigma = exp(log_sigma_mean + log_sigma_spread .* log_sigma_raw[s]);
    for (m in 1:M) {
      vector[5] theta = mu_p + sigma .* theta_raw[s, m];
      session_params[s, m, 1] = Phi_approx(theta[1]);
      session_params[s, m, 2] = Phi_approx(theta[2]);
      session_params[s, m, 3] = Phi_approx(theta[3]);
      session_params[s, m, 4] = Phi_approx(theta[4]) * beta_max;
      session_params[s, m, 5] = theta[5];       // bias_l is unbounded
    }
  }
}

model {
  // -- Population priors, matching the NumPyro model exactly --
  population_mean ~ std_normal();
  population_scale ~ std_normal();                 // half-normal, constrained above
  log_sigma_mean ~ normal(log_sigma_loc, log_sigma_scale);
  log_sigma_spread ~ std_normal();                 // half-normal

  // -- Non-centred offsets --
  for (s in 1:S) {
    mu_raw[s] ~ std_normal();
    log_sigma_raw[s] ~ std_normal();
    for (m in 1:M) {
      theta_raw[s, m] ~ std_normal();
    }
  }

  // Padded session slots still carry their std_normal prior above, exactly as the NumPyro
  // model does: `theta_raw` is sampled over the full padded grid there too, and masked out
  // of the likelihood only. Keeping that identical matters -- dropping the padded slots here
  // would change the parameter count and make the two posteriors different objects.

  // grainsize is passed as data and set to 1 by the trainer, which is Stan's AUTOMATIC
  // partitioning setting rather than "one subject per task". Worth stating because the first
  // diagnosis of the slowdown blamed grainsize; it was never the problem. The problem was
  // passing parameters as shared arguments, fixed by slicing session_params above.
  target += reduce_sum(partial_sum_subjects, session_params, grainsize,
                       choice, reward, n_sessions, n_trials);
}

generated quantities {
  // Per-session log likelihood, the same quantity the NumPyro model records as
  // `session_log_lik`. Recomputed here rather than carried out of the model block because
  // Stan has no deterministic-site mechanism.
  array[S, M] real session_log_lik;

  for (s in 1:S) {
    for (m in 1:M) {
      session_log_lik[s, m] = 0;
      if (m <= n_sessions[s]) {
        vector[2] Q = rep_vector(0.0, 2);
        for (t in 1:n_trials[s, m]) {
          session_log_lik[s, m] += bernoulli_logit_lpmf(
              choice[s, m, t]
              | session_params[s, m, 4] * (Q[2] - Q[1]) - session_params[s, m, 5]);
          {
            real r = reward[s, m, t];
            real lr = r > 0 ? session_params[s, m, 1] : session_params[s, m, 2];
            if (choice[s, m, t] == 1) {
              Q[2] += lr * (r - Q[2]);
              Q[1] *= (1 - session_params[s, m, 3]);
            } else {
              Q[1] += lr * (r - Q[1]);
              Q[2] *= (1 - session_params[s, m, 3]);
            }
          }
        }
      }
    }
  }
}
