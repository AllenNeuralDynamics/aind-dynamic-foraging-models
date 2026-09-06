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
  /* Log likelihood of one subject's sessions.
   *
   * Sliced over subjects for `reduce_sum`, which is how Stan uses more than one core here:
   * the gradient of a recurrent scan is inherently sequential within a session, so the only
   * available parallelism is across sessions and subjects.
   */
  /* NOTE the name: a `_lpdf` suffix would make Stan treat this as a probability density
   * and demand a real variate as its first argument, which a reduce_sum slice
   * (`array[] int`) is not. reduce_sum takes an ordinary function.
   */
  real partial_sum_subjects(array[] int subject_slice,
                    int start, int end,
                    array[,,] int choice,
                    array[,,] int reward,
                    array[] int n_sessions,
                    array[,] int n_trials,
                    // array[,] real, matching the transformed-parameter declarations
                    // below. Declaring these as array[] vector is the natural-looking
                    // guess and stanc rejects it: `array[S, M] real` is a 2-D real array,
                    // not an array of vectors.
                    array[,] real learn_rate_rew,
                    array[,] real learn_rate_unrew,
                    array[,] real forget_rate,
                    array[,] real beta,
                    array[,] real bias_l) {
    real lp = 0;
    for (i in 1:size(subject_slice)) {
      int s = subject_slice[i];
      for (m in 1:n_sessions[s]) {
        int T_sm = n_trials[s, m];
        if (T_sm == 0) continue;
        // Q[1] = left, Q[2] = right, both start at 0 -- the initialisation the
        // reward-branching equivalence above depends on.
        vector[2] Q = rep_vector(0.0, 2);
        real a_rew = learn_rate_rew[s, m];
        real a_unrew = learn_rate_unrew[s, m];
        real f = forget_rate[s, m];
        real b = beta[s, m];
        real bl = bias_l[s, m];

        for (t in 1:T_sm) {
          // choice: 0 = left, 1 = right. The NumPyro model applies the bias to the LEFT
          // option, so a right-referenced logit carries it with a minus sign.
          lp += bernoulli_logit_lpmf(choice[s, m, t] | b * (Q[2] - Q[1]) - bl);

          {
            real r = reward[s, m, t];
            real lr = r > 0 ? a_rew : a_unrew;
            // Chosen option moves toward the outcome; unchosen decays.
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
  // Bounded session parameters. Phi() with a standard-normal argument IS the uniform prior
  // the published model describes as "non-informative" -- the transform carries the prior,
  // so there is no separate uniform statement anywhere in this file.
  array[S, M] real<lower=0, upper=1> learn_rate_rew;
  array[S, M] real<lower=0, upper=1> learn_rate_unrew;
  array[S, M] real<lower=0, upper=1> forget_rate;
  array[S, M] real<lower=0> beta;
  array[S, M] real bias_l;                         // unbounded, passes through untransformed

  {
    for (s in 1:S) {
      vector[5] mu_p = population_mean + population_scale .* mu_raw[s];
      vector[5] sigma = exp(log_sigma_mean + log_sigma_spread .* log_sigma_raw[s]);
      for (m in 1:M) {
        vector[5] theta = mu_p + sigma .* theta_raw[s, m];
        learn_rate_rew[s, m]   = Phi_approx(theta[1]);
        learn_rate_unrew[s, m] = Phi_approx(theta[2]);
        forget_rate[s, m]      = Phi_approx(theta[3]);
        beta[s, m]             = Phi_approx(theta[4]) * beta_max;
        bias_l[s, m]           = theta[5];
      }
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

  {
    array[S] int subject_idx;
    for (s in 1:S) subject_idx[s] = s;
    target += reduce_sum(partial_sum_subjects, subject_idx, grainsize,
                         choice, reward, n_sessions, n_trials,
                         learn_rate_rew, learn_rate_unrew, forget_rate, beta, bias_l);
  }
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
              choice[s, m, t] | beta[s, m] * (Q[2] - Q[1]) - bias_l[s, m]);
          {
            real r = reward[s, m, t];
            real lr = r > 0 ? learn_rate_rew[s, m] : learn_rate_unrew[s, m];
            if (choice[s, m, t] == 1) {
              Q[2] += lr * (r - Q[2]);
              Q[1] *= (1 - forget_rate[s, m]);
            } else {
              Q[1] += lr * (r - Q[1]);
              Q[2] *= (1 - forget_rate[s, m]);
            }
          }
        }
      }
    }
  }
}
