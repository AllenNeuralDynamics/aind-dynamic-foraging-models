# Reference Stan model

`stan_qLearning_5params.stan` is copied **verbatim** from
[AllenNeuralDynamics/aind_stan_fit_sim](https://github.com/AllenNeuralDynamics/aind_stan_fit_sim)
(`code/stan_qLearning_5params.stan`), MIT licensed, © 2023 Allen Institute for Neural Dynamics.

It is kept byte-identical on purpose: it is the reference our NumPyro reimplementation is
validated and benchmarked against, so any edit would undermine the comparison.

## Parameter mapping

The reference and this package parameterise the same model differently. See `hattori2019_stan_reference` for the full argument; in short:

| reference | this package |
|---|---|
| `aP` | `learn_rate_rew` |
| `aN` | `learn_rate_unrew` |
| `aF` | `1 - forget_rate_unchosen` (retention vs decay) |
| `beta` | `softmax_inverse_temperature` |
| `bias` | `-biasL` |

Because `1 - Phi(x) = Phi(-x)`, the retention and decay parameterisations are the same model;
comparing that parameter's posterior needs only a sign flip on `mu_p`.
