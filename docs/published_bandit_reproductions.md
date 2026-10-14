# Published two-arm-bandit reproductions

This file records the equation source and deliberate implementation choices for
published behavioral models exposed by this package. The model classes are
deterministic teacher-forced likelihood models; session state is reset for every
real session.

## Lebedeva (mouse): PR

- Paper: Lebedeva et al., *Dorsal prefrontal cortex drives perseverative
  behavior in mice*, Nature Communications (2026), Eqs. 9-11.
- Author code: `neurvanna/perseveration` at
  `cc7060cc238f062675880ee5354a2a39fe71dd8c`, class
  `figure1/agents/agent_rs_habits.m`.
- Implementation: `ForagerLebedevaPR` initializes the perseveration and
  reward-seeking states to zero and applies the paper's signed-choice and
  signed-feedback updates before the next trial.
- Fitting bounds: learning rates use the published `[0, 1]` bounds. The author
  code leaves the two magnitudes and bias unbounded; differential evolution
  requires finite bounds, so this implementation uses `[-20, 20]`, a range in
  which logistic probabilities are already numerically saturated.
- Source discrepancy: the public MATLAB class initializes its reward state to
  5, while the paper explicitly says both states start at 0. The class here
  follows the paper.

## Beron (mouse): RFLR

- Paper: Beron et al., *Mice exhibit stochastic and efficient action switching
  during probabilistic decision making*, PNAS (2022), Eqs. 3-5.
- Author code: `celiaberon/2ABT_behavior_models` at
  `29896ce339264b1ca430b7aa260ba3b274e5e961`, functions
  `_log_prob_single_rflr` and `RFLR`.
- Implementation: `ForagerBeronRFLR` assigns probability 0.5 to the first
  choice, adds one-back choice history to exponentially decayed rewarded-choice
  evidence, and uses the author's `exp(-1/tau)` decay.
- Fitting bounds: the original SGD code does not constrain its raw parameters.
  This implementation uses `[-20, 20]` for the two log-odds weights and
  `[0.05, 100]` trials for the necessarily positive time constant.

## Miller (rat): RHG

- Original paper: Miller, Botvinick and Brody, *From predictive models to
  cognitive models: Separable behavioral processes underlying reward learning
  in the rat*, bioRxiv (2021), doi:10.1101/461129.
- Executable specification: Castro et al., *Discovering Symbolic Cognitive
  Models from Human and Animal Behavior*, PMLR (2025), Appendix E.2. The
  downloaded PDF used for the audit had SHA-256
  `fda7b873c21d09b4dc776fba1b2a4334070713e3973466cec9817f019c3c7203`.
- Implementation: `ForagerMillerRHG` copies the three reward-seeking, habit, and
  gambler's-fallacy state updates from Appendix E.2. Retentions are sigmoid
  transforms of the fitted raw parameters. The published policy logits are
  `[-V, +V]`; therefore the probability of choosing right is `sigmoid(2V)`.
- Fitting bounds: the seven raw parameters are unconstrained in the executable
  specification. This implementation uses `[-20, 20]` for weights and bias and
  `[-10, 10]` for retention logits. These cover essentially the full numeric
  range of the corresponding logistic transformations.

## Eckstein (human): RL and BI co-winners

- Paper: Eckstein et al., *Reinforcement learning and Bayesian inference
  provide complementary models for the unique advantage of adolescents in
  stochastic reversal*, Developmental Cognitive Neuroscience (2022), Methods
  4.5.1-4.5.3.
- Author code: `MariaEckstein/SLCN` at
  `4fb5955c1142fcbd8ec80d7fccdf6b35dbfd1616`, files
  `models/PSAllModels.py` and `models/PSModelFunctions2.py`.
- RL implementation: `ForagerEcksteinRL` uses separate rewarded and unrewarded
  learning rates, symmetric counterfactual updates of the unchosen option,
  one-back perseveration, initial values of 0.5, and the author's probability
  floor.
- BI implementation: `ForagerEcksteinBI` filters the hidden state "right is
  correct", applies the subjective switch transition, adds one-back
  perseveration, and uses the same softmax and probability floor.
- Fitting deviation: the paper estimated participants jointly with hierarchical
  Bayesian priors. Study 09's matched-half comparison requires subject-level
  adaptation on an identical prefix, so these implementations use individual
  maximum likelihood with the exact individual-fit bounds in the author code:
  learning and subjective probabilities `[0, 1]`, inverse temperature `[0, 15]`,
  and perseveration `[-1, 1]`.
- Source discrepancy: the paper states an incorrect-action reward floor of
  `1e-4`; the released model code uses `1e-5`. This implementation follows the
  code and separately clips predicted choice probabilities to `[1e-4, 0.9999]`.

## Findling (human): Weber-imprecision Bayesian inference

- Paper: Findling et al., *Neural variability in the medial prefrontal cortex
  contributes to efficient adaptive behavior*, Nature Communications (2025).
- Author code: `Findling-Lab/Volnoise` at
  `ee688535b569a8af8c0531350ec06f34cb989f8e`, files
  `main/fitting_script.py` and
  `main/models/fit_functions/noisy_beta/smc_parallel_JS_variance.py`.
- Implementation: `filter_findling_weber_session` reproduces the released Beta
  belief-state particle filter, symmetric KL change signal, Weber-scaled
  variance noise, softmax choice rule, and stratified resampling. It resets the
  Beta(1, 1) belief at every real session.
- Parameter fitting: `fit_findling_weber_map` uses the exact 1,000-point Sobol
  grid released by the authors and selects the maximum marginal-likelihood
  temperature and Weber slope across adaptation sessions. The release fixes
  lapse and constant noise to zero in its example fit; this implementation does
  the same.
- Stochasticity: the released fit uses only two latent-state particles and does
  not set a random seed. This implementation retains two particles by default
  but requires a recorded seed, making the fit reproducible. Study reports must
  show a higher-particle sensitivity check before treating small performance
  differences as scientific signal.
