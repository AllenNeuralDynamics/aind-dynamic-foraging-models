GLM-HMM
=======

:mod:`aind_dynamic_foraging_models.glm_hmm` fits an input-driven HMM with GLM emissions
(Ashwood et al. 2022, *Nat Neurosci* 25:201) over **left / right / ignore** choices, on
`dynamax <https://github.com/probml/dynamax>`_.

.. code-block:: bash

    pip install aind-dynamic-foraging-models[glmhmm]

- Two emissions, both exactly normalised over the three outcomes: ``hurdle`` (per state, an
  engagement GLM on unsigned history and a side GLM on signed history) and ``reference`` (per
  state, one multinomial GLM with ignore as the zero-logit class).
- Regressors (:mod:`~aind_dynamic_foraging_models.glm_hmm.design`): RewC and UnrC lags
  (signed), Ignored and Rewarded lags (unsigned), and a bias.
- Fitting (:mod:`~aind_dynamic_foraging_models.glm_hmm.model`): EM over many variable-length
  sessions. Priors are arguments, so one model serves a near-flat global fit
  (``GLOBAL_PRIOR``) and per-subject MAP refits started from it (``SUBJECT_PRIOR``).
  ``Prior.mean`` centres the weight prior, for example on the global fit's weights.
  ``fit_with_restarts(..., max_workers=N)`` runs EM restarts at once.
- Scoring: ``predictive_log_probs`` is the forward filter's one-step-ahead
  log p(y_t | y_<t) per class, the held-out quantity. ``state_posteriors`` gives filtered and
  smoothed posteriors and Viterbi paths.

Further reading
---------------

- How the two emissions differ, which to use, and compute-device notes:
  `glm_hmm/README.md <https://github.com/AllenNeuralDynamics/aind-dynamic-foraging-models/blob/main/src/aind_dynamic_foraging_models/glm_hmm/README.md>`_
- Parameter recovery and the cross-check against ``ssm``:
  `recovery/RESULTS.md <https://github.com/AllenNeuralDynamics/aind-dynamic-foraging-models/blob/main/src/aind_dynamic_foraging_models/glm_hmm/recovery/RESULTS.md>`_
- CPU and GPU timing:
  `benchmarks/RESULTS.md <https://github.com/AllenNeuralDynamics/aind-dynamic-foraging-models/blob/main/src/aind_dynamic_foraging_models/glm_hmm/benchmarks/RESULTS.md>`_
