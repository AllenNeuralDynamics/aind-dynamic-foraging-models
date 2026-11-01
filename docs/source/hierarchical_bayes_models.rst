Hierarchical Bayesian fitting
=============================

:mod:`aind_dynamic_foraging_models.hierarchical_bayes` fits the foragers as hierarchical
Bayesian models (population → subject → session) with NumPyro, as a partially-pooled
counterpart to the per-session MLE fitting in
:mod:`~aind_dynamic_foraging_models.generative_model`.

.. code-block:: bash

    pip install aind-dynamic-foraging-models[bayes]

The ``bayes`` extra needs Python 3.12 or later: every ``arviz`` 1.x wheel requires it.

- Two forager likelihoods, chosen by ``model=``
  (``MODELS`` in :mod:`~aind_dynamic_foraging_models.hierarchical_bayes.model`): ``"hattori2019"``
  (the default) and ``"ctt"`` (compare-to-threshold).
- :mod:`~aind_dynamic_foraging_models.hierarchical_bayes.heldout` scores a held-out subject: it
  adapts the subject to its context sessions with the population frozen, then scores the
  subject's test sessions.
- :mod:`~aind_dynamic_foraging_models.hierarchical_bayes.two_stage` fits each subject alone,
  then fits the population to their posteriors.

Further reading
---------------

- **Start here:** a worked end-to-end example on synthetic data with known ground truth:
  `demo_hierarchical_bayes.ipynb <https://github.com/AllenNeuralDynamics/aind-dynamic-foraging-models/blob/main/notebook/demo_hierarchical_bayes.ipynb>`_
- Design decisions and their rationale:
  `design-decisions.md <https://github.com/AllenNeuralDynamics/aind-dynamic-foraging-models/blob/main/src/aind_dynamic_foraging_models/hierarchical_bayes/docs/design-decisions.md>`_
- Validation against the reference Stan implementation, and performance measurements:
  `benchmarks/RESULTS.md <https://github.com/AllenNeuralDynamics/aind-dynamic-foraging-models/blob/main/src/aind_dynamic_foraging_models/hierarchical_bayes/benchmarks/RESULTS.md>`_
