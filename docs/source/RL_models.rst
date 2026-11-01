Reinforcement Learning (RL) models
=========================================

Overview
--------

RL agents that can perform any dynamic foraging task in `aind-behavior-gym <https://github.com/AllenNeuralDynamics/aind-behavior-gym>`_ and can fit behavior using MLE.

.. image:: https://github.com/user-attachments/assets/1edbcdb4-932f-4674-bcdc-97d2c840fc72

Code structure
--------------

|classes_aind_dynamic_foraging_models|

- To add more generative models, please subclass `DynamicForagingAgentMLEBase <https://github.com/AllenNeuralDynamics/aind-dynamic-foraging-models/blob/11c858f93f67a0699ed23892364f3f51b08eab37/src/aind_dynamic_foraging_models/generative_model/base.py#L25C7-L25C34>`_.

Implemented foragers
--------------------

- `ForagerQLearning <https://github.com/AllenNeuralDynamics/aind-dynamic-foraging-models/blob/f9ab39bbdc2cbea350e5a8f11d3f935d6674e08b/src/aind_dynamic_foraging_models/generative_model/forager_q_learning.py>`_: Simple Q-learning agents that incrementally update Q-values.
  
  Available ``agent_kwargs``:
  
  .. code-block:: python

      number_of_learning_rate: Literal[1, 2] = 2,
      number_of_forget_rate: Literal[0, 1] = 1,
      choice_kernel: Literal["none", "one_step", "full"] = "none",
      action_selection: Literal["softmax", "epsilon-greedy"] = "softmax",

- `ForagerLossCounting <https://github.com/AllenNeuralDynamics/aind-dynamic-foraging-models/blob/f9ab39bbdc2cbea350e5a8f11d3f935d6674e08b/src/aind_dynamic_foraging_models/generative_model/forager_loss_counting.py>`_: Loss counting agents with probabilistic ``loss_count_threshold``.
  
  Available ``agent_kwargs``:
  
  .. code-block:: python

      win_stay_lose_switch: Literal[False, True] = False,
      choice_kernel: Literal["none", "one_step", "full"] = "none",

- :class:`~aind_dynamic_foraging_models.generative_model.forager_actor_critic.ForagerActorCritic`: Actor-critic agents. A softmax actor holds a preference per side; a single state-value critic gives the reward baseline.

  Available ``agent_kwargs``:

  .. code-block:: python

      choice_kernel: Literal["none", "one_step", "full"] = "none",

- :class:`~aind_dynamic_foraging_models.generative_model.forager_compare_threshold.ForagerCompareThreshold`: Compare-to-threshold (CTT) agents. One value is compared with a threshold: above it the agent stays, below it the agent switches.

  Available ``agent_kwargs``:

  .. code-block:: python

      number_of_learning_rate: Literal[1, 2] = 1,
      choice_kernel: Literal["none", "one_step", "full"] = "none",
      reset_to_threshold: Literal[True, False] = True,

Published-model reproductions
-----------------------------

:mod:`~aind_dynamic_foraging_models.generative_model.forager_published_bandits` re-implements published two-armed-bandit models as MLE
agents with fixed structure (no ``agent_kwargs``).
`published_bandit_reproductions.md <https://github.com/AllenNeuralDynamics/aind-dynamic-foraging-models/blob/main/docs/published_bandit_reproductions.md>`_
records each model's equation source, the author code it was checked against, and every
deliberate departure from it.

.. list-table::
   :header-rows: 1

   * - Class
     - Model
   * - :class:`~aind_dynamic_foraging_models.generative_model.forager_published_bandits.ForagerFeedbackDependentRL`
     - Costa: feedback-dependent Rescorla-Wagner
   * - :class:`~aind_dynamic_foraging_models.generative_model.forager_published_bandits.ForagerFeedbackDependentRLBias`
     - Costa RL plus a fixed choice bias
   * - :class:`~aind_dynamic_foraging_models.generative_model.forager_published_bandits.ForagerFeedbackDependentRLBiasCK1`
     - Costa dual-rate RL plus bias and Bari's one-step choice kernel
   * - :class:`~aind_dynamic_foraging_models.generative_model.forager_published_bandits.ForagerAlsioRL`
     - Alsiö et al.: dual-rate RL with one-step spatial stickiness
   * - :class:`~aind_dynamic_foraging_models.generative_model.forager_published_bandits.ForagerLopezDoubleTrace`
     - López-Yépez et al.: reward value plus fast and slow choice traces
   * - :class:`~aind_dynamic_foraging_models.generative_model.forager_published_bandits.ForagerRLCK`
     - Chen and Zid: four-parameter RL plus choice kernel
   * - :class:`~aind_dynamic_foraging_models.generative_model.forager_published_bandits.ForagerZidHistoryKernel`
     - Zid et al.: history-kernel-2 compare-to-threshold foraging RL
   * - :class:`~aind_dynamic_foraging_models.generative_model.forager_published_bandits.ForagerGrossmanMetaLearning`
     - Grossman et al.: uncertainty-dependent asymmetric meta-learning
   * - :class:`~aind_dynamic_foraging_models.generative_model.forager_published_bandits.ForagerLebedevaPR`
     - Lebedeva et al.: perseveration / reward-learning (PR)
   * - :class:`~aind_dynamic_foraging_models.generative_model.forager_published_bandits.ForagerBeronRFLR`
     - Beron et al.: recursively formulated logistic regression (RFLR)
   * - :class:`~aind_dynamic_foraging_models.generative_model.forager_published_bandits.ForagerMillerRHG`
     - Miller et al.: reward-seeking / habit / gambler's fallacy (RHG)
   * - :class:`~aind_dynamic_foraging_models.generative_model.forager_published_bandits.ForagerEcksteinRL`
     - Eckstein et al.: asymmetric, counterfactual RL with perseveration
   * - :class:`~aind_dynamic_foraging_models.generative_model.forager_published_bandits.ForagerEcksteinBI`
     - Eckstein et al.: hidden-state Bayesian inference

The Findling et al. Weber-imprecision Bayesian-inference model
(:mod:`~aind_dynamic_foraging_models.generative_model.findling_weber`) is not an MLE agent: its likelihood averages over a stochastic
belief state with a particle filter. Fit it with
:func:`~aind_dynamic_foraging_models.generative_model.findling_weber.fit_findling_weber_map` over the 1,000-candidate grid from
:func:`~aind_dynamic_foraging_models.generative_model.findling_weber.findling_weber_parameter_grid`.

Preset foragers
---------------

`Here is the list <https://foraging-behavior-browser.allenneuraldynamics-test.org/RL_model_playground#all-available-foragers>`_ of the preset Q-learning and loss-counting foragers (``ForagerCollection``):

.. image:: https://github.com/user-attachments/assets/db2e3b6c-f888-496c-a12b-06e030499165
.. image:: https://github.com/user-attachments/assets/4f7b669c-2f0e-49cc-8fb4-7fa948926e2e

Usage
-----

- `Jupyter notebook <https://github.com/AllenNeuralDynamics/aind-dynamic-foraging-models/blob/main/notebook/demo_RL_agents.ipynb>`_
- See also `these unittest functions <https://github.com/AllenNeuralDynamics/aind-dynamic-foraging-models/tree/main/tests>`_.

RL model playground
-------------------

Play with the generative models `here <https://foraging-behavior-browser.allenneuraldynamics-test.org/RL_model_playground>`_.

.. image:: https://github.com/user-attachments/assets/691986b0-114b-437c-8df9-3b7b18f83de9

