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

`Here is the full list <https://foraging-behavior-browser.allenneuraldynamics-test.org/RL_model_playground#all-available-foragers>`_ of available foragers:

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

