"""Hierarchical Bayesian fitting of foraging models.

This subpackage requires the optional ``bayes`` extra (JAX and NumPyro)::

    pip install aind-dynamic-foraging-models[bayes]
"""

from .likelihood import (  # noqa: F401
    hattori2019_choice_prob,
    hattori2019_log_likelihood,
)
