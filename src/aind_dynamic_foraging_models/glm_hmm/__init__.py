"""GLM-HMM (input-driven HMM with GLM emissions) over left / right / ignore choices.

After Ashwood et al. 2022 (Nat Neurosci 25:201), built on dynamax with hurdle or
reference-class emissions, an exact Newton M-step, and length-bucketed EM over many sessions.

This subpackage requires the optional ``glmhmm`` extra::

    pip install aind-dynamic-foraging-models[glmhmm]
"""

from .align import match_by_posterior_overlap, match_by_weights  # noqa: F401
from .design import (  # noqa: F401
    DESIGNS,
    IGNORE,
    LEFT,
    RIGHT,
    build_inputs,
    column_names,
    column_slices,
    default_design,
)
from .emissions import HurdleGLMEmissions, ReferenceGLMEmissions  # noqa: F401
from .model import (  # noqa: F401
    GLMHMM,
    GLOBAL_PRIOR,
    SUBJECT_PRIOR,
    FitResult,
    ParamsGLMHMM,
    Prior,
    SessionBucket,
    dirichlet_map,
    fit_em,
    fit_with_restarts,
    make_buckets,
    predictive_log_probs,
    state_posteriors,
)
