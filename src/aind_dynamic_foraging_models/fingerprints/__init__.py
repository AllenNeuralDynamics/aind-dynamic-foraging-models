"""Model-free behavioural fingerprints of two-armed-bandit choice, for any model.

One shape for every statistic: a key from the history before trial t and an outcome at t. Animal
sessions and rollouts fill each bin with observed outcomes; replay (the model run on the real
choices and rewards) fills it with the model's probability of that outcome. See ``spec`` for the
statistics; the design is study 15 of aind-dynamic-foraging-bfm-studies.
"""

from .compare import (  # noqa: F401
    MIN_TRIALS,
    coverage,
    floor_ratio,
    kept_bins,
    noise_floor,
    replay_gap,
    rollout_gap,
    values,
)
from .compat import mevolve_replay_gaps, wrapper_abstract_label, wrapper_post_switch  # noqa: F401
from .spec import BINS, SPEC_VERSION, STATISTICS  # noqa: F401
from .tally import (  # noqa: F401
    OBS,
    PRED,
    N,
    add,
    drop_ignores,
    empty_tally,
    from_json,
    session_tally,
    tally_groups,
    tally_sessions,
    to_json,
)
