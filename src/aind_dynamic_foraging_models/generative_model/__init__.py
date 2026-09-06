"""Package for generative models of dynamic foraging behavior"""

# Register the forager classes here
from .forager_actor_critic import ForagerActorCritic  # noqa: F401
from .forager_compare_threshold import ForagerCompareThreshold  # noqa: F401
from .forager_loss_counting import ForagerLossCounting  # noqa: F401
from .forager_published_bandits import (  # noqa: F401
    ForagerBeronRFLR,
    ForagerEcksteinBI,
    ForagerEcksteinRL,
    ForagerGrossmanMetaLearning,
    ForagerLebedevaPR,
    ForagerMillerRHG,
    ForagerRLCK,
    ForagerZidHistoryKernel,
)
from .forager_q_learning import ForagerQLearning  # noqa: F401
from .foragers import ForagerCollection  # noqa: F401
from .findling_weber import (  # noqa: F401
    FindlingWeberFilterResult,
    filter_findling_weber_session,
    findling_weber_parameter_grid,
    fit_findling_weber_map,
)
