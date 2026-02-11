"""Package for generative models of dynamic foraging behavior"""

# Register the forager classes here
from .forager_compare_threshold import ForagerCompareThreshold  # noqa: F401
from .forager_ctt_dualQ import ForagerCTTDualQ  # noqa: F401
from .forager_ctt_avg import ForagerCTTAvg  # noqa: F401
from .forager_ctt_mean_reset import ForagerCTTMeanReset  # noqa: F401
from .forager_loss_counting import ForagerLossCounting  # noqa: F401
from .forager_q_learning import ForagerQLearning  # noqa: F401
from .foragers import ForagerCollection  # noqa: F401
