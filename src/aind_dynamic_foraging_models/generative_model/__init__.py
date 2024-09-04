"""Package for generative models of dynamic foraging behavior"""

# Register the forager classes here
from .forager_loss_counting import ForagerLossCounting  # noqa: F401
from .forager_q_learning import ForagerQLearning  # noqa: F401
from .foragers import ForagerCollection  # noqa: F401
