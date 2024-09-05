"""Package for params"""

from enum import Enum


class ParamsSymbols(str, Enum):
    """Symbols for the parameters.

    The order determined the default order of parameters when output as a string.
    """

    loss_count_threshold_mean = R"$\mu_{LC}$"
    loss_count_threshold_std = R"$\sigma_{LC}$"
    learn_rate = R"$\alpha$"
    learn_rate_rew = R"$\alpha_{rew}$"
    learn_rate_unrew = R"$\alpha_{unr}$"
    forget_rate_unchosen = R"$\delta$"
    choice_kernel_step_size = R"$\alpha_{ck}$"
    choice_kernel_relative_weight = R"$w_{ck}$"
    biasL = R"$b_L$"
    softmax_inverse_temperature = R"$\beta$"
    epsilon = R"$\epsilon$"
