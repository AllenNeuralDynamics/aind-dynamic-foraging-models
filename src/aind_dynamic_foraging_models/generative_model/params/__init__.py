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
    learn_rate_actor = R"$\alpha_{actor}$"
    learn_rate_critic = R"$\alpha_{critic}$"
    forget_rate_unchosen = R"$\delta$"
    choice_kernel_step_size = R"$\alpha_{ck}$"
    choice_kernel_relative_weight = R"$w_{ck}$"
    choice_kernel_inverse_temperature = R"$\beta_{ck}$"
    biasL = R"$b_L$"
    choice_bias = R"$b$"
    softmax_inverse_temperature = R"$\beta$"
    epsilon = R"$\epsilon$"
    threshold = R"$\rho$"  # Adding the threshold parameter with symbol ρ (rho)
    reset_to_threshold = R"$\mathrm{reset}$"
    forgetting_factor = R"$\zeta$"
    expected_uncertainty_step_size = R"$\alpha_v$"
    negative_learning_rate_step_size = R"$\psi$"
    perseveration_learning_rate = R"$\alpha_p$"
    reward_learning_rate = R"$\alpha_r$"
    perseveration_weight = R"$\lambda_p$"
    reward_weight = R"$\lambda_r$"
    choice_history_weight = R"$\alpha_{choice}$"
    reward_evidence_weight = R"$\beta_{reward}$"
    evidence_time_constant = R"$\tau$"
    habit_weight = R"$\beta_h$"
    gambler_fallacy_weight = R"$\beta_g$"
    reward_retention_logit = R"$\mathrm{logit}(\alpha_r)$"
    habit_retention_logit = R"$\mathrm{logit}(\alpha_h)$"
    gambler_fallacy_retention_logit = R"$\mathrm{logit}(\alpha_g)$"
    positive_learning_rate = R"$\alpha_+$"
    negative_learning_rate = R"$\alpha_-$"
    perseveration_bonus = R"$p$"
    subjective_switch_probability = R"$p_{switch}$"
    subjective_reward_probability = R"$p_{reward}$"
