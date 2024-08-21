"""Maximum likelihood fitting of foraging models
"""

import logging

# %%
from typing import Optional

import numpy as np
import scipy.optimize as optimize
from aind_behavior_gym.dynamic_foraging.agent import DynamicForagingAgentBase
from aind_behavior_gym.dynamic_foraging.task import DynamicForagingTaskBase, L, R
from aind_dynamic_foraging_basic_analysis import plot_foraging_session
from pydantic import BaseModel, Field, model_validator

from .act_functions import act_softmax
from .learn_functions import learn_RWlike

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
console_handler = logging.StreamHandler()
console_handler.setFormatter(
    logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
)
logger.addHandler(console_handler)


class forager_Hattori2019(DynamicForagingAgentBase):
    """
    Base class for maximum likelihood estimation.
    """

    class Param(BaseModel):
        """Parameters for the model and their default values.
        After overriden in subclasses, calling ClassName.Param() returns default parameters.
        """

        # For example:
        # param1: float = Field(default=0.5, ge=0.0, le=2.0, description="Parameter 1")
        # param2: float = Field(default=0.2, ge=0.0, description="Parameter 2")
        # raise NotImplementedError("Params class must be defined in subclasses")

        learn_rate_rew: float = Field(
            default=0.5, ge=0.0, le=1.0, description="Learning rate for rewarded choice"
        )
        learn_rate_unrew: float = Field(
            default=0.1, ge=0.0, le=1.0, description="Learning rate for unrewarded choice"
        )
        forget_rate: float = Field(
            default=0.2, ge=0.0, le=1.0, description="Forgetting rate for unchosen options"
        )
        softmax_inverse_temperature: float = Field(
            default=10, ge=0.0, description="Softmax temperature"
        )
        biasL: float = Field(default=0.0, description="Bias term for softmax")

        class Config:
            extra = "forbid"  # This forbids extra fields

    class ParamFitBounds(BaseModel):
        """Bounds for fitting parameters.
        After overriden in subclasses, calling ClassName.ParamFitBounds() returns default bounds.
        """

        # For example:
        # param1: list = Field(default=[0.0, 1.0], description="Bounds for param1")
        # param2: list = Field(default=[0.0, 1.0], description="Bounds for param2")
        # raise NotImplementedError("ParamFitBounds class must be defined in subclasses")
        learn_rate_rew: list = Field(default=[0.0, 1.0])
        learn_rate_unrew: list = Field(default=[0.0, 1.0])
        forget_rate: list = Field(default=[0.0, 1.0])
        softmax_inverse_temperature: list = Field(default=[0.0, 100.0])
        biasL: list = Field(default=[-5.0, 5.0])

        @model_validator(mode="after")
        def check_all_params(cls, values):
            for key, value in values.__dict__.items():
                if not isinstance(value, list):
                    raise ValueError(f'Value of "{key}" must be a list')
                if len(value) != 2:
                    raise ValueError(f'List "{key}" must have exactly 2 elements')
                if value[1] < value[0]:
                    raise ValueError(f'Upper bound of "{key}" must be greater than the lower bound')
            return values

        class Config:
            extra = "forbid"  # This forbids extra fields

    def __init__(
        self,
        params: dict = {},
        **kwargs,
    ):
        """Init"""

        super().__init__(**kwargs)

        # Set and validate the model parameters. Use default parameters if some are not provided
        self.params = self.Param(**params)

        # Add model fitting related attributes
        self.fitting_result = None
        self.fitting_result_cross_validation = None

        # Some switches
        self.fit_choice_kernel = False

        # Some initializations
        self.n_actions = 2
        self.task = None

    def reset(self):
        """Reset the agent"""
        self.trial = 0

        # Latent variables have n_trials + 1 length to capture the update
        # after the last trial (HH20210726)
        self.q_estimation = np.full([self.n_actions, self.n_trials + 1], np.nan)
        self.q_estimation[:, 0] = 0  # Initial Q values as 0

        self.choice_prob = np.full([self.n_actions, self.n_trials + 1], np.nan)
        self.choice_prob[:, 0] = 1 / self.n_actions  # To be strict (actually no use)

        if self.fit_choice_kernel:
            self.choice_kernel = np.zeros([self.n_actions, self.n_trials + 1])

        # Choice and reward history have n_trials length
        self.choice_history = np.full(self.n_trials, fill_value=-1, dtype=int)  # Choice history
        # Reward history, separated for each port (Corrado Newsome 2005)
        self.reward_history = np.zeros(self.n_trials)

    def set_params(self, params):
        """Update the model parameters and validate"""
        self.params = self.params.model_copy(update=params)
        return self.get_params()

    def get_params(self):
        """Get the model parameters in a dictionary format"""
        return self.params.dict()

    def perform(
        self,
        task: DynamicForagingTaskBase,
    ):
        """Generative simulation of a task.

        Override the base class method to include choice_prob caching etc.
        """
        self.task = task
        self.n_trials = task.num_trials

        # --- Main task loop ---
        self.reset()  # Reset agent
        _, _ = self.task.reset()  # Reset task and get the initial observation
        task_done = False
        while not task_done:
            assert self.trial == self.task.trial  # Ensure the two timers are in sync

            # -- Agent performs an action
            choice, choice_prob = self.act(_)

            # -- Environment steps (enviromnet's timer ticks here!!!)
            _, reward, task_done, _, _ = task.step(choice)

            # -- Update agent history
            self.choice_prob[:, self.trial] = choice_prob
            self.choice_history[self.trial] = choice
            # In Sutton & Barto's convention, reward belongs to the next time step, but we put it
            # in the current time step for the sake of consistency with neuroscience convention
            self.reward_history[self.trial] = reward

            # -- Agent's timer ticks here !!!
            self.trial += 1

            # -- Update q values
            # Note that this will update the q values **after the last trial**, a final update that
            # will not be used to make the next action (because task is *done*) but may be used for
            # correlating with physiology recordings
            self.learn(_, choice, reward, _, task_done)

    def predictive_perform(self, fit_choice_history, fit_reward_history):
        """Simulates the agent over a fixed choice and reward history using its params.

        Unlike .perform() ("generative" simulation), this is called "predictive" simulation,
        which does not need a task and is used for model fitting.
        """
        self.n_trials = len(fit_choice_history)
        self.reset()

        while self.trial <= self.n_trials - 1:
            # -- Compute and cache choice_prob (key to model fitting)
            _, choice_prob = self.act(None)
            self.choice_prob[:, self.trial] = choice_prob

            # -- Clamp history to fit_history
            clamped_choice = fit_choice_history[self.trial].astype(int)
            clamped_reward = fit_reward_history[self.trial]
            self.choice_history[self.trial] = clamped_choice
            self.reward_history[self.trial] = clamped_reward

            # -- Agent's timer ticks here !!!
            self.trial += 1

            # -- Update q values
            self.learn(None, clamped_choice, clamped_reward, None, None)

    def act(self, _):
        """Action selection"""
        choice, choice_prob = act_softmax(
            q_estimation_t=self.q_estimation[:, self.trial],
            softmax_inverse_temperature=self.params.softmax_inverse_temperature,
            bias_terms=np.array([self.params.biasL, 0]),
            choice_softmax_inverse_temperature=None,
            choice_kernel=None,
            rng=self.rng,
        )
        return choice, choice_prob

    def learn(self, _observation, choice, reward, _next_observation, done):
        """Update Q values"""
        # Update Q values
        self.q_estimation[:, self.trial] = learn_RWlike(
            **{
                "choice": choice,
                "reward": reward,
                "q_estimation_tminus1": self.q_estimation[:, self.trial - 1],
                "learn_rates": [self.params.learn_rate_rew, self.params.learn_rate_unrew],
                "forget_rates": [self.params.forget_rate, 0],  # 0: unchosen, 1: chosen
            }
        )
        if self.fit_choice_kernel and (self.trial < self.n_trials):
            self.step_choice_kernel(choice)

    def get_choice_history(self):
        """Return the history of actions in format that is compatible with other library such as
        aind_dynamic_foraging_basic_analysis
        """
        if self.task is None:
            return None
        # Make sure agent's history is consistent with the task's history and return
        np.testing.assert_array_equal(self.choice_history, self.task.get_choice_history())
        return self.task.get_choice_history()

    def get_reward_history(self):
        """Return the history of reward in format that is compatible with other library such as
        aind_dynamic_foraging_basic_analysis
        """
        if self.task is None:
            return None
        # Make sure agent's history is consistent with the task's history and return
        np.testing.assert_array_equal(self.reward_history, self.task.get_reward_history())
        return self.task.get_reward_history()

    def get_p_reward(self):
        """Return the reward probabilities for each arm in each trial which is compatible with
        other library such as aind_dynamic_foraging_basic_analysis
        """
        if self.task is None:
            return None
        return self.task.get_p_reward()

    def fit(
        self,
        fit_choice_history,
        fit_reward_history,
        fit_bounds_override: dict = {},
        clamp_params: dict = {},
        k_fold_cross_validation: Optional[int] = None,
        agent_kwargs: dict = {},
        DE_kwargs: dict = {"workers": 1},
    ):
        """Fit the model to the data using differential evolution.

        It handles fit_bounds_override and clamp_params as follows:
        1. It will first clamp the parameters specified in clamp_params
        2. For other parameters, if it is specified in fit_bounds_override, the specified
           bound will be used; otherwise, the bound in the model's ParamFitBounds will be used.

        For example, if params_to_fit and clamp_params are all empty, all parameters will
        be fitted with default bounds in the model's ParamFitBounds.

        Parameters
        ----------
        fit_choice_history : _type_
            _description_
        fit_reward_history : _type_
            _description_
        fit_bounds_override : dict, optional
            Override the default bounds for fitting parameters in ParamFitBounds, by default {}
        clamp_params : dict, optional
            Specify parameters to fix to certain values, by default {}
        k_fold_cross_validation : Optional[int], optional
            Whether to do cross-validation, by default None (no cross-validation).
            If k_fold_cross_validation > 1, it will do k-fold cross-validation and return the
            prediction accuracy of the test set for model comparison.
        agent_kwargs : dict, optional
            Other kwargs to pass to the model, by default {}
        DE_kwargs : dict, optional
            kwargs for differential_evolution, by default {'workers': 1}
            For example:
                workers : int
                    Number of workers for differential evolution, by default 1.
                    In CO, fitting a typical session of 1000 trials takes:
                        1 worker: ~100 s
                        4 workers: ~35 s
                        8 workers: ~22 s
                        16 workers: ~20 s
                    (see https://github.com/AllenNeuralDynamics/aind-dynamic-foraging-models/blob/22075b85360c0a5db475a90bcb025deaa4318f05/notebook/demo_rl_mle_fitting_new_test_time.ipynb) # noqa E501
                    That is to say, the parallel speedup in DE is sublinear. Therefore, given a constant
                    number of total CPUs, it is more efficient to parallelize on the level of session,
                    instead of on DE's workers.

        Returns
        -------
        _type_
            _description_
        """
        # ===== Preparation =====
        # -- Sanity checks --
        # Ensure params_to_fit and clamp_params are not overlapping
        assert set(fit_bounds_override.keys()).isdisjoint(clamp_params.keys())
        # Validate clamp_params
        assert self.Param(**clamp_params)

        # -- Get fit_names and fit_bounds --
        # Validate fit_bounds_override and fill in the missing bounds with default bounds
        fit_bounds = self.ParamFitBounds(**fit_bounds_override).model_dump()
        # Remove clamped parameters from fit_bounds
        for name in clamp_params.keys():
            fit_bounds.pop(name)
        # Get the names of the parameters to fit
        fit_names = list(fit_bounds.keys())
        # Parse bounds
        lower_bounds = [fit_bounds[name][0] for name in fit_names]
        upper_bounds = [fit_bounds[name][1] for name in fit_names]
        # Validate bounds themselves are valid parameters
        assert self.Param(**dict(zip(fit_names, lower_bounds)))
        assert self.Param(**dict(zip(fit_names, upper_bounds)))

        # # ===== Fit using the whole dataset ======
        logger.info("Fitting the model using the whole dataset...")
        fitting_result = self.__class__._optimize_DE(
            fit_choice_history=fit_choice_history,
            fit_reward_history=fit_reward_history,
            fit_names=fit_names,
            lower_bounds=lower_bounds,
            upper_bounds=upper_bounds,
            fit_trial_set=None,  # None means use all trials to fit
            clamp_params=clamp_params,
            agent_kwargs=agent_kwargs,
            DE_kwargs=DE_kwargs,
        )

        # -- Save fitting results --
        self.fitting_result = fitting_result

        # -- Rerun the predictive simulation with the fitted params--
        # To fill in the latent variables like q_estimation and choice_prob
        self.set_params(fitting_result.params)
        self.predictive_perform(fit_choice_history, fit_reward_history)
        # Compute prediction accuracy
        predictive_choice = np.argmax(self.choice_prob[:, :-1], axis=0)  # Exclude the last update
        fitting_result.prediction_accuracy = (
            np.sum(predictive_choice == fit_choice_history) / fitting_result.n_trials
        )

        if k_fold_cross_validation is None:  # Skip cross-validation
            return fitting_result, None

        # ======  Cross-validation ======
        logger.info(
            f"Cross-validating the model using {k_fold_cross_validation}-fold cross-validation..."
        )
        n_trials = len(fit_choice_history)
        trial_numbers_shuffled = np.arange(n_trials)
        self.rng.shuffle(trial_numbers_shuffled)

        prediction_accuracy_fit = []
        prediction_accuracy_test = []
        prediction_accuracy_test_bias_only = []
        fitting_results_all_folds = []

        for kk in range(k_fold_cross_validation):
            logger.info(f"Cross-validation fold {kk+1}/{k_fold_cross_validation}...")
            # -- Split the data --
            test_idx_begin = int(kk * np.floor(n_trials / k_fold_cross_validation))
            test_idx_end = int(
                n_trials
                if (kk == k_fold_cross_validation - 1)
                else (kk + 1) * np.floor(n_trials / k_fold_cross_validation)
            )
            test_set_this = trial_numbers_shuffled[test_idx_begin:test_idx_end]
            fit_set_this = np.hstack(
                (trial_numbers_shuffled[:test_idx_begin], trial_numbers_shuffled[test_idx_end:])
            )

            # -- Fit data using fit_set_this --
            fitting_result_this_fold = self.__class__._optimize_DE(
                fit_choice_history=fit_choice_history,
                fit_reward_history=fit_reward_history,
                fit_names=fit_names,
                lower_bounds=lower_bounds,
                upper_bounds=upper_bounds,
                fit_trial_set=fit_set_this,
                clamp_params=clamp_params,
                agent_kwargs=agent_kwargs,
                DE_kwargs=DE_kwargs,
            )
            fitting_results_all_folds.append(fitting_result_this_fold)

            # -- Compute the prediction accuracy of testing set --
            # Run PREDICTIVE simulation using temp_agent with the fitted parms of this fold
            tmp_agent = self.__class__(fitting_result_this_fold.params, **agent_kwargs)
            tmp_agent.predictive_perform(fit_choice_history, fit_reward_history)

            # Compute prediction accuracy
            predictive_choice_prob_this_fold = np.argmax(
                tmp_agent.choice_prob[:, :-1], axis=0
            )  # Exclude the last update

            correct_predicted = predictive_choice_prob_this_fold == fit_choice_history
            prediction_accuracy_fit.append(
                np.sum(correct_predicted[fit_set_this]) / len(fit_set_this)
            )
            prediction_accuracy_test.append(
                np.sum(correct_predicted[test_set_this]) / len(test_set_this)
            )
            # Also return cross-validated prediction_accuracy_bias_only
            if "biasL" in fitting_result_this_fold.params:
                bias_this = fitting_result_this_fold.params["biasL"]
                prediction_correct_bias_only = (
                    int(bias_this <= 0) == fit_choice_history
                )  # If bias_this < 0, bias predicts all rightward choices
                prediction_accuracy_test_bias_only.append(
                    sum(prediction_correct_bias_only[test_set_this]) / len(test_set_this)
                )

        # --- Save all cross_validation results, including raw fitting result of each fold ---
        fitting_result_cross_validation = dict(
            prediction_accuracy_test=prediction_accuracy_test,
            prediction_accuracy_fit=prediction_accuracy_fit,
            prediction_accuracy_test_bias_only=prediction_accuracy_test_bias_only,
            fitting_results_all_folds=fitting_results_all_folds,
        )
        self.fitting_result_cross_validation = fitting_result_cross_validation
        return fitting_result, fitting_result_cross_validation

    @classmethod
    def negLL_func_wrapper_for_de(
        cls,
        current_values,  # the current fitting values of params in fit_names (passed by DE)
        fit_choice_history,
        fit_reward_history,
        fit_trial_set,
        fit_names,
        clamp_params,
        agent_kwargs,
    ):
        """The core function that interacts with optimize.differential_evolution().
        For given params, run simulation using clamped history and
        return negative log likelihood.

        Note that this is a class method.
        """

        # -- Parse params and initialize a new agent --
        params = dict(zip(fit_names, current_values))  # Current fitting values
        params.update(clamp_params)  # Add clamped params
        agent = cls(params, **agent_kwargs)

        # -- Run **PREDICTIVE** simulation --
        # (clamp the history and do only one forward step on each trial)
        agent.predictive_perform(fit_choice_history, fit_reward_history)

        # Note that, again, we have an extra update after the last trial,
        # which is not used for fitting
        choice_prob = agent.choice_prob[:, :-1]

        return negLL(
            choice_prob, fit_choice_history, fit_reward_history, fit_trial_set
        )  # Return total negative log likelihood of the fit_trial_set

    @classmethod
    def _optimize_DE(
        cls,
        fit_choice_history,
        fit_reward_history,
        fit_names,
        lower_bounds,
        upper_bounds,
        fit_trial_set,
        clamp_params,
        agent_kwargs,
        DE_kwargs,
    ):
        """A wrapper of DE fitting for the model. It returns fitting results."""
        # --- Arguments for differential_evolution ---
        kwargs = dict(
            mutation=(0.5, 1),
            recombination=0.7,
            popsize=16,
            strategy="best1bin",
            disp=False,
            workers=1,
            updating="immediate",
            callback=None,
        )  # Default DE kwargs
        kwargs.update(DE_kwargs)  # Update user specified kwargs
        if kwargs["workers"] > 1:
            kwargs["updating"] = "deferred"

        # --- Heavy lifting here!! ---
        fitting_result = optimize.differential_evolution(
            func=cls.negLL_func_wrapper_for_de,
            bounds=optimize.Bounds(lower_bounds, upper_bounds),
            args=(
                fit_choice_history,
                fit_reward_history,
                fit_trial_set,  # subset of trials to fit; if empty, use all trials)
                fit_names,  # Pass names so that negLL_func_for_de knows which parameters to fit
                clamp_params,  # Clamped parameters
                agent_kwargs,  # Other kwargs to pass to the model
            ),
            **kwargs,
        )

        # --- Post-processing ---
        fitting_result.fit_settings = dict(
            fit_choice_history=fit_choice_history,
            fit_reward_history=fit_reward_history,
            fit_names=fit_names,
            lower_bounds=lower_bounds,
            upper_bounds=upper_bounds,
            clamp_params=clamp_params,
            agent_kwargs=agent_kwargs,
        )
        # Full parameter set
        params = dict(zip(fit_names, fitting_result.x))
        params.update(clamp_params)
        fitting_result.params = params
        fitting_result.k_model = len(fit_names)
        fitting_result.n_trials = len(fit_choice_history)
        fitting_result.log_likelihood = -fitting_result.fun

        fitting_result.AIC = -2 * fitting_result.log_likelihood + 2 * fitting_result.k_model
        fitting_result.BIC = -2 * fitting_result.log_likelihood + fitting_result.k_model * np.log(
            fitting_result.n_trials
        )

        # Likelihood-Per-Trial. See Wilson 2019 (but their formula was wrong...)
        fitting_result.LPT = np.exp(
            fitting_result.log_likelihood / fitting_result.n_trials
        )  # Raw LPT without penality
        fitting_result.LPT_AIC = np.exp(-fitting_result.AIC / 2 / fitting_result.n_trials)
        fitting_result.LPT_BIC = np.exp(-fitting_result.BIC / 2 / fitting_result.n_trials)

        return fitting_result

    def plot_session(self):
        """Plot session after .perform(task)"""
        fig, axes = plot_foraging_session(
            choice_history=self.task.get_choice_history(),
            reward_history=self.task.get_reward_history(),
            p_reward=self.task.get_p_reward(),
        )
        # Add Q value
        axes[0].plot(self.q_estimation[L, :], label="Q_left", color="red", lw=0.5)
        axes[0].plot(self.q_estimation[R, :], label="R_left", color="blue", lw=0.5)
        axes[0].legend(fontsize=6, loc="upper left", bbox_to_anchor=(0.6, 1.3), ncol=3)
        return fig, axes

    def plot_fitted_session(self):
        """Plot session after .fit()

        1. choice and reward history will be the history used for fitting
        2. laten variables q_estimate and choice_prob will be plotted
        3. p_reward will be missing (since it is not used for fitting)
        """
        if self.fitting_result is None:
            print("No fitting result found. Please fit the model first.")
            return

        # -- Retrieve fitting results and perform the predictive simiulation
        self.set_params(self.fitting_result.params)
        fit_choice_history = self.fitting_result.fit_settings["fit_choice_history"]
        fit_reward_history = self.fitting_result.fit_settings["fit_reward_history"]
        self.predictive_perform(fit_choice_history, fit_reward_history)

        # -- Plot the target choice and reward history
        # Note that the p_reward could be agnostic to the model fitting.
        fig, axes = plot_foraging_session(
            choice_history=fit_choice_history,
            reward_history=fit_reward_history,
            p_reward=np.full((2, len(fit_choice_history)), np.nan),  # Dummy p_reward
        )

        # -- Plot fitted Q values
        axes[0].plot(self.q_estimation[0], lw=1, color="red", ls=":", label="fitted_Q(L)")
        axes[0].plot(self.q_estimation[1], lw=1, color="blue", ls=":", label="fitted_Q(R)")

        # -- Plot fitted choice_prob
        axes[0].plot(
            self.choice_prob[1] / self.choice_prob.sum(axis=0),
            lw=2,
            color="green",
            ls=":",
            label="fitted_choice_prob(R/R+L)",
        )
        axes[0].legend(fontsize=6, loc="upper left", bbox_to_anchor=(0.6, 1.3), ncol=4)

        return fig, axes


def negLL(choice_prob, fit_choice_history, fit_reward_history, fit_trial_set=None):
    """Compute total negLL of the trials in fit_trial_set given the data."""

    # Compute negative likelihood
    likelihood_each_trial = choice_prob[
        fit_choice_history.astype(int), range(len(fit_choice_history))
    ]  # Get the actual likelihood for each trial

    # TODO: check this!
    # Deal with numerical precision (in rare cases, likelihood can be < 0 or > 1)
    likelihood_each_trial[(likelihood_each_trial <= 0) & (likelihood_each_trial > -1e-5)] = (
        1e-16  # To avoid infinity, which makes the number of zero likelihoods informative!
    )
    likelihood_each_trial[likelihood_each_trial > 1] = 1

    # Return total likelihoods
    if fit_trial_set is None:  # Use all trials
        return -np.sum(np.log(likelihood_each_trial))
    else:
        return -np.sum(np.log(likelihood_each_trial[fit_trial_set]))
