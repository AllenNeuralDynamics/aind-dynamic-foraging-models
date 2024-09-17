"""Base class for DynamicForagingAgent with MLE fitting
"""

import logging
from typing import Optional, Tuple, Type

import numpy as np
import scipy.optimize as optimize
from aind_behavior_gym.dynamic_foraging.agent import DynamicForagingAgentBase
from aind_behavior_gym.dynamic_foraging.task import DynamicForagingTaskBase
from aind_dynamic_foraging_basic_analysis import plot_foraging_session
from pydantic import BaseModel

from .params import ParamsSymbols

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
console_handler = logging.StreamHandler()
console_handler.setFormatter(
    logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
)
logger.addHandler(console_handler)


class DynamicForagingAgentMLEBase(DynamicForagingAgentBase):
    """Base class of "DynamicForagingAgentBase" + "MLE fitting" """

    def __init__(
        self,
        agent_kwargs: dict = {},
        params: dict = {},
        **kwargs,
    ):
        """Init

        Parameters
        ----------
        agent_kwargs : dict, optional
            The hyperparameters that define the agent type, by default {}
            For example, number_of_learning_rate, number_of_forget_rate, etc.
        params : dict, optional
            The kwargs that define the agent's parameters, by default {}
        **kwargs : dict
            Other kwargs that are passed to the base class, like rng's seed, by default {}
        """
        super().__init__(**kwargs)  # Set self.rng etc.

        # Get pydantic model for the parameters and bounds
        self.ParamModel, self.ParamFitBoundModel = self._get_params_model(agent_kwargs)

        # Set and validate the model parameters. Use default parameters if some are not provided
        self.params = self.ParamModel(**params)
        self._get_params_list()  # Get the number of free parameters of the agent etc.

        # Add model fitting related attributes
        self.fitting_result = None
        self.fitting_result_cross_validation = None

        # Some initializations
        self.n_actions = 2
        self.task = None
        self.n_trials = 0  # Should be set in perform or perform_closed_loop

    def _get_params_model(self, agent_kwargs, params) -> Tuple[Type[BaseModel], Type[BaseModel]]:
        """Dynamically generate the Pydantic model for parameters and fitting bounds.

        This should be overridden by the subclass!!
        It should return ParamModel and ParamFitBoundModel here.
        """
        raise NotImplementedError("This should be overridden by the subclass!!")

    def _get_params_list(self):
        """Get the number of free parameters of the agent etc."""
        self.params_list_all = list(self.ParamModel.model_fields.keys())
        self.params_list_frozen = {
            name: field.default
            for name, field in self.ParamModel.model_fields.items()
            if field.frozen
        }  # Parameters that are frozen by construction
        self.params_list_free = list(set(self.params_list_all) - set(self.params_list_frozen))

    def set_params(self, **params):
        """Update the model parameters and validate"""
        # This is safer than model_copy(update) because it will NOT validate the input params
        _params = self.params.model_dump()
        _params.update(params)
        self.params = self.ParamModel(**_params)
        return self.get_params()

    def get_agent_alias(self):
        """Get the agent alias for the model

        Should be overridden by the subclass.
        """
        return ""

    def get_params(self):
        """Get the model parameters in a dictionary format"""
        return self.params.model_dump()

    def get_params_str(self, if_latex=True, if_value=True, decimal=3):
        """Get string of the model parameters

        Parameters
        -----------
        if_latex: bool, optional
            if True, return the latex format of the parameters, by default True
        if_value: bool, optional
            if True, return the value of the parameters, by default True
        decimal: int, optional

        """
        # Sort the parameters by the order of ParamsSymbols
        params_default_order = list(ParamsSymbols.__members__.keys())
        params_list = sorted(
            self.get_params().items(), key=lambda x: params_default_order.index(x[0])
        )

        # Get fixed parameters if any
        if self.fitting_result is not None:
            # Effective fixed parameters (agent's frozen parameters + user specified clamp_params)
            fixed_params = self.fitting_result.fit_settings["clamp_params"].keys()
        else:
            # Frozen parameters (by construction)
            fixed_params = self.params_list_frozen.keys()

        ps = []
        for p in params_list:
            name_str = ParamsSymbols[p[0]] if if_latex else p[0]
            value_str = f" = {p[1]:.{decimal}f}" if if_value else ""
            fix_str = " (fixed)" if p[0] in fixed_params else ""
            ps.append(f"{name_str}{value_str}{fix_str}")

        return ", ".join(ps)

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

    def _reset(self):
        """Reset the agent"""
        self.trial = 0

        # MLE agent must have choice_prob
        self.choice_prob = np.full([self.n_actions, self.n_trials], np.nan)
        self.choice_prob[:, 0] = 1 / self.n_actions  # To be strict (actually no use)

        # Choice and reward history have n_trials length
        self.choice_history = np.full(self.n_trials, fill_value=-1, dtype=int)  # Choice history
        # Reward history, separated for each port (Corrado Newsome 2005)
        self.reward_history = np.zeros(self.n_trials)

    def perform(
        self,
        task: DynamicForagingTaskBase,
    ):
        """Generative simulation of a task, or "open-loop" simulation

        Override the base class method to include choice_prob caching etc.

        In each trial loop (note when time ticks):
                              agent.act()     task.step()    agent.learn()
            latent variable [t]  -->  choice [t]  --> reward [t] ---->  update latent variable [t+1]
        """
        self.task = task
        self.n_trials = task.num_trials

        # --- Main task loop ---
        self._reset()  # Reset agent
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

    def perform_closed_loop(self, fit_choice_history, fit_reward_history):
        """Simulates the agent over a fixed choice and reward history using its params.
        Also called "teacher forcing" or "closed-loop" simulation.

        Unlike .perform() ("generative" simulation), this is called "predictive" simulation,
        which does not need a task and is used for model fitting.
        """
        self.n_trials = len(fit_choice_history)
        self._reset()

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

    def act(self, observation):
        """
        Chooses an action based on the current observation.
        I just copy and paste this from DynamicForagingAgentBase here for clarity.

        Args:
            observation: The current observation from the environment.

        Returns:
            action: The action chosen by the agent.
        """
        raise NotImplementedError("The 'act' method should be overridden by subclasses.")

    def learn(self, observation, action, reward, next_observation, done):
        """
        Updates the agent's knowledge or policy based on the last action and its outcome.
        I just copy and paste this from DynamicForagingAgentBase here for clarity.

        This is the core method that should be implemented by all non-trivial agents.
        It could be Q-learning, policy gradients, neural networks, etc.

        Args:
            observation: The observation before the action was taken.
            action: The action taken by the agent.
            reward: The reward received after taking the action.
            next_observation: The next observation after the action.
            done: Whether the episode has ended.
        """
        raise NotImplementedError("The 'learn' method should be overridden by subclasses.")

    def fit(
        self,
        fit_choice_history,
        fit_reward_history,
        fit_bounds_override: Optional[dict] = {},
        clamp_params: Optional[dict] = {},
        k_fold_cross_validation: Optional[int] = None,
        DE_kwargs: Optional[dict] = {"workers": 1},
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
        assert self.ParamModel(**clamp_params)

        # -- Get fit_names and fit_bounds --
        # Validate fit_bounds_override and fill in the missing bounds with default bounds
        fit_bounds = self.ParamFitBoundModel(**fit_bounds_override).model_dump()
        # Add agent's frozen parameters (by construction) to clamp_params (user specified)
        clamp_params = clamp_params.copy()  # Make a copy to avoid modifying the default dict!!
        clamp_params.update(self.params_list_frozen)
        # Remove clamped parameters from fit_bounds
        for name in clamp_params.keys():
            fit_bounds.pop(name)
        # In the remaining parameters, check whether there are still collapsed bounds
        # If yes, clamp them to the collapsed value and remove them from fit_bounds
        _to_remove = []
        for name, bounds in fit_bounds.items():
            if bounds[0] == bounds[1]:
                clamp_params.update({name: bounds[0]})
                _to_remove.append(name)
                logger.warning(
                    f"Parameter {name} is clamped to {bounds[0]} "
                    f"because of collapsed bounds. "
                    f"Please specify it in clamp_params instead."
                )
        for name in _to_remove:
            fit_bounds.pop(name)

        # Get the names of the parameters to fit
        fit_names = list(fit_bounds.keys())
        # Parse bounds
        lower_bounds = [fit_bounds[name][0] for name in fit_names]
        upper_bounds = [fit_bounds[name][1] for name in fit_names]
        # Validate bounds themselves are valid parameters
        try:
            self.ParamModel(**dict(zip(fit_names, lower_bounds)))
            self.ParamModel(**dict(zip(fit_names, upper_bounds)))
        except ValueError as e:
            raise ValueError(
                f"Invalid bounds for {e}.\n"
                f"Bounds must be within the [ge, le] of the ParamModel.\n"
                f"Please check the bounds in fit_bounds_override."
            )

        # # ===== Fit using the whole dataset ======
        logger.info("Fitting the model using the whole dataset...")
        fitting_result = self._optimize_DE(
            fit_choice_history=fit_choice_history,
            fit_reward_history=fit_reward_history,
            fit_names=fit_names,
            lower_bounds=lower_bounds,
            upper_bounds=upper_bounds,
            clamp_params=clamp_params,
            fit_trial_set=None,  # None means use all trials to fit
            agent_kwargs=self.agent_kwargs,  # the class AND agent_kwargs fully define the agent
            DE_kwargs=DE_kwargs,
        )

        # -- Save fitting results --
        self.fitting_result = fitting_result

        # -- Rerun the predictive simulation with the fitted params--
        # To fill in the latent variables like q_value and choice_prob
        self.set_params(**fitting_result.params)
        self.perform_closed_loop(fit_choice_history, fit_reward_history)
        # Compute prediction accuracy
        predictive_choice = np.argmax(self.choice_prob, axis=0)
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
            fitting_result_this_fold = self._optimize_DE(
                agent_kwargs=self.agent_kwargs,  # the class AND agent_kwargs fully define the agent
                fit_choice_history=fit_choice_history,
                fit_reward_history=fit_reward_history,
                fit_names=fit_names,
                lower_bounds=lower_bounds,
                upper_bounds=upper_bounds,
                clamp_params=clamp_params,
                fit_trial_set=fit_set_this,
                DE_kwargs=DE_kwargs,
            )
            fitting_results_all_folds.append(fitting_result_this_fold)

            # -- Compute the prediction accuracy of testing set --
            # Run PREDICTIVE simulation using temp_agent with the fitted parms of this fold
            tmp_agent = self.__class__(params=fitting_result_this_fold.params, **self.agent_kwargs)
            tmp_agent.perform_closed_loop(fit_choice_history, fit_reward_history)

            # Compute prediction accuracy
            predictive_choice_prob_this_fold = np.argmax(tmp_agent.choice_prob, axis=0)

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

    def _optimize_DE(
        self,
        agent_kwargs,
        fit_choice_history,
        fit_reward_history,
        fit_names,
        lower_bounds,
        upper_bounds,
        clamp_params,
        fit_trial_set,
        DE_kwargs,
    ):
        """A wrapper of DE fitting for the model. It returns fitting results."""
        # --- Arguments for differential_evolution ---
        kwargs = dict(
            mutation=(0.5, 1),
            recombination=0.7,
            popsize=16,
            polish=True,
            strategy="best1bin",
            disp=False,
            workers=1,
            updating="immediate",
            callback=None,
        )  # Default DE kwargs
        kwargs.update(DE_kwargs)  # Update user specified kwargs
        # Special treatments
        if kwargs["workers"] > 1:
            kwargs["updating"] = "deferred"
        if "seed" in kwargs and isinstance(kwargs["seed"], (int, float)):
            # Convert seed to a numpy random number generator
            # because there seems to be a bug in DE when using int as a seed (not reproducible)
            kwargs["seed"] = np.random.default_rng(kwargs["seed"])

        # --- Heavy lifting here!! ---
        fitting_result = optimize.differential_evolution(
            func=self.__class__._cost_func_for_DE,
            bounds=optimize.Bounds(lower_bounds, upper_bounds),
            args=(
                agent_kwargs,  # Other kwargs to pass to the model
                fit_choice_history,
                fit_reward_history,
                fit_trial_set,  # subset of trials to fit; if empty, use all trials)
                fit_names,  # Pass names so that negLL_func_for_de knows which parameters to fit
                clamp_params,  # Clamped parameters
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
        fitting_result.k_model = len(fit_names)  # Number of free parameters of the model
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

        # Always save the result without polishing, regardless of the polish setting
        # (sometimes polishing will move parameters to boundaries, so I add this for sanity check)
        # - About `polish` in DE:
        #   - If `polish=False`, final `x` will be exactly the one in `population` that has the
        #     lowest `population_energy` (typically the first one).
        #     Its energy will also be the final `-log_likelihood`.
        #   - If `polish=True`, an additional gradient-based optimization will
        #     work on `population[0]`, resulting in the final `x`, and override the likelihood
        #     `population_energy[0]` . But it will not change `population[0]`!
        #   - That is to say, `population[0]` is always the result without `polish`.
        #     And if polished, we should rerun a `_cost_func_for_DE` to retrieve
        #     its likelihood, because it has been overridden by `x`.
        idx_lowest_energy = fitting_result.population_energies.argmin()
        x_without_polishing = fitting_result.population[idx_lowest_energy]

        log_likelihood_without_polishing = -self._cost_func_for_DE(
            x_without_polishing,
            agent_kwargs,  # Other kwargs to pass to the model
            fit_choice_history,
            fit_reward_history,
            fit_trial_set,  # subset of trials to fit; if empty, use all trials)
            fit_names,  # Pass names so that negLL_func_for_de knows which parameters to fit
            clamp_params,
        )
        fitting_result.x_without_polishing = x_without_polishing
        fitting_result.log_likelihood_without_polishing = log_likelihood_without_polishing

        params_without_polishing = dict(zip(fit_names, fitting_result.x_without_polishing))
        params_without_polishing.update(clamp_params)
        fitting_result.params_without_polishing = params_without_polishing
        return fitting_result

    @classmethod
    def _cost_func_for_DE(
        cls,
        current_values,  # the current fitting values of params in fit_names (passed by DE)
        # ---- Below are the arguments passed by args. The order must be the same! ----
        agent_kwargs,
        fit_choice_history,
        fit_reward_history,
        fit_trial_set,
        fit_names,
        clamp_params,
    ):
        """The core function that interacts with optimize.differential_evolution().
        For given params, run simulation using clamped history and
        return negative log likelihood.

        Note that this is a class method.
        """

        # -- Parse params and initialize a new agent --
        params = dict(zip(fit_names, current_values))  # Current fitting values
        params.update(clamp_params)  # Add clamped params
        agent = cls(params=params, **agent_kwargs)

        # -- Run **PREDICTIVE** simulation --
        # (clamp the history and do only one forward step on each trial)
        agent.perform_closed_loop(fit_choice_history, fit_reward_history)
        choice_prob = agent.choice_prob

        return negLL(
            choice_prob, fit_choice_history, fit_reward_history, fit_trial_set
        )  # Return total negative log likelihood of the fit_trial_set

    def plot_session(self, if_plot_latent=True):
        """Plot session after .perform(task)

        Parameters
        ----------
        if_plot_latent : bool, optional
            Whether to plot latent variables, by default True
        """
        fig, axes = plot_foraging_session(
            choice_history=self.task.get_choice_history(),
            reward_history=self.task.get_reward_history(),
            p_reward=self.task.get_p_reward(),
        )

        if if_plot_latent:
            # Plot latent variables
            self.plot_latent_variables(axes[0], if_fitted=False)
            # Plot choice_prob
            axes[0].plot(
                np.arange(self.n_trials) + 1,
                self.choice_prob[1] / self.choice_prob.sum(axis=0),
                lw=0.5,
                color="green",
                label="choice_prob(R/R+L)",
            )

        axes[0].legend(fontsize=6, loc="upper left", bbox_to_anchor=(0.6, 1.3), ncol=3)

        # 　Add the model parameters
        params_str = self.get_params_str()
        fig.suptitle(params_str, fontsize=10, horizontalalignment="left", x=fig.subplotpars.left)

        return fig, axes

    def plot_fitted_session(self, if_plot_latent=True):
        """Plot session after .fit()

        1. choice and reward history will be the history used for fitting
        2. laten variables q_estimate and choice_prob will be plotted
        3. p_reward will be missing (since it is not used for fitting)

        Parameters
        ----------
        if_plot_latent : bool, optional
            Whether to plot latent variables, by default True
        """
        if self.fitting_result is None:
            print("No fitting result found. Please fit the model first.")
            return

        # -- Retrieve fitting results and perform the predictive simiulation
        self.set_params(**self.fitting_result.params)
        fit_choice_history = self.fitting_result.fit_settings["fit_choice_history"]
        fit_reward_history = self.fitting_result.fit_settings["fit_reward_history"]
        self.perform_closed_loop(fit_choice_history, fit_reward_history)

        # -- Plot the target choice and reward history
        # Note that the p_reward could be agnostic to the model fitting.
        fig, axes = plot_foraging_session(
            choice_history=fit_choice_history,
            reward_history=fit_reward_history,
            p_reward=np.full((2, len(fit_choice_history)), np.nan),  # Dummy p_reward
        )

        # -- Plot fitted latent variables and choice_prob --
        if if_plot_latent:
            # Plot latent variables
            self.plot_latent_variables(axes[0], if_fitted=True)
            # Plot fitted choice_prob
            axes[0].plot(
                np.arange(self.n_trials) + 1,
                self.choice_prob[1] / self.choice_prob.sum(axis=0),
                lw=2,
                color="green",
                ls=":",
                label="fitted_choice_prob(R/R+L)",
            )

        axes[0].legend(fontsize=6, loc="upper left", bbox_to_anchor=(0.6, 1.3), ncol=4)

        # 　Add the model parameters
        params_str = self.get_params_str()
        fig.suptitle(
            f"fitted: {params_str}", fontsize=10, horizontalalignment="left", x=fig.subplotpars.left
        )

        return fig, axes

    def plot_latent_variables(self, ax, if_fitted=False):
        """Add agent-specific latent variables to the plot

        if_fitted: whether the latent variables are from the fitted model (styling purpose)
        """
        pass

    def get_latent_variables(self):
        """Return the latent variables of the agent

        This is agent-specific and should be implemented by the subclass.
        """
        return None

    @staticmethod
    def _fitting_result_to_dict(fitting_result_object, if_include_choice_reward_history=True):
        """Turn each fitting_result object (all data or cross-validation) into a dict

        if_include_choice_reward_history: whether to include choice and reward history in the dict.
        To save space, we may not want to include them for each fold in cross-validation.
        """

        # -- fit_settings --
        fit_settings = fitting_result_object.fit_settings.copy()
        if if_include_choice_reward_history:
            fit_settings["fit_choice_history"] = fit_settings["fit_choice_history"].tolist()
            fit_settings["fit_reward_history"] = fit_settings["fit_reward_history"].tolist()
        else:
            fit_settings.pop("fit_choice_history")
            fit_settings.pop("fit_reward_history")

        # -- fit_stats --
        fit_stats = {}
        fit_stats_fields = [
            "params",
            "log_likelihood",
            "AIC",
            "BIC",
            "LPT",
            "LPT_AIC",
            "LPT_BIC",
            "k_model",
            "n_trials",
            "nfev",
            "nit",
            "success",
            "population",
            "population_energies",
            "params_without_polishing",
            "log_likelihood_without_polishing",
        ]
        for field in fit_stats_fields:
            value = fitting_result_object[field]

            # If numpy array, convert to list
            if isinstance(value, np.ndarray):
                value = value.tolist()
            fit_stats[field] = value

        return {
            "fit_settings": fit_settings,
            **fit_stats,
        }

    def get_fitting_result_dict(self):
        """Return the fitting result in a json-compatible dict for uploading to docDB etc."""
        if self.fitting_result is None:
            print("No fitting result found. Please fit the model first.")
            return

        # -- result of fitting with all data --
        dict_fit_on_whole_data = self._fitting_result_to_dict(
            self.fitting_result, if_include_choice_reward_history=True
        )
        # Add prediction accuracy because it is treated separately for the whole dataset fitting
        dict_fit_on_whole_data["prediction_accuracy"] = self.fitting_result.prediction_accuracy

        # Add class name and agent alias to fit_settings for convenience
        dict_fit_on_whole_data["fit_settings"]["agent_class_name"] = self.__class__.__name__
        dict_fit_on_whole_data["fit_settings"]["agent_alias"] = self.get_agent_alias()

        # -- latent variables --
        latent_variables = self.get_latent_variables()

        # -- Pack all results --
        fitting_result_dict = {
            **dict_fit_on_whole_data,
            "fitted_latent_variables": latent_variables,
        }

        # -- Add cross validation if available --
        if self.fitting_result_cross_validation is not None:
            # Overall goodness of fit
            cross_validation = {
                "prediction_accuracy_test": self.fitting_result_cross_validation[
                    "prediction_accuracy_test"
                ],
                "prediction_accuracy_fit": self.fitting_result_cross_validation[
                    "prediction_accuracy_fit"
                ],
                "prediction_accuracy_test_bias_only": self.fitting_result_cross_validation[
                    "prediction_accuracy_test_bias_only"
                ],
            }

            # Fitting results of each fold
            fitting_results_each_fold = {}
            for kk, fitting_result_fold in enumerate(
                self.fitting_result_cross_validation["fitting_results_all_folds"]
            ):
                fitting_results_each_fold[f"{kk}"] = self._fitting_result_to_dict(
                    fitting_result_fold, if_include_choice_reward_history=False
                )
            cross_validation["fitting_results_each_fold"] = fitting_results_each_fold
            fitting_result_dict["cross_validation"] = cross_validation

        return fitting_result_dict


# -- Helper function --
def negLL(choice_prob, fit_choice_history, fit_reward_history, fit_trial_set=None):
    """Compute total negLL of the trials in fit_trial_set given the data."""

    # Compute negative likelihood
    likelihood_each_trial = choice_prob[
        fit_choice_history.astype(int), range(len(fit_choice_history))
    ]  # Get the actual likelihood for each trial

    # Deal with numerical precision (in rare cases, likelihood can be < 0 or > 1)
    likelihood_each_trial[(likelihood_each_trial <= 0) & (likelihood_each_trial > -1e-5)] = (
        1e-16  # To avoid infinity, which makes the number of zero likelihoods informative!
    )
    likelihood_each_trial[likelihood_each_trial > 1] = 1

    # Return total likelihoods
    if fit_trial_set is None:  # Use all trials
        return -np.sum(np.log(likelihood_each_trial))
    else:  # Use subset of trials in cross-validation
        return -np.sum(np.log(likelihood_each_trial[fit_trial_set]))
