"""Maximum likelihood fitting of foraging models
"""
# %%
import numpy as np
import scipy.optimize as optimize
import multiprocessing as mp
from pydantic import BaseModel, Field, model_validator

from aind_behavior_gym.dynamic_foraging.agent import DynamicForagingAgentBase
from aind_behavior_gym.dynamic_foraging.task import DynamicForagingTaskBase, L, R, IGNORE

from .act_functions import act_softmax
from .learn_functions import learn_RWlike

from aind_dynamic_foraging_basic_analysis import plot_foraging_session

class Bounds(BaseModel):
    lower: float = Field(..., description="Lower bound for the parameter")
    upper: float = Field(..., description="Upper bound for the parameter")
    
    @model_validator(mode="after")
    def check_bounds(cls, values):
        if values.lower >= values.upper:
            raise ValueError("Upper bound must be greater than lower bound")

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
        
        learn_rate_rew: float = Field(default=0.5, ge=0.0, le=1.0, description="Learning rate for rewarded choice")
        learn_rate_unrew : float = Field(default=0.1, ge=0.0, le=1.0, description="Learning rate for unrewarded choice")
        forget_rate: float = Field(default=0.2, ge=0.0, le=1.0, description="Forgetting rate for unchosen options")
        softmax_inverse_temperature: float = Field(default=10, ge=0.0, description="Softmax temperature")
        biasL: float = Field(default=0.0, description="Bias term for softmax")
        pass

    class ParamFitBounds(BaseModel):
        """Bounds for fitting parameters.
        After overriden in subclasses, calling ClassName.ParamFitBounds() returns default bounds.
        """
        # For example:
        # param1: Bounds = Field(default=Bounds(lower=0.0, upper=1.0), description="Bounds for param1")
        # param2: Bounds = Field(default=Bounds(lower=0.0, upper=1.0), description="Bounds for param2")
        # raise NotImplementedError("ParamFitBounds class must be defined in subclasses")
        learn_rate_rew: Bounds = Field(default=Bounds(lower=0.0, upper=1.0))
        learn_rate_unrew: Bounds = Field(default=Bounds(lower=0.0, upper=1.0))
        forget_rate: Bounds = Field(default=Bounds(lower=0.0, upper=1.0))
        softmax_inverse_temperature: Bounds = Field(default=Bounds(lower=0.0, upper=100.0))
        biasL: Bounds = Field(default=Bounds(lower=-5.0, upper=5.0))
        pass

    def __init__(
        self,
        params: dict = {},
        **kwargs,
        ):

        super().__init__(**kwargs)

        # Set and validate the model parameters. Use default parameters if some are not provided
        self.params = self.Param(**params)
        
        # Add model fitting related attributes
        self.fitting_result = None
        self.fit_bounds_default = dict()
        
        # Some switches
        self.fit_choice_kernel = False
        
        # Some initializations
        self.task = None

    def reset(self):
        self.trial = 0
        
        # Latent variables have n_trials + 1 length to capture the update after the last trial (HH20210726)
        self.q_estimation = np.full([self.n_actions, self.n_trials + 1], np.nan)
        self.q_estimation[:, 0] = 0  # Initial Q values as 0
        
        self.choice_prob = np.full([self.n_actions, self.n_trials + 1], np.nan)
        self.choice_prob[:, 0] = 1 / self.n_actions  # To be strict (actually no use)   
             
        if self.fit_choice_kernel:
            self.choice_kernel = np.zeros([self.n_actions, self.n_trials + 1])
        
        # Choice and reward history have n_trials length
        self.choice_history = np.full([1, self.n_trials], fill_value=-1, dtype=int)  # Choice history        
        # Reward history, separated for each port (Corrado Newsome 2005)
        self.reward_history = np.zeros([self.n_actions, self.n_trials])


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
        """Override the base class method to include choice_prob caching etc."""
        self.task = task
        self.n_actions = task.action_space.n
        self.n_trials = task.num_trials

        # --- Main task loop ---
        self.reset()  # Reset agent
        observation, info = self.task.reset()  # Reset task and get the initial observation
        task_done = False
        while not task_done:
            assert self.trial == self.task.trial  # Ensure the two timers are in sync

            # -- Agent performs an action
            choice, choice_prob = self.act(observation)
            
            # -- Environment steps (enviromnet's timer ticks here!!!)
            next_observation, reward, task_done, _, _ = task.step(choice)

            # -- Update agent history
            self.choice_prob[:, self.trial] = choice_prob
            self.choice_history[0, self.trial] = choice     
            # In Sutton & Barto's convention, reward belongs to the next time step, but we put it
            # in the current time step for the sake of consistency with neuroscience convention
            self.reward_history[choice, self.trial] = reward
            
            # -- Agent's timer ticks here !!!
            self.trial += 1
            
            # -- Update q values
            # Note that this will update the q values **after the last trial**, a final update that
            # will not be used to make the next action (because task is *done*) but may be used for
            # correlating with physiology recordings
            self.learn(observation, choice, reward, next_observation, task_done)  
            observation = next_observation

    def act(self, observation):
        choice, choice_prob = act_softmax(
            q_estimation_t=self.q_estimation[:, self.trial],
            softmax_inverse_temperature=self.params.softmax_inverse_temperature,
            bias_terms=np.array([self.params.biasL, 0]),
            choice_softmax_inverse_temperature=None,
            choice_kernel=None,
            rng=self.rng,
        )
        return choice, choice_prob

    def learn(self, observation, choice, reward, next_observation, done):
        """Update Q values"""
        # Update Q values
        self.q_estimation[:, self.trial] = learn_RWlike(
            **{
                "choice": choice,
                "reward": reward,
                "q_estimation_tminus1": self.q_estimation[:, self.trial - 1],
                "learn_rates": [self.params.learn_rate_rew, self.params.learn_rate_unrew],
                "forget_rates": [self.params.forget_rate, 0],   # 0: unchosen, 1: chosen
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
        np.testing.assert_array_equal(self.choice_history[0], self.task.get_choice_history())
        return self.task.get_choice_history()
    
    def get_reward_history(self):
        """Return the history of reward in format that is compatible with other library such as
        aind_dynamic_foraging_basic_analysis
        """
        if self.task is None:
            return None
        # Make sure agent's history is consistent with the task's history and return
        np.testing.assert_array_equal(np.sum(self.reward_history, axis=0), 
                                      self.task.get_reward_history())
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
        fit_bounds: dict = {},  # Specify parameters to fit and their bounds. It must be validated 
        # by the ParamFitBounds class of the model, but only params in the fit_bounds will be fitted
        DE_pop_size=16,
        pool="",
    ):
        """A class method for fitting the model
        """

        fit_names = list(fit_bounds.keys()) # Get the names of the parameters to fit
        assert self.ParamFitBounds(fit_bounds) # Only assert that the fit_bounds are valid, 
        # but the user input fit_bounds will be used in the fitting function

        lower_bounds = [fit_bounds[name].lower for name in fit_names]
        upper_bounds = [fit_bounds[name].upper for name in fit_names]

        fitting_result = optimize.differential_evolution(
            func=self.__class__.negLL_func_for_de,
            bounds=optimize.Bounds(lower_bounds, upper_bounds),
            args=(
                fit_choice_history,
                fit_reward_history,
                fit_names,  # Pass names so that negLL_func_for_de knows which parameters to fit
            ),
            mutation=(0.5, 1),
            recombination=0.7,
            popsize=DE_pop_size,
            strategy="best1bin",
            disp=False,
            workers=1 if pool == "" else int(mp.cpu_count()),
            # For DE, use pool to control if_parallel, although we don't use pool for DE
            updating="immediate" if pool == "" else "deferred",
            callback=None,
        )

        fitting_result.k_model = np.sum(
            np.diff(np.array(fit_bounds), axis=0) > 0
        )  # Get the number of fitted parameters with non-zero range of bounds
        fitting_result.n_trials = np.shape(fit_choice_history)[1]
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

        self.firting_result = fitting_result
        return fitting_result

    @classmethod
    def negLL_func_for_de(
        cls,
        fit_values, # the current fitting values of params in fit_names
        *args
    ):
        """ The core function that interacts with optimize.differential_evolution(). 
        For given params, run simulation using clamped history and 
        return negative log likelihood.
        
        Note that this is a class method.
        """
        # Arguments interpretation
        fit_choice_history, fit_reward_history, fit_names, fit_bandit_kwargs = args

        for name, value in zip(fit_names, fit_values):
            fit_bandit_kwargs.update({name: value})

        # Run **PREDICTIVE** simulation
        # (clamp the history and do only one forward step on each trial)
        # Use the current class to initialize a new agent for simulation
        bandit = cls(**fit_bandit_kwargs)
        bandit.simulate_fit(fit_choice_history, fit_reward_history)
        negLL = bandit.negLL(fit_choice_history, fit_reward_history)
        return negLL

    def set_fitparams_random(self):
        x0 = []
        for lb, ub in zip(self.fit_bounds[0], self.fit_bounds[1]):
            x0.append(self.rng.uniform(lb, ub))
        for i_name, name in enumerate(self.fit_names):
            setattr(self, name, x0[i_name])
        return x0

    def set_fitparams_values(self, x0):
        for i_name, name in enumerate(self.fit_names):
            setattr(self, name, x0[i_name])

    def get_fitparams_values(self):
        x0 = list()
        for i_name, name in enumerate(self.fit_names):
            x0.append(getattr(self, name))
        return x0

    def negLL(self, fit_choice_history, fit_reward_history):
        likelihood_all_trial = []

        # Compute negative likelihood
        choice_prob = self.choice_prob[
            :, :-1
        ]  # Get all predictive choice probability [K, num_trials], exclude the final update after the last trial
        likelihood_each_trial = choice_prob[
            fit_choice_history[0, :], range(len(fit_choice_history[0]))
        ]  # Get the actual likelihood for each trial

        # TODO: check this!
        # Deal with numerical precision
        likelihood_each_trial[(likelihood_each_trial <= 0) & (likelihood_each_trial > -1e-5)] = (
            1e-16  # To avoid infinity, which makes the number of zero likelihoods informative!
        )
        likelihood_each_trial[likelihood_each_trial > 1] = 1

        # Cache likelihoods
        likelihood_all_trial.extend(likelihood_each_trial)
        likelihood_all_trial = np.array(likelihood_all_trial)
        negLL = -sum(np.log(likelihood_all_trial))

        return negLL

    def simulate_fit(
        self, fit_choice_history, fit_reward_history
    ):  # This simulates the agent over a fixed choice and reward history
        self.n_actions, self.n_trials = np.shape(fit_reward_history)  # Use the targeted histories
        self.fit_choice_history = fit_choice_history
        self.fit_reward_history = fit_reward_history
        self.reset()
        for t in range(self.n_trials):
            choice, choice_prob = (
                self.act()
            )  # Compute choice and choice probabilities, updates choice history and choice probability
            choice = fit_choice_history[0, self.trial]  # Override choice
            reward = fit_reward_history[choice, self.trial]  # get reward from data
            self.choice_prob[:, self.trial] = choice_prob
            self.choice_history[0, self.trial] = choice
            self.reward_history[choice, self.trial] = reward
            self.step(choice, reward)  # updates reward history, and update time
            
    def plot_session(self):
        fig, axes = plot_foraging_session(
            choice_history=self.task.get_choice_history(),
            reward_history=self.task.get_reward_history(),
            p_reward=self.task.get_p_reward(),
        )
        
        # Add Q value
        axes[0].plot(self.q_estimation[L, :], label="Q_left", color='red', lw=0.5) 
        axes[0].plot(self.q_estimation[R, :], label="R_left", color='blue', lw=0.5) 
        axes[0].legend(fontsize=6, loc="upper left", bbox_to_anchor=(0.6, 1.3), ncol=3)
        
        return fig


