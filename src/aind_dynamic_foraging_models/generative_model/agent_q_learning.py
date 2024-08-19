"""Maximum likelihood fitting of foraging models
"""
# %%
import numpy as np
import scipy.optimize as optimize
import multiprocessing as mp
from pydantic import BaseModel, Field, model_validator
from .util import moving_average, softmax, choose_ps

from aind_behavior_gym.dynamic_foraging.agent import DynamicForagingAgentBase
from aind_behavior_gym.dynamic_foraging.task import DynamicForagingTaskBase, L, R, IGNORE
from aind_dynamic_foraging_basic_analysis import plot_foraging_session

class Bounds(BaseModel):
    lower: float = Field(..., description="Lower bound for the parameter")
    upper: float = Field(..., description="Upper bound for the parameter")
    
    @model_validator(mode="after")
    def check_bounds(cls, values):
        if values.lower >= values.upper:
            raise ValueError("Upper bound must be greater than lower bound")

class DynamicForagingAgentMLEBase(DynamicForagingAgentBase):
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
        softmax_temperature: float = Field(default=0.3, ge=0.0, description="Softmax temperature")
        biasL: float = Field(default=0.0, description="Bias term for softmax")
        pass

    class ParamFitBounds(BaseModel):
        """Bounds for fitting parameters.
        After overriden in subclasses, calling ClassName.ParamFitBounds() returns default bounds.
        """
        # For example:
        #
        param1: Bounds = Field(default=Bounds(lower=0.0, upper=1.0), description="Bounds for param1")
        param2: Bounds = Field(default=Bounds(lower=0.0, upper=1.0), description="Bounds for param2")
        # raise NotImplementedError("ParamFitBounds class must be defined in subclasses")
        pass

    def __init__(
        self,
        params: dict = {},
        ):

        super().__init__()

        # Set and validate the model parameters. Use default parameters if some are not provided
        self.params = self.Param(**params)
        
        # Add model fitting related attributes
        self.fitting_result = None
        self.fit_bounds_default = dict()
        
        # Some switches
        self.fit_choice_kernel = False

    def reset(self):
        self.trial = 0
        # All latent variables have n_trials + 1 length to capture the update after the last trial (HH20210726)
        self.q_estimation = np.full([self.n_actions, self.n_trials + 1], np.nan)
        self.q_estimation[:, 0] = 0
        self.choice_prob = np.full([self.n_actions, self.n_trials + 1], np.nan)
        self.choice_prob[:, 0] = 1 / self.n_actions  # To be strict (actually no use)        
        # Generative mode is now default mode of agent
        self.choice_history = np.zeros([1, self.n_trials + 1], dtype=int)  # Choice history        
        # Reward history, separated for each port (Corrado Newsome 2005)
        self.reward_history = np.zeros([self.n_actions, self.n_trials + 1])

        if self.fit_choice_kernel:
            self.choice_kernel = np.zeros([self.n_actions, self.n_trials + 1])

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
        done = False
        while not done:
            choice, choice_prob = self.act(observation)
            next_observation, reward, done, truncated, info = task.step(choice)

            assert self.trial == next_observation['trial']  # Ensure the trial number is consistent
            self.choice_prob[:, self.trial] = choice_prob
            self.choice_history[0, self.trial] = choice     

            # In Sutton & Barto's convention, reward belongs to the next time step, but we put it
            # in the current time step for the sake of consistency with the choice
            self.reward_history[choice, self.trial] = reward

            self.learn(observation, choice, reward, next_observation, done)  # Where self.trial increases
            observation = next_observation

    def act(self, observation):
        choice, choice_prob = act_softmax(
            q_estimation_t=self.q_estimation[:, self.trial],
            softmax_temperature=self.params.softmax_temperature,
            bias_terms=np.array([self.params.biasL, 0]),
            choice_softmax_temperature=None,
            choice_kernel=None,
        )
        return choice, choice_prob

    def learn(self, observation, choice, reward, next_observation, done):
        """Update Q values"""
        # Agent's time ticks here !!!
        self.trial += 1

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
            x0.append(np.random.uniform(lb, ub))
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
        axes[0].plot(self.q_estimation[L, :], label="Q_left", color='red') 
        axes[0].plot(self.q_estimation[R, :], label="R_left", color='blue') 
        axes[0].legend(fontsize=6, loc="upper left", bbox_to_anchor=(0.6, 1.3), ncol=3)


def sigmoid(x):
    return 1 / (1 + math.exp(-x))


def act_softmax(
    q_estimation_t=0,
    softmax_temperature=0,
    bias_terms=0,
    choice_softmax_temperature=None,
    choice_kernel=None,
):
    if choice_kernel is not None:
        q_estimation_t = np.vstack(
            [q_estimation_t, choice_kernel]
        ).transpose()  # the first dimension is the choice and the second is usual valu in position 0 and kernel in position 1
        softmax_temperature = np.array([softmax_temperature, choice_softmax_temperature])[
            np.newaxis, :
        ]
    choice_prob = softmax(q_estimation_t, temperature=softmax_temperature, bias=bias_terms)
    choice = choose_ps(choice_prob)
    return choice, choice_prob


def learn_RWlike(choice, reward, q_estimation_tminus1, forget_rates, learn_rates):
    # Reward-dependent step size ('Hattori2019')
    learn_rate_rew, learn_rate_unrew = learn_rates[0], learn_rates[1]
    if reward:
        learn_rate = learn_rate_rew
    else:
        learn_rate = learn_rate_unrew
    # Choice-dependent forgetting rate ('Hattori2019')
    # Chosen:   Q(n+1) = (1- forget_rate_chosen) * Q(n) + step_size * (Reward - Q(n))
    q_estimation_t = np.zeros_like(q_estimation_tminus1)
    K = q_estimation_tminus1.shape[0]
    q_estimation_t[choice] = (1 - forget_rates[1]) * q_estimation_tminus1[choice] + learn_rate * (
        reward - q_estimation_tminus1[choice]
    )
    # Unchosen: Q(n+1) = (1-forget_rate_unchosen) * Q(n)
    unchosen_idx = [cc for cc in range(K) if cc != choice]
    q_estimation_t[unchosen_idx] = (1 - forget_rates[0]) * q_estimation_tminus1[unchosen_idx]
    return q_estimation_t


class forager_Hattori2019(DynamicForagingAgentMLEBase):
    def __init__(
        self,
        softmax_temperature=None,
        learn_rate_rew=None,
        learn_rate_unrew=None,
        **kwargs,
    ):
        super(forager_Hattori2019, self).__init__(**kwargs)
        
        self.softmax_temperature = softmax_temperature
        self.learn_rate_rew = learn_rate_rew
        self.learn_rate_unrew = learn_rate_unrew

        self.fit_names.extend(
            ["learn_rate_rew", "learn_rate_unrew", "forget_rate", "softmax_temperature", "biasL"]
        )
        
        
        
        self.fit_std_values.extend([0.5, 0.5, 0.2, 0.1, 0.3])  # typical parameters
        self.fit_bounds[0].extend([0, 0, 0, 1e-2, -5])
        self.fit_bounds[1].extend([1, 1, 1, 15, 5])

        self.model_name = "Hattori2019"
        self.banditmodel = forager_Hattori2019
        self.step_function = learn_RWlike
        self.act_function = act_Probabilistic

    def step(self, choice, reward):
        step_kwargs = {
            "q_estimation_tminus1": self.q_estimation[:, self.trial],
            "learn_rates": [self.learn_rate_rew, self.learn_rate_unrew],
            "forget_rates": self.forget_rates,
        }
        return super().step(choice, reward, **step_kwargs)

    def act(
        self,
    ):  # Compatible with either fitting mode (predictive) or not (generative). It's much clear now!!
        act_kwargs = {
            "q_estimation_t": self.q_estimation[:, self.trial],
            "softmax_temperature": self.softmax_temperature,
            "bias_terms": self.bias_terms,
        }
        return super().act(**act_kwargs)
