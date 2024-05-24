import numpy as np
import scipy.optimize as optimize
import seaborn as sns
import matplotlib.pyplot as plt
import math
from foragingmodels.util import moving_average, softmax, choose_ps
import multiprocessing as mp
from collections import defaultdict

def sigmoid(x):
    return 1 / (1 + math.exp(-x))

def step_RWlike(choice, reward, q_estimation_tminus1, forget_rates, learn_rates):
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
    q_estimation_t[choice] = (1 - forget_rates[1]) * q_estimation_tminus1[choice] + learn_rate * (reward - q_estimation_tminus1[choice])
    # Unchosen: Q(n+1) = (1-forget_rate_unchosen) * Q(n)
    unchosen_idx = [cc for cc in range(K) if cc != choice]
    q_estimation_t[unchosen_idx] = (1 - forget_rates[0]) * q_estimation_tminus1[unchosen_idx]
    return q_estimation_t

def step_RWlike_ignore(choice, reward, valid_reward_history, q_estimation_tminus1, forget_rates, learn_rates, ignore_rates):
    if reward:
        learn_rate = learn_rates[1]
    else:
        learn_rate = learn_rates[0]
    w1, w2 = ignore_rates[0], ignore_rates[1]
    cumulative_reward = np.sum(valid_reward_history)
    q_estimation_t = np.zeros_like(q_estimation_tminus1)
    K = q_estimation_tminus1.shape[0]
    q_estimation_t[choice] = (1 - forget_rates[1]) * q_estimation_tminus1[choice] + learn_rate * (reward - q_estimation_tminus1[choice])
    # Unchosen: Q(n+1) = (1-forget_rate_unchosen) * Q(n)
    unchosen_idx = [cc for cc in range(K-1) if cc != choice]
    q_estimation_t[unchosen_idx] = (1 - forget_rates[0]) * q_estimation_tminus1[unchosen_idx]
    q_estimation_t[K-1] = w1 * cumulative_reward - w2 * np.sum(q_estimation_tminus1[:-1])
    return q_estimation_t


    # def step_function(self, a, r):
    #     self.et *= self.lam
    #     self.et[0, a] += 1.

    #     if np.random.rand() < self.br:
    #         self.updated = True
    #         self.mbsa += self.lr * (self.et - self.mbsa)
    #         self.mbs += self.lr * (self.et.sum(-1) - self.mbs)

    #         if r > 0:
    #             self.updated = True
    #             self.m[..., 0] += self.lr * (self.et - self.m[..., 0])
        
    #     return self.contingency


def step_ANCCR2(choice, reward, q_estimation_tminus1, m, mbs, mbsa, w):
    # et *= lam
    # et[0, choice] += 1.
    # if np.random.rand() < base_rate:        
    #     mbsa += learn_rate * (et - mbsa)
    #     mbs += learn_rate * (et.sum(-1) - mbs)
    # if reward > 0:            
    #     m[:,:, 0] += learn_rate * (et - m[:,:, 0])        
    cp = m - mbsa[:,:,np.newaxis] 
    cs = cp * mbs[np.newaxis, np.newaxis,:] / mbsa[:,:,np.newaxis].clip(min=10)
    contingency = cp * w + cs * (1 - w)
    # return contingency.flatten()

    # Reward-dependent step size ('Hattori2019')
    # learn_rate_rew, learn_rate_unrew = learn_rates[0], learn_rates[1]
    # if reward:
        # learn_rate = learn_rate_rew
    # else:
        # learn_rate = learn_rate_unrew
    # Choice-dependent forgetting rate ('Hattori2019')
    # Chosen:   Q(n+1) = (1- forget_rate_chosen) * Q(n) + step_size * (Reward - Q(n))
    
    # forget_rate=0.1
    # q_estimation_t = np.zeros_like(q_estimation_tminus1)
    # K = q_estimation_tminus1.shape[0]
    # q_estimation_t[choice] = (1 - forget_rate) * q_estimation_tminus1[choice] + learn_rate * (reward - q_estimation_tminus1[choice])    
    # unchosen_idx = [cc for cc in range(K) if cc != choice]
    # q_estimation_t[unchosen_idx] = (1 - forget_rate) * q_estimation_tminus1[unchosen_idx]
    return contingency.flatten()#q_estimation_t


def step_ANCCR(choice, reward, q_estimation_tminus1, learn_rate):#, et, lam, base_rate,  m, mbs, mbsa, w):
    # et *= lam
    # et[0, choice] += 1.
    # if np.random.rand() < base_rate:        
    # mbsa += learn_rate * (et - mbsa)
    # mbs += learn_rate * (et.sum(-1) - mbs)
    # if reward > 0:            
    #     m[..., 0] += learn_rate * (et - m[..., 0])        
    # cp = m - mbsa[..., None] 
    # cs = cp * mbs[None, None, :] / mbsa[..., None].clip(min=10)
    # contingency = cp * w + cs * (1 - w)    

    forget_rate=0.1
    q_estimation_t = np.zeros_like(q_estimation_tminus1)
    K = q_estimation_tminus1.shape[0]
    q_estimation_t[choice] = (1 - forget_rate) * q_estimation_tminus1[choice] + learn_rate * (reward - q_estimation_tminus1[choice])
    # Unchosen: Q(n+1) = (1-forget_rate_unchosen) * Q(n)
    unchosen_idx = [cc for cc in range(K) if cc != choice]
    q_estimation_t[unchosen_idx] = (1 - forget_rate) * q_estimation_tminus1[unchosen_idx]           
    return q_estimation_t

def act_Probabilistic(q_estimation_t=0, softmax_temperature=0, bias_terms=0, choice_softmax_temperature=None, choice_kernel=None):    
    if choice_kernel is not None:
        q_estimation_t = np.vstack([q_estimation_t, choice_kernel]).transpose() #the first dimension is the choice and the second is usual valu in position 0 and kernel in position 1
        softmax_temperature = np.array([softmax_temperature, choice_softmax_temperature])[np.newaxis,:]
    choice_prob = softmax(q_estimation_t, temperature=softmax_temperature, bias=bias_terms)
    choice = choose_ps(choice_prob)
    return choice, choice_prob

def act_Probabilistic_ignore(q_estimation_t=0, softmax_temperature=0, ignore_softmax_temperature=0, bias_terms=0, ignore_bias=1., choice_softmax_temperature=None, choice_kernel=None):
    if choice_kernel is not None:
        q_estimation_t = np.vstack([q_estimation_t, choice_kernel]).transpose() #the first dimension is the choice and the second is usual valu in position 0 and kernel in position 1
        softmax_temperature = np.array([softmax_temperature, choice_softmax_temperature])[np.newaxis,:]
        ignore_sotfmax_temperature = np.array([ignore_softmax_temperature, choice_softmax_temperature])[np.newaxis,:]
    choice_prob = np.zeros(q_estimation_t.shape[:1])    
    choice_prob[-1] = sigmoid(np.sum((q_estimation_t[-1]-ignore_bias)/ ignore_softmax_temperature))
    choice_prob[:-1] =  (1.-choice_prob[-1]) * softmax(q_estimation_t[:-1], temperature=softmax_temperature)
    choice = choose_ps(choice_prob)
    return choice, choice_prob

class ForagerModel:
    '''
    Foragers that can simulate and fit bandit models
    '''

    # @K_arm: # of arms
    # @epsilon: probability for exploration in epsilon-greedy algorithm
    # @initial: initial estimation for each action
    # @step_size: constant step size for updating estimations

    def __init__(self, K_arm=2, n_trials=1000,
                 biasL=0,
                 biasR=0,
                 forget_rate=None,
                 step_function=None,
                 act_function=None,
                 fit_choice_history=None,
                 fit_reward_history=None,                      
                 fit_choice_kernel=False,                                       
                 choice_step_size=None,
                 choice_softmax_temperature=None,
                 seed='',
                 ):
        self.K = K_arm
        self.n_trials = n_trials
        self.fit_choice_history = fit_choice_history
        self.fit_reward_history = fit_reward_history        
        self.step_function = step_function
        self.act_function = act_function
        self.fit_choice_kernel = fit_choice_kernel
        self.seed = seed        
        self.fit_names = list()
        self.fit_bounds = [list(),list()]        
        self.fit_std_values = list()
        self.fit_bandit_kwargs = {}
        
        self.optogenetic_perturbation = False
        self.opto_prob = 0.3
        self.opto_reward = 0.3

        # =============================================================================
        #   Parameter check and prepration
        # =============================================================================
        # -- Bias terms --
        # 2. for those involve softmax: b_undefined = 0, no constraint. cp_k = exp(Q/sigma + b_i) / sum(Q/sigma + b_i). Putting b_i outside /sigma to make it comparable across different softmax_temperatures
        if self.K == 2:                        
            self.bias_terms = np.array([biasL, 0])  # Relative to right
        elif self.K == 3:
            self.bias_terms = np.array([biasL, biasR, 0])  # Relative to middle
        # No constraints

        if forget_rate is None:
            # Allow Hattori2019 to not have forget_rate. In that case, it is an extension of RW1972.
            forget_rate = 0
            # 0: unrewarded, 1: rewarded
        self.forget_rates = [forget_rate, 0]  # 0: unchosen, 1: chosen
        
        # Choice kernel can be added to any reward-based forager
        if self.fit_choice_kernel:            
            # self.fit_names = (self.fit_names).extend(['choice_step_size', 'choice_softmax_temperature'])
            self.fit_names.extend(['choice_step_size', 'choice_softmax_temperature'])
            self.fit_std_values.extend([0.5, 0.3])
            self.fit_bounds[0].extend([0, 1e-2])
            self.fit_bounds[1].extend([1, 20])            
            self.fit_bandit_kwargs = {'fit_choice_kernel':True}
            self.choice_step_size = choice_step_size
            self.choice_softmax_temperature = choice_softmax_temperature

    def reset(self):
        # Initialization
        if self.seed != '':
            np.random.seed(self.seed)
        self.time = 0
        # All latent variables have n_trials + 1 length to capture the update after the last trial (HH20210726)
        self.q_estimation = np.full([self.K, self.n_trials + 1], np.nan)
        self.q_estimation[:, 0] = 0
        self.choice_prob = np.full([self.K, self.n_trials + 1], np.nan)
        self.choice_prob[:, 0] = 1 / self.K  # To be strict (actually no use)        
        # Generative mode is now default mode of agent
        self.choice_history = np.zeros([1, self.n_trials + 1], dtype=int)  # Choice history        
        # Reward history, separated for each port (Corrado Newsome 2005)
        self.reward_history = np.zeros([self.K, self.n_trials + 1])
        if self.optogenetic_perturbation:
            self.opto_reward_history = np.zeros([self.n_trials + 1])

        if self.fit_choice_kernel:
            self.choice_kernel = np.zeros([self.K, self.n_trials + 1])
    
    def fit_history(self, banditmodel, fit_choice_history, fit_reward_history, fit_names, fit_bounds, fit_bandit_kwargs, DE_pop_size = 16, pool = ''):
        fitting_result = optimize.differential_evolution(func=self.negLL_func, args=(banditmodel, fit_choice_history, fit_reward_history, fit_names, fit_bandit_kwargs),
                                                         bounds=optimize.Bounds(fit_bounds[0], fit_bounds[1]),
                                                         mutation=(0.5, 1), recombination=0.7, popsize=DE_pop_size,
                                                         strategy='best1bin',
                                                         disp=False,
                                                         workers=1 if pool == '' else int(mp.cpu_count()),
                                                         # For DE, use pool to control if_parallel, although we don't use pool for DE
                                                         updating='immediate' if pool == '' else 'deferred',
                                                         callback=None)

        fitting_result.k_model = np.sum(np.diff(np.array(fit_bounds), axis=0) > 0)  # Get the number of fitted parameters with non-zero range of bounds
        fitting_result.n_trials = np.shape(fit_choice_history)[1]
        fitting_result.log_likelihood = - fitting_result.fun

        fitting_result.AIC = -2 * fitting_result.log_likelihood + 2 * fitting_result.k_model
        fitting_result.BIC = -2 * fitting_result.log_likelihood + fitting_result.k_model * np.log(fitting_result.n_trials)

        # Likelihood-Per-Trial. See Wilson 2019 (but their formula was wrong...)
        fitting_result.LPT = np.exp(fitting_result.log_likelihood / fitting_result.n_trials)  # Raw LPT without penality
        fitting_result.LPT_AIC = np.exp(- fitting_result.AIC / 2 / fitting_result.n_trials)
        fitting_result.LPT_BIC = np.exp(- fitting_result.BIC / 2 / fitting_result.n_trials)
        return fitting_result

    @staticmethod
    def negLL_func(fit_values, *args):
        # Arguments interpretation
        banditmodel, fit_choice_history, fit_reward_history, fit_names, fit_bandit_kwargs = args        
        for name, value in zip(fit_names, fit_values):
            fit_bandit_kwargs.update({name:value})

        # Run **PREDICTIVE** simulation
        bandit = banditmodel(**fit_bandit_kwargs)
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
        choice_prob = self.choice_prob[:,:-1]  # Get all predictive choice probability [K, num_trials], exclude the final update after the last trial
        likelihood_each_trial = choice_prob[fit_choice_history[0, :], range(len(fit_choice_history[0]))]  # Get the actual likelihood for each trial

        # Deal with numerical precision
        likelihood_each_trial[(likelihood_each_trial <= 0) & (likelihood_each_trial > -1e-5)] = 1e-16  # To avoid infinity, which makes the number of zero likelihoods informative!
        likelihood_each_trial[likelihood_each_trial > 1] = 1

        # Cache likelihoods
        likelihood_all_trial.extend(likelihood_each_trial)
        likelihood_all_trial = np.array(likelihood_all_trial)
        negLL = - sum(np.log(likelihood_all_trial))

        return negLL

    def fit_each_init(banditmodel, fit_names, fit_bounds, fit_choice_history, fit_reward_history, session_num, fit_method='L-BFGS-B'):
        '''
        For local optimizers, fit using ONE certain initial condition
        '''
        x0 = []
        for lb, ub in zip(fit_bounds[0], fit_bounds[1]):
            x0.append(np.random.uniform(lb, ub))
        fitting_result = optimize.minimize(negLL_func, x0, args=(banditmodel, fit_names, fit_choice_history, fit_reward_history, {}), method=fit_method, bounds=optimize.Bounds(fit_bounds[0], fit_bounds[1]))
        return fitting_result

    def act(self, **kwargs):  # Compatible with either fitting mode (predictive) or not (generative). It's much clear now!!        
        if self.fit_choice_kernel:
            kwargs = {**kwargs}            
            kwargs['choice_kernel'] = self.choice_kernel[:,self.time]
            kwargs['choice_softmax_temperature'] = self.choice_softmax_temperature            
        choice, choice_prob = self.act_function(**kwargs)
        # self.choice_prob[:, self.time] = choice_prob
        # self.choice_history[0, self.time] = choice
        return choice, choice_prob

    def step(self, choice, reward, **kwargs):  # Compatible with either fitting mode (predictive) or not (generative). It's much clear now!!
        # Note that according to Sutton & Barto's convention,        
        # this update should belong to time t+1, but here I use t for simplicity.
        self.time += 1  # Time ticks here !!!        
        self.q_estimation[:, self.time] = self.step_function(choice, reward, **kwargs)
        if self.fit_choice_kernel and (self.time < self.n_trials):
            self.step_choice_kernel(choice)

    def step_choice_kernel(self, choice):
        # Choice vector
        choice_vector = np.zeros([self.K])
        choice_vector[choice] = 1
        # Update choice kernel (see Model 5 of Wilson and Collins, 2019)
        # Note that if chocie_step_size = 1, degenerates to Bari 2019 (choice kernel = the last choice only)
        self.choice_kernel[:, self.time] = self.choice_kernel[:, self.time - 1] + self.choice_step_size * (choice_vector - self.choice_kernel[:, self.time - 1])

    def simulate_fit(self, fit_choice_history, fit_reward_history): # This simulates the agent over a fixed choice and reward history
        self.K, self.n_trials = np.shape(fit_reward_history)  # Use the targeted histories
        self.fit_choice_history = fit_choice_history
        self.fit_reward_history = fit_reward_history
        self.reset()
        for t in range(self.n_trials):
            choice, choice_prob = self.act() # Compute choice and choice probabilities, updates choice history and choice probability
            choice = fit_choice_history[0, self.time]  # Override choice
            reward = fit_reward_history[choice, self.time] #get reward from data
            self.choice_prob[:, self.time] = choice_prob
            self.choice_history[0, self.time] = choice       
            self.reward_history[choice, self.time] = reward 
            self.step(choice, reward) # updates reward history, and update time

    def perform_task(self, task, n_trials=None):
        if n_trials is not None:
            self.n_trials = n_trials
        self.reset()
        for t in range(self.n_trials):
            choice, choice_prob = self.act()
            reward = task.step(choice)
            self.choice_prob[:, self.time] = choice_prob
            self.choice_history[0, self.time] = choice       
            self.reward_history[choice, self.time] = reward
            if self.optogenetic_perturbation:
                if (reward==0) and (np.random.rand() < self.opto_prob):
                    self.opto_reward_history[self.time] = self.opto_reward
                    self.reward_history[choice, self.time] += self.opto_reward
            self.step(choice, reward)

    def plot_session_lightweight(self, task=None, fit_choice_history=None, smooth_factor=5):
        sns.set(style="ticks", context="paper", font_scale=1.4)
        # smooth_factor, fit_choice_history, task = 5, None, None
        # choice_history, reward_history, choice_prob = agent.fit_choice_history, agent.fit_reward_history, agent.choice_prob
        choice_history = self.choice_history
        reward_history = self.reward_history
        choice_prob = self.choice_prob
        n_trials = choice_history.shape[1]        

        rewarded_trials = np.any(reward_history, axis=0)
        unrewarded_trials = np.logical_not(rewarded_trials)       
        ignored_trials = choice_history[0,:] > 1
        unignored_trials = np.logical_not(ignored_trials)       

        # == Choice trace ==
        fig = plt.figure(figsize=(25, 4), dpi=150)
        ax = fig.add_subplot(111)
        fig.subplots_adjust(left=0.1, right=0.8)

        # Rewarded trials
        ax.plot(np.nonzero(rewarded_trials)[0], 0.5 + (choice_history[0, rewarded_trials] - 0.5) * 1.4, '|', color='black', markersize=20, markeredgewidth=2)

        # Unrewarded trials
        ax.plot(np.nonzero(unignored_trials*unrewarded_trials)[0], 0.5 + (choice_history[0, unignored_trials * unrewarded_trials] - 0.5) * 1.4, '|', color='gray', markersize=10, markeredgewidth=1)

        # Ignored trials 
        ax.plot(np.nonzero(ignored_trials*unrewarded_trials)[0], -0.5 + (choice_history[0, ignored_trials * unrewarded_trials] - 0.5) * 1.4, '|', color='gray', markersize=10, markeredgewidth=1)

        # Plot ignore trials if present
        if choice_history.max() > 1:
            choice_history[0,ignored_trials] = 0.5
            choice_prob = choice_prob[:-1] / choice_prob[:-1].sum(axis=0)

        # Choice probability
        p_smooth_factor = 1
        y = moving_average(choice_prob[1,:], smooth_factor)
        x = np.arange(0, len(y)) + int(smooth_factor / 2)
        ax.plot(x, y, linewidth=1., color='blue', label='model choice probability (smooth = %g)' % p_smooth_factor)

        # Smoothed choice history
        y = moving_average(choice_history, smooth_factor)
        x = np.arange(0, len(y)) + int(smooth_factor / 2)
        ax.plot(x, y, linewidth=1., color='black', label='choice (smooth = %g)' % smooth_factor)

        # Base probability
        if task is not None:
            p_reward = task.p_reward[:,:n_trials]
            p_reward_fraction = p_reward[1, :] / (np.sum(p_reward, axis=0))
            ax.plot(np.arange(0, len(p_reward_fraction)), p_reward_fraction, color='y', label='base rew. prob.', lw=1.5)

        # For each session, if any
        if fit_choice_history is not None:
            y = moving_average(fit_choice_history, smooth_factor)
            x = np.arange(0, len(y)) + int(smooth_factor / 2)
            ax.plot(x, y, linewidth=1., color='blue', label='choice history (smooth = %g)' % smooth_factor)
            # ax.plot(np.arange(0, n_trials), fitted_data[1, :], linewidth=1., label='model')                

        ax.legend(fontsize=10, loc=1, bbox_to_anchor=(0.985, 0.89), bbox_transform=plt.gcf().transFigure)

        ax.set_yticks([0, 1])
        ax.set_yticklabels(['Left', 'Right'])

        sns.despine(trim=True)
        return ax

class forager_Hattori2019(ForagerModel):
    def __init__(self,
                 softmax_temperature=None,
                 learn_rate_rew=None,
                 learn_rate_unrew=None,                 
                 **kwargs,
                 ):
        super(forager_Hattori2019, self).__init__(**kwargs)
        # super().__init__(**kwargs)
        self.softmax_temperature = softmax_temperature
        self.learn_rate_rew = learn_rate_rew
        self.learn_rate_unrew = learn_rate_unrew
        
        self.fit_names.extend(['learn_rate_rew', 'learn_rate_unrew', 'forget_rate', 'softmax_temperature', 'biasL'])
        self.fit_std_values.extend([0.5, 0.5, 0.2, 0.1, 0.3]) # typical parameters     
        self.fit_bounds[0].extend([0, 0, 0, 1e-2, -5])
        self.fit_bounds[1].extend([1, 1, 1, 15, 5])
        
        self.model_name = 'Hattori2019'
        self.banditmodel = forager_Hattori2019
        self.step_function = step_RWlike
        self.act_function = act_Probabilistic

    def step(self, choice, reward):
        step_kwargs = {'q_estimation_tminus1':self.q_estimation[:,self.time],
                       'learn_rates':[self.learn_rate_rew, self.learn_rate_unrew],
                       'forget_rates':self.forget_rates}
        return super().step(choice, reward, **step_kwargs)

    def act(self):  # Compatible with either fitting mode (predictive) or not (generative). It's much clear now!!
        act_kwargs = {'q_estimation_t': self.q_estimation[:, self.time],
                      'softmax_temperature': self.softmax_temperature,
                      'bias_terms': self.bias_terms}
        return super().act(**act_kwargs)

    def fit_history(self, fit_choice_history, fit_reward_history, pool=''):        
        fitting_result = super().fit_history(self.banditmodel, fit_choice_history, fit_reward_history, self.fit_names, self.fit_bounds, self.fit_bandit_kwargs, pool=pool)
        return fitting_result


class forager_Hattori2019_ignore(ForagerModel):
    def __init__(self,
                 softmax_temperature=None,
                 learn_rate_rew=None,
                 learn_rate_unrew=None,
                 ignore_rate_satiety=None,
                 ignore_rate_Qs=None,
                 ignore_softmax_temperature=None,
                 **kwargs,
                 ):
        # super(bandit_Hattori2019, self).__init__(*args)
        {**kwargs}.update({'K_arm':3})
        super().__init__(**kwargs)
        self.softmax_temperature = softmax_temperature
        self.learn_rate_rew = learn_rate_rew
        self.learn_rate_unrew = learn_rate_unrew
        self.ignore_rate_satiety = ignore_rate_satiety
        self.ignore_rate_Qs = ignore_rate_Qs
        self.ignore_softmax_temperature = ignore_softmax_temperature
        self.fit_names.extend(['learn_rate_rew', 'learn_rate_unrew', 'forget_rate', 'softmax_temperature','ignore_rate_satiety','ignore_rate_Qs', 'ignore_softmax_temperature', 'biasL', 'biasR'])
        self.fit_std_values.extend([0.5, 0.1, 0.2, 0.3, 0.01, 0.4, 12., 0.3, 0.3]) 
        self.fit_bounds[0].extend([0, 0, 0, 1e-2, 0, 0, 1e-2, -5, -5])
        self.fit_bounds[1].extend([1, 1, 1, 15, 1, 1, 15, 5, 5])
        self.model_name = 'Hattori2019_ignore'
        self.banditmodel = forager_Hattori2019_ignore
        self.step_function = step_RWlike_ignore
        self.act_function = act_Probabilistic_ignore
        self.K = 3

    def step(self, choice, reward):
        step_kwargs = {'valid_reward_history':self.reward_history[:,:self.time],
                       'q_estimation_tminus1':self.q_estimation[:,self.time],
                       'learn_rates':[self.learn_rate_rew, self.learn_rate_unrew],
                       'forget_rates':self.forget_rates,
                       'ignore_rates':[self.ignore_rate_satiety, self.ignore_rate_Qs]}
        return super().step(choice, reward, **step_kwargs)

    def act(self):
        act_kwargs = {'q_estimation_t': self.q_estimation[:, self.time],
                      'softmax_temperature': self.softmax_temperature,
                      'ignore_softmax_temperature': self.ignore_softmax_temperature,
                      'bias_terms': self.bias_terms}
        return super().act(**act_kwargs)

    def fit_history(self, fit_choice_history, fit_reward_history, pool=''):        
        fitting_result = super().fit_history(self.banditmodel, fit_choice_history, fit_reward_history, self.fit_names, self.fit_bounds, self.fit_bandit_kwargs,
                    pool=pool)
        return fitting_result


class forager_Bari2019(ForagerModel):
    def __init__(self,
                 softmax_temperature=None,
                 learn_rate=None,
                 forget_rate=None,
                 **kwargs,
                 ):
        super().__init__(**kwargs)
        self.softmax_temperature = softmax_temperature
        self.learn_rate = learn_rate
        self.forget_rate = forget_rate
        self.fit_names.extend(['learn_rate', 'forget_rate', 'softmax_temperature'])
        self.fit_std_values.extend([0.5, 0.1, 0.3]) #forget rate 0.5        
        self.fit_bounds[0].extend([0, 0, 1e-2])
        self.fit_bounds[1].extend([1, 1, 15])
        self.model_name = 'Bari2019'
        self.banditmodel = forager_Bari2019
        self.step_function = step_RWlike
        self.act_function = act_Probabilistic

    def step(self, choice, reward):
        step_kwargs = {'q_estimation_tminus1': self.q_estimation[:, self.time],
            'learn_rates': [self.learn_rate, self.learn_rate],
            'forget_rates': [self.forget_rate, self.forget_rate]}
        return super().step(choice, reward, **step_kwargs)

    def act(self):  # Compatible with either fitting mode (predictive) or not (generative). It's much clear now!!
        act_kwargs = {'q_estimation_t': self.q_estimation[:, self.time],
            'softmax_temperature': self.softmax_temperature,
            'bias_terms': self.bias_terms}
        return super().act(**act_kwargs)

    def fit_history(self, fit_choice_history, fit_reward_history, pool=''):
        fitting_result = super().fit_history(self.banditmodel, fit_choice_history, fit_reward_history, self.fit_names, self.fit_bounds, self.fit_bandit_kwargs, pool=pool)
        return fitting_result

## ---- ANCCR model 


class forager_ANCCR2(ForagerModel):
    def __init__(self,
                 learn_rate=None,                 
                 softmax_temperature=None,                 
                 lam=None,
                 w=None,
                 **kwargs,
                 ):
        super().__init__(**kwargs)
        self.softmax_temperature = softmax_temperature
        self.learn_rate = learn_rate        
        self.lam = lam
        self.w = w

        self.n = 1                
        self.lam = lam        
        self.base_rate = 1.0
        # self.et = np.full([self.n, self.n_trials + 1], np.nan)#np.zeros((self.n, self.n_trials))
        # self.m = np.full([self.n, self.n_trials + 1, self.n], np.nan)#np.zeros((self.n, self.n_trials, self.n))
        # self.mbs = np.full([self.n,], np.nan)#np.zeros((self.n,))
        # self.mbsa = np.full([self.n, self.n_trials + 1], np.nan)#np.zeros((self.n, self.n_trials))                                            

        N_actions = 2
        self.et, self.m, self.mbs, self.mbsa = (
            np.zeros((self.n, self.K)),
            np.zeros((self.n, self.K, self.n)),
            np.zeros((self.n,)),
            np.zeros((self.n, self.K))
        )
        
        self.fit_names.extend(['learn_rate', 'softmax_temperature', 'lam', 'w'])
        self.fit_std_values.extend([0.1, 1.0, 0.5, 0.5]) #forget rate 0.5        
        self.fit_bounds[0].extend([0, 1e-2, 0, 0])
        self.fit_bounds[1].extend([1, 15, 1, 1])

        self.model_name = 'ANCCR2'
        self.banditmodel = forager_ANCCR2
        self.step_function = step_ANCCR2
        self.act_function = act_Probabilistic

    def step(self, choice, reward):
        self.et *= self.lam
        self.et[0, choice] += 1.
        if np.random.rand() < self.base_rate:        
            self.mbsa += self.learn_rate * (self.et - self.mbsa)
            self.mbs += self.learn_rate * (self.et.sum(-1) - self.mbs)
        if reward > 0:            
            self.m[:,:, 0] += self.learn_rate * (self.et - self.m[:,:, 0])        

        cp = self.m - self.mbsa[:,:,np.newaxis] 
        cs = cp * self.mbs[np.newaxis, np.newaxis,:] / self.mbsa[:,:,np.newaxis].clip(min=10)
        contingency = cp * self.w + cs * (1 - self.w)
        return contingency.flatten()
        # step_kwargs = {'q_estimation_tminus1': self.q_estimation[:, self.time],            
            # 'm':self.m, 
            # 'mbs':self.mbs, 
            # 'mbsa':self.mbsa, 
            # 'w':self.w}            
        # return super().step(choice, reward, **step_kwargs)

    def act(self):  # Compatible with either fitting mode (predictive) or not (generative). It's much clear now!!
        act_kwargs = {'q_estimation_t': self.q_estimation[:, self.time],
            'softmax_temperature': self.softmax_temperature,
            'bias_terms': self.bias_terms}
        return super().act(**act_kwargs)

    def fit_history(self, fit_choice_history, fit_reward_history, pool=''):
        fitting_result = super().fit_history(self.banditmodel, fit_choice_history, fit_reward_history, self.fit_names, self.fit_bounds, self.fit_bandit_kwargs, pool=pool)
        return fitting_result

## ---- ANCCR model 


class forager_ANCCR(ForagerModel):
    def __init__(self,                 
                 softmax_temperature=None,                 
                 learn_rate=None,
                 lam=None,                 
                 w=None,                
                 **kwargs,
                 ):
        super(forager_ANCCR, self).__init__(**kwargs)

        self.softmax_temperature = softmax_temperature
        self.n = 1        
        self.lam = lam
        self.w = w
        base_rate=1.0,
        forget_rate=0.1,                  
        self.base_rate = base_rate
        self.learn_rate = learn_rate
        self.forget_rate = forget_rate
        self.et = np.zeros((self.n, self.n_trials)),
        self.m = np.zeros((self.n, self.n_trials, self.n))
        self.mbs = np.zeros((self.n,))
        self.mbsa = np.zeros((self.n, self.n_trials))            
        # self.states = defaultdict(Counter())
        self.updated = True
        self.contingency = None

        self.fit_names.extend(['learn_rate', 'softmax_temperature', 'lam', 'w'])
        self.fit_std_values.extend([0.1, 1.0, 0.5, 0.5]) #forget rate 0.5        
        self.fit_bounds[0].extend([0, 1e-2, 0, 0])
        self.fit_bounds[1].extend([1, 15, 1, 1])

        self.model_name = 'ANCCR'
        self.banditmodel = forager_ANCCR
        self.act_function = act_Probabilistic
        self.step_function = step_ANCCR        
    
    # def step_function(self, a, r):
    #     self.et *= self.lam
    #     self.et[0, a] += 1.
    #     if np.random.rand() < self.base_rate:
    #         self.updated = True
    #         self.mbsa += self.learn_rate * (self.et - self.mbsa)
    #         self.mbs += self.learn_rate * (self.et.sum(-1) - self.mbs)
    #         if r > 0:
    #             self.updated = True
    #             self.m[..., 0] += self.learn_rate * (self.et - self.m[..., 0])
    #     return self.contingency

    # def step(self, choice, reward):
        # return super().step(choice, reward)

    def step(self, choice, reward):
        step_kwargs = {'q_estimation_t': self.q_estimation[:, self.time], 
            'learn_rate':self.learn_rate, 
            'forget_rate':self.forget_rate}#,
            # 'et':self.et, 
            # 'lam':self.lam, 
            # 'base_rate':self.base_rate,              
            # 'm':self.m, 
            # 'mbs':self.mbs, 
            # 'mbsa':self.mbsa, 
            # 'w':self.w}
        # step_kwargs = {'q_estimation_tminus1': self.q_estimation[:, self.time],
            # 'learn_rates': [self.learn_rate, self.learn_rate]}#,
            # 'forget_rates': [self.forget_rate, self.forget_rate]}
        return super().step(choice, reward, **step_kwargs)
    
    def act(self):
        act_kwargs = {'q_estimation_t': self.q_estimation[:, self.time],
            'softmax_temperature': self.softmax_temperature,
            'bias_terms': self.bias_terms}
        return super().act(**act_kwargs)

    def fit_history(self, fit_choice_history, fit_reward_history, pool=''):
        fitting_result = super().fit_history(self.banditmodel, fit_choice_history, fit_reward_history, self.fit_names, self.fit_bounds, self.fit_bandit_kwargs, pool=pool)
        return fitting_result

    # @property
    # def contingency(self):
    #     if self.updated:
    #         self.updated = False
    #         cp = self.m - self.mbsa[..., None] 
    #         cs = cp * self.mbs[None, None, :] / self.mbsa[..., None].clip(min=10)
    #         self.contingency_ = cp * self.w + cs * (1 - self.w)
    #     return self.contingency_

    # def __call__(self, s, a):
    #     s, a = self.states[s], self.actions[a]
    #     if s >= self.et.shape[0]:
    #         self.expand_memory()
    #     return self.contingency[s,a].sum()

#     def expand_memory(self):
#         n, a = self.et.shape
#         et, m, mbs, mbsa, c_ = self.et, self.m, self.mbs, self.mbsa, self.contingency_
#         self.et, self.m, self.mbs, self.mbsa, self.contingency_ = (
#             np.zeros((2*n, a)),
#             np.zeros((2*n, a, 2*n)),
#             np.zeros((2*n,)),
#             np.zeros((2*n, a)),
#             np.zeros((2 * n, a, 2 * n))
#         )
#         self.et[:n], self.m[:n,:,:n], self.mbs[:n], self.mbsa[:n], self.contingency_[:n,:,:n] = et, m, mbs, mbsa, c_

# class Counter:
#     def __init__(self):
#         self.i = -1

#     def __call__(self):
#         self.i += 1
#         return self.i

