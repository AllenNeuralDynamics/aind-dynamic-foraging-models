import numpy as np

from .util import moving_average, softmax, choose_ps

LEFT = 0
RIGHT = 1

class CoupledBlocks:
    def __init__(self, K_arm=2, n_trials=1000, if_baited=True, seed='',  p_reward_sum=0.45, p_reward_pairs=None):
        self.K = K_arm
        self.if_baited = if_baited
        self.seed = seed
        self.p_reward_sum = p_reward_sum
        self.p_reward_pairs = p_reward_pairs
        self.n_trials = n_trials
        self.left = 0
        self.right = 1
        self.K_max = K_arm # This allows to model ignore choices, or choices outside of the task

    def generate_p_reward(self, block_size_base = 80, block_size_sd = 20, p_reward_pairs=[[.4, .05], [.3857, .0643], [.3375, .1125], [.225, .225]]):
        # If para_optim, fix the random seed to ensure that p_reward schedule is fixed for all candidate parameters
        # However, we should make it random during a session (see the last line of this function)
        if self.seed != '':
            np.random.seed(self.seed)
        if self.p_reward_pairs == None:
            p_reward_pairs = np.array(
                p_reward_pairs) / 0.45 * self.p_reward_sum
        else:  # Full override of p_reward
            p_reward_pairs = self.p_reward_pairs
        # Adapted from Marton's code
        n_trials_now = 0
        block_size = []
        n_trials = self.n_trials + 1
        p_reward = np.zeros([2, n_trials])

        # Fill in trials until the required length
        while n_trials_now < n_trials:
            # Number of trials in each block (Gaussian distribution)
            # I treat p_reward[0,1] as the ENTIRE lists of reward probability. RIGHT = 0, LEFT = 1. HH
            n_trials_this_block = np.rint(np.random.normal(block_size_base, block_size_sd)).astype(int)
            n_trials_this_block = min(n_trials_this_block, n_trials - n_trials_now)
            block_size.append(n_trials_this_block)

            # Get values to fill for this block
            # If 0, the first block is set to 50% reward rate (as Marton did)
            if n_trials_now == -1:
                p_reward_this_block = np.array(
                    [[sum(p_reward_pairs[0]) / 2] * 2])  # Note the outer brackets
            else:
                # Choose reward_ratio_pair
                # If we had equal p_reward in the last block
                if n_trials_now > 0 and not (np.diff(p_reward_this_block)):
                    # We should not let it happen again immediately
                    pair_idx = np.random.choice(range(len(p_reward_pairs) - 1))
                else:
                    pair_idx = np.random.choice(range(len(p_reward_pairs)))

                p_reward_this_block = np.array(
                    [p_reward_pairs[pair_idx]])  # Note the outer brackets

                # To ensure flipping of p_reward during transition (Marton)
                if len(block_size) % 2:
                    p_reward_this_block = np.flip(p_reward_this_block)

            # Fill in trials for this block
            p_reward[:, n_trials_now: n_trials_now + n_trials_this_block] = p_reward_this_block.T

            # Next block
            n_trials_now += n_trials_this_block

        self.n_blocks = len(block_size)
        self.p_reward = p_reward
        self.block_size = np.array(block_size)
        self.p_reward_fraction = p_reward[RIGHT, :] / (np.sum(p_reward, axis=0))  # For future use
        self.p_reward_ratio = p_reward[RIGHT, :] / p_reward[LEFT, :]  # For future use

    def reset(self):
        self.time = 0
        self.choice_history = np.zeros([1, self.n_trials + 1], dtype=int)  # Choice history
        # Reward history, separated for each port (Corrado Newsome 2005)
        self.reward_history = np.zeros([self.K_max, self.n_trials + 1])

        # Generate baiting prob in block structure
        self.generate_p_reward()

        # Prepare reward for the first trial
        # For example, [0,1] represents there is reward baited at the RIGHT but not LEFT port.
        # Reward history, separated for each port (Corrado Newsome 2005)
        self.reward_available = np.zeros([self.K_max, self.n_trials + 1])
        self.reward_available[:self.K, 0] = (np.random.uniform(0, 1, self.K) < self.p_reward[:, self.time]).astype(int)

    def step(self, choice):         
        #  In generative mode, generate reward and make the state transition
        reward = int(self.reward_available[choice, self.time])
        # Note that according to Sutton & Barto's convention,
        self.reward_history[choice, self.time] = reward
        # this update should belong to time t+1, but here I use t for simplicity.

        # An intermediate reward status. Note the .copy()!
        reward_available_after_choice = self.reward_available[:, self.time].copy()
        # The reward is depleted at the chosen lick port.
        reward_available_after_choice[choice] = 0
        # =================================================
        self.time += 1  # Time ticks here !!!
        # Doesn't terminate here to finish the final update after the last trial
        # if self.time == self.n_trials:
        #     return   # Session terminates
        # =================================================
        # Prepare reward for the next trial (if sesson did not end)
        # Generate the next reward status, the "or" statement ensures the baiting property, gated by self.if_baited.
        if self.time < self.n_trials:
            self.reward_available[:self.K, self.time] = np.logical_or(reward_available_after_choice[:self.K] * self.if_baited, np.random.uniform(0, 1, self.K) < self.p_reward[:, self.time]).astype(int)
        return reward
