import numpy as np
import matplotlib.pyplot as plt

# ==========================================
# Part 1: Simulation (The "Subject")
# ==========================================

def softmax(q, beta):
    """Compute softmax probabilities for actions."""
    # Subtract max to prevent overflow
    exp_q = np.exp(beta * (q - np.max(q)))
    return exp_q / np.sum(exp_q)

def simulate_behavior(n_trials=300):
    """
    Simulates a Q-learning agent with dynamic parameters performing
    a two-armed bandit task.
    """
    # 1. Define True Parameters (Dynamic)
    # Learning rate (alpha) changes sinusoidally
    true_alpha = 0.5 + 0.3 * np.sin(np.linspace(0, 4*np.pi, n_trials))
    # Inverse temp (beta) ramps up (agent gets more decisive)
    true_beta = np.linspace(1.5, 4.0, n_trials)
    
    # Task Reward Probabilities (Drifting)
    p_reward = np.zeros((n_trials, 2))
    p_reward[:, 0] = 0.2 + 0.6 * (np.sin(np.linspace(0, 2*np.pi, n_trials)) > 0) # Square waveish
    p_reward[:, 1] = 1 - p_reward[:, 0]

    # Simulation containers
    actions = np.zeros(n_trials, dtype=int)
    rewards = np.zeros(n_trials)
    q_values = np.zeros((n_trials, 2))
    
    # Initialize Q-values (e.g., to 0.5)
    q = np.array([0.5, 0.5])

    for t in range(n_trials):
        # Store current Q
        q_values[t] = q.copy()
        
        # Select Action using Softmax
        probs = softmax(q, true_beta[t])
        action = np.random.choice([0, 1], p=probs)
        actions[t] = action
        
        # Observe Reward (Bernoulli trial)
        reward = 1 if np.random.rand() < p_reward[t, action] else 0
        rewards[t] = reward
        
        # Update Q-values (Standard Q-Learning)
        # Q(t+1) = Q(t) + alpha * (R - Q(t))
        q[action] = q[action] + true_alpha[t] * (reward - q[action])

    return actions, rewards, true_alpha, true_beta, q_values

# ==========================================
# Part 2: Particle Filter Algorithm
# ==========================================

def particle_filter_estimation(actions, rewards, n_particles=1000, 
                               sigma_alpha=0.05, sigma_beta=0.1):
    """
    Estimates alpha, beta, and Q-values using a Particle Filter.
    
    Args:
        actions (array): Observed actions.
        rewards (array): Observed rewards.
        n_particles (int): Number of particles to use.
        sigma_alpha (float): Random walk std dev for alpha evolution.
        sigma_beta (float): Random walk std dev for beta evolution.
    """
    n_trials = len(actions)
    
    # --- Initialize Particles ---
    # Particles for parameters
    p_alpha = np.random.uniform(0, 1, n_particles)
    p_beta = np.random.uniform(0, 5, n_particles)
    
    # Particles for Q-values (2 arms)
    p_q = 0.5 * np.ones((n_particles, 2)) 
    
    # Particle weights (uniform initially)
    weights = np.ones(n_particles) / n_particles

    # Storage for estimates (Weighted Means)
    est_alpha = np.zeros(n_trials)
    est_beta = np.zeros(n_trials)
    est_q = np.zeros((n_trials, 2))

    for t in range(n_trials):
        # 1. Prediction / Evolution Step
        # -------------------------------
        # Add noise to parameters (Random Walk) to allow them to change
        noise_alpha = np.random.normal(0, sigma_alpha, n_particles)
        noise_beta = np.random.normal(0, sigma_beta, n_particles)
        
        p_alpha = np.clip(p_alpha + noise_alpha, 0.01, 0.99) # Bound alpha
        p_beta = np.clip(p_beta + noise_beta, 0.1, 10.0)     # Bound beta > 0

        # 2. Observation / Weight Update Step
        # -------------------------------
        # Calculate likelihood of the *observed* action given each particle's Q & Beta
        # P(a_t | Q_t, beta_t)
        # We use the Log-Sum-Exp trick for numerical stability in softmax
        q_obs = p_q[:, actions[t]]
        q_max = np.max(p_q, axis=1)
        
        # log_prob = beta * q_obs - (beta * q_max + log(sum(exp(beta * (q_all - q_max)))))
        # Calculating raw probabilities for clarity:
        numerators = np.exp(p_beta * (q_obs - q_max))
        denominators = np.sum(np.exp(p_beta[:, None] * (p_q - q_max[:, None])), axis=1)
        likelihoods = numerators / denominators
        
        # Update weights
        weights *= likelihoods
        weights += 1.e-300 # Avoid zero division
        weights /= np.sum(weights) # Normalize

        # 3. Estimation (before resampling)
        # -------------------------------
        est_alpha[t] = np.sum(weights * p_alpha)
        est_beta[t] = np.sum(weights * p_beta)
        est_q[t] = np.sum(weights[:, None] * p_q, axis=0)

        # 4. Resampling Step (Systematic Resampling)
        # -------------------------------
        # Only resample if effective sample size is low (optional, but good practice)
        # Here we resample every step for simplicity as per standard PF text.
        cumulative_sum = np.cumsum(weights)
        cumulative_sum[-1] = 1.0 # Ensure last is 1.0
        step = 1.0 / n_particles
        u = np.random.uniform(0, step)
        indices = []
        
        # Fast systematic resampling
        idx = 0
        for i in range(n_particles):
            pointer = u + i * step
            while pointer > cumulative_sum[idx]:
                idx += 1
            indices.append(idx)
            
        # Resample all states
        p_alpha = p_alpha[indices]
        p_beta = p_beta[indices]
        p_q = p_q[indices]
        weights = np.ones(n_particles) / n_particles # Reset weights

        # 5. Deterministic Update of Q-values (for t+1)
        # -------------------------------
        # Apply Q-learning rule for each particle using its specific alpha
        # Q(t+1) = Q(t) + alpha * (R - Q(t))
        # Note: We use the *observed* reward
        act = actions[t]
        rew = rewards[t]
        p_q[:, act] = p_q[:, act] + p_alpha * (rew - p_q[:, act])

    return est_alpha, est_beta, est_q

# ==========================================
# Part 3: Run and Visualize
# ==========================================

# 1. Generate Data
print("Generating synthetic data...")
actions, rewards, true_alpha, true_beta, true_q = simulate_behavior(n_trials=500)

# 2. Run Particle Filter
print("Running Particle Filter...")
est_alpha, est_beta, est_q = particle_filter_estimation(actions, rewards)

# 3. Plotting
fig, axes = plt.subplots(3, 1, figsize=(10, 12), sharex=True)

# Plot Learning Rate (Alpha)
axes[0].plot(true_alpha, 'k--', label='True Alpha', linewidth=2)
axes[0].plot(est_alpha, 'r-', label='Estimated Alpha', alpha=0.8)
axes[0].set_ylabel(r'Learning Rate ($\alpha$)')
axes[0].set_title('Estimation of Internal Parameters')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Plot Inverse Temperature (Beta)
axes[1].plot(true_beta, 'k--', label='True Beta', linewidth=2)
axes[1].plot(est_beta, 'g-', label='Estimated Beta', alpha=0.8)
axes[1].set_ylabel(r'Inverse Temp ($\beta$)')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

# Plot Q-values (Action Values)
axes[2].plot(true_q[:, 0], 'k--', label='True Q (Option A)', alpha=0.5)
axes[2].plot(est_q[:, 0], 'b-', label='Est Q (Option A)', alpha=0.8)
# axes[2].plot(true_q[:, 1], 'grey', linestyle='--', label='True Q (Option B)', alpha=0.5) # Optional
axes[2].set_ylabel('Q-Values')
axes[2].set_xlabel('Trial')
axes[2].legend()
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()