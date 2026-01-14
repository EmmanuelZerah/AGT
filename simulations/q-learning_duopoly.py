import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from tqdm import tqdm

# --- 1. CONFIGURATION & PARAMETERS ---
# Parameters taken from Section 5 "Anatomy of Collusion"
ALPHA = 0.2  # Learning rate (speed of learning)
BETA = 15e-6  # Experimentation decay (exploration parameter)
DELTA = 0.95  # Discount factor
MU = 0.25  # Horizontal differentiation (substitutability)
A_val = 2.0  # Product quality (a_i - c_i = 1, so if c=1, a=2)
A_0 = 0.0  # Outside good quality
C_val = 1.0  # Marginal cost
M = 15  # Number of discrete prices in the grid
XI = 0.1  # Grid expansion parameter (how far outside Nash/Monopoly to go)

# Simulation Duration
# The paper typically sees convergence around 850,000 steps.
NUM_EPISODES = 3_000_000


# --- 2. ECONOMIC ENVIRONMENT ---

def get_demand(p1, p2):
    """Logit demand function: Calculates market share for both firms."""
    # Numerators
    exp1 = np.exp((A_val - p1) / MU)
    exp2 = np.exp((A_val - p2) / MU)
    # Denominator (including outside good)
    denom = exp1 + exp2 + np.exp(A_0 / MU)
    return exp1 / denom, exp2 / denom


def get_profit(p1, p2):
    """Calculates profit for both firms given prices p1 and p2."""
    q1, q2 = get_demand(p1, p2)
    pi1 = (p1 - C_val) * q1
    pi2 = (p2 - C_val) * q2
    return pi1, pi2


# --- 3. CALCULATE BENCHMARKS (NASH & MONOPOLY) ---

def solve_nash():
    """
    Finds the Static Bertrand-Nash Equilibrium.
    Iteratively optimizes one firm's price while holding the other fixed
    until the prices stop changing (Fixed Point).
    """

    def neg_profit_i(p_i, p_j_fixed):
        # We minimize negative profit (maximize profit)
        return -get_profit(p_i, p_j_fixed)[0]  # [0] returns profit for firm i

    # Start with a guess
    p_curr = 1.5

    # Iterate to find the fixed point (Best Response)
    for _ in range(50):
        # Find best price against current opponent price
        res = minimize(neg_profit_i, p_curr, args=(p_curr,), bounds=[(1.0, 5.0)])
        p_new = res.x[0]

        # Check for convergence
        if np.abs(p_new - p_curr) < 1e-6:
            return p_new
        p_curr = p_new
    return p_curr


def solve_monopoly():
    """
    Finds the Joint Profit Maximizing (Monopoly) Price.
    """

    def neg_joint_profit(p):
        # Assume symmetric prices p1 = p2 = p[0]
        pi1, pi2 = get_profit(p[0], p[0])
        return -(pi1 + pi2)

    res = minimize(neg_joint_profit, [2.0], bounds=[(1.0, 5.0)])
    return res.x[0]


# Calculate benchmarks
p_nash = solve_nash()
p_monop = solve_monopoly()
pi_nash = get_profit(p_nash, p_nash)[0]
pi_monop = get_profit(p_monop, p_monop)[0]

print(f"Reference: Nash Price = {p_nash:.4f}, Monopoly Price = {p_monop:.4f}")

# --- 4. ACTION SPACE (PRICE GRID) ---
# The agents can only choose from M discrete prices.
interval = p_monop - p_nash
p_min = p_nash - XI * interval
p_max = p_monop + XI * interval
action_space = np.linspace(p_min, p_max, M)

print(f"Action Space (Allowed Prices): \n{np.round(action_space, 3)}")


# --- 5. Q-LEARNING AGENT CLASS ---

class Agent:
    def __init__(self, n_actions, n_states, initial_q_value):
        # Initialize Q-Table with the 'initial_q_value'
        self.q_table = np.full((n_states, n_actions), initial_q_value)
        self.n_actions = n_actions

        # Memory to store the previous state/action for updating
        self.last_state_idx = None
        self.last_action_idx = None

    def choose_action(self, state_idx, epsilon):
        """Epsilon-Greedy Action Selection"""
        if np.random.rand() < epsilon:
            # Exploration: Random action
            return np.random.randint(self.n_actions)
        else:
            # Exploitation: Best known action
            # Note: We find all actions with the max Q-value and pick randomly among them to break ties
            max_q = np.max(self.q_table[state_idx])
            best_actions = np.where(self.q_table[state_idx] == max_q)[0]
            return np.random.choice(best_actions)

    def update(self, reward, new_state_idx):
        """Q-Learning Update Rule (Equation 4 in the paper)"""
        # Q(s,a) = (1-alpha)*Q(s,a) + alpha*[reward + delta * max(Q(s', a'))]

        # Current estimate
        old_q = self.q_table[self.last_state_idx, self.last_action_idx]

        # Estimate of future value (max Q of new state)
        future_val = np.max(self.q_table[new_state_idx])

        # Update
        new_q = (1 - ALPHA) * old_q + ALPHA * (reward + DELTA * future_val)

        # Write back to table
        self.q_table[self.last_state_idx, self.last_action_idx] = new_q


for _ in range(3):
    # --- 6. INITIALIZATION ---
    # Calculate the initial Q-value (Equation 8)
    # This represents the expected profit if the opponent plays randomly.
    avg_profits = []
    for p_i in action_space:
        p_i_profits = []
        for p_j in action_space:
            pi, _ = get_profit(p_i, p_j)
            p_i_profits.append(pi)
        avg_profits.append(np.mean(p_i_profits))
    initial_q = np.mean(avg_profits) / (1 - DELTA)

    print(f"Initial Q-Table Value: {initial_q:.4f}")

    # --- 7. MAIN SIMULATION LOOP ---

    # State Space: The state is defined by the previous prices of both agents.
    # We map (p1_idx, p2_idx) to a single integer state_idx.
    n_states = M * M

    # Create Agents
    agent1 = Agent(M, n_states, initial_q)
    agent2 = Agent(M, n_states, initial_q)

    # Random start
    p1_idx = np.random.randint(M)
    p2_idx = np.random.randint(M)
    state_idx = p1_idx * M + p2_idx

    history_delta = []

    # Variables to store previous reward for updating
    pi1_prev = 0
    pi2_prev = 0

    print(f"\nRunning {NUM_EPISODES} episodes...")

    for t in tqdm(range(NUM_EPISODES), desc="Running Simulation"):
        # Calculate current Exploration Rate (epsilon)
        epsilon = np.exp(-BETA * t)

        # 1. Agents choose actions based on CURRENT state
        a1_idx = agent1.choose_action(state_idx, epsilon)
        a2_idx = agent2.choose_action(state_idx, epsilon)

        # 2. Market determines profits
        p1 = action_space[a1_idx]
        p2 = action_space[a2_idx]
        pi1, pi2 = get_profit(p1, p2)

        # 3. Determine NEW state
        new_state_idx = a1_idx * M + a2_idx

        # 4. Learning Step (Update Q-values based on PREVIOUS actions and CURRENT state)
        if t > 0:
            agent1.update(pi1_prev, new_state_idx)
            agent2.update(pi2_prev, new_state_idx)

        # 5. Store current info for next step
        agent1.last_state_idx = state_idx
        agent1.last_action_idx = a1_idx
        agent2.last_state_idx = state_idx
        agent2.last_action_idx = a2_idx

        pi1_prev = pi1
        pi2_prev = pi2
        state_idx = new_state_idx

        # 6. Record Statistics (Delta) every 1000 steps
        if t % 1000 == 0:
            avg_pi = (pi1 + pi2) / 2
            # Delta = (Actual Profit - Nash Profit) / (Monopoly Profit - Nash Profit)
            delta = (avg_pi - pi_nash) / (pi_monop - pi_nash)
            history_delta.append(delta)

    # --- 8. VISUALIZATION ---

    print("Simulation finished.")
    window = 100  # Moving average window
    smoothed_delta = np.convolve(history_delta, np.ones(window) / window, mode='valid')

    plt.figure(figsize=(12, 6))
    plt.plot(smoothed_delta, linewidth=1.5)
    plt.title(f"Q-Learning: Average Profit Gain ($\Delta$) over Time\nAlpha={ALPHA}, Beta={BETA}")
    plt.xlabel("Time (x1000 periods)")
    plt.ylabel("Degree of Collusion ($\Delta$)")
    plt.axhline(0, color='red', linestyle='--', label="Nash Equilibrium")
    plt.axhline(1, color='green', linestyle='--', label="Monopoly")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.show()