import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from tqdm import tqdm

# --- 1. CONFIGURATION & PARAMETERS ---
# [cite_start]Parameters taken from Section 5 "Anatomy of Collusion" [cite: 2001, 2133]
# Note: Beta is critical here. If Beta is too high (decay too fast), SARSA acts like Q-learning.
# If Beta is low (exploration stays active), SARSA's conservatism kicks in.
ALPHA = 0.15  # Learning rate
BETA = 4e-6  # Experimentation decay (Slower decay to highlight SARSA's risk aversion)
DELTA = 0.95  # Discount factor
MU = 0.25  # Horizontal differentiation
A_val = 2.0  # Product quality
A_0 = 0.0  # Outside good quality
C_val = 1.0  # Marginal cost
M = 15  # Number of discrete prices
XI = 0.1  # Grid expansion parameter

# Simulation Duration
NUM_EPISODES = 1_500_000  # Extended to ensure convergence


# --- 2. ECONOMIC ENVIRONMENT ---

def get_demand(p1, p2):
    """Logit demand function: Calculates market share [cite: 1957]"""
    exp1 = np.exp((A_val - p1) / MU)
    exp2 = np.exp((A_val - p2) / MU)
    denom = exp1 + exp2 + np.exp(A_0 / MU)
    return exp1 / denom, exp2 / denom


def get_profit(p1, p2):
    """Calculates profit for both firms [cite: 1962]"""
    q1, q2 = get_demand(p1, p2)
    pi1 = (p1 - C_val) * q1
    pi2 = (p2 - C_val) * q2
    return pi1, pi2


# --- 3. CALCULATE BENCHMARKS ---

def solve_nash():
    """Finds Static Bertrand-Nash Equilibrium [cite: 1966]"""
    def neg_profit_i(p_i, p_j_fixed):
        return -get_profit(p_i, p_j_fixed)[0]

    p_curr = 1.5
    for _ in range(50):
        res = minimize(neg_profit_i, p_curr, args=(p_curr,), bounds=[(1.0, 5.0)])
        p_new = res.x[0]
        if np.abs(p_new - p_curr) < 1e-6:
            return p_new
        p_curr = p_new
    return p_curr


def solve_monopoly():
    """Finds Joint Profit Maximizing Price [cite: 1966]"""

    def neg_joint_profit(p):
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

# --- 4. ACTION SPACE ---
interval = p_monop - p_nash
p_min = p_nash - XI * interval
p_max = p_monop + XI * interval
action_space = np.linspace(p_min, p_max, M)
print(f"Action Space: \n{np.round(action_space, 3)}")


# --- 5. SARSA AGENT CLASS ---

class SarsaAgent:
    def __init__(self, n_actions, n_states, initial_q_value):
        # [cite_start]Q-Table initialization [cite: 1994]
        self.q_table = np.full((n_states, n_actions), initial_q_value)
        self.n_actions = n_actions

        # Memory for SARSA update
        self.last_state_idx = None
        self.last_action_idx = None

    def choose_action(self, state_idx, epsilon):
        """Epsilon-Greedy Action Selection [cite: 1897]"""
        if np.random.rand() < epsilon:
            return np.random.randint(self.n_actions)
        else:
            # Exploitation: Tie-breaking argmax
            max_q = np.max(self.q_table[state_idx])
            best_actions = np.where(self.q_table[state_idx] == max_q)[0]
            return np.random.choice(best_actions)

    def update(self, reward, new_state_idx, next_action_idx):
        """
        SARSA Update Rule.
        Difference from Q-Learning: Uses Q(s', a') instead of max Q(s', a')
        """
        # 1. Retrieve Q(s, a) (Old estimate)
        old_q = self.q_table[self.last_state_idx, self.last_action_idx]

        # 2. Retrieve Q(s', a') (Actual future path)
        # This is the "Realist" step: Value depends on what we ACTUALLY picked
        future_val = self.q_table[new_state_idx, next_action_idx]

        # [cite_start]3. Weighted Average Update [cite: 1882]
        new_q = (1 - ALPHA) * old_q + ALPHA * (reward + DELTA * future_val)

        # 4. Write back
        self.q_table[self.last_state_idx, self.last_action_idx] = new_q


# --- 6. INITIALIZATION & LOOP ---

# [cite_start]Initialize Q-value to average profit (Equation 8) [cite: 1996]
avg_profits = []
for p_i in action_space:
    p_i_profits = []
    for p_j in action_space:
        pi, _ = get_profit(p_i, p_j)
        p_i_profits.append(pi)
    avg_profits.append(np.mean(p_i_profits))
initial_q = np.mean(avg_profits) / (1 - DELTA)

print(f"Initial Q-Table Value: {initial_q:.4f}")

# State mapping: (p1_idx, p2_idx) -> single integer
n_states = M * M

# Create Agents
agent1 = SarsaAgent(M, n_states, initial_q)
agent2 = SarsaAgent(M, n_states, initial_q)

# Random start
p1_idx = np.random.randint(M)
p2_idx = np.random.randint(M)
state_idx = p1_idx * M + p2_idx

history_delta = []

# Previous step storage for the delayed update loop
pi1_prev = 0
pi2_prev = 0

print(f"\nRunning {NUM_EPISODES} episodes with SARSA...")

for t in tqdm(range(NUM_EPISODES), desc="Simulating SARSA"):
    # [cite_start]Calculate epsilon [cite: 1982]
    epsilon = np.exp(-BETA * t)

    # 1. Agents choose actions for CURRENT state
    # In the context of the previous step, this is a_{t+1}
    a1_idx = agent1.choose_action(state_idx, epsilon)
    a2_idx = agent2.choose_action(state_idx, epsilon)

    # 2. Update Previous Step (SARSA Logic)
    # We update the PREVIOUS action using the CURRENT action as the target
    if t > 0:
        agent1.update(pi1_prev, state_idx, a1_idx)
        agent2.update(pi2_prev, state_idx, a2_idx)

    # 3. Market Step
    p1 = action_space[a1_idx]
    p2 = action_space[a2_idx]
    pi1, pi2 = get_profit(p1, p2)

    # 4. Prepare for next loop
    new_state_idx = a1_idx * M + a2_idx

    # Store history for the update that will happen in the NEXT iteration
    agent1.last_state_idx = state_idx
    agent1.last_action_idx = a1_idx
    agent2.last_state_idx = state_idx
    agent2.last_action_idx = a2_idx

    pi1_prev = pi1
    pi2_prev = pi2
    state_idx = new_state_idx

    # 5. Record Delta
    if t % 1000 == 0:
        avg_pi = (pi1 + pi2) / 2
        delta = (avg_pi - pi_nash) / (pi_monop - pi_nash)
        history_delta.append(delta)

# --- 7. VISUALIZATION ---

print("Simulation finished.")
window = 1000
smoothed_delta = np.convolve(history_delta, np.ones(window) / window, mode='valid')

plt.figure(figsize=(12, 6))
plt.plot(smoothed_delta, linewidth=1.5, color='darkorange', label="SARSA")
plt.title(f"SARSA Agents: Average Profit Gain ($\Delta$) over Time\nAlpha={ALPHA}, Beta={BETA}")
plt.xlabel("Time (x1000 periods)")
plt.ylabel("Degree of Collusion ($\Delta$)")
plt.axhline(0, color='red', linestyle='--', label="Nash Equilibrium")
plt.axhline(1, color='green', linestyle='--', label="Monopoly")
plt.legend()
plt.grid(alpha=0.3)
plt.show()

# Final stats
final_avg = np.mean(history_delta[-50:])
print(f"Final Delta: {final_avg:.4f}")
if final_avg < 0.3:
    print("RESULT: SARSA failed to collude (Optimism Hypothesis Supported).")
else:
    print("RESULT: SARSA colluded (Optimism not required).")