import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

# --- 1. CONFIGURATION ---
# We can use a higher Alpha because we are more confident in our "full info" updates
ALPHA = 0.2
BETA =15e-6
DELTA = 0.95
MU = 0.25
A_val = 2.0
A_0 = 0.0
C_val = 1.0
M = 15
XI = 0.1
NUM_EPISODES = 3_000_000

# np.random.seed(42)


# --- 2. ECONOMIC ENVIRONMENT (Same as before) ---
def get_demand(p1, p2):
    exp1 = np.exp((A_val - p1) / MU)
    exp2 = np.exp((A_val - p2) / MU)
    denom = exp1 + exp2 + np.exp(A_0 / MU)
    return exp1 / denom, exp2 / denom


def get_profit(p1, p2):
    q1, q2 = get_demand(p1, p2)
    pi1 = (p1 - C_val) * q1
    pi2 = (p2 - C_val) * q2
    return pi1, pi2


# --- 3. CALCULATE BENCHMARKS ---
# (Using the simplified iterative solver from previous fix)
def solve_nash():
    p = 1.5
    for _ in range(50):
        # Best response against self
        # Minimize negative profit
        from scipy.optimize import minimize
        res = minimize(lambda x: -get_profit(x, p)[0], p, bounds=[(1.0, 5.0)])
        if np.abs(res.x[0] - p) < 1e-6: break
        p = res.x[0]
    return p


def solve_monopoly():
    from scipy.optimize import minimize
    res = minimize(lambda x: -sum(get_profit(x[0], x[0])), [2.0], bounds=[(1.0, 5.0)])
    return res.x[0]


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

# --- 4. THE NEW "Q-EXPERTS" AGENT ---

class ExpertAgent:
    def __init__(self, player_id, n_actions, n_states, initial_q):
        self.player_id = player_id  # 0 for Agent 1, 1 for Agent 2
        self.q_table = np.full((n_states, n_actions), initial_q)
        self.n_actions = n_actions
        self.last_state_idx = None

    def choose_action(self, state_idx, epsilon):
        # Epsilon-greedy is still useful to explore different STATES
        if np.random.rand() < epsilon:
            return np.random.randint(self.n_actions)
        else:
            # Random tie-breaking for max
            max_q = np.max(self.q_table[state_idx])
            return np.random.choice(np.where(self.q_table[state_idx] == max_q)[0])

    def update_expert(self, rival_action_idx):
        """
        THE EXPERT UPDATE:
        Instead of updating just one cell, we update the entire row
        for the previous state, calculating what would have happened
        for EVERY possible price we could have chosen.
        """
        # Get the rival's price (which is fixed for this counterfactual calculation)
        p_rival = action_space[rival_action_idx]

        # Iterate over ALL possible actions I could have taken (0 to M-1)
        for potential_action_idx in range(self.n_actions):
            p_potential = action_space[potential_action_idx]

            # 1. Calculate Counterfactual Reward
            if self.player_id == 0:
                # I am P1, rival is P2
                r, _ = get_profit(p_potential, p_rival)
                # My potential action determines the first part of next state
                # Next state = (MyPotential, RivalActual)
                hypothetical_next_state = potential_action_idx * M + rival_action_idx
            else:
                # I am P2, rival is P1
                _, r = get_profit(p_rival, p_potential)
                # Next state = (RivalActual, MyPotential)
                hypothetical_next_state = rival_action_idx * M + potential_action_idx

            # 2. Calculate Continuation Value (Max Q of hypothetical next state)
            max_next_q = np.max(self.q_table[hypothetical_next_state])

            # 3. Update Q-Table
            old_q = self.q_table[self.last_state_idx, potential_action_idx]
            new_q = (1 - ALPHA) * old_q + ALPHA * (r + DELTA * max_next_q)

            self.q_table[self.last_state_idx, potential_action_idx] = new_q

for _ in range(3):
    # --- 5. INITIALIZATION ---
    # Same init logic
    avg_profits = []
    for p_i in action_space:
        p_i_profits = [get_profit(p_i, p_j)[0] for p_j in action_space]
        avg_profits.append(np.mean(p_i_profits))
    initial_q = np.mean(avg_profits) / (1 - DELTA)

    # Initialize Agents with IDs
    n_states = M * M
    agent1 = ExpertAgent(0, M, n_states, initial_q)
    agent2 = ExpertAgent(1, M, n_states, initial_q)

    # --- 6. SIMULATION LOOP ---

    p1_idx = np.random.randint(M)
    p2_idx = np.random.randint(M)
    state_idx = p1_idx * M + p2_idx

    history_delta = []

    print(f"Running Q-Experts Simulation for {NUM_EPISODES} episodes...")

    for t in tqdm(range(NUM_EPISODES), desc="Running Simulation"):
        epsilon = np.exp(-BETA * t)

        # 1. Choose Actions
        a1_idx = agent1.choose_action(state_idx, epsilon)
        a2_idx = agent2.choose_action(state_idx, epsilon)

        # 2. Get Real Profit (just for stats)
        pi1, pi2 = get_profit(action_space[a1_idx], action_space[a2_idx])

        # 3. Expert Update
        # Agent 1 updates knowing Agent 2 chose a2_idx
        if t > 0:
            agent1.update_expert(a2_idx)  # Uses agent1.last_state_idx internally
            agent2.update_expert(a1_idx)

        # 4. Advance State
        agent1.last_state_idx = state_idx
        agent2.last_state_idx = state_idx

        state_idx = a1_idx * M + a2_idx

        if t % 1000 == 0:
            delta = ((pi1 + pi2) / 2 - pi_nash) / (pi_monop - pi_nash)
            history_delta.append(delta)

    # --- 7. PLOT ---
    window = 100
    smoothed_delta = np.convolve(history_delta, np.ones(window) / window, mode='valid')

    plt.figure(figsize=(12, 6))
    plt.plot(smoothed_delta, linewidth=1.5)
    plt.title(f"Q-Experts: Average Profit Gain ($\Delta$) over Time\nAlpha={ALPHA}, Beta={BETA}")
    plt.xlabel("Time (x1000 periods)")
    plt.ylabel("Degree of Collusion ($\Delta$)")
    plt.axhline(0, color='red', linestyle='--', label="Nash Equilibrium")
    plt.axhline(1, color='green', linestyle='--', label="Monopoly")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.show()

    print(f"Final Delta: {np.mean(history_delta[-10:]) :.4f}")
