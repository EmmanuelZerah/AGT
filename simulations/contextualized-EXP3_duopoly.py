import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.special import softmax
from tqdm import tqdm


# --- 1. EXP3 AGENT & POLICY ---

class EXP3Agent:
    def __init__(self, n_actions, epsilon=0.005, seed=None, name="EXP3"):
        self.n = int(n_actions)
        self.epsilon = float(epsilon)
        self.rng = np.random.default_rng(seed)
        self.log_weights = np.zeros(self.n)
        self.prob = np.ones(self.n) / self.n

    def _update_probs(self, force_greedy=False):
        safe_logits = self.log_weights - np.max(self.log_weights)
        weights_ratio = softmax(safe_logits)
        if force_greedy:
            self.prob = weights_ratio
        else:
            K = self.n
            self.prob = (1 - K * self.epsilon) * weights_ratio + self.epsilon

    def choose_action(self, force_greedy=False):
        self._update_probs(force_greedy)
        self.prob /= np.sum(self.prob)
        return int(self.rng.choice(self.n, p=self.prob))

    def update(self, chosen_arm, reward):
        if np.max(self.log_weights) > 100:
            self.log_weights -= np.max(self.log_weights)
        arm_prob = float(self.prob[chosen_arm])
        if arm_prob < 1e-12: arm_prob = 1e-12
        xhat = reward / arm_prob
        self.log_weights[chosen_arm] += self.epsilon * xhat


class ContextualPolicy:
    def __init__(self, n_actions, make_agent, seed=None, name="CTX"):
        self.n_actions = n_actions
        self.make_agent = make_agent
        self.policies = {}
        self.rng = np.random.default_rng(seed)

    def _get_policy(self, state_key):
        if state_key not in self.policies:
            self.policies[state_key] = self.make_agent()
        return self.policies[state_key]

    def act(self, state_key, force_greedy=False):
        return self._get_policy(state_key).choose_action(force_greedy)

    def update_bandit(self, state_key, chosen_arm, reward_01):
        self._get_policy(state_key).update(chosen_arm, reward_01)


# --- 2. CONFIGURATION ---

NUM_EPISODES = 3_000_000
EVAL_EPISODES = 1_000
EPSILON = 0.003
SEED = 45
MU = 0.25
A_val = 2.0
A_0 = 0.0
C_val = 1.0
M = 15
XI = 0.1
np.random.seed(SEED)


# --- 3. ENVIRONMENT & SETUP ---

def get_profit(p1, p2):
    exp1 = np.exp((A_val - p1) / MU)
    exp2 = np.exp((A_val - p2) / MU)
    denom = exp1 + exp2 + np.exp(A_0 / MU)
    return (p1 - C_val) * (exp1 / denom), (p2 - C_val) * (exp2 / denom)


# Benchmarks
def solve_nash():
    p = 1.5
    for _ in range(50):
        res = minimize(lambda x: -get_profit(x, p)[0], p, bounds=[(1.0, 5.0)])
        if np.abs(res.x[0] - p) < 1e-6: break
        p = res.x[0]
    return p


def solve_monopoly():
    res = minimize(lambda x: -sum(get_profit(x[0], x[0])), [2.0], bounds=[(1.0, 5.0)])
    return res.x[0]

p_nash = solve_nash()
p_monop = solve_monopoly()
pi_nash = get_profit(p_nash, p_nash)[0]
pi_monop = get_profit(p_monop, p_monop)[0]
interval = p_monop - p_nash
action_space = np.linspace(p_nash - XI * interval, p_monop + XI * interval, M)

max_possible_profit = 0
for p1 in action_space:
    for p2 in action_space:
        pi = get_profit(p1, p2)[0]
        if pi > max_possible_profit: max_possible_profit = pi

print(f"Reference: Nash Price = {p_nash:.4f}, Monopoly Price = {p_monop:.4f}")
print(f"Action Space (Allowed Prices): \n{np.round(action_space, 3)}")

# Agents
def make_agent_1(): return EXP3Agent(M, epsilon=EPSILON, seed=SEED)


def make_agent_2(): return EXP3Agent(M, epsilon=EPSILON, seed=SEED + 1)


agent1 = ContextualPolicy(M, make_agent_1, seed=SEED)
agent2 = ContextualPolicy(M, make_agent_2, seed=SEED + 1)

# --- 4. SIMULATION ---

a1 = np.random.randint(M)
a2 = np.random.randint(M)
state_key = (a1, a2)
history_delta = []

print(f"Phase 1: Training ({NUM_EPISODES} steps)...")
for t in tqdm(range(NUM_EPISODES), desc="Training Agents"):
    a1 = agent1.act(state_key, force_greedy=False)
    a2 = agent2.act(state_key, force_greedy=False)

    p1, p2 = action_space[a1], action_space[a2]
    pi1, pi2 = get_profit(p1, p2)

    agent1.update_bandit(state_key, a1, np.clip(pi1 / max_possible_profit, 0, 1))
    agent2.update_bandit(state_key, a2, np.clip(pi2 / max_possible_profit, 0, 1))

    state_key = (a1, a2)
    # 5. Record Delta (CORRECTED)
    if t % 100 == 0:
        avg_pi = (pi1 + pi2) / 2
        delta = (avg_pi - pi_nash) / (pi_monop - pi_nash)
        history_delta.append(delta)

print(f"Phase 2: Evaluation ({EVAL_EPISODES} steps)...")
eval_p1, eval_p2 = [], []
for t in tqdm(range(EVAL_EPISODES), desc="Evaluating Agents"):
    a1 = agent1.act(state_key, force_greedy=True)
    a2 = agent2.act(state_key, force_greedy=True)
    state_key = (a1, a2)
    eval_p1.append(action_space[a1])
    eval_p2.append(action_space[a2])

# --- 5. PLOTTING TIME SERIES ---
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
# Training
window = 2000
smooth_train = np.convolve(history_delta, np.ones(window) / window, mode='valid')
ax1.plot(smooth_train, color='tab:blue')
ax1.set_title("Training: Delta (Smoothed)")
ax1.set_ylim(-0.2, 1.1)
ax1.grid(alpha=0.3)
# Evaluation Prices
ax2.plot(eval_p1, label='Ag1', alpha=0.8)
ax2.plot(eval_p2, label='Ag2', alpha=0.8, linestyle='--')
ax2.set_title(f"Evaluation: Prices (Last {EVAL_EPISODES})")
ax2.legend()
ax2.grid(alpha=0.3)
plt.show()

# --- 6. PLOTTING THE LEARNED POLICY (HEATMAP) ---

print("Generating Policy Heatmaps for Agent 1...")

# Initialize grids (Rows: Own Prev Price, Cols: Rival Prev Price)
policy_grid = np.zeros((M, M))
prob_grid = np.zeros((M, M))

# Fill grids
for i in range(M):  # Own previous price index
    for j in range(M):  # Rival previous price index
        state = (i, j)

        # Check if this state was ever visited/initialized
        if state in agent1.policies:
            agent = agent1.policies[state]

            # Force greedy to see the "learned" best action
            agent._update_probs(force_greedy=True)

            # Find most probable action and its probability
            best_action_idx = np.argmax(agent.prob)
            max_prob = np.max(agent.prob)

            # Store price (not index) for better readability
            policy_grid[i, j] = action_space[best_action_idx]
            prob_grid[i, j] = max_prob
        else:
            # If state never visited, assume random (or mark as nan)
            policy_grid[i, j] = np.nan
            prob_grid[i, j] = 0.0

# Plotting
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# Heatmap 1: The Strategy (Most Probable Price)
# Use 'pcolormesh' or 'imshow'. Imshow is easier for grids.
# Origin 'lower' puts index 0 at the bottom.
im1 = ax1.imshow(policy_grid, origin='lower', cmap='viridis', aspect='auto',
                 extent=[action_space[0], action_space[-1], action_space[0], action_space[-1]])
ax1.set_title("Agent 1 Strategy\n(Price chosen given previous prices)")
ax1.set_xlabel("Rival's Previous Price ($P_{2, t-1}$)")
ax1.set_ylabel("Own Previous Price ($P_{1, t-1}$)")
fig.colorbar(im1, ax=ax1, label='Next Price ($P_{1, t}$)')

# Heatmap 2: The Confidence (Probability of that Price)
im2 = ax2.imshow(prob_grid, origin='lower', cmap='plasma', aspect='auto', vmin=0, vmax=1,
                 extent=[action_space[0], action_space[-1], action_space[0], action_space[-1]])
ax2.set_title("Agent 1 Confidence\n(Probability of chosen action)")
ax2.set_xlabel("Rival's Previous Price ($P_{2, t-1}$)")
ax2.set_ylabel("Own Previous Price ($P_{1, t-1}$)")
fig.colorbar(im2, ax=ax2, label='Probability')

plt.tight_layout()
plt.show()
