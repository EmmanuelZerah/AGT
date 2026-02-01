import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize_scalar
from tqdm import tqdm
from collections import deque

# Assuming EXP3IXAgent.py is in the same directory
from algorithms.EXP3IXAgent import EXP3IX


# ==========================================
# 1. The Environment (Same as before)
# ==========================================
class BertrandDuopoly:
    def __init__(self, mu=0.25, c=1.0, a=2.0, a0=0.0, m=15, xi=0.1):
        self.mu = mu
        self.c = c
        self.a = a
        self.a0 = a0
        self.m = m
        self.p_nash = self._find_nash_price()
        self.p_monopoly = self._find_monopoly_price()

        interval = self.p_monopoly - self.p_nash
        low = self.p_nash - xi * interval
        high = self.p_monopoly + xi * interval
        self.actions = np.linspace(low, high, m)
        self.max_profit = self._get_max_possible_profit()

    def _demand(self, p_own, p_rival):
        exp_own = np.exp((self.a - p_own) / self.mu)
        exp_rival = np.exp((self.a - p_rival) / self.mu)
        exp_out = np.exp(self.a0 / self.mu)
        return exp_own / (exp_own + exp_rival + exp_out)

    def _get_profit(self, p_own, p_rival):
        q = self._demand(p_own, p_rival)
        return (p_own - self.c) * q

    def _find_nash_price(self):
        p = 1.5
        for _ in range(50):
            res = minimize_scalar(lambda x: -self._get_profit(x, p), bounds=(1, 3), method='bounded')
            p = res.x
        return p

    def _find_monopoly_price(self):
        res = minimize_scalar(lambda x: -2 * self._get_profit(x, x), bounds=(1, 5), method='bounded')
        return res.x

    def _get_max_possible_profit(self):
        max_p = 0
        for p_rival in np.linspace(self.p_nash, self.p_monopoly * 1.2, 20):
            res = minimize_scalar(lambda x: -self._get_profit(x, p_rival), bounds=(1, 3), method='bounded')
            profit = -res.fun
            if profit > max_p: max_p = profit
        return max_p * 1.1

    def step(self, idx1, idx2):
        p1 = self.actions[idx1]
        p2 = self.actions[idx2]
        pi1 = self._get_profit(p1, p2)
        pi2 = self._get_profit(p2, p1)
        return pi1, pi2, p1, p2


# ==========================================
# 2. Simulation Setup
# ==========================================
N_EPISODES = 2_000_000
LEARNING_RATE_SCALE = 2.0

env = BertrandDuopoly()
agent1 = EXP3IX(n_arms=env.m, learning_rate_scale=LEARNING_RATE_SCALE)
agent2 = EXP3IX(n_arms=env.m, learning_rate_scale=LEARNING_RATE_SCALE)

# --- NEW: Buffers for 3-Step Delay ---
# Each buffer will store tuples of (action_index, reward_at_that_time)
# We need to store t-2 and t-1 to combine with t.
buffer_a1 = deque(maxlen=2)
buffer_a2 = deque(maxlen=2)

history_p1 = []
history_p2 = []

print(f"Simulation Started: 3-Step Delayed Additive Rewards")
print(f"Nash Price: {env.p_nash:.3f}, Monopoly Price: {env.p_monopoly:.3f}")

# ==========================================
# 3. Main Loop
# ==========================================
for t in tqdm(range(N_EPISODES), desc="Simulating 2-Step Delay"):

    # --- A. Choose Actions ---
    idx1, prob1 = agent1.choose_action()
    idx2, prob2 = agent2.choose_action()

    # --- B. Environment Step ---
    profit1, profit2, p1, p2 = env.step(idx1, idx2)

    # --- C. Normalize Rewards [0, 1] ---
    r1_t = profit1 / env.max_profit
    r2_t = profit2 / env.max_profit

    # --- D. 3-Step Delayed Update Logic ---
    # Logic:
    # If buffer has 2 items: [(a_{t-2}, r_{t-2}), (a_{t-1}, r_{t-1})]
    # We can now fully update a_{t-2} using r_{t-2} + r_{t-1} + r_t

    # Agent 1 Update
    if len(buffer_a1) == 2:
        # Pop the oldest event (t-2)
        idx_t_minus_2, r_t_minus_2 = buffer_a1[0]
        # Peek at the middle event (t-1)
        _, r_t_minus_1 = buffer_a1[1]

        # Calculate 3-step average reward
        combined_reward = (r_t_minus_2 + r_t_minus_1 + r1_t) / 3.0

        # Update the agent for the action taken 2 steps ago
        agent1.update(idx_t_minus_2, combined_reward)

    # Agent 2 Update
    if len(buffer_a2) == 2:
        idx_t_minus_2, r_t_minus_2 = buffer_a2[0]
        _, r_t_minus_1 = buffer_a2[1]

        combined_reward = (r_t_minus_2 + r_t_minus_1 + r2_t) / 3.0
        agent2.update(idx_t_minus_2, combined_reward)

    # --- E. Update Buffers ---
    # Add current round (t) to the right.
    # If len was 2, appending will automatically push out the oldest (t-2) due to maxlen=2,
    # effectively sliding the window forward.
    buffer_a1.append((idx1, r1_t))
    buffer_a2.append((idx2, r2_t))

    # --- F. Tracking ---
    history_p1.append(p1)
    history_p2.append(p2)

# ==========================================
# 4. Analysis & Plotting
# ==========================================
window = 1000


def moving_avg(x, w):
    return np.convolve(x, np.ones(w) / w, 'valid')


ma_p1 = moving_avg(history_p1, window)
ma_p2 = moving_avg(history_p2, window)

plt.figure(figsize=(10, 6))
plt.plot(ma_p1, label="Agent 1 (Avg Price)", alpha=0.8)
plt.plot(ma_p2, label="Agent 2 (Avg Price)", alpha=0.8, linestyle='--')
plt.axhline(env.p_nash, color='red', linestyle=':', linewidth=2, label="Bertrand-Nash")
plt.axhline(env.p_monopoly, color='green', linestyle=':', linewidth=2, label="Monopoly")

plt.title("EXP3-IX with 2-Step Delayed Rewards")
plt.xlabel("Episode")
plt.ylabel("Price")
plt.legend()
plt.grid(alpha=0.3)
plt.show()

# Final stats
final_avg = np.mean(history_p1[-5000:])
print(f"Final Average Price: {final_avg:.3f}")
print(f"Gap to Monopoly: {env.p_monopoly - final_avg:.3f}")
print(f"Gap to Nash: {final_avg - env.p_nash:.3f}")