import numpy as np
from scipy.special import softmax

class EXP3Agent:
    def __init__(self, n_actions, learning_rate=0.07, gamma=0.07, seed=None, name="EXP3"):
        self.n = int(n_actions)
        self.learning_rate = float(learning_rate)
        self.gamma = float(gamma)
        self.rng = np.random.default_rng(seed)
        self.log_weights = np.zeros(self.n)
        self.prob = np.ones(self.n) / self.n
        self.name = name

    def _update_probs(self):
        exp_z = softmax(self.log_weights)
        self.prob = (1 - self.gamma) * exp_z + self.gamma / self.n

    def choose_action(self):
        self._update_probs()
        return int(self.rng.choice(self.n, p=self.prob))

    def update(self, chosen_arm, reward):
        self.log_weights -= np.max(self.log_weights)
        arm_prob = float(self.prob[chosen_arm])
        xhat = reward / arm_prob  # reward should be between 0 and 1
        self.log_weights[chosen_arm] += self.learning_rate * xhat
        self._update_probs()
