import numpy as np


class Exp3IX:
    def __init__(self, n_arms):
        self.n_arms = n_arms
        self.t = 0
        self.cumulative_losses = np.zeros(n_arms)  # Track losses, not rewards

    def choose_action(self):
        self.t += 1

        # 1. Anytime Learning Rate (Neu 2015 parameterization)
        # eta_t = sqrt(2 * log(K) / (K * t))
        eta = np.sqrt(2 * np.log(self.n_arms) / (self.n_arms * self.t))

        # 2. Implicit Exploration Parameter
        # Standard setting is gamma = eta / 2
        self.current_eta = eta
        self.current_gamma = eta / 2

        # 3. Calculate Weights (using Losses)
        # w_i = exp(-eta * CumulativeLoss_i)
        log_weights = -self.current_eta * self.cumulative_losses
        log_weights -= np.max(log_weights)  # Numerical stability
        weights = np.exp(log_weights)

        self.probs = weights / np.sum(weights)

        chosen_arm = np.random.choice(self.n_arms, p=self.probs)
        return chosen_arm, self.probs[chosen_arm]

    def update(self, arm_index, observed_reward):
        # Clip reward
        observed_reward = np.clip(observed_reward, 0, 1)

        # Convert to Loss (Standard Exp3-IX formulation)
        observed_loss = 1.0 - observed_reward

        # IX Estimator for Loss: l / (p + gamma)
        # This underestimates loss (optimism), which is theoretically preferred
        prob = self.probs[arm_index]
        estimated_loss = observed_loss / (prob + self.current_gamma)

        self.cumulative_losses[arm_index] += estimated_loss
