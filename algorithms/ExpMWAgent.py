import numpy as np
from MWAgent import MWAgent


class ExpMWAgent(MWAgent):
    """Agent using the Multiplicative Weights algorithm to maximize payoff."""

    def __init__(self, n_actions, learning_rate=0.1, name="Agent"):
        """
        Initialize an agent using the MW algorithm.

        Args:
            n_actions: Number of possible actions
            learning_rate: The learning rate parameter
            name: Name identifier for the agent
        """
        super().__init__(n_actions, learning_rate, name)

    def update(self, payoffs):
        """
        Update weights based on observed payoffs.

        Args:
            payoffs: Vector of payoffs for each action
        """
        self.log_weights += self.learning_rate * payoffs

        # Numerical stability: subtract max. doesn't change the softmax
        self.log_weights -= np.max(self.log_weights)

        # Track payoff statistics
        realized_payoff = payoffs[self.action_history[-1]]
        self.payoff_history.append(realized_payoff)
        self.cumulative_payoff += realized_payoff

        # Track best fixed action in hindsight
        if len(self.payoff_history) == 1:
            self.best_action_payoffs = payoffs.copy()
        else:
            self.best_action_payoffs += payoffs

        current_best_action = np.argmax(self.best_action_payoffs)
        current_best_payoff = np.max(self.best_action_payoffs)
        if current_best_payoff > self.best_action_payoff:
            self.best_action_payoff = current_best_payoff
            self.best_action = current_best_action

        # Track regret
        self.regrets.append(self.calculate_regret())
