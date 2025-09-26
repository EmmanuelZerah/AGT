import numpy as np
from scipy.special import softmax
from NoRegretAgent import NoRegretAgent


class MWAgent(NoRegretAgent):
    """Agent using the Multiplicative Weights algorithm to maximize payoff."""

    def __init__(self, n_actions, learning_rate=0.1, name="Agent"):
        """
        Initialize an agent using the MW algorithm.

        Args:
            n_actions: Number of possible actions
            learning_rate: The learning rate parameter
            name: Name identifier for the agent
        """
        super().__init__(n_actions, name)
        self.learning_rate = learning_rate

        # Initialize weights uniformly
        self.log_weights = np.zeros(n_actions)

        # Keep track of history for analysis
        self.action_history = []
        self.distribution_history = []
        self.payoff_history = []
        self.cumulative_payoff = 0
        self.best_action_payoff = 0
        self.best_action = None
        self.best_action_payoffs = None
        self.regrets = []

    def choose_action(self):
        """Choose an action based on the current weights."""
        # Create probability distribution from weights
        probabilities = softmax(self.log_weights)

        # Store the distribution for analysis
        self.distribution_history.append(probabilities.copy())

        # Sample action according to this distribution
        action = np.random.choice(self.n_actions, p=probabilities)
        self.action_history.append(action)

        return action

    def update(self, payoffs):
        """
        Update weights based on observed payoffs.

        Args:
            payoffs: Vector of payoffs for each action
        """
        self.log_weights += np.log(1 + self.learning_rate * payoffs)

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

    def calculate_regret(self):
        """Calculate the regret compared to the best fixed action in hindsight."""
        if not self.payoff_history or self.best_action_payoffs is None:
            return 0

        # Calculate payoff of the best fixed action in hindsight
        best_fixed_action_payoff = np.max(self.best_action_payoffs)

        # Calculate regret (best fixed action payoff - algorithm's payoff)
        regret = best_fixed_action_payoff - self.cumulative_payoff

        # Return time-averaged regret
        return regret / len(self.payoff_history)

    def reset(self):
        """Reset the agent's state for a new game."""
        self.log_weights = np.zeros(self.n_actions)
        self.action_history.clear()
        self.distribution_history.clear()
        self.payoff_history.clear()
        self.cumulative_payoff = 0
        self.best_action = None
        self.best_action_payoff = 0
        self.best_action_payoffs = None
        self.regrets.clear()
