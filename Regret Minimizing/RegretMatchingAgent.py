
import numpy as np
from NoRegretAgent import NoRegretAgent


class RegretMatchingAgent(NoRegretAgent):
    """Agent using the Regret Matching algorithm to maximize payoff."""

    def __init__(self, n_actions, name="Agent"):
        """
        Initialize an agent using the Regret Matching algorithm.

        Args:
            n_actions: Number of possible actions
            name: Name identifier for the agent
        """
        super().__init__(n_actions, name)

        # Regret matching: track cumulative regret for not playing each action
        self.regret_sums = np.zeros(n_actions)

        # Keep track of history for analysis
        self.action_history = []
        self.distribution_history = []
        self.payoff_history = []
        self.cumulative_payoff = 0
        self.best_action = None
        self.best_action_payoff = 0
        self.best_action_payoffs = None
        self.regrets = []

    def choose_action(self):
        """Choose an action based on current regret-matching probabilities."""
        positive_regrets = np.maximum(self.regret_sums, 0)
        if np.sum(positive_regrets) > 0:
            probabilities = positive_regrets / np.sum(positive_regrets)
        else:
            probabilities = np.ones(self.n_actions) / self.n_actions  # uniform

        self.distribution_history.append(probabilities.copy())
        action = np.random.choice(self.n_actions, p=probabilities)
        self.action_history.append(action)
        return action

    def update(self, payoffs):
        """
        Update regrets based on observed payoffs.

        Args:
            payoffs: Vector of payoffs for each action
        """
        last_action = self.action_history[-1]
        actual = payoffs[last_action]
        self.regret_sums += payoffs - actual  # Regret for not having played other actions

        # Track payoff statistics
        self.payoff_history.append(actual)
        self.cumulative_payoff += actual

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
        best_fixed_action_payoff = np.max(self.best_action_payoffs)
        regret = best_fixed_action_payoff - self.cumulative_payoff
        return regret / len(self.payoff_history)

    def reset(self):
        """Reset the agent's state for a new game."""
        self.regret_sums = np.zeros(self.n_actions)
        self.action_history.clear()
        self.distribution_history.clear()
        self.payoff_history.clear()
        self.cumulative_payoff = 0
        self.best_action = None
        self.best_action_payoff = 0
        self.best_action_payoffs = None
        self.regrets.clear()
