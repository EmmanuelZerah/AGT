import numpy as np
from scipy.special import softmax

class MWDeterministicAgent:
    """
    Deterministic Multiplicative Weights Update (MWU) agent.

    Initialized with:
      - A payoff matrix U where U[i, j] = payoff of this player
        when they play action i and the opponent plays action j.

    At each step, given the opponent's mixed strategy q,
    the agent updates its log-weights using the expected payoff
    vector U @ q (deterministic update, no random sampling).
    """

    def __init__(self, payoff_matrix: np.ndarray, learning_rate: float = 0.05, name: str = "Agent"):
        self.U = np.asarray(payoff_matrix, dtype=float)   # shape = (n_actions_self, n_actions_opp)
        self.n_actions = self.U.shape[0]
        self.learning_rate = learning_rate
        self.name = name
        self.log_weights = np.zeros(self.n_actions, dtype=float)

    @property
    def distribution(self) -> np.ndarray:
        """Return the current mixed strategy (softmax over log-weights)."""
        return softmax(self.log_weights)

    def expected_payoff_vector(self, opp_dist: np.ndarray) -> np.ndarray:
        """
        Compute expected payoff for each action:
            u[i] = Σ_j  U[i, j] * opp_dist[j]
        """
        return self.U @ np.asarray(opp_dist, dtype=float)

    def update(self, opp_dist: np.ndarray):
        """
        Perform a deterministic MWU update against opponent's distribution.
        """
        u_exp = self.expected_payoff_vector(opp_dist)
        self.log_weights += self.learning_rate * u_exp
        # normalize log_weights to improve numerical stability
        self.log_weights -= np.max(self.log_weights)

    def step(self, opp_dist: np.ndarray):
        """Alias for update() — provided for semantic clarity."""
        self.update(opp_dist)
