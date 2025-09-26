# repeated_first_price_auction.py
import numpy as np
from matplotlib import pyplot as plt
from MWAgent import MWAgent


def payoff_vector_first_price(value: int, opp_bid: int, n_actions: int = 11) -> np.ndarray:
    """
    Return full-information payoffs for all possible bids given the opponent's bid.
    Bids are integers in [0, n_actions-1]. First-price auction with value 'value'.
    Tie-breaking: split the surplus evenly.
    """
    bids = np.arange(n_actions)
    # winner if bid > opp; tie if == opp; else lose
    win_mask = bids > opp_bid
    tie_mask = bids == opp_bid
    lose_mask = bids < opp_bid

    payoffs = np.zeros(n_actions, dtype=float)
    payoffs[win_mask] = value - bids[win_mask]
    payoffs[tie_mask] = 0.5 * (value - bids[tie_mask])
    payoffs[lose_mask] = 0.0
    return payoffs


def run_simulation(
    T: int = 100000,
    value_1: int = 10,
    value_2: int = 10,
    eta_1: float = 0.01,
    eta_2: float = 0.01,
    seed: int = 42
):
    """
    Run a repeated 2-player first-price auction with two MW agents.
    Returns results dict containing actions, bids, payoffs, regrets, and distributions.
    """
    if seed is not None:
        np.random.seed(seed)

    n_actions = 11  # bids 0..10
    agent1 = MWAgent(n_actions=n_actions, learning_rate=eta_1, name="P1")
    agent2 = MWAgent(n_actions=n_actions, learning_rate=eta_2, name="P2")

    # Logs
    bids_1, bids_2 = [], []
    payoff_1, payoff_2 = [], []

    for t in range(T):
        b1 = agent1.choose_action()   # chosen index == bid
        b2 = agent2.choose_action()
        bids_1.append(b1)
        bids_2.append(b2)

        # Construct full-information payoff vectors for both players
        p1_all = payoff_vector_first_price(value=value_1, opp_bid=b2, n_actions=n_actions)
        p2_all = payoff_vector_first_price(value=value_2, opp_bid=b1, n_actions=n_actions)

        # Update agents
        agent1.update(p1_all)
        agent2.update(p2_all)

        # Log realized payoffs for the chosen bids
        payoff_1.append(p1_all[b1])
        payoff_2.append(p2_all[b2])

        if t+1 in [1, 10, 50, 100, 500, 1000, 5000, 10_000, 50_000, 100_000]:
            final_dist_p1 = agent1.distribution_history[-1]
            final_dist_p2 = agent2.distribution_history[-1]

            fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharey=True)

            # Player 1
            axes[0].bar(range(11), final_dist_p1, color="skyblue")
            axes[0].set_title("P1 Distribution")
            axes[0].set_xlabel("Bid")
            axes[0].set_ylabel("Probability")
            axes[0].set_xticks(range(11))
            axes[0].set_ylim(0, 1)

            # Player 2
            axes[1].bar(range(11), final_dist_p2, color="salmon")
            axes[1].set_title("P2 Distribution")
            axes[1].set_xlabel("Bid")
            axes[1].set_xticks(range(11))
            axes[0].set_ylim(0, 1)

            plt.suptitle(f"Action Distributions at t={t+1}")
            plt.tight_layout()
            plt.show()

    results = {
        "bids_1": np.array(bids_1),
        "bids_2": np.array(bids_2),
        "payoffs_1": np.array(payoff_1),
        "payoffs_2": np.array(payoff_2),
        "regret_1": np.array(agent1.regrets),
        "regret_2": np.array(agent2.regrets),
        "dist_history_1": np.array(agent1.distribution_history),  # shape (T, 11)
        "dist_history_2": np.array(agent2.distribution_history),
        "agent1": agent1,
        "agent2": agent2,
    }
    return results


def main():
    for _ in range(1):
        res = run_simulation(seed=None)


if __name__ == "__main__":
    main()


