import numpy as np
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
from tqdm import tqdm
from MWAgent import MWAgent

N_ROUNDS = 5000
LR1 = 0.1
LR2 = 0.1

# Chicken game payoffs matrix
chicken_game = np.array([
    [[-5, -5], [3, -1]],
    [[-1, 3], [0, 0]]
])

# Battle of the sexes payoffs matrix
battle_of_the_sexes = np.array([
    [[1, 2], [0, 0]],
    [[0, 0], [2, 1]]
])


def normalize(payoffs_matrix):
    """Normalize a payoffs_matrix to [0, 1]."""
    min_val = np.min(payoffs_matrix)
    max_val = np.max(payoffs_matrix)
    if max_val == min_val:
        return np.zeros_like(payoffs_matrix)
    return (payoffs_matrix - min_val) / (max_val - min_val)


def simulate_game(payoffs_matrix, n_rounds, lr1=0.1, lr2=0.1):
    """
    Simulate a game between two agents using the MW algorithm.

    Args:
        payoffs_matrix: Payoff matrix for the game [n_actions x n_actions x 2]
        n_rounds: Number of rounds to play
        lr1: Learning rate for agent 1
        lr2: Learning rate for agent 2

    Returns:
        The two agents after simulation
    """
    norm_payoffs_matrix = normalize(payoffs_matrix)
    n = payoffs_matrix.shape[0]

    # Create agents
    agent1 = MWAgent(n, lr1, "Player 1")
    agent2 = MWAgent(n, lr2, "Player 2")

    for t in range(n_rounds):
        # Agents choose actions
        action1 = agent1.choose_action()
        action2 = agent2.choose_action()

        # Calculate payoffs
        payoffs_vec1 = norm_payoffs_matrix[:, action2, 0]  # Player 1's payoffs
        payoffs_vec2 = norm_payoffs_matrix[action1, :, 1]  # Player 2's payoffs

        # Update agents
        agent1.update(payoffs_vec1)
        agent2.update(payoffs_vec2)

    return agent1, agent2


def plot_regret(agent1, agent2, game_name):
    """
    Plot the regret of both agents over time.

    Args:
        agent1: MWAgent (Player 1)
        agent2: MWAgent (Player 2)
        game_name: Name of the game for the title
    """
    plt.plot(agent1.regrets, label=agent1.name)
    plt.plot(agent2.regrets, label=agent2.name)
    plt.xlabel("Round")
    plt.ylabel("Regret")
    plt.title(f"Regret over time in {game_name}")
    plt.grid(True)
    plt.legend()
    plt.show()


def plot_joint_strategy_matrix(agent1, agent2, payoffs_matrix, labels=("Action 0", "Action 1")):
    """
    Plot a 2x2 matrix of joint probabilities based on final strategies of both agents,
    with payoffs for each joint action.

    Args:
        agent1: MWAgent (Player 1)
        agent2: MWAgent (Player 2)
        payoffs_matrix: [n_actions x n_actions x 2] payoffs for players
        labels: tuple of action names for axis labeling (optional)
    """
    p1 = agent1.distribution_history[-1]  # Final strategy of Player 1
    p2 = agent2.distribution_history[-1]  # Final strategy of Player 2

    joint_probs = np.outer(p1, p2)  # 2x2 matrix of joint outcome probabilities

    fig, ax = plt.subplots()
    cax = ax.matshow(joint_probs, cmap="Greens", vmin=0, vmax=1)

    n = joint_probs.shape[0]
    for i in range(n):
        for j in range(n):
            prob = joint_probs[i, j]
            payoff1 = payoffs_matrix[i, j, 0]
            payoff2 = payoffs_matrix[i, j, 1]
            text = f"({payoff1}, {payoff2})\nP={prob:.2f}"
            ax.text(j, i, text, va='center', ha='center', color='black', fontsize=9)

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)
    ax.set_xlabel(f"{agent2.name}'s Actions")
    ax.set_ylabel(f"{agent1.name}'s Actions")
    plt.title("Agent's Final Joint Action Distribution with Payoffs")
    fig.colorbar(cax, label="Probability")
    plt.tight_layout()
    plt.show()


def plot_empirical_joint_distribution(agent1, agent2, payoffs_matrix, labels=("Action 0", "Action 1")):
    """
    Plot a 2x2 matrix of empirical joint probabilities based on action history of both agents,
    with payoffs for each joint action.
    Args:
        agent1: MWAgent (Player 1)
        agent2: MWAgent (Player 2)
        payoffs_matrix: [n_actions x n_actions x 2] payoffs for players
        labels: tuple of action names for axis labeling (optional)
    """
    n = agent1.n_actions
    joint_counts = np.zeros((n, n))

    for a1, a2 in zip(agent1.action_history, agent2.action_history):
        joint_counts[a1, a2] += 1

    joint_probs = joint_counts / len(agent1.action_history)

    fig, ax = plt.subplots()
    cax = ax.matshow(joint_probs, cmap="Blues", vmin=0, vmax=1)

    for i in range(n):
        for j in range(n):
            prob = joint_probs[i, j]
            payoff1 = payoffs_matrix[i, j, 0]
            payoff2 = payoffs_matrix[i, j, 1]
            text = f"P={prob:.2f}\n({payoff1}, {payoff2})"
            ax.text(j, i, text, va='center', ha='center', fontsize=9)

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)
    ax.set_xlabel(f"{agent2.name}'s Actions")
    ax.set_ylabel(f"{agent1.name}'s Actions")
    plt.title("Empirical Joint Action Distribution with Payoffs")
    fig.colorbar(cax, label="Probability")
    plt.tight_layout()
    plt.show()


def experiment_varying_opponent_lr(
    payoffs_matrix,
    n_rounds,
    fixed_lr1=0.1,
    delta=0.09,
    n_points=21,
    target_action_index=0,
    target_action_name="Action 0",
    n_repeats=10
):
    # Create symmetric range of lr2 values around fixed_lr1
    lr2_values = np.linspace(fixed_lr1 - delta, fixed_lr1 + delta, n_points)
    lr2_values = np.clip(lr2_values, 1e-4, 1.0)

    def run_single_simulation(lr2):
        agent1, agent2 = simulate_game(payoffs_matrix, n_rounds, fixed_lr1, lr2)
        return agent1.distribution_history[-1][target_action_index]

    avg_probs = []

    for lr2 in tqdm(lr2_values, desc="Varying LR2"):
        repeat_probs = Parallel(n_jobs=-1)(
            delayed(run_single_simulation)(lr2) for _ in range(n_repeats)
        )
        avg_final_prob = np.mean(repeat_probs)
        avg_probs.append(avg_final_prob)

    # Plot the result
    plt.plot(lr2_values, avg_probs, marker='o')
    plt.axvline(fixed_lr1, color='gray', linestyle='--', label='Player 1 LR')
    plt.xlabel("Player 2 Learning Rate")
    plt.ylabel(f"Avg. Final Prob of Player 1 Choosing {target_action_name}")
    plt.title("Effect of Opponent's LR on Player 1's Strategy")
    plt.ylim(-0.05, 1.05)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()


def main():
    # Simulate Chicken game
    agent1_chicken, agent2_chicken = simulate_game(chicken_game, N_ROUNDS, LR1, LR2)
    plot_regret(agent1_chicken, agent2_chicken, "Chicken Game")
    plot_joint_strategy_matrix(agent1_chicken, agent2_chicken, chicken_game, labels=("Swerve", "Straight"))
    plot_empirical_joint_distribution(agent1_chicken, agent2_chicken, chicken_game, labels=("Swerve", "Straight"))
    experiment_varying_opponent_lr(chicken_game, N_ROUNDS, target_action_name="Swerve")

    # Simulate Battle of sexes
    agent1_battle, agent2_battle = simulate_game(battle_of_the_sexes, N_ROUNDS, LR1, LR2)
    plot_regret(agent1_battle, agent2_battle, "Battle of the sexes")
    plot_joint_strategy_matrix(agent1_battle, agent2_battle, chicken_game, labels=("Prize Fight", "Ballet"))
    plot_empirical_joint_distribution(agent1_battle, agent2_battle, chicken_game, labels=("Prize Fight", "Ballet"))
    experiment_varying_opponent_lr(battle_of_the_sexes, N_ROUNDS, target_action_name="Prize Fight")


if __name__ == "__main__":
    main()
