import numpy as np
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
from tqdm import tqdm
from MWAgent import MWAgent
from RegretMatchingAgent import RegretMatchingAgent
from ExpMWAgent import ExpMWAgent

N_ROUNDS = 20000
LR1 = 0.01
LR2 = 0.01

# Chicken game payoffs matrix
chicken_game = np.array([
    [[0.0, 0.0], [1.0, 0.1]],
    [[0.1, 1.0], [0.4, 0.4]]
])

BotS_game = np.array([
    [[3, 2], [0, 0]],
    [[0, 0], [2, 3]]
])


def normalize(payoffs_matrix):
    """
    Normalize each player's payoff matrix independently to the range [0, 1].

    Args:
        payoffs_matrix: A 3D numpy array of shape (n_actions_p1, n_actions_p2, 2),
                        where the last dimension corresponds to (p1_payoff, p2_payoff).

    Returns:
        A normalized payoff matrix in the range [0, 1] for each player.
    """
    norm_payoffs_matrix = np.zeros_like(payoffs_matrix, dtype=np.float64)

    for player in [0, 1]:  # 0 for player 1, 1 for player 2
        player_payoffs = payoffs_matrix[:, :, player]
        max_val = np.max(np.abs(player_payoffs))
        min_val = np.min(np.abs(player_payoffs))

        if min_val < 0:
            raise ValueError("Payoff matrix contains negative values, normalization requires non-negative payoffs.")

        # Avoid division by zero
        if max_val > 0:
            norm = player_payoffs / max_val
        else:
            norm = np.zeros_like(player_payoffs)

        norm_payoffs_matrix[:, :, player] = norm

    return norm_payoffs_matrix


def simulate_game(payoffs_matrix, n_rounds, agent1, agent2):
    """
    Simulate a game between two agents using the MW algorithm.

    Args:
        payoffs_matrix: Payoff matrix for the game [n_actions x n_actions x 2]
        n_rounds: Number of rounds to play
        agent1: Player 1
        agent2: Player 2
    Returns:
        The two agents after simulation
    """
    norm_payoffs_matrix = normalize(payoffs_matrix)
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

    joint_probs = np.outer(p1, p2)  # 2x2 matrix of joint outcome probabilities. assumes independent actions

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
    plt.title("Agent's Final Joint Action Distribution")
    fig.colorbar(cax, label="Probability")
    plt.tight_layout()
    plt.show()


def plot_empirical_joint_distribution(agent1, agent2, payoffs_matrix, labels=("Action 0", "Action 1")):
    """
    Plot a 2x2 matrix of empirical joint probabilities based on action history of both agents,
    with payoffs for each joint action.
    Args:
        agent1: Player 1
        agent2: Player 2
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
            text = f"({payoff1}, {payoff2})\nP={prob:.2f}"
            ax.text(j, i, text, va='center', ha='center', fontsize=9)

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)
    ax.set_xlabel(f"{agent2.name}'s Actions")
    ax.set_ylabel(f"{agent1.name}'s Actions")
    plt.title("Empirical Joint Action Distribution")
    fig.colorbar(cax, label="Probability")
    plt.tight_layout()
    plt.show()


def experiment_varying_opponent_lr(
        payoffs_matrix,
        n_rounds,
        fixed_lr1=0.25,
        delta=0.025,
        target_action_index=0,
        target_action_name="Action 0",
        n_repeats=300
):
    # Create symmetric range of lr2 values around fixed_lr1
    lr2_values = np.arange(0.025, 0.5 + delta, delta)

    def run_single_simulation(lr2):
        agent1 = MWAgent(n_actions=payoffs_matrix.shape[0], learning_rate=fixed_lr1, name="Player 1")
        agent2 = MWAgent(n_actions=payoffs_matrix.shape[0], learning_rate=lr2, name="Player 2")
        agent1, agent2 = simulate_game(payoffs_matrix, n_rounds, agent1, agent2)
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


def search_for_non_pure_equilibrium(payoffs_matrix, n_repeats=1000):
    non_pure_equilibrium = []

    for _ in tqdm(range(n_repeats), desc="Searching for non-pure equilibria"):
        agent1 = MWAgent(n_actions=2, learning_rate=LR1, name="MW Agent 1")
        agent2 = MWAgent(n_actions=2, learning_rate=LR2, name="MW Agent 2")
        agent1, agent2 = simulate_game(payoffs_matrix, N_ROUNDS, agent1, agent2)

        if (not np.isclose(agent1.distribution_history[-1][0], 1.0, atol=1e-2) and
                not np.isclose(agent1.distribution_history[-1][0], 0.0, atol=1e-2)):
            non_pure_equilibrium.append((agent1.distribution_history[-1], agent2.distribution_history[-1]))

    if non_pure_equilibrium:
        print("Found non-pure equilibrium strategies:")
        for i, (dist1, dist2) in enumerate(non_pure_equilibrium):
            print(f"Player 1: {dist1}, Player 2: {dist2}")
    else:
        print(f"No non-pure equilibrium strategies found in {n_repeats} simulations.")


def simulate_fixed_vs_random_lr(payoffs_matrix=None, n_rounds=N_ROUNDS, n_repeats=500):
    """
    Simulate n_repeats games where Player 1 uses fixed_lr1 and Player 2 uses a random lr2 in (0, 0.5).
    Returns the average probability that Player 1 chooses Hawk (action 0) at the end.
    """
    results = []
    flag = False
    for _ in tqdm(range(n_repeats), desc="Simulating fixed vs random LR"):

        if payoffs_matrix is None or flag:
            # Randomly generate a payoff matrix for the Hawk-Dove game
            a = 0
            d = np.random.uniform(0, 1)
            b = np.random.uniform(0, d)
            c = 1

            payoffs_matrix = np.array([
                [[a, a], [c, b]],
                [[b, c], [d, d]]
            ])

            flag = True

        else:
            a = payoffs_matrix[0, 0, 0]  # Payoff for Player 1 choosing Hawk vs Hawk
            b = payoffs_matrix[0, 1, 1]  # Payoff for Player 1 choosing Dove vs Hawk
            c = payoffs_matrix[0, 1, 0]  # Payoff for Player 1 choosing Hawk vs Dove
            d = payoffs_matrix[1, 1, 1]  # Payoff for Player 1 choosing Dove vs Dove

        q_inc = ((c - d) * (1 + 0.5 * b) ** 2) / (((c - d) * (1 + 0.5 * b) ** 2) - ((a - b) * (1 + 0.5 * d) ** 2))
        q_dec = (c - d) / ((c - d) - (a - b))

        if q_inc > 0.5:
            fixed_lr1 = 0.5
        elif q_dec < 0.5:
            fixed_lr1 = 0.01
        else:
            fixed_lr1 = 0.1

        # lr2 = np.random.uniform(0, 0.5)
        lr2 = 0.1
        agent1 = MWAgent(n_actions=payoffs_matrix.shape[0], learning_rate=fixed_lr1, name="Player 1")
        agent2 = MWAgent(n_actions=payoffs_matrix.shape[0], learning_rate=lr2, name="Player 2")
        agent1, agent2 = simulate_game(payoffs_matrix, n_rounds, agent1, agent2)
        results.append(agent1.distribution_history[-1][0])  # Probability of Hawk

    mean_prob_hawk = np.mean(results)

    print(f"Payoff Matrix:")
    for i in range(payoffs_matrix.shape[0]):
        row = []
        for j in range(payoffs_matrix.shape[1]):
            cell = payoffs_matrix[i, j]
            row.append(f"({cell[0]:.2f}, {cell[1]:.2f})")
        print("  ".join(row))

    if fixed_lr1 == 0.01:
        print("\nq_dec < 0.5 scenario => fixed_lr1 = 0.01")
    elif fixed_lr1 == 0.5:
        print("\nq_inc > 0.5 scenario => fixed_lr1 = 0.5")
    else:
        print("\nq_inc <= 0.5 <= q_dec scenario => fixed_lr1 = 0.1")

    print(f"\nAverage probability of Player 1 choosing Hawk in final round: {mean_prob_hawk:.2f}")

    return mean_prob_hawk


def main():
    # Simulate MW Agent vs. MW Agent
    agent1 = MWAgent(n_actions=2, learning_rate=LR1, name="MW Agent 1")
    agent2 = MWAgent(n_actions=2, learning_rate=LR2, name="MW Agent 2")
    agent1, agent2 = simulate_game(BotS_game, N_ROUNDS, agent1, agent2)
    plot_regret(agent1, agent2, "BotS - MW vs MW")
    plot_joint_strategy_matrix(agent1, agent2, BotS_game, labels=("Prize Fight", "Ballet"))
    plot_empirical_joint_distribution(agent1, agent2, BotS_game, labels=("Prize Fight", "Ballet"))
    #
    # # Simulate MW Agent vs. Regret Matching Agent
    # agent1 = RegretMatchingAgent(n_actions=2, name="Regret Matching Player")
    # agent2 = MWAgent(n_actions=2, learning_rate=LR2, name="MW Player")
    # agent1_regret, agent2_regret = simulate_game(chicken_game, N_ROUNDS, agent1, agent2)
    # plot_regret(agent1_regret, agent2_regret, "Hawk-Dove (Chicken Game) - RM vs MW")
    # plot_joint_strategy_matrix(agent1_regret, agent2_regret, chicken_game, labels=("Hawk", "Dove"))
    # plot_empirical_joint_distribution(agent1_regret, agent2_regret, chicken_game, labels=("Hawk", "Dove"))
    #
    # Simulate Regret Matching Agent vs. Regret Matching Agent
    agent1 = RegretMatchingAgent(n_actions=2, name="Regret Matching 1")
    agent2 = RegretMatchingAgent(n_actions=2, name="Regret Matching 2")
    agent1, agent2 = simulate_game(BotS_game, N_ROUNDS, agent1, agent2)
    plot_regret(agent1, agent2, "BotS - RM vs RM")
    plot_joint_strategy_matrix(agent1, agent2, BotS_game, labels=("Prize Fight", "Ballet"))
    plot_empirical_joint_distribution(agent1, agent2, BotS_game, labels=("Prize Fight", "Ballet"))

    # # Simulate MW Agent vs. MW Agent
    # agent1 = ExpMWAgent(n_actions=2, learning_rate=LR1, name="Exp MW Agent")
    # agent2 = MWAgent(n_actions=2, learning_rate=LR2, name="MW Agent")
    # agent1, agent2 = simulate_game(chicken_game, N_ROUNDS, agent1, agent2)
    # plot_regret(agent1, agent2, "Hawk-Dove (Chicken Game) - Exp MW vs MW")
    # plot_joint_strategy_matrix(agent1, agent2, chicken_game, labels=("Hawk", "Dove"))
    # plot_empirical_joint_distribution(agent1, agent2, chicken_game, labels=("Hawk", "Dove"))

    # # Experiment with varying opponent's learning rate
    # experiment_varying_opponent_lr(chicken_game, N_ROUNDS, target_action_index=0, target_action_name="Hawk")

    # # Search for non-pure equilibrium strategies
    # search_for_non_pure_equilibrium(chicken_game, n_repeats=20000)

    # # Simulate fixed vs random learning rate scenario
    # simulate_fixed_vs_random_lr(None, N_ROUNDS, n_repeats=250)
    #
    # for _ in range(10):
    #     # simulate_fixed_vs_random_lr(None, N_ROUNDS, n_repeats=100)


if __name__ == "__main__":
    main()
