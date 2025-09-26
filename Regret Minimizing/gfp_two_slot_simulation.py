import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
from tqdm import tqdm

from MWAgent import MWAgent  # uses your agent exactly

# -----------------------------
# Auction / environment helpers
# -----------------------------
def gfp_payoff_vector(value: float, opp_bid: float, bids: np.ndarray) -> np.ndarray:
    """
    Full-information utility for *every* possible bid in a 2-slot GFP auction,
    given opponent's realized bid 'opp_bid'.

    Two slots with CTRs: top=1, bottom=0.5. First-price per click.
    If my bid b > opp_bid  -> I win top: utility = (value - b)
    If my bid b < opp_bid  -> I get bottom: utility = 0.5 * (value - b)
    If tie b == opp_bid    -> expected utility = 0.75 * (value - b)  (coin for top)
    """
    b = bids
    win_mask  = b > opp_bid
    lose_mask = b < opp_bid
    tie_mask  = ~win_mask & ~lose_mask  # equal

    u = np.zeros_like(b, dtype=float)
    u[win_mask]  = (value - b[win_mask])            # top slot CTR=1
    u[lose_mask] = 0.5 * (value - b[lose_mask])     # bottom slot CTR=0.5
    u[tie_mask]  = 0.75 * (value - b[tie_mask])     # expected: 1/2*top + 1/2*bottom = 3/4
    return u


def run_gfp_simulation(
    T: int = 1_000_000,
    v_high: float = 2.0,
    w_low: float = 1.0,
    epsilon: float = 0.01,
    eta_high: float = 0.02,
    eta_low: float = 0.02,
    seed=None
):
    """
    Repeated GFP auction with two multiplicative-weights agents (full-information feedback).
    Returns logs and the 2D joint histogram of bids.
    """
    if seed is not None:
        np.random.seed(seed)

    # Discrete ε-grid of bids (per-click)
    bids = np.round(np.arange(0.0, 1.0 + 1e-9, epsilon), 10)  # 0.00 ... 1.00
    n_actions = len(bids)

    agent_hi = MWAgent(n_actions=n_actions, learning_rate=eta_high, name="High")
    agent_lo = MWAgent(n_actions=n_actions, learning_rate=eta_low,  name="Low")

    # Logs (store only what we need for the figure to keep memory reasonable)
    bid_samples_hi = []
    bid_samples_lo = []

    # Main loop
    for t in tqdm(range(T), desc="GFP simulation"):
        # Choose discrete actions (indices), map to numeric bids
        a_hi = agent_hi.choose_action()
        a_lo = agent_lo.choose_action()
        b_hi = bids[a_hi]
        b_lo = bids[a_lo]

        # Full-information payoff vectors against realized opponent bid
        u_all_hi = gfp_payoff_vector(v_high, b_lo, bids)
        u_all_lo = gfp_payoff_vector(w_low,  b_hi, bids)

        # MW updates
        agent_hi.update(u_all_hi)
        agent_lo.update(u_all_lo)


        bid_samples_hi.append(b_hi)
        bid_samples_lo.append(b_lo)

    return {
        "bids_axis": bids,
        "samples_hi": np.array(bid_samples_hi),
        "samples_lo": np.array(bid_samples_lo),
        "agent_hi": agent_hi,
        "agent_lo": agent_lo,
    }

def plot_joint_with_marginals(x, y, bins=100, title="Joint distribution of bids (GFP, v=2, w=1)"):
    """
    Heatmap + marginals.
    Marginals show *probabilities* (discrete PMFs), not densities.
    """
    fig = plt.figure(figsize=(6, 6))
    gs = gridspec.GridSpec(2, 2, width_ratios=[4, 1], height_ratios=[1, 4],
                           wspace=0.05, hspace=0.05)

    ax_histx = plt.subplot(gs[0, 0])
    ax_main  = plt.subplot(gs[1, 0])
    ax_histy = plt.subplot(gs[1, 1])

    # Main 2D histogram (still counts, relative coloring is fine)
    ax_main.hist2d(x, y, bins=bins, range=[[0, 1], [0, 1]], cmap="Blues")
    ax_main.set_xlabel("Agent 1 bid")
    ax_main.set_ylabel("Agent 2 bid")
    ax_main.set_xlim(0, 1)
    ax_main.set_ylim(0, 1)

    # ---- Top marginal: empirical PMF of Agent 1 ----
    counts, _ = np.histogram(x, bins=bins, range=(0, 1))
    pmf = counts / counts.sum()
    centers = np.linspace(0, 1, bins, endpoint=False) + 0.5 / bins
    ax_histx.bar(centers, pmf, width=1/bins, align="center")
    ax_histx.set_xticks([])

    # ---- Right marginal: empirical PMF of Agent 2 ----
    counts, _ = np.histogram(y, bins=bins, range=(0, 1))
    pmf = counts / counts.sum()
    centers = np.linspace(0, 1, bins, endpoint=False) + 0.5 / bins
    ax_histy.barh(centers, pmf, height=1/bins, align="center")
    ax_histy.set_xticks([])
    ax_histy.set_yticks([])

    fig.suptitle(title, y=0.96)
    plt.show()


def main():
    # Settings to mirror the paper’s Figure 6a:
    # - v = 2, w = 1
    # - large T to get a clear empirical density
    out = run_gfp_simulation(
        T=10_000,
        v_high=2.0,
        w_low=1.0,
        epsilon=0.01,
        eta_high=0.01,
        eta_low=0.01,
    )

    # Build the joint density figure
    plot_joint_with_marginals(
        out["samples_hi"],
        out["samples_lo"],
        bins=int(1.0 / 0.01),
        title="(a) Joint distribution of bids of agents with v=2 and w=1"
    )


if __name__ == "__main__":
    main()
