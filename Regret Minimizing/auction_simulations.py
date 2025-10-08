import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
from tqdm import tqdm
from matplotlib.ticker import MultipleLocator, FormatStrFormatter

from MWAgent import MWAgent


# -----------------------------
# Auction / environment helpers
# -----------------------------

def fp_payoff_vector(value: float, opp_bid: float, bids: np.ndarray) -> np.ndarray:
    """
    Full-information utility for *every* possible bid in a classic
    2-player First Price Auction (single slot).

    Winner pays their *own* bid.
    Tie is broken uniformly: expected utility = 0.5 * (value - b).
    """
    b = bids

    win_mask = b > opp_bid
    lose_mask = b < opp_bid
    tie_mask = ~(win_mask | lose_mask)

    u = np.zeros_like(b, dtype=float)

    # Win: pay own bid
    u[win_mask] = value - b[win_mask]

    # Lose: utility = 0 (already set)

    # Tie: 50% chance to win and pay own bid
    u[tie_mask] = 0.5 * (value - b[tie_mask])

    return u


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


def sp_payoff_vector(value: float, opp_bid: float, bids: np.ndarray) -> np.ndarray:
    """
    Full-information utility for *every* possible bid in a classic
    2-player Second Price Auction (Vickrey auction).
    """
    b = bids

    win_mask  = b > opp_bid
    lose_mask = b < opp_bid
    tie_mask  = ~(win_mask | lose_mask)

    u = np.zeros_like(b, dtype=float)

    # Win: pay opp_bid (not my own bid!)
    u[win_mask] = value - opp_bid

    # Lose: utility = 0 (already set)

    # Tie: 50% chance to win at price = opp_bid
    u[tie_mask] = 0.5 * (value - opp_bid)

    return u


def build_bid_grid(max_bid: float, epsilon: float, decimals: int = 10) -> np.ndarray:
    steps = int(np.floor((max_bid + 1e-12) / epsilon))
    grid = epsilon * np.arange(steps + 1, dtype=float)  # 0, ε, 2ε, ..., steps*ε ≤ max_bid
    return np.round(np.clip(grid, 0.0, max_bid), decimals)


def run_auction_simulation(
    T: int = 1_000_000,
    v_high: float = 1.0,
    w_low: float = 0.5,
    epsilon: float = 0.01,
    eta_high: float = 0.05,
    eta_low: float = 0.05,
    utility_function=sp_payoff_vector,
    seed=None,
    no_overbidding=False
):
    """
    Repeated auction (SPA/FPA/GFP) with two multiplicative-weights agents (full information).
    Overbidding banned by using per-player bid grids capped at their value.
    """
    if seed is not None:
        np.random.seed(seed)

    if no_overbidding:
        bids_hi = build_bid_grid(v_high, epsilon)  # ⊆ [0, v_high]
        bids_lo = build_bid_grid(w_low, epsilon)  # ⊆ [0, w_low]
    else:
        max_value = max(v_high, w_low)
        bids_hi = build_bid_grid(max_value, epsilon)
        bids_lo = bids_hi.copy()

    agent_hi = MWAgent(n_actions=len(bids_hi), learning_rate=eta_high, name="High")
    agent_lo = MWAgent(n_actions=len(bids_lo), learning_rate=eta_low,  name="Low")

    bid_samples_hi, bid_samples_lo = [], []

    for _ in tqdm(range(T), desc="Auction simulation"):
        a_hi = agent_hi.choose_action()
        a_lo = agent_lo.choose_action()
        b_hi = bids_hi[a_hi]
        b_lo = bids_lo[a_lo]

        # utility vectors computed on each player's *own* action set
        u_all_hi = utility_function(v_high, b_lo, bids_hi)  # shape = len(bids_hi)
        u_all_lo = utility_function(w_low,  b_hi, bids_lo)  # shape = len(bids_lo)

        agent_hi.update(u_all_hi)
        agent_lo.update(u_all_lo)

        bid_samples_hi.append(b_hi)
        bid_samples_lo.append(b_lo)

    return {
        "bids_hi": bids_hi,
        "bids_lo": bids_lo,
        "samples_hi": np.array(bid_samples_hi),
        "samples_lo": np.array(bid_samples_lo),
        "agent_hi": agent_hi,
        "agent_lo": agent_lo,
    }


def plot_bid_histograms(samples_hi, samples_lo, bids_axis_hi, bids_axis_lo):
    # --- Player 1 (high value) ---
    eps_hi = float(np.round(bids_axis_hi[1] - bids_axis_hi[0], 12)) if len(bids_axis_hi) > 1 else 1.0
    bins_hi = np.r_[bids_axis_hi - eps_hi/2, bids_axis_hi[-1] + eps_hi/2]
    counts_hi, _ = np.histogram(samples_hi, bins=bins_hi)
    pmf_hi = counts_hi / counts_hi.sum()
    centers_hi = (bins_hi[:-1] + bins_hi[1:]) / 2

    # --- Player 2 (low value) ---
    eps_lo = float(np.round(bids_axis_lo[1] - bids_axis_lo[0], 12)) if len(bids_axis_lo) > 1 else 1.0
    bins_lo = np.r_[bids_axis_lo - eps_lo/2, bids_axis_lo[-1] + eps_lo/2]
    counts_lo, _ = np.histogram(samples_lo, bins=bins_lo)
    pmf_lo = counts_lo / counts_lo.sum()
    centers_lo = (bins_lo[:-1] + bins_lo[1:]) / 2

    # --- Shared axis limits ---
    max_bid = max(bids_axis_hi[-1], bids_axis_lo[-1])
    max_prob = max(pmf_hi.max(), pmf_lo.max())

    fig, axes = plt.subplots(2, 1, figsize=(7, 6), sharex=True, sharey=True)

    axes[0].bar(centers_hi, pmf_hi, width=eps_hi, align="center")
    axes[0].set_ylabel("Probability")
    axes[0].set_title("Distribution of Player 1 (High Value) bids")

    axes[1].bar(centers_lo, pmf_lo, width=eps_lo, align="center", color="orange")
    axes[1].set_ylabel("Probability")
    axes[1].set_title("Distribution of Player 2 (Low Value) bids")
    axes[1].set_xlabel("Bid")

    # Apply the same limits to both plots
    for ax in axes:
        ax.set_xlim(0, max_bid)
        ax.set_ylim(0, max_prob * 1.05)  # small headroom for aesthetics

    plt.tight_layout()
    plt.show()



def plot_joint_with_marginals(x, y, bids_axis, title="Joint distribution of bids (GFP, v=2, w=1)"):
    """
    Heatmap + marginals using bin EDGES aligned to the discrete ε-grid in bids_axis.
    Marginals show probabilities (PMFs) from empirical samples.
    """
    eps = float(np.round(bids_axis[1] - bids_axis[0], 12))
    # One bin per bid value: edges are centered around grid points
    edges = np.r_[bids_axis - eps/2, bids_axis[-1] + eps/2]
    lo, hi = float(bids_axis[0]), float(bids_axis[-1])

    fig = plt.figure(figsize=(6, 6))
    gs = gridspec.GridSpec(2, 2, width_ratios=[4, 1], height_ratios=[1, 4],
                           wspace=0.05, hspace=0.05)

    ax_histx = plt.subplot(gs[0, 0])
    ax_main  = plt.subplot(gs[1, 0])
    ax_histy = plt.subplot(gs[1, 1])

    # --- Joint empirical distribution (counts normalized to probability) ---
    H, _, _ = np.histogram2d(x, y, bins=[edges, edges])
    H_prob = H / H.sum() if H.sum() > 0 else H
    ax_main.imshow(H_prob.T, origin="lower",
                   extent=[lo, hi, lo, hi], aspect="equal", cmap="Blues")
    ax_main.set_xlim(lo, hi); ax_main.set_ylim(lo, hi)
    ax_main.set_xlabel("Agent 1 bid"); ax_main.set_ylabel("Agent 2 bid")

    # --- Add 0.1 major ticks and grid on the main heatmap ---
    locator = MultipleLocator(0.1)
    ax_main.xaxis.set_major_locator(locator)
    ax_main.yaxis.set_major_locator(locator)
    ax_main.xaxis.set_major_formatter(FormatStrFormatter("%.1f"))

    # --- Top marginal: PMF of Agent 1 ---
    counts_x, _ = np.histogram(x, bins=edges)
    pmf_x = counts_x / counts_x.sum() if counts_x.sum() > 0 else counts_x
    ax_histx.bar(bids_axis, pmf_x, width=eps, align="center")
    ax_histx.set_ylabel("Probability"); ax_histx.set_xticks([])

    # --- Right marginal: PMF of Agent 2 ---
    counts_y, _ = np.histogram(y, bins=edges)
    pmf_y = counts_y / counts_y.sum() if counts_y.sum() > 0 else counts_y
    ax_histy.barh(bids_axis, pmf_y, height=eps, align="center")
    ax_histy.set_xlabel("Probability"); ax_histy.set_xticks([]); ax_histy.set_yticks([])

    fig.suptitle(title, y=0.96)
    plt.show()

def main():

    out = run_auction_simulation(
        T=1_000_000,
        v_high=1.0,
        w_low=0.5,
        epsilon=0.01,
        eta_high=0.01,
        eta_low=0.01,
        utility_function=fp_payoff_vector,
        no_overbidding=True,
    )

    # # Build the joint density figure
    # plot_joint_with_marginals(
    #     out["samples_hi"],
    #     out["samples_lo"],
    #     bids_axis=out["bids_axis"],
    #     title="(a) Joint distribution of bids of agents with v=2 and w=1"
    # )

    plot_bid_histograms(out["samples_hi"], out["samples_lo"], out["bids_hi"], out["bids_lo"])

    # ---- Save empirical bids + joint probs to NPZ ----
    # bids_axis = out["bids_axis"]
    # x = out["samples_hi"]
    # y = out["samples_lo"]

    # eps = float(np.round(bids_axis[1] - bids_axis[0], 12))
    # edges = np.r_[bids_axis - eps/2, bids_axis[-1] + eps/2]

    # # joint counts on the ε-grid (one bin per bid), then normalize to probs
    # joint_counts, _, _ = np.histogram2d(x, y, bins=[edges, edges])
    # joint_probs = joint_counts / joint_counts.sum() if joint_counts.sum() > 0 else joint_counts
    #
    # np.savez(
    #     "empirical_bids_and_joint.npz",
    #     bids=bids_axis,
    #     samples_hi=x,
    #     samples_lo=y,
    #     joint_probs=joint_probs,
    # )
    # print("Saved -> empirical_bids_and_joint.npz")



if __name__ == "__main__":
    main()
