import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
from tqdm import tqdm
from matplotlib.ticker import MultipleLocator, FormatStrFormatter

from MWAgent import MWAgent


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


def run_gfp_simulation(
    T: int = 1_000_000,
    v_high: float = 1.0,
    w_low: float = 0.5,
    epsilon: float = 0.05,
    eta_high: float = 0.1,
    eta_low: float = 0.1,
    seed=None
):
    """
    Repeated GFP auction with two multiplicative-weights agents (full-information feedback).
    Returns logs and the 2D joint histogram of bids.
    """
    if seed is not None:
        np.random.seed(seed)

    # Discrete ε-grid of bids (per-click), now from 0.00 to 1.00 in 0.05 steps
    bids = np.round(np.arange(0.0, 1.0 + 1e-9, epsilon), 10)  # 0.00 ... 1.00
    n_actions = len(bids)

    agent_hi = MWAgent(n_actions=n_actions, learning_rate=eta_high, name="High")
    agent_lo = MWAgent(n_actions=n_actions, learning_rate=eta_low,  name="Low")

    bid_samples_hi = []
    bid_samples_lo = []


    for t in tqdm(range(T), desc="GFP simulation"):
        a_hi = agent_hi.choose_action()
        a_lo = agent_lo.choose_action()
        b_hi = bids[a_hi]
        b_lo = bids[a_lo]

        u_all_hi = sp_payoff_vector(v_high, b_lo, bids)
        u_all_lo = sp_payoff_vector(w_low,  b_hi, bids)

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


def plot_bid_histograms(samples_hi, samples_lo, bids_axis, bins=None):
    """
    Plot two histograms: one for Player 1 (high value) and one for Player 2 (low value).
    Both are normalized to probabilities.
    """
    if bins is None:
        # use exact grid bins (aligned to bids_axis)
        eps = float(np.round(bids_axis[1] - bids_axis[0], 12))
        bins = np.r_[bids_axis - eps/2, bids_axis[-1] + eps/2]

    fig, axes = plt.subplots(2, 1, figsize=(7, 6), sharex=True)

    # Player 1 (high value)
    counts_hi, _ = np.histogram(samples_hi, bins=bins)
    pmf_hi = counts_hi / counts_hi.sum()
    centers = (bins[:-1] + bins[1:]) / 2
    axes[0].bar(centers, pmf_hi, width=centers[1] - centers[0], align="center")
    axes[0].set_ylabel("Probability")
    axes[0].set_title("Distribution of Player 1 (High Value) bids")

    # Player 2 (low value)
    counts_lo, _ = np.histogram(samples_lo, bins=bins)
    pmf_lo = counts_lo / counts_lo.sum()
    axes[1].bar(centers, pmf_lo, width=centers[1] - centers[0], align="center", color="orange")
    axes[1].set_ylabel("Probability")
    axes[1].set_title("Distribution of Player 2 (Low Value) bids")
    axes[1].set_xlabel("Bid")

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
    # Settings to mirror the paper’s Figure 6a:
    # - v = 2, w = 1
    # - large T to get a clear empirical density
    out = run_gfp_simulation()

    # Build the joint density figure
    plot_joint_with_marginals(
        out["samples_hi"],
        out["samples_lo"],
        bids_axis=out["bids_axis"],
        title="(a) Joint distribution of bids of agents with v=2 and w=1"
    )

    plot_bid_histograms(out["samples_hi"], out["samples_lo"], out["bids_axis"])

    # ---- Save empirical bids + joint probs to NPZ ----
    bids_axis = out["bids_axis"]
    x = out["samples_hi"]
    y = out["samples_lo"]

    eps = float(np.round(bids_axis[1] - bids_axis[0], 12))
    edges = np.r_[bids_axis - eps/2, bids_axis[-1] + eps/2]

    # joint counts on the ε-grid (one bin per bid), then normalize to probs
    joint_counts, _, _ = np.histogram2d(x, y, bins=[edges, edges])
    joint_probs = joint_counts / joint_counts.sum() if joint_counts.sum() > 0 else joint_counts

    np.savez(
        "empirical_bids_and_joint.npz",
        bids=bids_axis,
        samples_hi=x,
        samples_lo=y,
        joint_probs=joint_probs,
    )
    print("Saved -> empirical_bids_and_joint.npz")



if __name__ == "__main__":
    main()
