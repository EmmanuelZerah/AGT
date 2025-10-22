import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm

from utils import build_bid_grid, spa_payoff_vector, fpa_payoff_vector, gfpa_payoff_vector
from MWDeterministicAgent import MWDeterministicAgent


def build_payoff_matrices(bids_hi: np.ndarray, bids_lo: np.ndarray,
                          v_high: float, w_low: float,
                          utility_fn_hi, utility_fn_lo=None) -> tuple[np.ndarray, np.ndarray]:
    """
    Returns (U_hi, U_lo) where:
      U_hi[i,j] = payoff to HIGH bidding bids_hi[i] vs LOW bidding bids_lo[j]
      U_lo[j,i] = payoff to LOW  bidding bids_lo[j] vs HIGH bidding bids_hi[i]
    If utility_fn_lo is None we reuse utility_fn_hi with appropriate (value, opp_bid, own_grid).
    """
    n_hi, n_lo = len(bids_hi), len(bids_lo)
    U_hi = np.empty((n_hi, n_lo), dtype=float)
    U_lo = np.empty((n_lo, n_hi), dtype=float)
    for j, b_lo in enumerate(bids_lo):
        U_hi[:, j] = utility_fn_hi(v_high, b_lo, bids_hi)
    util_lo = utility_fn_lo if utility_fn_lo is not None else utility_fn_hi
    for i, b_hi in enumerate(bids_hi):
        U_lo[:, i] = util_lo(w_low, b_hi, bids_lo)
    return U_hi, U_lo


def run_deterministic_simulation(U_hi: np.ndarray, U_lo: np.ndarray,
                                 eta_high: float, eta_low: float,
                                 T: int, log_every: int = 0):
    hi_agent = MWDeterministicAgent(U_hi, learning_rate=eta_high, name="High")
    lo_agent = MWDeterministicAgent(U_lo, learning_rate=eta_low,  name="Low")

    # logs
    log = {
        "steps": [],
        "p_hi": [],   # list of np.ndarray, one per checkpoint
        "p_lo": []
    } if log_every else None

    for t in tqdm(range(1, T + 1)):
        p_hi = hi_agent.distribution
        p_lo = lo_agent.distribution

        hi_agent.update(p_lo)
        lo_agent.update(p_hi)

        if log_every and (t % log_every == 0):
            log["steps"].append(t)
            log["p_hi"].append(p_hi.copy())
            log["p_lo"].append(p_lo.copy())

    return hi_agent, lo_agent, log


def _bar_edges(ax, grid, pmf, color=None, label=None):
    """
    Draw histogram-like bars whose left edges sit exactly on the bid grid.
    Ensures the 0-bin is fully visible and the last bar is not clipped.
    """
    grid = np.asarray(grid, dtype=float)
    bw = (grid[1] - grid[0]) if len(grid) > 1 else 1.0
    ax.bar(grid, pmf, width=bw, align="edge", color=color, label=label, zorder=2)
    ax.set_xlim(grid[0], grid[-1] + bw)   # include full width of the last bar
    ax.set_axisbelow(True)


def plot_final_pmfs(bids_hi, p_hi, bids_lo, p_lo, title="FPA"):
    fig, axes = plt.subplots(2, 1, figsize=(7, 6), sharex=False, sharey=False)

    _bar_edges(axes[0], bids_hi, p_hi)
    axes[0].set_title(f"High player (final PMF) — {title}")
    axes[0].set_ylabel("Probability")

    _bar_edges(axes[1], bids_lo, p_lo, color="orange")
    axes[1].set_title(f"Low player (final PMF) — {title}")
    axes[1].set_ylabel("Probability")
    axes[1].set_xlabel("Bid")

    plt.tight_layout()
    plt.show()


def plot_pmfs_checkpoints(bids_hi, bids_lo, steps, p_hi_list, p_lo_list, max_cols=3, title="PMFs over time"):
    import math
    k = len(steps)
    cols = min(max_cols, k)
    rows = math.ceil(k / cols)

    fig, axes = plt.subplots(rows*2, cols, figsize=(4.8*cols, 3.2*rows))
    axes = np.array(axes).reshape(rows*2, cols)

    for idx, step in enumerate(steps):
        r, c = divmod(idx, cols)

        ax_hi = axes[2*r, c]
        _bar_edges(ax_hi, bids_hi, p_hi_list[idx])
        ax_hi.set_title(f"High — t={step:,}")
        ax_hi.set_ylabel("Prob")

        ax_lo = axes[2*r+1, c]
        _bar_edges(ax_lo, bids_lo, p_lo_list[idx], color="orange")
        ax_lo.set_title(f"Low — t={step:,}")
        ax_lo.set_xlabel("Bid")
        ax_lo.set_ylabel("Prob")


    fig.suptitle(title, y=0.995)
    fig.tight_layout()
    plt.show()

    for idx, step in enumerate(steps):
        plot_joint_from_mixtures(bids_hi, p_hi_list[idx], bids_lo, p_lo_list[idx], title=f"GFPA (v=1, w=1), t={step}")


def plot_joint_from_mixtures(bids_hi, p_hi, bids_lo, p_lo, title="Joint (p_hi ⊗ p_lo)"):
    eps_hi = (bids_hi[1]-bids_hi[0]) if len(bids_hi)>1 else 1.0
    eps_lo = (bids_lo[1]-bids_lo[0]) if len(bids_lo)>1 else 1.0
    edges_x = np.r_[bids_hi - eps_hi/2, bids_hi[-1] + eps_hi/2]
    edges_y = np.r_[bids_lo - eps_lo/2, bids_lo[-1] + eps_lo/2]

    H = np.outer(p_hi, p_lo)  # (n_hi, n_lo), sums to 1
    fig = plt.figure(figsize=(6, 6))
    gs = plt.GridSpec(2, 2, width_ratios=[4,1], height_ratios=[1,4], wspace=0.05, hspace=0.05)
    ax_top = plt.subplot(gs[0,0]); ax_main = plt.subplot(gs[1,0]); ax_right = plt.subplot(gs[1,1])

    ax_main.pcolormesh(edges_x, edges_y, H.T, cmap="Blues", shading="auto")
    ax_main.set_xlabel("High bid"); ax_main.set_ylabel("Low bid")
    ax_main.set_xlim(edges_x[0], edges_x[-1]); ax_main.set_ylim(edges_y[0], edges_y[-1])

    ax_top.bar(bids_hi, p_hi, width=eps_hi, align="center"); ax_top.set_xticks([]); ax_top.set_ylabel("Prob")
    ax_right.barh(bids_lo, p_lo, height=eps_lo, align="center"); ax_right.set_yticks([]); ax_right.set_xlabel("Prob")
    fig.suptitle(title, y=0.96); plt.show()


def plot_argmax_bid_over_time(steps, p_hi_list, p_lo_list, bids_hi, bids_lo, title="Argmax bid over time"):
    """
    Plot the bid (not the probability) with the highest probability mass at each checkpoint.
    """
    if not steps:
        print("No checkpoints to plot.")
        return

    # find which bid has the highest probability at each logged step
    arg_hi = [bids_hi[np.argmax(p)] for p in p_hi_list]
    arg_lo = [bids_lo[np.argmax(p)] for p in p_lo_list]

    # plot the argmax bids over time
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.step(steps, arg_hi, where="post", lw=2, label="High agent", color="tab:blue")
    ax.step(steps, arg_lo, where="post", lw=2, label="Low agent", color="orange")

    ax.set_xlabel("Step")
    ax.set_ylabel("Bid with highest probability")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend()
    plt.tight_layout()
    plt.show()


def main():
    epsilon = 0.01
    v_high, w_low = 1.0, 1.0
    bids_hi = build_bid_grid(v_high, epsilon)
    bids_lo = build_bid_grid(w_low, epsilon)

    U_hi, U_lo = build_payoff_matrices(bids_hi, bids_lo, v_high, w_low, gfpa_payoff_vector)

    hi, lo, log = run_deterministic_simulation(
        U_hi, U_lo,
        eta_high=0.01, eta_low=0.01,
        T=1_000_000,
        log_every=1_000,
    )

    # Final PMFs (as before)
    p_hi, p_lo = hi.distribution, lo.distribution
    plot_final_pmfs(bids_hi, p_hi, bids_lo, p_lo, title="GFPA (v=2, w=1)")
    plot_joint_from_mixtures(bids_hi, p_hi, bids_lo, p_lo, title="GFPA (v=2, w=1)")

    # New: PMFs at checkpoints
    if log is not None and log["steps"]:
        # plot_pmfs_checkpoints(
        #     bids_hi, bids_lo,
        #     steps=log["steps"],
        #     p_hi_list=log["p_hi"],
        #     p_lo_list=log["p_lo"],
        #     max_cols=3,
        #     title="GFPA (v=2, w=1) — PMFs every 100k steps"
        # )

        plot_argmax_bid_over_time(
            steps=log["steps"],
            p_hi_list=log["p_hi"],
            p_lo_list=log["p_lo"],
            bids_hi=bids_hi,
            bids_lo=bids_lo,
            title="Most probable bid over time"
        )


if __name__ == "__main__":
    main()
