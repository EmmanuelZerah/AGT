# simulate_mw_duopoly.py
import numpy as np
from dataclasses import dataclass
from typing import Tuple, List
from matplotlib import pyplot as plt
from tqdm import tqdm
from algorithms.ContextualPolicy import ContextualPolicy
from algorithms.EXP3Agent import EXP3Agent
from algorithms.MWAgent import MWAgent

# -----------------------------
# Economic environment (logit)
# -----------------------------


@dataclass
class LogitParams:
    a0: float = 0.0       # "outside good" quality (inverse demand shifter)
    a1: float = 2.0       # quality of firm 1's product
    a2: float = 2.0       # quality of firm 2's product
    c1: float = 1.0       # marginal cost firm 1
    c2: float = 1.0       # marginal cost firm 2
    mu: float = 0.25      # horizontal differentiation (lower -> tougher competition)


def logit_shares(p1: float, p2: float, P: LogitParams) -> Tuple[float, float, float]:
    """Return (q1, q2, q0) given prices and parameters."""
    u1 = (P.a1 - p1) / P.mu
    u2 = (P.a2 - p2) / P.mu
    u0 = P.a0 / P.mu
    den = np.exp(u1) + np.exp(u2) + np.exp(u0)
    q1 = np.exp(u1) / den
    q2 = np.exp(u2) / den
    q0 = np.exp(u0) / den
    return q1, q2, q0


def profits(p1: float, p2: float, P: LogitParams) -> Tuple[float, float]:
    q1, q2, _ = logit_shares(p1, p2, P)
    return (p1 - P.c1) * q1, (p2 - P.c2) * q2

# --------------------------------------
# Static benchmarks: p^N and p^M (cont)
# --------------------------------------


def best_response_against(p_other: float, for_player: int, grid: np.ndarray, P: LogitParams) -> float:
    """Discrete best response price (argmax profit) vs opponent price p_other."""
    if for_player == 1:
        profs = np.array([profits(p, p_other, P)[0] for p in grid])
    else:
        profs = np.array([profits(p_other, p, P)[1] for p in grid])
    return grid[int(np.argmax(profs))]


def nash_prices_continuous(P, coarse_min, coarse_max, n_grid=1500,
                           max_iter=10000, tol=1e-8, gamma=0.5):
    """
    Compute a 'continuous' Nash via iterated discrete best-responses on a fine grid
    spanning a wide, reasonable range.
    """
    grid = np.linspace(coarse_min, coarse_max, n_grid)
    p1 = P.c1 + 0.5
    p2 = P.c2 + 0.5
    for _ in range(max_iter):
        p1_br = best_response_against(p2, for_player=1, grid=grid, P=P)
        p2_br = best_response_against(p1_br, for_player=2, grid=grid, P=P)
        p1_new = (1-gamma)*p1 + gamma*p1_br
        p2_new = (1-gamma)*p2 + gamma*p2_br
        if abs(p1_new - p1) + abs(p2_new - p2) < tol:
            return float(p1_new), float(p2_new)
        p1, p2 = p1_new, p2_new
    return float(p1), float(p2)


def _refine_window(lo, hi, center, shrink=0.35):
    span = hi - lo
    half = 0.5 * shrink * span
    new_lo = max(lo, center - half)
    new_hi = min(hi, center + half)
    return new_lo, new_hi


def monopoly_prices_continuous(P, coarse_min, coarse_max, n_grid=600, n_refine=3):
    """Joint profit maximization over the same fine grid."""
    lo, hi = coarse_min, coarse_max
    p1_star, p2_star = None, None
    for _ in range(n_refine):
        grid = np.linspace(lo, hi, n_grid)
        best_val = -1.0
        best = (grid[0], grid[0])
        for p1 in grid:
            p2s = grid
            q1, q2, _ = logit_shares(p1, p2s, P)
            pi1 = (p1 - P.c1) * q1
            pi2 = (p2s - P.c2) * q2
            j = int(np.argmax(pi1 + pi2))
            val = float(pi1[j] + pi2[j])
            if val > best_val:
                best_val = val
                best = (float(p1), float(p2s[j]))
        p1_star, p2_star = best
        # refine window around best of both coordinates
        lo, hi = _refine_window(coarse_min, coarse_max, 0.5 * (p1_star + p2_star))
        # optional: also check if best hits boundary; if so, expand:
        if p1_star >= hi - 1e-9 or p2_star >= hi - 1e-9:
            hi += 2.0
        if p1_star <= lo + 1e-9 or p2_star <= lo + 1e-9:
            lo = max(coarse_min - 2.0, 0.0)
    return p1_star, p2_star


# -------------------------------------
# Action grid built around p^N and p^M
# -------------------------------------

def precompute_profit_caps(grid: np.ndarray, P: LogitParams) -> tuple[float, float]:
    """
    Upper bounds U1, U2 on per-period profits over grid×grid.
    Used to normalize rewards into [0,1] for EXP3.
    """
    G = grid
    U1 = -np.inf
    U2 = -np.inf

    # Max profit for firm 1 across (p1, p2 in grid)
    for p1 in G:
        q1, q2, _ = logit_shares(p1, G, P)  # vectorized over p2
        pi1 = (p1 - P.c1) * q1
        U1 = max(U1, float(pi1.max()))

    # Max profit for firm 2 across (p1, p2 in grid)
    for p2 in G:
        q1, q2, _ = logit_shares(G, p2, P)  # vectorized over p1
        pi2 = (p2 - P.c2) * q2
        U2 = max(U2, float(pi2.max()))

    # guard against degenerate cases
    return max(U1, 1e-12), max(U2, 1e-12)


def build_action_grid(pN: Tuple[float, float], pM: Tuple[float, float], m: int = 15, xi: float = 0.1) -> np.ndarray:
    """
    Follow the paper: m equally spaced points in [p^N - xi*(p^M - p^N), p^M + xi*(p^M - p^N)].
    With symmetry we center around the average of each pair.
    """
    # If asymmetric, use midpoint of each pair to get a scalar center, then a scalar span
    center_N = 0.5 * (pN[0] + pN[1])
    center_M = 0.5 * (pM[0] + pM[1])
    span = center_M - center_N
    lo = center_N - xi * span
    hi = center_M + xi * span
    return np.linspace(lo, hi, m)

# ------------------------------------------------
# Discrete-stage benchmarks (to compute Δ properly)
# ------------------------------------------------

def nash_on_discrete_grid(grid: np.ndarray, P: LogitParams, max_iter: int = 200, tol: float = 1e-8) -> Tuple[float, float, float, float]:
    """Return (p1N, p2N, pi1N, pi2N) Nash best-response fixed point on the discrete action set."""
    # Start at grid midpoints
    p1 = grid[len(grid)//2]
    p2 = grid[len(grid)//2]
    for _ in range(max_iter):
        p1_new = best_response_against(p2, 1, grid, P)
        p2_new = best_response_against(p1_new, 2, grid, P)
        if abs(p1_new - p1) + abs(p2_new - p2) < tol:
            pi1N, pi2N = profits(p1_new, p2_new, P)
            return float(p1_new), float(p2_new), float(pi1N), float(pi2N)
        p1, p2 = p1_new, p2_new
    pi1N, pi2N = profits(p1, p2, P)
    return float(p1), float(p2), float(pi1N), float(pi2N)


def monopoly_on_discrete_grid(grid: np.ndarray, P: LogitParams) -> Tuple[float, float, float, float]:
    """Return (p1M, p2M, pi1M, pi2M) joint-profit maximizer on the discrete action set."""
    best_val = -1.0
    best_tuple = (grid[0], grid[0], 0.0, 0.0)
    for p1 in grid:
        # vectorized search over p2
        p2s = grid
        q1, q2, _ = logit_shares(p1, p2s, P)
        pi1 = (p1 - P.c1) * q1
        pi2 = (p2s - P.c2) * q2
        tot = pi1 + pi2
        j = int(np.argmax(tot))
        if tot[j] > best_val:
            best_val = float(tot[j])
            best_tuple = (float(p1), float(p2s[j]), float(pi1[j]), float(pi2[j]))
    return best_tuple

# -------------------------
# Plotting utilities
# -------------------------

def _rolling_mean(x: np.ndarray, win: int) -> np.ndarray:
    if win is None or win <= 1:
        return x
    win = int(win)
    if win >= len(x):
        return np.full_like(x, np.mean(x))
    kernel = np.ones(win, dtype=float) / win
    y = np.convolve(x, kernel, mode='same')
    return y


def plot_prices(prices_1, prices_2, pN_cont, pM_cont, title="Price paths",
                smooth=None, max_points=20000):
    """
    Plot the two firms' chosen prices over time. Also draw horizontal lines at p^N and p^M.
    - smooth: optional rolling window size (e.g., 5000) to smooth jagged bandit series
    - max_points: downsample to at most this many points for speed
    """
    P1 = np.asarray(prices_1, dtype=float)
    P2 = np.asarray(prices_2, dtype=float)
    T  = len(P1)

    # Downsample uniformly if very long
    step = max(1, T // max_points)
    idx  = np.arange(0, T, step)
    t    = idx + 1
    p1   = P1[idx]
    p2   = P2[idx]

    # Optional smoothing (applied before downsampling if desired)
    if smooth is not None and smooth > 1:
        P1s = _rolling_mean(P1, smooth)
        P2s = _rolling_mean(P2, smooth)
        p1s = P1s[idx]
        p2s = P2s[idx]
    else:
        p1s = p1
        p2s = p2

    # Horizontal benchmarks (use mean of the two in case of asymmetry)
    pN_line = 0.5 * (float(pN_cont[0]) + float(pN_cont[1]))
    pM_line = 0.5 * (float(pM_cont[0]) + float(pM_cont[1]))

    plt.figure(figsize=(11, 5.5))
    # light raw traces
    if smooth:
        alpha = 0.35
    else:
        alpha = 1
    plt.plot(t, p1, alpha=alpha, linewidth=0.8, label="Firm 1 (raw)")
    plt.plot(t, p2, alpha=alpha, linewidth=0.8, label="Firm 2 (raw)")
    # thicker smoothed traces
    if smooth:
        plt.plot(t, p1s, linewidth=2.0, label=f"Firm 1 (rolling mean, w={smooth})")
        plt.plot(t, p2s, linewidth=2.0, label=f"Firm 2 (rolling mean, w={smooth})")

    # Benchmarks
    plt.axhline(pN_line, linestyle="--", linewidth=2, label=r"$p^N$")
    plt.axhline(pM_line, linestyle=":",  linewidth=2, label=r"$p^M$")

    plt.title(title)
    plt.xlabel("Round")
    plt.ylabel("Price")
    plt.legend(loc="best")
    plt.tight_layout()
    plt.show()

# -------------------------
# simulations
# -------------------------


@dataclass
class SimConfig:
    T: int = 1_000_000           # repetitions (MW has built-in exploration via its randomization)
    m: int = 15                  # number of discrete actions (price points)
    xi: float = 0.10             # range extension around [p^N, p^M]
    eta1: float = 0.10           # MW learning rate for player 1
    eta2: float = 0.10           # MW learning rate for player 2
    seed: int = 42               # RNG seed for reproducibility
    log_every: int = 100_000     # progress prints


@dataclass
class BanditConfig:
    T: int = 2_000_000        # bandit learns slower than full-info MW
    m: int = 15
    xi: float = 0.10
    eta1: float = 0.07        # EXP3 learning rate (≈ gamma)
    eta2: float = 0.07
    gamma1: float = 0.07      # EXP3 exploration
    gamma2: float = 0.07
    seed: int = 123
    log_every: int = 100_000


@dataclass
class ContextBanditConfig:
    T: int = 2_000_000
    m: int = 15
    xi: float = 0.10
    eta1: float = 0.07
    eta2: float = 0.07
    gamma1: float = 0.07
    gamma2: float = 0.07
    seed: int = 123
    log_every: int = 100_000
    # Optional annealing for exploration (keep small floor)
    anneal_gamma: bool = False
    gamma_floor: float = 0.01
    gamma_power: float = 0.2   # γ_t = max(floor, γ0 * t^{-power})


def payoff_vector_for_player(player: int, opp_price: float, grid: np.ndarray, P: LogitParams) -> np.ndarray:
    """Full-information bandit feedback: payoff for each feasible action against the realized opponent price."""
    if player == 1:
        # vectorize over my price grid
        my = grid
        q1, _, _ = logit_shares(my, opp_price, P)
        return (my - P.c1) * q1
    else:
        my = grid
        _, q2, _ = logit_shares(opp_price, my, P)
        return (my - P.c2) * q2


def run_simulation(P: LogitParams, C: SimConfig):
    if C.seed is not None:
        np.random.seed(C.seed)

    # 1) Find wide-range continuous p^N and p^M to build action grid the same way as the paper
    coarse_min = min(P.c1, P.c2)       # cannot profit below cost; we’ll start there
    coarse_max = max(P.c1, P.c2) + 4.0 # generous upper bound (works for baseline)
    pN_cont = nash_prices_continuous(P, coarse_min, coarse_max)
    pM_cont = monopoly_prices_continuous(P, coarse_min, coarse_max)

    # 2) Action grid around those
    grid = build_action_grid(pN_cont, pM_cont, m=C.m, xi=C.xi)

    # 3) Static benchmarks on the discrete set (used for Δ)
    p1N, p2N, pi1N, pi2N = nash_on_discrete_grid(grid, P)
    p1M, p2M, pi1M, pi2M = monopoly_on_discrete_grid(grid, P)
    piN_bar = 0.5 * (pi1N + pi2N)
    piM_bar = 0.5 * (pi1M + pi2M)

    # 4) Two MW agents over the discrete price grid
    #    Your MWAgent samples actions according to softmax(log_weights) and updates via log(1 + eta * payoffs)
    #    (see file). :contentReference[oaicite:3]{index=3}
    agent1 = MWAgent(n_actions=len(grid), learning_rate=C.eta1, name="MW-1")
    agent2 = MWAgent(n_actions=len(grid), learning_rate=C.eta2, name="MW-2")

    # 5) Run repeated play
    prices_1: List[float] = []
    prices_2: List[float] = []
    profs_1: List[float]  = []
    profs_2: List[float]  = []

    for t in tqdm(range(1, C.T + 1), desc="Simulating MW duopoly"):
        a1 = agent1.choose_action()
        a2 = agent2.choose_action()
        p1, p2 = float(grid[a1]), float(grid[a2])

        pi1, pi2 = profits(p1, p2, P)

        # Full-information feedback vectors (all hypothetical payoffs vs realized opp price)
        pay1 = payoff_vector_for_player(1, opp_price=p2, grid=grid, P=P)
        pay2 = payoff_vector_for_player(2, opp_price=p1, grid=grid, P=P)

        agent1.update(pay1)
        agent2.update(pay2)

        # record
        prices_1.append(p1)
        prices_2.append(p2)
        profs_1.append(pi1)
        profs_2.append(pi2)

        if (t % C.log_every) == 0:
            avg_p = 0.5 * (np.mean(prices_1[-C.log_every:]) + np.mean(prices_2[-C.log_every:]))
            avg_pi = 0.5 * (np.mean(profs_1[-C.log_every:]) + np.mean(profs_2[-C.log_every:]))
            print(f"[t={t:,}] avg price (last block) ≈ {avg_p:.3f}, avg profit per firm ≈ {avg_pi:.4f}")

    # 6) Post-stats (ignore burn-in to mimic “limit behavior”)
    burn = C.T // 5
    P1 = np.array(prices_1[burn:], dtype=float)
    P2 = np.array(prices_2[burn:], dtype=float)
    Pi1 = np.array(profs_1[burn:], dtype=float)
    Pi2 = np.array(profs_2[burn:], dtype=float)

    avg_price = 0.5 * (P1.mean() + P2.mean())
    avg_profit = 0.5 * (Pi1.mean() + Pi2.mean())

    # Normalized profit gain Δ = (π - π^N) / (π^M - π^N)
    Delta = (avg_profit - piN_bar) / max(1e-12, (piM_bar - piN_bar))

    out = {
        "grid": grid,
        "pN_cont": pN_cont,
        "pM_cont": pM_cont,
        "pN_discrete": (p1N, p2N),
        "pM_discrete": (p1M, p2M),
        "piN_bar": piN_bar,
        "piM_bar": piM_bar,
        "avg_price": avg_price,
        "avg_profit": avg_profit,
        "Delta": float(Delta),
        "agent1_regret_Tavg": agent1.regrets[-1] if agent1.regrets else 0.0,
        "agent2_regret_Tavg": agent2.regrets[-1] if agent2.regrets else 0.0,
        "prices_1": prices_1,
        "prices_2": prices_2
    }

    return out


def run_simulation_bandit(P: LogitParams, C: BanditConfig):
    # 1) build the action grid using your existing “continuous” benchmarks
    coarse_min = min(P.c1, P.c2)
    coarse_max = max(P.c1, P.c2) + 4.0
    pN_cont = nash_prices_continuous(P, coarse_min, coarse_max)
    pM_cont = monopoly_prices_continuous(P, coarse_min, coarse_max)
    grid = build_action_grid(pN_cont, pM_cont, m=C.m, xi=C.xi)

    # 2) discrete benchmarks for Δ (exactly as before)
    p1N, p2N, pi1N, pi2N = nash_on_discrete_grid(grid, P)
    p1M, p2M, pi1M, pi2M = monopoly_on_discrete_grid(grid, P)
    piN_bar = 0.5 * (pi1N + pi2N)
    piM_bar = 0.5 * (pi1M + pi2M)

    # 3) precompute profit caps (for reward normalization)
    U1, U2 = precompute_profit_caps(grid, P)

    # 4) two EXP3 agents (bandit feedback)
    agent1 = EXP3Agent(n_actions=len(grid), learning_rate=C.eta1, gamma=C.gamma1, seed=(None if C.seed is None else C.seed+1), name="EXP3-1")
    agent2 = EXP3Agent(n_actions=len(grid), learning_rate=C.eta2, gamma=C.gamma2, seed=(None if C.seed is None else C.seed+2), name="EXP3-2")

    prices_1: List[float] = []
    prices_2: List[float] = []
    profs_1:  List[float] = []
    profs_2:  List[float] = []

    rng = np.random.default_rng(C.seed)

    for t in tqdm(range(1, C.T + 1), desc="Simulating EXP3 duopoly (bandit)"):
        # choose actions from current distributions
        a1 = agent1.choose_action()
        a2 = agent2.choose_action()
        p1, p2 = float(grid[a1]), float(grid[a2])

        # observe ONLY chosen actions' payoffs (bandit feedback)
        pi1, pi2 = profits(p1, p2, P)

        # normalize to [0,1] for EXP3 stability/guarantees
        r1 = float(np.clip(pi1 / U1, 0.0, 1.0))
        r2 = float(np.clip(pi2 / U2, 0.0, 1.0))

        # importance-weighted updates happen inside your EXP3Agent
        agent1.update(a1, r1)
        agent2.update(a2, r2)

        # record for stats
        prices_1.append(p1); prices_2.append(p2)
        profs_1.append(pi1); profs_2.append(pi2)

        if (t % C.log_every) == 0:
            avg_p_block  = 0.5 * (np.mean(prices_1[-C.log_every:]) + np.mean(prices_2[-C.log_every:]))
            avg_pi_block = 0.5 * (np.mean(profs_1[-C.log_every:])  + np.mean(profs_2[-C.log_every:]))
            print(f"[t={t:,}] avg price≈{avg_p_block:.3f}, avg profit≈{avg_pi_block:.5f}")

    # 5) post-stats (ignore burn-in to mimic limit behavior)
    burn = C.T // 5
    P1  = np.array(prices_1[burn:], dtype=float)
    P2  = np.array(prices_2[burn:], dtype=float)
    Pi1 = np.array(profs_1[burn:],  dtype=float)
    Pi2 = np.array(profs_2[burn:],  dtype=float)

    avg_price  = 0.5 * (P1.mean() + P2.mean())
    avg_profit = 0.5 * (Pi1.mean() + Pi2.mean())
    Delta = (avg_profit - piN_bar) / max(1e-12, (piM_bar - piN_bar))

    return {
        "grid": grid,
        "pN_cont": pN_cont, "pM_cont": pM_cont,
        "pN_discrete": (p1N, p2N), "pM_discrete": (p1M, p2M),
        "piN_bar": piN_bar, "piM_bar": piM_bar,
        "avg_price": avg_price, "avg_profit": avg_profit, "Delta": float(Delta),
        "U1": U1, "U2": U2,
        "prices_1": prices_1,
        "prices_2": prices_2
    }


def state_key_from_actions(a1_prev, a2_prev):
    if a1_prev is None or a2_prev is None:
        return ("START",)
    return int(a1_prev), int(a2_prev)


def run_simulation_contextual_mw(P: LogitParams, C: SimConfig):
    # same grid & benchmarks as your current pipeline
    coarse_min = min(P.c1, P.c2)
    coarse_max = max(P.c1, P.c2) + 4.0
    pN_cont = nash_prices_continuous(P, coarse_min, coarse_max)
    pM_cont = monopoly_prices_continuous(P, coarse_min, coarse_max)
    grid = build_action_grid(pN_cont, pM_cont, m=C.m, xi=C.xi)

    p1N, p2N, pi1N, pi2N = nash_on_discrete_grid(grid, P)
    p1M, p2M, pi1M, pi2M = monopoly_on_discrete_grid(grid, P)
    piN_bar = 0.5 * (pi1N + pi2N); piM_bar = 0.5 * (pi1M + pi2M)

    # contextual policies for each player (separate MW per state)
    def mk1(): return MWAgent(n_actions=len(grid), learning_rate=C.eta1, name="MW-1")
    def mk2(): return MWAgent(n_actions=len(grid), learning_rate=C.eta2, name="MW-2")
    pol1 = ContextualPolicy(len(grid), mk1, seed=C.seed, name="CTX-MW-1")
    pol2 = ContextualPolicy(len(grid), mk2, seed=C.seed+1 if C.seed is not None else None, name="CTX-MW-2")

    prices_1, prices_2, profs_1, profs_2 = [], [], [], []
    a1_prev = a2_prev = None

    for t in tqdm(range(1, C.T + 1), desc="Contextual MW Duopoly (k=1)"):
        s = state_key_from_actions(a1_prev, a2_prev)
        a1 = pol1.act(s)
        a2 = pol2.act(s)
        p1, p2 = float(grid[a1]), float(grid[a2])

        # realized payoff
        pi1, pi2 = profits(p1, p2, P)

        # full-information payoff vectors vs realized opponent price
        # these are the learning signals for the *policy used in state s*
        my = grid
        q1, _, _ = logit_shares(my, p2, P); r1_vec = (my - P.c1) * q1
        _, q2, _ = logit_shares(p1, my, P); r2_vec = (my - P.c2) * q2

        pol1.update_full_info(s, r1_vec)
        pol2.update_full_info(s, r2_vec)

        prices_1.append(p1); prices_2.append(p2)
        profs_1.append(pi1); profs_2.append(pi2)

        a1_prev, a2_prev = a1, a2

        if (t % C.log_every) == 0:
            avg_p = 0.5*(np.mean(prices_1[-C.log_every:]) + np.mean(prices_2[-C.log_every:]))
            avg_pi = 0.5*(np.mean(profs_1[-C.log_every:]) + np.mean(profs_2[-C.log_every:]))
            print(f"[t={t:,}] avg price≈{avg_p:.3f}, avg profit≈{avg_pi:.5f}")

    burn = C.T // 5
    P1 = np.array(prices_1[burn:], float); P2 = np.array(prices_2[burn:], float)
    Pi1 = np.array(profs_1[burn:], float); Pi2 = np.array(profs_2[burn:], float)
    avg_price = 0.5*(P1.mean()+P2.mean()); avg_profit = 0.5*(Pi1.mean()+Pi2.mean())
    Delta = (avg_profit - piN_bar) / max(1e-12, (piM_bar - piN_bar))

    return dict(
        grid=grid, pN_cont=pN_cont, pM_cont=pM_cont,
        pN_discrete=(p1N, p2N), pM_discrete=(p1M, p2M),
        piN_bar=piN_bar, piM_bar=piM_bar,
        avg_price=avg_price, avg_profit=avg_profit, Delta=float(Delta),
        prices_1=prices_1, prices_2=prices_2
    )


def run_simulation_contextual_exp3(P: LogitParams, C: ContextBanditConfig):
    # 1) grid & continuous benchmarks
    coarse_min = min(P.c1, P.c2)
    coarse_max = max(P.c1, P.c2) + 4.0
    pN_cont = nash_prices_continuous(P, coarse_min, coarse_max)
    pM_cont = monopoly_prices_continuous(P, coarse_min, coarse_max)
    grid = build_action_grid(pN_cont, pM_cont, m=C.m, xi=C.xi)

    # 2) discrete benchmarks for Δ
    p1N, p2N, pi1N, pi2N = nash_on_discrete_grid(grid, P)
    p1M, p2M, pi1M, pi2M = monopoly_on_discrete_grid(grid, P)
    piN_bar = 0.5 * (pi1N + pi2N)
    piM_bar = 0.5 * (pi1M + pi2M)

    # 3) normalize rewards to [0,1]
    U1, U2 = precompute_profit_caps(grid, P)

    # 4) contextual policies: one EXP3 per state
    def mk1():
        return EXP3Agent(n_actions=len(grid), learning_rate=C.eta1, gamma=C.gamma1, seed=None, name="EXP3-1")
    def mk2():
        return EXP3Agent(n_actions=len(grid), learning_rate=C.eta2, gamma=C.gamma2, seed=None, name="EXP3-2")

    pol1 = ContextualPolicy(len(grid), mk1, seed=C.seed, name="CTX-EXP3-1")
    pol2 = ContextualPolicy(len(grid), mk2, seed=(C.seed + 1) if C.seed is not None else None, name="CTX-EXP3-2")

    prices_1, prices_2, profs_1, profs_2 = [], [], [], []
    a1_prev = a2_prev = None

    # optional annealing helper
    def _anneal(g0, t):
        if not C.anneal_gamma:
            return g0
        return max(C.gamma_floor, g0 * (t ** (-C.gamma_power)))

    for t in tqdm(range(1, C.T + 1), desc="Contextual EXP3 Duopoly (k=1)"):
        s = state_key_from_actions(a1_prev, a2_prev)

        # If you want annealing: reach into the per-state agent and update gamma before acting
        if C.anneal_gamma:
            # gentle: only if policy already exists (avoid constructing it just to set gamma)
            if s in pol1.policies:
                pol1.policies[s].gamma = _anneal(pol1.policies[s].gamma, t)
            if s in pol2.policies:
                pol2.policies[s].gamma = _anneal(pol2.policies[s].gamma, t)

        a1 = pol1.act(s)
        a2 = pol2.act(s)
        p1, p2 = float(grid[a1]), float(grid[a2])

        pi1, pi2 = profits(p1, p2, P)

        # bandit feedback: normalize to [0,1]
        r1 = float(np.clip(pi1 / U1, 0.0, 1.0))
        r2 = float(np.clip(pi2 / U2, 0.0, 1.0))

        # importance-weighted updates happen inside EXP3Agent
        pol1.update_bandit(s, a1, r1)
        pol2.update_bandit(s, a2, r2)

        prices_1.append(p1); prices_2.append(p2)
        profs_1.append(pi1);  profs_2.append(pi2)

        a1_prev, a2_prev = a1, a2

        if (t % C.log_every) == 0:
            avg_p = 0.5 * (np.mean(prices_1[-C.log_every:]) + np.mean(prices_2[-C.log_every:]))
            avg_pi = 0.5 * (np.mean(profs_1[-C.log_every:]) + np.mean(profs_2[-C.log_every:]))
            print(f"[t={t:,}] avg price≈{avg_p:.3f}, avg profit≈{avg_pi:.5f}")

    # 5) post-stats (ignore burn-in)
    burn = C.T // 5
    P1 = np.array(prices_1[burn:], float); P2 = np.array(prices_2[burn:], float)
    Pi1 = np.array(profs_1[burn:],  float); Pi2 = np.array(profs_2[burn:],  float)

    avg_price  = 0.5 * (P1.mean() + P2.mean())
    avg_profit = 0.5 * (Pi1.mean() + Pi2.mean())
    Delta = (avg_profit - piN_bar) / max(1e-12, (piM_bar - piN_bar))

    return dict(
        grid=grid, pN_cont=pN_cont, pM_cont=pM_cont,
        pN_discrete=(p1N, p2N), pM_discrete=(p1M, p2M),
        piN_bar=piN_bar, piM_bar=piM_bar,
        avg_price=avg_price, avg_profit=avg_profit, Delta=float(Delta),
        U1=U1, U2=U2,
        prices_1=prices_1, prices_2=prices_2
    )


def full_info_simulation():
    # Paper's baseline params: c_i=1, a_i - c_i = 1 => a_i = 2, a0 = 0, mu = 1/4
    P = LogitParams(a0=0.0, a1=2.0, a2=2.0, c1=1.0, c2=1.0, mu=0.1)

    # Feel free to reduce T while iterating, then scale up
    C = SimConfig(
        T=200000,  # try 200_000 for quick test; use 1e6+ for “stable” behavior
        m=15,
        xi=0.1,
        eta1=0.05,
        eta2=0.05,
        seed=None,
        log_every=100_000
    )

    res = run_simulation(P, C)

    print("\n==== STATIC BENCHMARKS ====")
    print(f"p^N: ({res['pN_cont'][0]:.3f}, {res['pN_cont'][1]:.3f}),  "
          f"π̄^N ≈ {res['piN_bar']:.5f}")
    print(f"p^M: ({res['pM_cont'][0]:.3f}, {res['pM_cont'][1]:.3f}),  "
          f"π̄^M ≈ {res['piM_bar']:.5f}")

    print("\n==== MW DYNAMICS (post burn-in averages) ====")
    print(f"Average price (firms’ mean): {res['avg_price']:.3f}")
    print(f"Average profit per firm:     {res['avg_profit']:.5f}")
    print(f"Δ (normalized profit gain):  {res['Delta']:.3f}")

    print("\n==== No-regret diagnostics ====")
    print(f"Time-avg regret (MW-1): {res['agent1_regret_Tavg']:.6f}")
    print(f"Time-avg regret (MW-2): {res['agent2_regret_Tavg']:.6f}")

    plot_prices(
        res["prices_1"], res["prices_2"],
        res["pN_cont"], res["pM_cont"],
        title="MW (full-information) — price paths",
        max_points=20000
    )


def bandit_simulation():
    P = LogitParams(a0=0.0, a1=2.0, a2=2.0, c1=1.0, c2=1.0, mu=0.25)

    # ---- EXP3 (bandit) run ----
    C_b = BanditConfig(
        T=2_000_000,  # bandit needs longer
        m=15,
        xi=0.10,
        eta1=0.05, eta2=0.05,
        gamma1=0.05, gamma2=0.05,
        seed=None,
        log_every=100_000
    )
    res_b = run_simulation_bandit(P, C_b)

    print("\n==== STATIC BENCHMARKS ====")
    print(f"p^N: ({res_b['pN_cont'][0]:.3f}, {res_b['pN_cont'][1]:.3f}),  "
          f"π̄^N ≈ {res_b['piN_bar']:.5f}")
    print(f"p^M: ({res_b['pM_cont'][0]:.3f}, {res_b['pM_cont'][1]:.3f}),  "
          f"π̄^M ≈ {res_b['piM_bar']:.5f}")

    print("\n==== EXP3 DYNAMICS (post burn-in averages) ====")
    print(f"Average price (firms’ mean): {res_b['avg_price']:.3f}")
    print(f"Average profit per firm:     {res_b['avg_profit']:.5f}")
    print(f"Δ (normalized profit gain):  {res_b['Delta']:.3f}")

    plot_prices(
        res_b["prices_1"], res_b["prices_2"],
        res_b["pN_cont"], res_b["pM_cont"],
        title="EXP3 (bandit) — price paths",
        smooth=5000,          # try 2,000–10,000; None for raw
        max_points=20000      # plots up to ~20k points for speed
    )


def mw_contextual_simulation():
    # Paper's baseline params: c_i=1, a_i - c_i = 1 => a_i = 2, a0 = 0, mu = 1/4
    P = LogitParams(a0=0.0, a1=2.0, a2=2.0, c1=1.0, c2=1.0, mu=0.25)

    # Feel free to reduce T while iterating, then scale up
    C = SimConfig(
        T=2_000_000,  # try 200_000 for quick test; use 1e6+ for “stable” behavior
        m=15,
        xi=0.10,
        eta1=0.05,
        eta2=0.05,
        seed=None,
        log_every=100_000
    )

    res = run_simulation_contextual_mw(P, C)

    print("\n==== STATIC BENCHMARKS ====")
    print(f"p^N: ({res['pN_cont'][0]:.3f}, {res['pN_cont'][1]:.3f}),  "
          f"π̄^N ≈ {res['piN_bar']:.5f}")
    print(f"p^M: ({res['pM_cont'][0]:.3f}, {res['pM_cont'][1]:.3f}),  "
          f"π̄^M ≈ {res['piM_bar']:.5f}")

    print("\n==== MW DYNAMICS (post burn-in averages) ====")
    print(f"Average price (firms’ mean): {res['avg_price']:.3f}")
    print(f"Average profit per firm:     {res['avg_profit']:.5f}")
    print(f"Δ (normalized profit gain):  {res['Delta']:.3f}")


    plot_prices(
        res["prices_1"], res["prices_2"],
        res["pN_cont"], res["pM_cont"],
        title="Contextual MW (full-information) — price paths",
        max_points=20000      # plots up to ~20k points for speed
    )


def contextual_bandit_simulation():
    P = LogitParams(a0=0.0, a1=2.0, a2=2.0, c1=1.0, c2=1.0, mu=0.25)

    # ---- EXP3 (bandit) run ----
    C_b = ContextBanditConfig(
        T=2_000_000,  # bandit needs longer
        m=15,
        xi=0.10,
        eta1=0.05, eta2=0.05,
        gamma1=0.05, gamma2=0.05,
        seed=None,
        log_every=100_000,
        anneal_gamma=False
    )

    res_b = run_simulation_contextual_exp3(P, C_b)

    print("\n==== STATIC BENCHMARKS ====")
    print(f"p^N: ({res_b['pN_cont'][0]:.3f}, {res_b['pN_cont'][1]:.3f}),  "
          f"π̄^N ≈ {res_b['piN_bar']:.5f}")
    print(f"p^M: ({res_b['pM_cont'][0]:.3f}, {res_b['pM_cont'][1]:.3f}),  "
          f"π̄^M ≈ {res_b['piM_bar']:.5f}")

    print("\n==== Contextual EXP3 DYNAMICS (post burn-in averages) ====")
    print(f"Average price (firms’ mean): {res_b['avg_price']:.3f}")
    print(f"Average profit per firm:     {res_b['avg_profit']:.5f}")
    print(f"Δ (normalized profit gain):  {res_b['Delta']:.3f}")

    plot_prices(
        res_b["prices_1"], res_b["prices_2"],
        res_b["pN_cont"], res_b["pM_cont"],
        title="Contextual EXP3 (bandit) — price paths",
        smooth=5000,          # try 2,000–10,000; None for raw
        max_points=20000      # plots up to ~20k points for speed
    )



if __name__ == "__main__":
    # print("Running full-information MW simulation...")
    # full_info_simulation()
    print("Running bandit EXP3 simulation...")
    bandit_simulation()
    # print("Running contextual MW simulation...")
    # mw_contextual_simulation()
    # print("Running contextual bandit EXP3 simulation...")
    contextual_bandit_simulation()

