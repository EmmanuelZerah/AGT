# mw_duopoly_experiment.py
import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from MWAgent import MWAgent


# -----------------------------
# Environment (Logit demand)
# -----------------------------
@dataclass
class LogitParams:
    alpha: float = 1.0        # currency scale
    beta: float = 100.0       # market size scale
    mu: float = 0.25          # horizontal differentiation (lower => closer substitutes)
    a_i: float = 2.0          # vertical differentiation (same for both firms here)
    a0: float = 0.0           # outside option attractiveness
    c_i: float = 1.0          # marginal cost before scaling (so cost in currency is alpha*c_i)


def demand_logit_two_firms(p1, p2, par: LogitParams):
    """
    Return (q1, q2) under the 2-firm logit with outside option.
    q_i = beta * exp((a_i - p_i)/(alpha*mu)) / (sum_j exp((a_j - p_j)/(alpha*mu)) + exp(a0/mu))
    """
    z1 = np.exp((par.a_i - p1) / (par.alpha * par.mu))
    z2 = np.exp((par.a_i - p2) / (par.alpha * par.mu))
    z0 = np.exp(par.a0 / par.mu)
    denom = z1 + z2 + z0
    s1 = z1 / denom
    s2 = z2 / denom
    q1 = par.beta * s1
    q2 = par.beta * s2
    return q1, q2


def profit_i(p_i, q_i, par: LogitParams):
    return (p_i - par.alpha * par.c_i) * q_i


# -----------------------------
# Price grid / actions
# -----------------------------
def monopoly_price_grid_hint(par: LogitParams, p_min=None, p_max_mult=2.5, grid_size=101):
    """
    Build a price grid with a reasonable upper bound relative to the *joint* monopoly price (p^M).
    We approximate p^M by brute-force on a fine dense grid and then set an upper cap = p_max_mult * p^M.
    """
    # Lower bound: at or just above cost
    if p_min is None:
        p_min = par.alpha * par.c_i * 1.0001  # tiny epsilon above cost

    # Find joint-monopoly symmetric price (p1 = p2 = p) by scanning
    scan = np.linspace(p_min, par.alpha * par.c_i + 10.0 * par.alpha, 2001)
    best = None
    best_p = scan[0]
    for p in scan:
        q1, q2 = demand_logit_two_firms(p, p, par)
        pi1 = profit_i(p, q1, par)
        pi2 = profit_i(p, q2, par)
        tot = pi1 + pi2
        if best is None or tot > best:
            best = tot
            best_p = p

    p_max = p_max_mult * best_p
    if p_max <= p_min:
        p_max = p_min + par.alpha  # fallback

    grid = np.linspace(p_min, p_max, grid_size)
    return grid, best_p


def payoff_vector_vs_price(opponent_price, my_price_grid, par: LogitParams):
    """
    Return payoffs for *each* action (price) in my_price_grid given opponent's fixed price.
    """
    payoffs = np.empty_like(my_price_grid)
    for k, p_i in enumerate(my_price_grid):
        q_i, _ = demand_logit_two_firms(p_i, opponent_price, par)
        payoffs[k] = profit_i(p_i, q_i, par)
    return payoffs


# -----------------------------
# Experiment runner
# -----------------------------
@dataclass
class RunConfig:
    T: int = 300                 # periods
    alpha_choices=(1.0, 3.2, 10.0)
    seed: int = 0
    lr_agent1: float = 0.1
    lr_agent2: float = 0.1
    grid_size: int = 101         # number of discrete actions (prices)


def run_single_duopoly(config: RunConfig):
    rng = np.random.default_rng(config.seed)
    alpha = rng.choice(config.alpha_choices)
    par = LogitParams(alpha=alpha)

    # Build price grid bounded by joint-monopoly hint
    price_grid, p_monopoly_hint = monopoly_price_grid_hint(par, grid_size=config.grid_size)

    # Initialize agents (each action = one price on grid)
    A1 = MWAgent(n_actions=len(price_grid), learning_rate=config.lr_agent1, name="A1")
    A2 = MWAgent(n_actions=len(price_grid), learning_rate=config.lr_agent2, name="A2")

    prices1, prices2 = [], []
    profits1, profits2 = [], []

    for t in range(config.T):
        # action indices -> prices
        a1 = A1.choose_action()
        a2 = A2.choose_action()
        p1 = price_grid[a1]
        p2 = price_grid[a2]

        # record realized outcome
        q1, q2 = demand_logit_two_firms(p1, p2, par)
        pi1 = profit_i(p1, q1, par)
        pi2 = profit_i(p2, q2, par)

        prices1.append(p1 / par.alpha)   # normalize by alpha like the paper
        prices2.append(p2 / par.alpha)
        profits1.append(pi1 / par.alpha) # normalize profits by alpha as well
        profits2.append(pi2 / par.alpha)

        # build full payoff vectors for each agent vs opponent's realized price
        payoff_vec_1 = payoff_vector_vs_price(opponent_price=p2, my_price_grid=price_grid, par=par) / par.alpha
        payoff_vec_2 = payoff_vector_vs_price(opponent_price=p1, my_price_grid=price_grid, par=par) / par.alpha

        # MW updates
        A1.update(payoff_vec_1)
        A2.update(payoff_vec_2)

    results = {
        "alpha": par.alpha,
        "price_grid": price_grid,
        "p_monopoly_hint": p_monopoly_hint,
        "prices1": np.array(prices1),
        "prices2": np.array(prices2),
        "profits1": np.array(profits1),
        "profits2": np.array(profits2),
    }
    return results


def summarize_last_50(res):
    p1 = res["prices1"][-50:].mean()
    p2 = res["prices2"][-50:].mean()
    prof_sum = (res["profits1"][-50:].mean() + res["profits2"][-50:].mean())
    return p1, p2, prof_sum


def main():
    # Example: run a few seeds and print summary like the paper
    seeds = [0, 1, 2, 3, 4]
    configs = [RunConfig(seed=s) for s in seeds]
    summaries = []

    for cfg in configs:
        res = run_single_duopoly(cfg)
        p1, p2, tot = summarize_last_50(res)
        summaries.append((res["alpha"], p1, p2, tot))

    print("alpha | avg price firm1 | avg price firm2 | avg total profit (last 50, normalized)")
    for (alpha, p1, p2, tot) in summaries:
        print(f"{alpha:4.1f} | {p1:15.3f} | {p2:15.3f} | {tot:24.3f}")

    # Optional: plot one run’s price dynamics (seed 0) normalized by alpha
    res0 = run_single_duopoly(RunConfig(seed=0))
    t = np.arange(len(res0["prices1"]))
    plt.figure()
    plt.plot(t, res0["prices1"], label="Firm 1 price / alpha")
    plt.plot(t, res0["prices2"], label="Firm 2 price / alpha")
    plt.xlabel("Period")
    plt.ylabel("Price (normalized by alpha)")
    plt.title("MW vs MW: Price Dynamics")
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
