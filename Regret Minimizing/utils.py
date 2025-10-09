import numpy as np
from scipy.special import softmax
import matplotlib.pyplot as plt


# ---- grids ----
def build_bid_grid(max_bid: float, epsilon: float, decimals: int = 10) -> np.ndarray:
    steps = int(np.floor((max_bid + 1e-12) / epsilon))
    grid = epsilon * np.arange(steps + 1, dtype=float)
    return np.round(np.clip(grid, 0.0, max_bid), decimals)


# ---- payoff vectors for one player against a realized opp bid ----
def spa_payoff_vector(value: float, opp_bid: float, bids: np.ndarray) -> np.ndarray:
    b = bids
    win = b > opp_bid
    lose = b < opp_bid
    tie = ~(win | lose)
    u = np.zeros_like(b, dtype=float)
    u[win] = value - opp_bid
    u[tie] = 0.5 * (value - opp_bid)
    return u


def fpa_payoff_vector(value: float, opp_bid: float, bids: np.ndarray) -> np.ndarray:
    b = bids
    win = b > opp_bid
    lose = b < opp_bid
    tie = ~(win | lose)
    u = np.zeros_like(b, dtype=float)
    u[win] = value - b[win]
    u[tie] = 0.5 * (value - b[tie])
    return u


def gfpa_payoff_vector(value: float, opp_bid: float, bids: np.ndarray,
                       ctr_top: float = 1.0, ctr_bottom: float = 0.5) -> np.ndarray:
    b = bids
    win = b > opp_bid
    lose = b < opp_bid
    tie = ~(win | lose)
    u = np.zeros_like(b, dtype=float)
    u[win]  = ctr_top    * (value - b[win])
    u[lose] = ctr_bottom * (value - b[lose])
    u[tie]  = 0.5 * (ctr_top + ctr_bottom) * (value - b[tie])
    return u
