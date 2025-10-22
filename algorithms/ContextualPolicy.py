import numpy as np
from typing import Dict, Tuple


# ---- contextual wrapper: one agent per state ----
class ContextualPolicy:
    """
    Maintains a separate learner per state. A state here is the previous (a1_prev, a2_prev).
    START is used at t=1 when there is no history.
    """
    def __init__(self, n_actions: int, make_agent, seed=None, name="CTX"):
        self.n_actions = n_actions
        self.make_agent = make_agent
        self.policies: Dict[Tuple[int,int], object] = {}
        self.START = ("START",)
        self.rng = np.random.default_rng(seed)
        self.name = name

    def _get_policy(self, state_key):
        if state_key not in self.policies:
            # create a fresh learner for this state
            self.policies[state_key] = self.make_agent()
        return self.policies[state_key]

    def act(self, state_key):
        return self._get_policy(state_key).choose_action()

    # For MW: rewards_vec is a vector for all actions in that state.
    def update_full_info(self, state_key, rewards_vec):
        self._get_policy(state_key).update(rewards_vec)

    # For EXP3: chosen action index and normalized reward in [0,1]
    def update_bandit(self, state_key, chosen_arm, reward_01):
        self._get_policy(state_key).update(chosen_arm, reward_01)