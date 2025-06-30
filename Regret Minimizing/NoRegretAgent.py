
from abc import ABC, abstractmethod


class NoRegretAgent(ABC):
    def __init__(self, n_actions, name="Agent"):
        self.n_actions = n_actions
        self.name = name

    @abstractmethod
    def choose_action(self):
        pass

    @abstractmethod
    def update(self, payoffs):
        pass

    @abstractmethod
    def calculate_regret(self):
        pass

    @abstractmethod
    def reset(self):
        pass
