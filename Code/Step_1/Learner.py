import numpy as np


class Learner:
    def __init__(self, prices):
        self.prices = np.array(prices)
        self.n_arms = len(prices)
        self.t = 0
        self.rewards_per_arm = [[] for i in range(self.n_arms)]
        self.collected_rewards = np.array([])

    def update_observations(self, pulled_arm, _reward):
        # once the reward is returned by the environment
        self.rewards_per_arm[pulled_arm].append(_reward)
        self.collected_rewards = np.append(self.collected_rewards, _reward)