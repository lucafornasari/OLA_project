import numpy as np


class Learner:
    def __init__(self, prices, prod_cost, clicks, costs):
        self.prices = np.array(prices)
        self.prod_cost = prod_cost
        self.clicks = clicks
        self.costs = costs
        self.n_arms = len(prices)
        self.t = 0
        self.rewards_per_arm = [[] for i in range(self.n_arms)]
        self.collected_rewards = np.array([])

    def update_observations(self, pulled_arm, decision):
        # once the reward is returned by the environment
        _margin = self.prices[pulled_arm] - self.prod_cost
        _reward = self.clicks * _margin - self.costs
        self.rewards_per_arm[pulled_arm].append(_reward * decision)
        self.collected_rewards = np.append(self.collected_rewards, _reward * decision)
