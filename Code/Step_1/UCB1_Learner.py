from Learner import *


class UCB1_Learner(Learner):
    def __init__(self, prices, prod_cost, clicks, costs):
        super().__init__(prices, prod_cost, clicks, costs)
        self.n_arms = self.n_arms
        self.cumulative_rewards = np.zeros(self.n_arms)  # cumulative rewards for each arm
        self.pulls = np.zeros(self.n_arms)  # number of pulls for each arm

    def pull_arms(self):
        if np.any(self.pulls == 0):
            return np.where(self.pulls == 0)[0][0]
        t = np.sum(self.pulls) + 1
        exploration_bonus = np.sqrt((2 * np.log(t)) / self.pulls)
        ucb_values = self.cumulative_rewards / self.pulls + exploration_bonus
        idx = np.argmax(ucb_values)
        return idx

    def update(self, pulled_arm, decision):
        self.t += 1
        self.update_observations(pulled_arm, decision)
        _margin = self.prices[pulled_arm] - self.prod_cost
        _reward = self.clicks * _margin - self.costs
        self.cumulative_rewards[pulled_arm] += _reward
        self.pulls[pulled_arm] += 1
