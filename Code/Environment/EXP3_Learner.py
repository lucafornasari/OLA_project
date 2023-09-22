import math
from Code.Environment.Learner import *


class EXP3_Learner(Learner):
    def __init__(self, prices, gamma):
        super().__init__(prices)
        self.gamma = gamma
        self.weights = np.ones(self.n_arms)
        self.probability = np.zeros(self.n_arms)
        self.update_probability()

    def update_probability(self):
        total_weight = np.sum(self.weights)
        self.probability = (1.0 - self.gamma) * (self.weights / total_weight) + (self.gamma / self.n_arms)

    def pull_arms(self):
        idx = np.random.uniform(0, np.sum(self.weights))
        return np.argmax(np.cumsum(self.weights) >= idx)

    def update(self, pulled_arm, reward):
        self.t += 1
        self.update_observations(pulled_arm, reward[0])
        rewards = max(0, reward[0] / 365)
        estimated_reward = rewards / self.probability[pulled_arm]
        self.weights[pulled_arm] *= math.exp(estimated_reward * self.gamma / self.n_arms)
        self.update_probability()
