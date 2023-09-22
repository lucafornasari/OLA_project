import numpy as np
from Code.Environment.Learner import *


class EXP3_Learner(Learner):
    def __init__(self, prices, gamma):
        super().__init__(prices)
        self.gamma = gamma
        self.weights = np.ones(self.n_arms)
        self.probabilities = []

    def pull_arms(self):
        self.probabilities = (1 - self.gamma) * (self.weights / np.sum(self.weights)) + (self.gamma / self.n_arms)
        pulled_arm = np.random.choice(self.n_arms, p=self.probabilities)
        return pulled_arm

    def update(self, pulled_arm, reward):
        self.t += 1
        self.update_observations(pulled_arm, reward[0])
        if isinstance(reward[0], (float, int)):
            estimated_reward = reward[0] / (self.probabilities[pulled_arm] + 1e-1)
            self.weights[pulled_arm] *= np.exp(estimated_reward * self.gamma / self.n_arms)
        else:
            # Handle the case where reward is not a scalar (e.g., print an error message)
            print("Error: reward is not a scalar")
