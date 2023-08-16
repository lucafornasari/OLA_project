import numpy as np


class Learner:
    def __init__(self, rewards):
        self.rewards = rewards
        self.n_arms = len(rewards)
        self.t = 0
        self.rewards_per_arm = [[] for i in range(self.n_arms)]
        self.collected_rewards = np.array([])
        self.arms_pulled= []

    def update_observations(self, pulled_arm, decision):
        # once the reward is returned by the environment
        self.rewards_per_arm[pulled_arm].append(decision)
        self.collected_rewards = np.append(self.collected_rewards, decision)
        self.arms_pulled.append(pulled_arm)
