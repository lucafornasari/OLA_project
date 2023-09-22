from Code.Environment.Learner import *
import numpy as np


class UCB1_Passive_Learner(Learner):
    def __init__(self, prices, delta):
        super().__init__(prices)
        self.empirical_means = np.zeros(self.n_arms)
        self.confidence = np.array([np.inf] * self.n_arms)
        self.n_tests = np.zeros(self.n_arms)
        self.delta = delta

    def pull_arms(self):
        upper_conf = (self.empirical_means + self.confidence) * self.rewards
        return np.random.choice(np.where(upper_conf == upper_conf.max())[0])

    def update(self, pulled_arm, reward):
        self.t += 1
        self.update_observations(pulled_arm, reward[0])
        self.n_tests[pulled_arm] += reward[1]
        pulled_arm_delta = self.arms_pulled[max(0, self.t - self.delta):]
        collected_rewards_delta = self.collected_rewards[max(0, self.t - self.delta):]
        cumulative_rewards_delta = np.zeros(self.n_arms)
        n_tests_delta = np.zeros(self.n_arms)
        count = 0
        for i in pulled_arm_delta:
            n_tests_delta[i] += reward[1]
            cumulative_rewards_delta[i] += collected_rewards_delta[count]
            count += 1

        self.empirical_means[pulled_arm] = (cumulative_rewards_delta[pulled_arm]) / (n_tests_delta[pulled_arm])
        for a in range(self.n_arms):
            self.confidence[a] = (2 * np.log(max(0, self.t - self.delta)) / n_tests_delta[a]) ** 0.5 if n_tests_delta[
                                                                                                            a] > 0 else np.inf
