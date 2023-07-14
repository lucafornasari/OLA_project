from Learner import *


class UCB1_Learner(Learner):
    def __init__(self, prices):
        super().__init__(prices)
        # self.n_arms = self.n_arms
        # self.cumulative_rewards = np.zeros(self.n_arms)  # cumulative rewards for each arm
        # self.pulls = np.zeros(self.n_arms)  # number of pulls for each arm

        self.empirical_means = np.zeros(self.n_arms)
        self.confidence = np.array([np.inf] * self.n_arms)
        self.n_samples = np.zeros(self.n_arms)

    def pull_arms(self):
        # t = np.sum(self.pulls) + 1
        # exploration_bonus = np.sqrt((2 * np.log(t)) / (self.pulls + 1e-3))
        # ucb_values = self.cumulative_rewards / (self.pulls + 1e-3) + exploration_bonus
        # idx = np.argmax(ucb_values)
        # print("UCB pulled arm: ", idx)
        # return idx
        upper_conf = (self.empirical_means + self.confidence)*self.prices
        return np.random.choice(np.where(upper_conf == upper_conf.max())[0])


    def update(self, pulled_arm, reward):
        # self.t += 1
        # self.update_observations(pulled_arm, reward)
        # self.cumulative_rewards[pulled_arm] += reward
        # self.pulls[pulled_arm] += 1
        self.t += 1
        self.n_samples[pulled_arm] += reward[0] + reward[1]
        self.empirical_means[pulled_arm] = (self.empirical_means[pulled_arm] * self.n_samples[pulled_arm] + reward[0]) / (self.n_samples[pulled_arm])
        for a in range(self.n_arms):
            self.confidence[a] = (2 * np.log(self.t) / self.n_samples[a]) ** 0.5 if self.n_samples[a] > 0 else np.inf
        self.update_observations(pulled_arm, reward[2])
