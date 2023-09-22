from Code.Environment.Learner import *


class UCB1_Learner(Learner):
    def __init__(self, prices):
        super().__init__(prices)

        self.empirical_means = np.zeros(self.n_arms)
        self.confidence = np.array([np.inf] * self.n_arms)
        self.n_tests = np.zeros(self.n_arms)

    def pull_arms(self):
        upper_conf = (self.empirical_means + self.confidence) * self.rewards
        return np.random.choice(np.where(upper_conf == upper_conf.max())[0])

    def update(self, pulled_arm, reward):
        self.t += 1
        self.empirical_means[pulled_arm] = (self.empirical_means[pulled_arm] * self.n_tests[pulled_arm] + reward[0] *
                                            reward[1]) / (self.n_tests[pulled_arm] + reward[1])
        self.n_tests[pulled_arm] += reward[1]

        for a in range(self.n_arms):
            self.confidence[a] = (2 * np.log(self.t) / self.n_tests[a]) ** 0.5 if self.n_tests[a] > 0 else np.inf
        self.update_observations(pulled_arm, reward[0])
