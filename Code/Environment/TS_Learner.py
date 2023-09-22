from Code.Environment.Learner import *


class TS_Learner(Learner):
    def __init__(self, prices):
        super().__init__(prices)
        self.beta_parameters = np.ones((self.n_arms, 2))  # parameters of Beta distribution

    def pull_arms(self):
        idx = np.argmax(np.random.beta(self.beta_parameters[:, 0], self.beta_parameters[:, 1]) * self.rewards)
        return idx

    def update(self, pulled_arm, _reward):
        self.update_observations(pulled_arm, _reward[0])
        self.beta_parameters[pulled_arm, 0] = self.beta_parameters[pulled_arm, 0] + _reward[2]
        self.beta_parameters[pulled_arm, 1] = self.beta_parameters[pulled_arm, 1] + (_reward[1]-_reward[2])
