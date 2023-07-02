from Learner import *


class TS_Learner(Learner):
    def __init__(self, rewards):
        super().__init__(rewards)  # calling the construction of super class passing the n_arms parameter
        self.beta_parameters = np.ones((self.n_arms, 2))  # parameters of Beta distribution

    def pull_arms(self):
        idx = np.argmax(np.random.beta(self.beta_parameters[:, 0], self.beta_parameters[:, 1]) * self.rewards)
        # Beta distributions are defined by 2 parameters: alpha and beta
        # Beta distribution draws the probability of each pulled arm
        return idx

    def update(self, pulled_arm, reward):
        self.t += 1
        b_reward = (1 if reward > 0 else 0)  # binary reward
        self.update_observations(pulled_arm, b_reward)
        self.beta_parameters[pulled_arm, 0] += b_reward
        self.beta_parameters[pulled_arm, 1] += 1.0 - b_reward
