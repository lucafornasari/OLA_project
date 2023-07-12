from Learner import *


class TS_Learner(Learner):
    def __init__(self, prices, prod_cost, clicks, costs):
        super().__init__(prices, prod_cost, clicks, costs)  # calling the construction of super class passing the n_arms parameter
        self.beta_parameters = np.ones((self.n_arms, 2))  # parameters of Beta distribution

    def pull_arms(self):
        _margin = self.prices - self.prod_cost
        _rewards = self.clicks * _margin - self.costs
        idx = np.argmax(np.random.beta(self.beta_parameters[:, 0], self.beta_parameters[:, 1]) * _rewards)
        # Beta distributions are defined by 2 parameters: alpha and beta
        # Beta distribution draws the probability of each pulled arm
        return idx

    def update(self, pulled_arm, decision):
        self.t += 1
        self.update_observations(pulled_arm, decision)
        self.beta_parameters[pulled_arm, 0] += decision
        self.beta_parameters[pulled_arm, 1] += 1.0 - decision
