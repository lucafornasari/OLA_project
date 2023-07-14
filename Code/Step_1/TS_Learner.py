from Learner import *


class TS_Learner(Learner):
    def __init__(self, prices):
        super().__init__(prices)  # calling the construction of super class passing the n_arms parameter
        self.beta_parameters = np.ones((self.n_arms, 2))  # parameters of Beta distribution

    def pull_arms(self):
        idx = np.argmax(np.random.beta(self.beta_parameters[:, 0], self.beta_parameters[:, 1]) * self.prices)
        print("TS pulled arm: ", idx)
        # Beta distributions are defined by 2 parameters: alpha and beta
        # Beta distribution draws the probability of each pulled arm
        return idx

    def update(self, pulled_arm, _reward):
        self.update_observations(pulled_arm, _reward[2])
        # Add the success (if any) to alpha
        self.beta_parameters[pulled_arm, 0] = self.beta_parameters[pulled_arm, 0] + _reward[0]
        # Add the failure (if any) to beta (1-rew=1 iff rew=0 iff fail)
        self.beta_parameters[pulled_arm, 1] = self.beta_parameters[pulled_arm, 1] + _reward[1]


    # def update(self, pulled_arm, _reward):
    #     self.t += 1
    #     self.update_observations(pulled_arm, _reward)
    #     decision = 0
    #     if _reward > 0:
    #         decision = 1
    #     self.beta_parameters[pulled_arm, 0] += decision
    #     self.beta_parameters[pulled_arm, 1] += 1.0 - decision
