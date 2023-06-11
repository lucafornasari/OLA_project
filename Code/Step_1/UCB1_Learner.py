from Learner import *


class UCB1_Learner(Learner):
    def __init__(self, n_arms):
        super().__init__(n_arms)
        self.n_arms = n_arms
        self.rewards = np.zeros(n_arms)  # cumulative rewards for each arm
        self.pulls = np.zeros(n_arms)  # number of pulls for each arm

    def pull_arms(self):
        t = np.sum(self.pulls) + 1
        exploration_bonus = np.sqrt((2 * np.log(t)) / self.pulls)
        ucb_values = self.rewards / self.pulls + exploration_bonus
        idx = np.argmax(ucb_values)
        return idx

    def update(self, pulled_arms, reward):
        self.t += 1
        self.update_observations(pulled_arms, reward)
        self.rewards[pulled_arms] += reward
        self.pulls[pulled_arms] += 1
