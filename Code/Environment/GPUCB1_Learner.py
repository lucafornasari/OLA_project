from Code.Environment.Learner import *
from Code.Environment.Environment import Environment
import numpy as np
import math
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C


class GPUCB1_Learner(Learner):
    def __init__(self, rewards):
        super().__init__(rewards)
        self.means = np.zeros(self.n_arms)
        self.sigmas = np.ones(self.n_arms) * 10
        self.pulled_arms = []
        self.pulls = np.zeros(self.n_arms)
        self.alpha = 1.0
        self.kernel = C(1.0, (1e-9, 1e9)) * RBF(1.0, (1e-10, 1e7))  #
        self.gp = GaussianProcessRegressor(kernel=self.kernel, alpha=self.alpha ** 2, n_restarts_optimizer=7)
        self.gp._max_iter = 100000
        self.interval = 1
        self.sigma_modifier = 1

    def update_observations(self, pulled_arm, decision):
        super().update_observations(pulled_arm, decision)
        self.pulled_arms.append(pulled_arm)

    def update_model(self):
        x = np.atleast_2d(self.pulled_arms).T
        y = self.collected_rewards

        if self.t % self.interval == 0:
            self.gp.fit(x, y)
        self.means, self.sigmas = self.gp.predict(np.atleast_2d(range(self.n_arms)).T, return_std=True)
        self.sigmas = np.maximum(self.sigmas, 1e-2)

    def update(self, pulled_arm, reward):
        self.t += 1
        self.update_observations(pulled_arm, reward)
        self.pulls[pulled_arm] += 1
        self.update_model()

    def pull_arm(self):
        t = np.sum(self.pulls) + 1
        exploration_bonus = np.sqrt(self.sigma_modifier * (np.log(t)) / (self.pulls + 1.2 * math.exp(-4)))
        ucb_values = self.means + self.sigmas * exploration_bonus
        # select all maximum values and randomly select one
        idx = np.random.choice(np.where(ucb_values == ucb_values.max())[0])
        return idx