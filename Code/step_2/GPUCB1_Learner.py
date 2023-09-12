from Code.step_2.Learner import *
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
        self.kernel = RBF(1.0, (1e-3, 1e3)) #C(1e1, (1e-7, 1e7)) *
        self.gp = GaussianProcessRegressor(kernel=self.kernel, alpha=self.alpha**2, n_restarts_optimizer = 9, normalize_y = True)

    def update_observations(self, pulled_arm, decision):
        super().update_observations(pulled_arm, decision)
        self.pulled_arms.append(pulled_arm)

    def update_model(self):
        x = np.atleast_2d(self.pulled_arms).T
        y = self.collected_rewards
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
        exploration_bonus = np.sqrt(1.5*(np.log(t)) / (self.pulls + 1*math.exp(-4)))
        ucb_values = self.means +self.sigmas * exploration_bonus
        idx = np.argmax(ucb_values)
        return idx