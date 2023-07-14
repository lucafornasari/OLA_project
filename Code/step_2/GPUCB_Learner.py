from Learner import *
from Environment import Environment
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel as C, RBF

class GPUCB_Learner(Learner):
    def __init__(self, arms):
        super().__init__(arms.shape[0])
        self.arms = arms
        self.n_arms = len(arms)
        self.means = np.zeros(self.n_arms)
        self.sigmas = np.ones(self.n_arms) * 10
        self.pulled_arms = []
        self.pulls = np.zeros(self.n_arms)
        self.alpha = 10.0
        self.kernel = C(1.0, (1e-4, 1e4)) * RBF(1.0, (1e-4, 1e4))
        self.gp = GaussianProcessRegressor(kernel=self.kernel, alpha=self.alpha**2, n_restarts_optimizer=9, normalize_y = True)

    def update_observations(self, arm_idx, reward):
        super().update_observations(arm_idx, reward)
        self.pulled_arms.append(self.arms[arm_idx])

    def update_model(self):
        x = np.atleast_2d(self.pulled_arms).T
        y = self.collected_rewards
        self.gp.fit(x, y)
        self.means, self.sigmas = self.gp.predict(np.atleast_2d(self.arms).T, return_std=True)
        self.sigmas = np.maximum(self.sigmas, 1e-2)

    def update(self, pulled_arm, reward):
        self.t += 1
        self.update_observations(pulled_arm, reward)
        self.pulls[pulled_arm] += 1
        self.update_model()

    def pull_arm(self):
        t = np.sum(self.pulls) + 1
        exploration_bonus = np.sqrt((np.log(t)) / (self.pulls+e^-3))
        ucb_values = self.means + exploration_bonus
        idx = np.argmax(ucb_values)
        return idx