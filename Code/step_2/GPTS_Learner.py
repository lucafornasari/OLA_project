from Code.step_2.Learner import *
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C

class GPTS_Learner(Learner):
    def __init__(self, arms):
        super().__init__(arms)
        self.arms=arms
        self.n_arms = len(arms)
        self.means=np.zeros(self.n_arms)
        self.sigmas=np.ones(self.n_arms)*10
        self.pulled_arms = []
        self.alpha= 1.0
        kernel = C(1.0, (1e-9, 1e9)) * RBF(1.0, (1e-10, 1e7))
        self.gp=GaussianProcessRegressor(kernel=self.kernel,alpha=self.alpha**2, n_restarts_optimizer = 9, normalize_y = True)
        self.gp._max_iter = 100000

    def update_observations(self,arm_idx, reward):
      super().update_observations(arm_idx, reward)
      self.pulled_arms.append(self.arms[arm_idx])


    def update_model(self):
      x= np.atleast_2d(self.pulled_arms).T
      y= self.collected_rewards
      self.gp.fit(x,y)
      self.means, self.sigmas= self.gp.predict(np.atleast_2d(self.arms).T, return_std= True)
      self.sigmas = np.maximum(self.sigmas, 1e-2)

    
    def update(self,pulled_arms, reward):
      self.t +=1
      self.update_observations(pulled_arms, reward)
      self.update_model()

    def pull_arm(self):
      sampled_values = np.random.normal(self.means,self.sigmas)
      return np.argmax(sampled_values)