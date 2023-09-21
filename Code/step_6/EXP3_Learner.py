import numpy as np
from Learner import *

class EXP3_Learner(Learner):
    def __init__(self, prices, gamma):
        super().__init__(prices)
        self.gamma = gamma
        self.weights = np.ones(self.n_arms)
        self.probabilities = []
        
    #def distr(self):
    #    theSum = float(sum(self.weights))
    #    return tuple((1.0 - self.gamma) * (w / theSum) + (self.gamma / len(self.weights)) for w in self.weights)

    def pull_arms(self):
        print(f'gamma: {self.gamma}')
        print(f'weights: {self.weights}')
        print(f'n_arms: {self.n_arms}')
        #self.probabilities = self.distr()
        self.probabilities = (1 - self.gamma) * (self.weights / np.sum(self.weights)) + (self.gamma / self.n_arms)
        print(f'prob: {self.probabilities}')
        pulled_arm = np.random.choice(self.n_arms, p=self.probabilities)
        return pulled_arm

    def update(self, pulled_arm, reward):
        self.t +=1
        self.update_observations(pulled_arm, reward[0])
        if isinstance(reward[0], (float, int)):
            print(f'reward: {reward[0]}')
            estimated_reward = reward[0] / (self.probabilities[pulled_arm]+ 1e-1)
            self.weights[pulled_arm] *= np.exp(estimated_reward * self.gamma / self.n_arms)
            #self.weights[pulled_arm] *= np.exp(self.gamma * estimated_reward / float(self.n_arms))
        else:
            # Handle the case where reward is not a scalar (e.g., print an error message)
            print("Error: reward is not a scalar")
            print(reward)
        #estimated_reward = float(reward) / self.probabilities[pulled_arm]
        #self.weights[pulled_arm] *= np.exp(self.gamma * estimated_reward / float(self.n_arms))
        
        

    