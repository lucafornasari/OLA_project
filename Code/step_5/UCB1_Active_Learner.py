from Learner import *


class UCB1_Active_Learner(Learner):
    def __init__(self, prices):
        super().__init__(prices)
        # self.n_arms = self.n_arms
        # self.cumulative_rewards = np.zeros(self.n_arms)  # cumulative rewards for each arm
        # self.pulls = np.zeros(self.n_arms)  # number of pulls for each arm

        self.empirical_means = np.zeros(self.n_arms)
        self.confidence = np.array([np.inf] * self.n_arms)
        self.n_tests = np.zeros(self.n_arms)
        self.detections= []
        self.threshold=7000
        self.num_obs=16

    def pull_arms(self):
        # t = np.sum(self.pulls) + 1
        # exploration_bonus = np.sqrt((2 * np.log(t)) / (self.pulls + 1e-3))
        # ucb_values = self.cumulative_rewards / (self.pulls + 1e-3) + exploration_bonus
        # idx = np.argmax(ucb_values)
        # print("UCB pulled arm: ", idx)
        # return idx
        upper_conf = (self.empirical_means + self.confidence)*self.rewards
        return np.random.choice(np.where(upper_conf == upper_conf.max())[0])


    def update(self, pulled_arm, reward):
        # self.t += 1
        # self.update_observations(pulled_arm, reward)
        # self.cumulative_rewards[pulled_arm] += reward
        # self.pulls[pulled_arm] += 1
        self.t += 1
        self.empirical_means[pulled_arm] = (self.empirical_means[pulled_arm] * self.n_tests[pulled_arm] + reward[0]*reward[1]) / (self.n_tests[pulled_arm]+reward[1])
        self.n_tests[pulled_arm] +=reward[1]

        for a in range(self.n_arms):
            self.confidence[a] = (2 * np.log(self.t) / self.n_tests[a]) ** 0.5 if self.n_tests[a] > 0 else np.inf
        self.update_observations(pulled_arm, reward[0])
        #controllo change detection
        for a in range(self.n_arms):
            if(len(self.rewards_per_arm[a])>self.num_obs):
                temp=self.rewards_per_arm[a]
                temp=temp[(len(self.rewards_per_arm[a])-self.num_obs):]
                b=self.change_detection(temp)
                if(b):
                    self.detections.append(self.t)
                    self.restart()

                

    def change_detection(self,temp):
        divisor=round(self.num_obs/2)
        temp=np.array(temp)
        first=temp
        first=first[:divisor]
        second=temp
        second=second[divisor:]
        sum_1=first.cumsum()
        sum_2=second.cumsum()
        difference=abs(sum_2[len(sum_2)-1]-sum_1[len(sum_1)-1])
        detection =False
        #sbu.append(difference)
        if(difference>self.threshold):
            detection= True
        
        return detection

    def restart(self):
        self.t=0
        self.empirical_means = np.zeros(self.n_arms)
        self.confidence = np.array([np.inf] * self.n_arms)
        self.n_tests = np.zeros(self.n_arms)
        self.rewards_per_arm = [[] for i in range(self.n_arms)]
        self.collected_rewards = np.array([])
