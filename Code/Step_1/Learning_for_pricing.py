import numpy as np
import matplotlib.pyplot as plt
from Code.Environment.Environment import Environment
from Code.Step_1.Greedy_Learner import Greedy_Learner
from Code.Step_1.TS_Learner import TS_Learner
from Code.Step_1.UCB1_Learner import UCB1_Learner

env = Environment()
customer_class = "C1"

T = 365

opt = np.array(env.prices)[-1]

n_experiments = 100
ts_rewards_per_experiment = []
ucb_rewards_per_experiment = []
gr_rewards_per_experiment = []

for e in range(0, n_experiments):
    print("Starting new experiment...")
    ts_learner = TS_Learner(len(env.prices))
    ucb_learner = UCB1_Learner(len(env.prices))
    gr_learner = Greedy_Learner(len(env.prices))

    for t in range(0, T):
        # Thompson Sampling Learner
        pulled_arm = ts_learner.pull_arms()
        reward = env.purchase_decision(env.prices[pulled_arm], customer_class)
        ts_learner.update(pulled_arm, reward)
        print("Reward for experiment " + str(e) + " at time " + str(t) + " for TS: " + str(reward))

        # UCB1 Learner
        pulled_arm = ucb_learner.pull_arms()
        reward = env.purchase_decision(env.prices[pulled_arm], customer_class)
        ucb_learner.update(pulled_arm, reward)
        print("Reward for experiment " + str(e) + " at time " + str(t) + " for UCB1: " + str(reward))

        # Greedy Learner
        pulled_arm = gr_learner.pull_arms()
        reward = env.purchase_decision(env.prices[pulled_arm], customer_class)
        gr_learner.update(pulled_arm, reward)
        print("Reward for experiment " + str(e) + " at time " + str(t) + " for Greedy: " + str(reward))

    ts_rewards_per_experiment.append(ts_learner.collected_rewards)
    ucb_rewards_per_experiment.append(ucb_learner.collected_rewards)
    gr_rewards_per_experiment.append(gr_learner.collected_rewards)

    print('Finished experiment #', e)

print("TS: ", np.sum(ts_rewards_per_experiment))
print("UCB: ", np.sum(ucb_rewards_per_experiment))
print("GR: ", np.sum(gr_rewards_per_experiment))

plt.figure(0)
plt.xlabel("t")
plt.ylabel("Regret")
plt.plot(np.cumsum(np.mean(opt - ts_rewards_per_experiment, axis=0)), 'r')
plt.plot(np.cumsum(np.mean(opt - ucb_rewards_per_experiment, axis=0)), 'g')
plt.plot(np.cumsum(np.mean(opt - gr_rewards_per_experiment, axis=0)), 'b')
plt.legend(['TS', 'UCB1', 'Greedy'])
plt.show()
