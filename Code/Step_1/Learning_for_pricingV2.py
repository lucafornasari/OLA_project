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

ts_cumulative_regret = np.zeros(T)
ts_cumulative_reward = np.zeros(T)
ts_instantaneous_regret = np.zeros(T)
ts_instantaneous_reward = np.zeros(T)

ucb_cumulative_regret = np.zeros(T)
ucb_cumulative_reward = np.zeros(T)
ucb_instantaneous_regret = np.zeros(T)
ucb_instantaneous_reward = np.zeros(T)

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

        ts_cumulative_reward[t] += reward
        ts_instantaneous_reward[t] = reward
        ts_cumulative_regret[t] += opt - reward
        ts_instantaneous_regret[t] = opt - reward

        # UCB1 Learner
        pulled_arm = ucb_learner.pull_arms()
        reward = env.purchase_decision(env.prices[pulled_arm], customer_class)
        ucb_learner.update(pulled_arm, reward)

        ucb_cumulative_reward[t] += reward
        ucb_instantaneous_reward[t] = reward
        ucb_cumulative_regret[t] += opt - reward
        ucb_instantaneous_regret[t] = opt - reward

    print('Finished experiment #', e)

# Calculate average values
ts_cumulative_regret /= n_experiments
ts_cumulative_reward /= n_experiments
ts_instantaneous_regret /= n_experiments
ts_instantaneous_reward /= n_experiments

ucb_cumulative_regret /= n_experiments
ucb_cumulative_reward /= n_experiments
ucb_instantaneous_regret /= n_experiments
ucb_instantaneous_reward /= n_experiments

# Plot results
plt.figure(figsize=(12, 8))

# Cumulative Regret
plt.subplot(2, 2, 1)
plt.plot(ts_cumulative_regret, label='TS')
plt.plot(ucb_cumulative_regret, label='UCB1')
plt.xlabel('Time')
plt.ylabel('Cumulative Regret')
plt.legend()

# Cumulative Reward
plt.subplot(2, 2, 2)
plt.plot(ts_cumulative_reward, label='TS')
plt.plot(ucb_cumulative_reward, label='UCB1')
plt.xlabel('Time')
plt.ylabel('Cumulative Reward')
plt.legend()

# Instantaneous Regret
plt.subplot(2, 2, 3)
plt.plot(ts_instantaneous_regret, label='TS')
plt.plot(ucb_instantaneous_regret, label='UCB1')
plt.xlabel('Time')
plt.ylabel('Instantaneous Regret')
plt.legend()

# Instantaneous Reward
plt.subplot(2, 2, 4)
plt.plot(ts_instantaneous_reward, label='TS')
plt.plot(ucb_instantaneous_reward, label='UCB1')
plt.xlabel('Time')
plt.ylabel('Instantaneous Reward')
plt.legend()

plt.tight_layout()
plt.show()
