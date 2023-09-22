import numpy as np
import matplotlib.pyplot as plt
from Code.Environment.EXP3_Learner import EXP3_Learner
from Code.Environment.Clairvoyant import *
from Code.Environment.UCB1_Active_Learner import UCB1_Active_Learner
from Code.Environment.UCB1_Passive_Learner import UCB1_Passive_Learner

env = Environment()
customer_class = "C1"

T = 365
_, opt_bids = optimize(env)
opt_bid = opt_bids["C1"]
phase = ["phase_1", "phase_2", "phase_3"]
opt_prices = [250, 200, 300]
opt_rewards = [0, 0, 0]
for i in range(len(phase)):
    opt_rewards[i] = env.get_clicks(opt_bid, customer_class) * env.seasonal_prob(phase[i], opt_prices[i]) * (
                opt_prices[i] - env.prod_cost) - env.get_costs(opt_bid, customer_class)

n_experiments = 5

ucb_active_rewards_per_experiment = []
ucb_active_regrets_per_experiment = []

ucb_passive_rewards_per_experiment = []
ucb_passive_regrets_per_experiment = []

EXP3_rewards_per_experiment = []
EXP3_regrets_per_experiment = []

for e in range(0, n_experiments):
    ucb_active_learner = UCB1_Active_Learner(env.prices)
    ucb_passive_learner = UCB1_Passive_Learner(env.prices, 20)
    EXP3_learner = EXP3_Learner(env.prices, gamma=0.01)

    ucb_active_regrets = []
    ucb_active_rewards = []
    ucb_passive_regrets = []
    EXP3_regrets = []
    EXP3_rewards = []
    phase = ["phase_1", "phase_2", "phase_3"]
    current_phase = phase[0]
    opt = opt_rewards[0]

    for t in range(T):
        if t == (T / 3):
            current_phase = phase[1]
            opt = opt_rewards[1]
        if t == ((T * 2) / 3):
            current_phase = phase[2]
            opt = opt_rewards[2]

        # EXP3 Learner passive
        pulled_arm = EXP3_learner.pull_arms()
        reward = env.seasonal_prob_round(customer_class, current_phase, pulled_arm, opt_bid)
        EXP3_learner.update(pulled_arm, reward)
        EXP3_regrets.append(opt - reward[0])
        # ucb1 Learner passive
        pulled_arm = ucb_passive_learner.pull_arms()
        reward = env.seasonal_prob_round(customer_class, current_phase, pulled_arm, opt_bid)
        ucb_passive_learner.update(pulled_arm, reward)
        ucb_passive_regrets.append(opt - reward[0])
        # ucb1 Learner active
        pulled_arm = ucb_active_learner.pull_arms()
        reward = env.seasonal_prob_round(customer_class, current_phase, pulled_arm, opt_bid)
        ucb_active_learner.update(pulled_arm, reward)
        ucb_active_regrets.append(opt - reward[0])
        ucb_active_rewards.append(reward[0])

    EXP3_rewards_per_experiment.append(EXP3_learner.collected_rewards.tolist())
    EXP3_regrets_per_experiment.append(EXP3_regrets)
    ucb_active_rewards_per_experiment.append(ucb_active_rewards)
    ucb_active_regrets_per_experiment.append(ucb_active_regrets)
    ucb_passive_rewards_per_experiment.append(ucb_passive_learner.collected_rewards.tolist())
    ucb_passive_regrets_per_experiment.append(ucb_passive_regrets)

EXP3_rewards_per_experiment = np.array(EXP3_rewards_per_experiment)
EXP3_regrets_per_experiment = np.array(EXP3_regrets_per_experiment)
ucb_active_rewards_per_experiment = np.array(ucb_active_rewards_per_experiment)
ucb_active_regrets_per_experiment = np.array(ucb_active_regrets_per_experiment)
ucb_passive_rewards_per_experiment = np.array(ucb_passive_rewards_per_experiment)
ucb_passive_regrets_per_experiment = np.array(ucb_passive_regrets_per_experiment)

plt.figure(2)
plt.subplot(2, 2, 1)

avg_ist_rewards_EXP3 = EXP3_rewards_per_experiment.mean(axis=0)
avg_ist_rewards_ucb_active = ucb_active_rewards_per_experiment.mean(axis=0)
avg_ist_rewards_ucb_passive = ucb_passive_rewards_per_experiment.mean(axis=0)

plt.plot(avg_ist_rewards_EXP3, "-", label="ist avg EXP3", color="g")
plt.plot(avg_ist_rewards_ucb_active, "-", label="ist avg ucb active", color="r")
plt.plot(avg_ist_rewards_ucb_passive, "-", label="ist avg passive", color="y")

plt.legend()
plt.title("Average of Instantaneous Reward")
plt.subplot(2, 2, 2)

std_ist_rewards_EXP3 = np.std(EXP3_rewards_per_experiment, axis=0)
std_ist_rewards_ucb_active = np.std(ucb_active_rewards_per_experiment, axis=0)
std_ist_rewards_ucb_passive = np.std(ucb_passive_rewards_per_experiment, axis=0)

plt.plot(std_ist_rewards_EXP3, "-", label="ist std EXP3", color="g")
plt.plot(std_ist_rewards_ucb_active, "-", label="ist std ucb active", color="r")
plt.plot(std_ist_rewards_ucb_passive, "-", label="ist std ucb passive", color="y")
plt.legend()
plt.title("Standard Deviation Instantaneous Reward")
plt.subplot(2, 2, 3)

avg_ist_regregpts_EXP3 = EXP3_regrets_per_experiment.mean(axis=0)
avg_ist_regregpts_ucb_active = ucb_active_regrets_per_experiment.mean(axis=0)
avg_ist_regregpts_ucb_passive = ucb_passive_regrets_per_experiment.mean(axis=0)

plt.plot(avg_ist_regregpts_EXP3, "-", label="ist avg regret EXP3", color="g")
plt.plot(avg_ist_regregpts_ucb_active, "-", label="ist avg regret ucb active", color="r")
plt.plot(avg_ist_regregpts_ucb_passive, "-", label="ist avg regret ucb passive", color="y")

plt.legend()
plt.title("Average of Instantaneous Regrets")
plt.subplot(2, 2, 4)

std_ist_regregpts_EXP3 = np.std(EXP3_regrets_per_experiment, axis=0)
std_ist_regregpts_ucb_active = np.std(ucb_active_regrets_per_experiment, axis=0)
std_ist_regregpts_ucb_passive = np.std(ucb_passive_regrets_per_experiment, axis=0)

plt.plot(std_ist_regregpts_EXP3, "-", label="ist std regrets EXP3", color="g")
plt.plot(std_ist_regregpts_ucb_active, "-", label="ist std regrets ucb active", color="r")
plt.plot(std_ist_regregpts_ucb_passive, "-", label="ist std regrets ucb passive", color="y")

plt.legend()
plt.title("Standard Deviation Instantaneous Regrets")

EXP3_rewards_per_experiment_cumsum = EXP3_rewards_per_experiment.cumsum(axis=1)
EXP3_regrets_per_experiment_cumsum = EXP3_regrets_per_experiment.cumsum(axis=1)
ucb_active_rewards_per_experiment_cumsum = ucb_active_rewards_per_experiment.cumsum(axis=1)
ucb_active_regrets_per_experiment_cumsum = ucb_active_regrets_per_experiment.cumsum(axis=1)
ucb_passive_rewards_per_experiment_cumsum = ucb_passive_rewards_per_experiment.cumsum(axis=1)
ucb_passive_regrets_per_experiment_cumsum = ucb_passive_regrets_per_experiment.cumsum(axis=1)

plt.figure(3)
plt.subplot(2, 2, 1)
avg_cumsum_rewards_EXP3 = EXP3_rewards_per_experiment_cumsum.mean(axis=0)
avg_cumsum_rewards_ucb_active = ucb_active_rewards_per_experiment_cumsum.mean(axis=0)
avg_cumsum_rewards_ucb_passive = ucb_passive_rewards_per_experiment_cumsum.mean(axis=0)

plt.plot(avg_cumsum_rewards_EXP3, "-", label="cumsum avg EXP3", color="g")
plt.plot(avg_cumsum_rewards_ucb_active, "-", label="cumsum avg ucb active", color="r")
plt.plot(avg_cumsum_rewards_ucb_passive, "-", label="cumsum avg ucb passive", color="y")

plt.legend()
plt.title("Average of Cumulative Reward")
plt.subplot(2, 2, 2)
std_cumsum_rewards_EXP3 = np.std(EXP3_rewards_per_experiment_cumsum, axis=0)
std_cumsum_rewards_ucb_active = np.std(ucb_active_rewards_per_experiment_cumsum, axis=0)
std_cumsum_rewards_ucb_passive = np.std(ucb_passive_rewards_per_experiment_cumsum, axis=0)

plt.plot(std_cumsum_rewards_EXP3, "-", label="cumsum std EXP3", color="g")
plt.plot(std_cumsum_rewards_ucb_active, "-", label="cumsum std ucb active", color="r")
plt.plot(std_cumsum_rewards_ucb_passive, "-", label="cumsum std ucb passive", color="y")

plt.legend()
plt.title("Standard Deviation of Cumulative Reward")
plt.subplot(2, 2, 3)
avg_cumsum_regregpts_EXP3 = EXP3_regrets_per_experiment_cumsum.mean(axis=0)
avg_cumsum_regregpts_ucb_active = ucb_active_regrets_per_experiment_cumsum.mean(axis=0)
avg_cumsum_regregpts_ucb_passive = ucb_passive_regrets_per_experiment_cumsum.mean(axis=0)

plt.plot(avg_cumsum_regregpts_EXP3, "-", label="cumsum avg regrets EXP3", color="g")
plt.plot(avg_cumsum_regregpts_ucb_active, "-", label="cumsum avg regrets ucb active", color="r")
plt.plot(avg_cumsum_regregpts_ucb_passive, "-", label="cumsum avg regrets ucb passive", color="y")

plt.legend()
plt.title("Average of Cumulative Regret")
plt.subplot(2, 2, 4)
std_cumsum_regrets_EXP3 = np.std(EXP3_regrets_per_experiment_cumsum, axis=0)
std_cumsum_regrets_ucb_active = np.std(ucb_active_regrets_per_experiment_cumsum, axis=0)
std_cumsum_regrets_ucb_passive = np.std(ucb_passive_regrets_per_experiment_cumsum, axis=0)

plt.plot(std_cumsum_regrets_EXP3, "-", label="cumsum std regrets EXP3", color="g")
plt.plot(std_cumsum_regrets_ucb_active, "-", label="cumsum std regrets ucb active", color="r")
plt.plot(std_cumsum_regrets_ucb_passive, "-", label="cumsum std regrets ucb", color="y")

plt.legend()
plt.title("Standard Deviation of Cumulative Regret")

plt.show()
