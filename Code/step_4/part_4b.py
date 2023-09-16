import numpy as np
import matplotlib.pyplot as plt
from Code.Environment.Environment import Environment
from Code.step_3.GPTS_Learner_3 import GPTS_Learner_3
from Code.step_3.GPUCB1_Learner_3 import GPUCB1_Learner_3
from Code.step_2.Clairvoyant import *
from Code.step_4.ContextHandler import ContextHandler

env = Environment()

T = 100
opt_prices, opt_bids = optimize(env)
opt = [env.get_clicks(opt_bids[customer_class], customer_class) * env.get_conversion_prob(opt_prices[customer_class],customer_class) * (opt_prices[customer_class] - env.prod_cost) - env.get_costs(opt_bids[customer_class], customer_class) for customer_class in env.classes]
opt_sum = sum(opt)

c_handler = ContextHandler()

n_experiments = 1
gpts_rewards_per_experiment = []
gpucb_rewards_per_experiment = []
gpts_regrets_per_experiment = []
gpucb_regrets_per_experiment = []

for e in range(0, n_experiments):
    print("Starting new experiment...")

    gpts_regrets = [np.array([]), np.array([]), np.array([])]
    gpucb_regrets = [np.array([]), np.array([]), np.array([])]

    for t in range(0, T):

        # if t % 14 == 0 and t != 0:
        #     c_handler.change_context()

        for i in range(len(c_handler.context_ts)):
            # Thompson Sampling Learner
            pulled_arm_bid, pulled_arm_price = c_handler.context_ts[i].pull_arm()
            reward = env.part4_round(c_handler.context_classes_ts[i], pulled_arm_price, pulled_arm_bid, t)
            c_handler.update_dataset_ts(reward[3])
            c_handler.context_ts[i].update(pulled_arm_price, pulled_arm_bid, reward)
            #gpts_regrets[class_id] = np.append(gpts_regrets[class_id], opt[class_id] - reward[2])

        for i in range(len(c_handler.context_ucb)):
            # gpucb1 Learner
            pulled_arm_bid, pulled_arm_price = c_handler.context_ucb[i].pull_arm()
            reward = env.part4_round(c_handler.context_classes_ucb[i], pulled_arm_price, pulled_arm_bid, t)
            c_handler.update_dataset_ucb(reward[3])
            c_handler.context_ucb[i].update(pulled_arm_price, pulled_arm_bid, reward)
            #gpucb_regrets[class_id] = np.append(gpucb_regrets[class_id], opt[class_id] - reward[2])

    # gpts_rewards_per_experiment.append(sum(gpts_learner[class_id].bid.collected_rewards for class_id in range(0, 3)))
    # gpucb_rewards_per_experiment.append(sum(gpucb_learner[class_id].bid.collected_rewards for class_id in range(0, 3)))
    # gpts_regrets_per_experiment.append(sum(gpts_regrets[class_id] for class_id in range(0, 3)))
    # gpucb_regrets_per_experiment.append(sum(gpucb_regrets[class_id] for class_id in range(0, 3)))

    print('Finished experiment #', e)
#
# gpts_rewards_per_experiment = np.array(gpts_rewards_per_experiment)
# gpucb_rewards_per_experiment = np.array(gpucb_rewards_per_experiment)
# gpts_regrets_per_experiment = np.array(gpts_regrets_per_experiment)
# gpucb_regrets_per_experiment = np.array(gpucb_regrets_per_experiment)
#
# # average instantaneous reward
# plt.figure(1)
# plt.subplot(2, 2, 1)  # 2 righe, 1 colonna, primo subplot
# avg_ist_rewards_gpts = gpts_rewards_per_experiment.mean(axis=0)
# avg_ist_rewards_gpucb = gpucb_rewards_per_experiment.mean(axis=0)
# plt.plot(avg_ist_rewards_gpts, "-", label="ist avg gpts", color="r")
# plt.plot(avg_ist_rewards_gpucb, "-", label="ist avg gpucb", color="g")
# plt.legend()
# plt.title("Average of Instantaneous Reward")
# plt.subplot(2, 2, 2)  # 2 righe, 1 colonna, primo subplot
# std_ist_rewards_gpts = np.std(gpts_rewards_per_experiment, axis=0)
# std_ist_rewards_gpucb = np.std(gpucb_rewards_per_experiment, axis=0)
# plt.plot(std_ist_rewards_gpts, "-", label="ist std gpts", color="r")
# plt.plot(std_ist_rewards_gpucb, "-", label="ist std gpucb", color="g")
# plt.legend()
# plt.title("Standard Deviation Instantaneous Reward")
# # average instantaneous regret
# plt.subplot(2, 2, 3)  # 2 righe, 1 colonna, primo subplot
# avg_ist_regregpts_gpts = gpts_regrets_per_experiment.mean(axis=0)
# avg_ist_regregpts_gpucb = gpucb_regrets_per_experiment.mean(axis=0)
# plt.plot(avg_ist_regregpts_gpts, "-", label="ist avg regregpts gpts", color="r")
# plt.plot(avg_ist_regregpts_gpucb, "-", label="ist avg regret gpucb", color="g")
# plt.legend()
# plt.title("Average of Instantaneous Regregpts")
# plt.subplot(2, 2, 4)  # 2 righe, 1 colonna, primo subplot
# std_ist_regregpts_gpts = np.std(gpts_regrets_per_experiment, axis=0)
# std_ist_regregpts_gpucb = np.std(gpucb_regrets_per_experiment, axis=0)
# plt.plot(std_ist_regregpts_gpts, "-", label="ist std regregpts gpts", color="r")
# plt.plot(std_ist_regregpts_gpucb, "-", label="ist std regregpts gpucb", color="g")
# plt.legend()
# plt.title("Standard Deviation Instantaneous Regregpts")
#
# gpts_rewards_per_experiment_cumsum = gpts_rewards_per_experiment.cumsum(axis=1)
# gpucb_rewards_per_experiment_cumsum = gpucb_rewards_per_experiment.cumsum(axis=1)
# gpts_regregpts_per_experiment_cumsum = gpts_regrets_per_experiment.cumsum(axis=1)
# gpucb_regregpts_per_experiment_cumsum = gpucb_regrets_per_experiment.cumsum(axis=1)
#
# plt.figure(2)
# plt.subplot(2, 2, 1)  # 2 righe, 1 colonna, primo subplot
# avg_cumsum_rewards_gpts = gpts_rewards_per_experiment_cumsum.mean(axis=0)
# avg_cumsum_rewards_gpucb = gpucb_rewards_per_experiment_cumsum.mean(axis=0)
# plt.plot(avg_cumsum_rewards_gpts, "-", label="cumsum avg gpts", color="r")
# plt.plot(avg_cumsum_rewards_gpucb, "-", label="cumsum avg gpucb", color="g")
# plt.legend()
# plt.title("Average of Cumulative Reward")
# plt.subplot(2, 2, 2)  # 2 righe, 1 colonna, primo subplot
# std_cumsum_rewards_gpts = np.std(gpts_rewards_per_experiment_cumsum, axis=0)
# std_cumsum_rewards_gpucb = np.std(gpucb_rewards_per_experiment_cumsum, axis=0)
# plt.plot(std_cumsum_rewards_gpts, "-", label="cumsum std gpts", color="r")
# plt.plot(std_cumsum_rewards_gpucb, "-", label="cumsum std gpucb", color="g")
# plt.legend()
# plt.title("Standard Deviation of Cumulative Reward")
# plt.subplot(2, 2, 3)  # 2 righe, 1 colonna, primo subplot
# avg_cumsum_regregpts_gpts = gpts_regregpts_per_experiment_cumsum.mean(axis=0)
# avg_cumsum_regregpts_gpucb = gpucb_regregpts_per_experiment_cumsum.mean(axis=0)
# plt.plot(avg_cumsum_regregpts_gpts, "-", label="cumsum avg regregpts gpts", color="r")
# plt.plot(avg_cumsum_regregpts_gpucb, "-", label="cumsum avg regregpts gpucb", color="g")
# plt.legend()
# plt.title("Average of Cumulative Regret")
# plt.subplot(2, 2, 4)  # 2 righe, 1 colonna, primo subplot
# std_cumsum_regregpts_gpts = np.std(gpts_regregpts_per_experiment_cumsum, axis=0)
# std_cumsum_regregpts_gpucb = np.std(gpucb_regregpts_per_experiment_cumsum, axis=0)
# plt.plot(std_cumsum_regregpts_gpts, "-", label="cumsum std regregpts gpts", color="r")
# plt.plot(std_cumsum_regregpts_gpucb, "-", label="cumsum std regregpts gpucb", color="g")
# plt.legend()
# plt.title("Standard Deviation of Cumulative Regret")
# plt.show()