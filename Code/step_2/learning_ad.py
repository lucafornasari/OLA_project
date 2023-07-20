import numpy as np
import matplotlib.pyplot as plt
from Environment import Environment
from GPTS_Learner import GPTS_Learner
from GPUCB1_Learner import GPUCB1_Learner
from GPUCB_Learner import GPUCB_Learner
from Clairvoyant import*

env = Environment()
customer_class = "C1"

T = 250
opt_prices, opt_bids =optimize(env)
opt_price=opt_prices["C1"]
margin = opt_price - env.prod_cost
opt_bid=opt_bids["C1"]
opt = env.get_clicks(opt_bid, customer_class) * env.get_conversion_prob(opt_price, customer_class) * margin - env.get_costs(opt_bid, customer_class)

n_experimengpts = 1
gpts_rewards_per_experiment = []
gpucb_rewards_per_experiment = []
gpts_regregpts_per_experiment = []
gpucb_regregpts_per_experiment = []


for e in range(0, n_experimengpts):
    print("Starting new experiment...")
    gpts_learner = GPTS_Learner(env.bids)
    gpucb_learner = GPUCB1_Learner(env.bids)

    gpts_regregpts = []
    gpucb_regregpts = []

    for t in range(0, T):
        # Thompson Sampling Learner
        pulled_arm = gpts_learner.pull_arm()
        clicks = max(0,env.sample_clicks(env.bids[pulled_arm], customer_class))
        conv = env.get_conversion_prob(opt_price, customer_class)
        cost = max(0,env.sample_costs(env.bids[pulled_arm], customer_class))
        reward = clicks * conv * margin - cost
        gpts_learner.update(pulled_arm, reward)
        gpts_regregpts.append(opt - reward)
        print(opt-reward)
        # gpucb1 Learner
        pulled_arm = gpucb_learner.pull_arm()
        clicks = max(0,env.sample_clicks(env.bids[pulled_arm], customer_class))
        conv = env.get_conversion_prob(opt_price, customer_class)
        cost = max(0,env.sample_costs(env.bids[pulled_arm], customer_class))
        reward = clicks * conv * margin - cost
        
        gpucb_learner.update(pulled_arm, reward)
        print(gpts_learner.pulled_arms)
        gpucb_regregpts.append(opt - reward)

    gpts_rewards_per_experiment.append(gpts_learner.collected_rewards.tolist())
    gpucb_rewards_per_experiment.append(gpucb_learner.collected_rewards.tolist())
    gpts_regregpts_per_experiment.append(gpts_regregpts)
    gpucb_regregpts_per_experiment.append(gpucb_regregpts)

    print('Finished experiment #', e)

gpts_rewards_per_experiment = np.array(gpts_rewards_per_experiment)
gpucb_rewards_per_experiment = np.array(gpucb_rewards_per_experiment)
gpts_regregpts_per_experiment = np.array(gpts_regregpts_per_experiment)
gpucb_regregpts_per_experiment = np.array(gpucb_regregpts_per_experiment)



print(gpucb_learner.pulled_arms)
# average instantaneous reward
plt.figure(1)
plt.subplot(2, 2, 1)  # 2 righe, 1 colonna, primo subplot
avg_ist_rewards_gpts = gpts_rewards_per_experiment.mean(axis=0)
avg_ist_rewards_gpucb = gpucb_rewards_per_experiment.mean(axis=0)
plt.plot(avg_ist_rewards_gpts, "-", label="ist avg gpts", color="r")
plt.plot(avg_ist_rewards_gpucb, "-", label="ist avg gpucb", color="g")
plt.legend()
plt.title("Average of Instantaneous Reward")
plt.subplot(2, 2, 2)  # 2 righe, 1 colonna, primo subplot
std_ist_rewards_gpts = np.std(gpts_rewards_per_experiment, axis=0)
std_ist_rewards_gpucb = np.std(gpucb_rewards_per_experiment, axis=0)
plt.plot(std_ist_rewards_gpts, "-", label="ist std gpts", color="r")
plt.plot(std_ist_rewards_gpucb, "-", label="ist std gpucb", color="g")
plt.legend()
plt.title("Standard Deviation Instantaneous Reward")
# average instantaneous regret
plt.subplot(2, 2, 3)  # 2 righe, 1 colonna, primo subplot
avg_ist_regregpts_gpts = gpts_regregpts_per_experiment.mean(axis=0)
avg_ist_regregpts_gpucb = gpucb_regregpts_per_experiment.mean(axis=0)
plt.plot(avg_ist_regregpts_gpts, "-", label="ist avg regregpts gpts", color="r")
plt.plot(avg_ist_regregpts_gpucb, "-", label="ist avg regret gpucb", color="g")
plt.legend()
plt.title("Average of Instantaneous Regregpts")
plt.subplot(2, 2, 4)  # 2 righe, 1 colonna, primo subplot
std_ist_regregpts_gpts = np.std(gpts_regregpts_per_experiment, axis=0)
std_ist_regregpts_gpucb = np.std(gpucb_regregpts_per_experiment, axis=0)
plt.plot(std_ist_regregpts_gpts, "-", label="ist std regregpts gpts", color="r")
plt.plot(std_ist_regregpts_gpucb, "-", label="ist std regregpts gpucb", color="g")
plt.legend()
plt.title("Standard Deviation Instantaneous Regregpts")

gpts_rewards_per_experiment_cumsum = gpts_rewards_per_experiment.cumsum(axis=1)
gpucb_rewards_per_experiment_cumsum = gpucb_rewards_per_experiment.cumsum(axis=1)
gpts_regregpts_per_experiment_cumsum = gpts_regregpts_per_experiment.cumsum(axis=1)
gpucb_regregpts_per_experiment_cumsum = gpucb_regregpts_per_experiment.cumsum(axis=1)

plt.figure(2)
plt.subplot(2, 2, 1)  # 2 righe, 1 colonna, primo subplot
avg_cumsum_rewards_gpts = gpts_rewards_per_experiment_cumsum.mean(axis=0)
avg_cumsum_rewards_gpucb = gpucb_rewards_per_experiment_cumsum.mean(axis=0)
plt.plot(avg_cumsum_rewards_gpts, "-", label="cumsum avg gpts", color="r")
plt.plot(avg_cumsum_rewards_gpucb, "-", label="cumsum avg gpucb", color="g")
plt.legend()
plt.title("Average of Cumulative Reward")
plt.subplot(2, 2, 2)  # 2 righe, 1 colonna, primo subplot
std_cumsum_rewards_gpts = np.std(gpts_rewards_per_experiment_cumsum, axis=0)
std_cumsum_rewards_gpucb = np.std(gpucb_rewards_per_experiment_cumsum, axis=0)
plt.plot(std_cumsum_rewards_gpts, "-", label="cumsum std gpts", color="r")
plt.plot(std_cumsum_rewards_gpucb, "-", label="cumsum std gpucb", color="g")
plt.legend()
plt.title("Standard Deviation of Cumulative Reward")
plt.subplot(2, 2, 3)  # 2 righe, 1 colonna, primo subplot
avg_cumsum_regregpts_gpts = gpts_regregpts_per_experiment_cumsum.mean(axis=0)
avg_cumsum_regregpts_gpucb = gpucb_regregpts_per_experiment_cumsum.mean(axis=0)
plt.plot(avg_cumsum_regregpts_gpts, "-", label="cumsum avg regregpts gpts", color="r")
plt.plot(avg_cumsum_regregpts_gpucb, "-", label="cumsum avg regregpts gpucb", color="g")
plt.legend()
plt.title("Average of Cumulative Regret")
plt.subplot(2, 2, 4)  # 2 righe, 1 colonna, primo subplot
std_cumsum_regregpts_gpts = np.std(gpts_regregpts_per_experiment_cumsum, axis=0)
std_cumsum_regregpts_gpucb = np.std(gpucb_regregpts_per_experiment_cumsum, axis=0)
plt.plot(std_cumsum_regregpts_gpts, "-", label="cumsum std regregpts gpts", color="r")
plt.plot(std_cumsum_regregpts_gpucb, "-", label="cumsum std regregpts gpucb", color="g")
plt.legend()
plt.title("Standard Deviation of Cumulative Regret")
plt.show()

print("-------------------------")
print(gpts_learner.pulled_arms)
print("-------------------------")
print(gpucb_learner.pulled_arms)
print(opt_bid)