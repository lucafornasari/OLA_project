import numpy as np
import matplotlib.pyplot as plt

from Code.ClairvoyantAlg import optimize
from Code.Environment.Environment import Environment
from Code.Step_1.TS_Learner import TS_Learner
from Code.Step_1.UCB1_Learner import UCB1_Learner

env = Environment()
customer_class = "C1"

T = 250

opt_bids, opt_prices = optimize(env)
opt_bid = opt_bids[customer_class]
opt_price = opt_prices[customer_class]

_margin = opt_price - env.prod_cost
clicks = env.get_clicks(opt_bid, customer_class)
costs = env.get_costs(opt_bid, customer_class)
opt = clicks * env.get_conversion_prob(opt_price, customer_class) * _margin - costs

n_experiments = 10
ts_rewards_per_experiment = []
ucb_rewards_per_experiment = []
ts_regrets_per_experiment = []
ucb_regrets_per_experiment = []

for e in range(0, n_experiments):
    print("Starting new experiment...")
    ts_learner = TS_Learner(env.prices)
    ucb_learner = UCB1_Learner(env.prices)

    ts_regrets = []
    ucb_regrets = []

    for t in range(0, T):
        # Thompson Sampling Learner
        pulled_arm = ts_learner.pull_arms()
        _reward = env.round(customer_class, pulled_arm, opt_bid)
        # _margin = env.prices[pulled_arm] - env.prod_cost
        # _reward = clicks * _margin * env.purchase_decision(env.prices[pulled_arm], customer_class) - costs
        ts_learner.update(pulled_arm, _reward)
        ts_regrets.append(opt - _reward[2])

        # UCB1 Learner
        pulled_arm = ucb_learner.pull_arms()
        _reward = env.round(customer_class, pulled_arm, opt_bid)
        # _margin = env.prices[pulled_arm] - env.prod_cost
        # _reward = clicks * _margin * env.purchase_decision(env.prices[pulled_arm], customer_class) - costs
        ucb_learner.update(pulled_arm, _reward)
        ucb_regrets.append(opt - _reward[2])

    ts_rewards_per_experiment.append(ts_learner.collected_rewards.tolist())
    ucb_rewards_per_experiment.append(ucb_learner.collected_rewards.tolist())
    ts_regrets_per_experiment.append(ts_regrets)
    ucb_regrets_per_experiment.append(ucb_regrets)

    print('Finished experiment #', e)

ts_rewards_per_experiment = np.array(ts_rewards_per_experiment)
ucb_rewards_per_experiment = np.array(ucb_rewards_per_experiment)
ts_regrets_per_experiment = np.array(ts_regrets_per_experiment)
ucb_regrets_per_experiment = np.array(ucb_regrets_per_experiment)

# average instantaneous reward
plt.figure(1)
plt.subplot(2, 2, 1)  # 2 righe, 1 colonna, primo subplot
avg_ist_rewards_ts = ts_rewards_per_experiment.mean(axis=0)
avg_ist_rewards_ucb = ucb_rewards_per_experiment.mean(axis=0)
plt.plot(avg_ist_rewards_ts, "-", label="ist avg ts", color="r")
plt.plot(avg_ist_rewards_ucb, "-", label="ist avg ucb", color="g")
plt.legend()
plt.title("Average of Instantaneous Reward")
plt.subplot(2, 2, 2)  # 2 righe, 1 colonna, primo subplot
std_ist_rewards_ts = np.std(ts_rewards_per_experiment, axis=0)
std_ist_rewards_ucb = np.std(ucb_rewards_per_experiment, axis=0)
plt.plot(std_ist_rewards_ts, "-", label="ist std ts", color="r")
plt.plot(std_ist_rewards_ucb, "-", label="ist std ucb", color="g")
plt.legend()
plt.title("Standard Deviation Instantaneous Reward")
# average instantaneous regret
plt.subplot(2, 2, 3)  # 2 righe, 1 colonna, primo subplot
avg_ist_regrets_ts = ts_regrets_per_experiment.mean(axis=0)
avg_ist_regrets_ucb = ucb_regrets_per_experiment.mean(axis=0)
plt.plot(avg_ist_regrets_ts, "-", label="ist avg regrets ts", color="r")
plt.plot(avg_ist_regrets_ucb, "-", label="ist avg regret ucb", color="g")
plt.legend()
plt.title("Average of Instantaneous Regrets")
plt.subplot(2, 2, 4)  # 2 righe, 1 colonna, primo subplot
std_ist_regrets_ts = np.std(ts_regrets_per_experiment, axis=0)
std_ist_regrets_ucb = np.std(ucb_regrets_per_experiment, axis=0)
plt.plot(std_ist_regrets_ts, "-", label="ist std regrets ts", color="r")
plt.plot(std_ist_regrets_ucb, "-", label="ist std regrets ucb", color="g")
plt.legend()
plt.title("Standard Deviation Instantaneous Regrets")

ts_rewards_per_experiment_cumsum = ts_rewards_per_experiment.cumsum(axis=1)
ucb_rewards_per_experiment_cumsum = ucb_rewards_per_experiment.cumsum(axis=1)
ts_regrets_per_experiment_cumsum = ts_regrets_per_experiment.cumsum(axis=1)
ucb_regrets_per_experiment_cumsum = ucb_regrets_per_experiment.cumsum(axis=1)

plt.figure(2)
plt.subplot(2, 2, 1)  # 2 righe, 1 colonna, primo subplot
avg_cumsum_rewards_ts = ts_rewards_per_experiment_cumsum.mean(axis=0)
avg_cumsum_rewards_ucb = ucb_rewards_per_experiment_cumsum.mean(axis=0)
plt.plot(avg_cumsum_rewards_ts, "-", label="cumsum avg ts", color="r")
plt.plot(avg_cumsum_rewards_ucb, "-", label="cumsum avg ucb", color="g")
plt.legend()
plt.title("Average of Cumulative Reward")
plt.subplot(2, 2, 2)  # 2 righe, 1 colonna, primo subplot
std_cumsum_rewards_ts = np.std(ts_rewards_per_experiment_cumsum, axis=0)
std_cumsum_rewards_ucb = np.std(ucb_rewards_per_experiment_cumsum, axis=0)
plt.plot(std_cumsum_rewards_ts, "-", label="cumsum std ts", color="r")
plt.plot(std_cumsum_rewards_ucb, "-", label="cumsum std ucb", color="g")
plt.legend()
plt.title("Standard Deviation of Cumulative Reward")
plt.subplot(2, 2, 3)  # 2 righe, 1 colonna, primo subplot
avg_cumsum_regrets_ts = ts_regrets_per_experiment_cumsum.mean(axis=0)
avg_cumsum_regrets_ucb = ucb_regrets_per_experiment_cumsum.mean(axis=0)
plt.plot(avg_cumsum_regrets_ts, "-", label="cumsum avg regrets ts", color="r")
plt.plot(avg_cumsum_regrets_ucb, "-", label="cumsum avg regrets ucb", color="g")
plt.legend()
plt.title("Average of Cumulative Regret")
plt.subplot(2, 2, 4)  # 2 righe, 1 colonna, primo subplot
std_cumsum_regrets_ts = np.std(ts_regrets_per_experiment_cumsum, axis=0)
std_cumsum_regrets_ucb = np.std(ucb_regrets_per_experiment_cumsum, axis=0)
plt.plot(std_cumsum_regrets_ts, "-", label="cumsum std regrets ts", color="r")
plt.plot(std_cumsum_regrets_ucb, "-", label="cumsum std regrets ucb", color="g")
plt.legend()
plt.title("Standard Deviation of Cumulative Regret")
plt.show()
