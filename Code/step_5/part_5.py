import numpy as np
import matplotlib.pyplot as plt
from Code.Environment.Environment import Environment
from UCB1_Learner import UCB1_Learner
from Clairvoyant import*
from tqdm import tqdm_gui
from Code.Environment.UCB1_Active_Learner import UCB1_Active_Learner
from Code.Environment.UCB1_Passive_Learner import UCB1_Passive_Learner

env = Environment()
customer_class = "C1"

T = 900
opt_prices, opt_bids =optimize(env)
opt_bid=opt_bids["C1"]
phase=["phase_1","phase_2","phase_3"]
opt_prices=[250,200,300]
opt_rewards=[0,0,0]
for i in range(len(phase)):
    opt_rewards[i] = env.get_clicks(opt_bid, customer_class) * env.seasonal_prob(phase[i], opt_prices[i]) * (opt_prices[i]-env.prod_cost) - env.get_costs(opt_bid, customer_class)

n_experimengpts = 1

ucb_rewards_per_experiment = []
ucb_regregpts_per_experiment = []

ucb_active_rewards_per_experiment = []
ucb_active_regregpts_per_experiment = []

ucb_passive_rewards_per_experiment = []
ucb_passive_regregpts_per_experiment = []



for e in range(0, n_experimengpts):
    print("Starting new experiment...")
    
    ucb_learner =UCB1_Learner(env.prices)
    ucb_active_learner=UCB1_Active_Learner(env.prices)
    ucb_passive_learner=UCB1_Passive_Learner(env.prices,20)

    ucb_active_regregpts = []
    ucb_active_rewards=[]
    ucb_passive_regregpts = []
    ucb_regregpts = []
    phase=["phase_1","phase_2","phase_3"]
    current_phase=phase[0]
    opt=opt_rewards[0]

    for t in range(T):
        if(t==(T/3)):
            current_phase=phase[1]
            opt=opt_rewards[1]
            print("oh my change")
        if(t==((T*2)/3)):
            current_phase=phase[2]
            opt=opt_rewards[2]
            print("oh my change numero dos")


        pulled_arm = ucb_learner.pull_arms()
        reward = env.seasonal_prob_round(customer_class,current_phase,pulled_arm,opt_bid)
        ucb_learner.update(pulled_arm, reward)
        ucb_regregpts.append(opt - reward[0])
        # ucb1 Learner passive
        pulled_arm = ucb_passive_learner.pull_arms()
        reward = env.seasonal_prob_round(customer_class,current_phase,pulled_arm,opt_bid)
        ucb_passive_learner.update(pulled_arm, reward)
        ucb_passive_regregpts.append(opt - reward[0])
         # ucb1 Learner active
        pulled_arm = ucb_active_learner.pull_arms()
        reward = env.seasonal_prob_round(customer_class,current_phase,pulled_arm,opt_bid)
        ucb_active_learner.update(pulled_arm, reward)
        ucb_active_regregpts.append(opt - reward[0])
        ucb_active_rewards.append(reward[0])
      
        

    
    ucb_rewards_per_experiment.append(ucb_learner.collected_rewards.tolist())
    ucb_regregpts_per_experiment.append(ucb_regregpts)
    ucb_active_rewards_per_experiment.append(ucb_active_rewards)
    ucb_active_regregpts_per_experiment.append(ucb_active_regregpts)
    ucb_passive_rewards_per_experiment.append(ucb_passive_learner.collected_rewards.tolist())
    ucb_passive_regregpts_per_experiment.append(ucb_passive_regregpts)

    print('Finished experiment #', e)

    print("arms pulled ucb")
    print(ucb_learner.arms_pulled)
    print("----------------------------------")
    print("active detections")
    print(ucb_active_learner.detections)
    print("----------------------------------")
    print("arms pulled active")
    print(ucb_active_learner.arms_pulled)
    print("----------------------------------")
    print("arms pulled passive")
    print(ucb_passive_learner.arms_pulled)
    print("----------------------------------")


ucb_rewards_per_experiment = np.array(ucb_rewards_per_experiment)
ucb_regregpts_per_experiment = np.array(ucb_regregpts_per_experiment)
ucb_active_rewards_per_experiment = np.array(ucb_active_rewards_per_experiment)
ucb_active_regregpts_per_experiment = np.array(ucb_active_regregpts_per_experiment)
ucb_passive_rewards_per_experiment = np.array(ucb_passive_rewards_per_experiment)
ucb_passive_regregpts_per_experiment = np.array(ucb_passive_regregpts_per_experiment)



# average instantaneous reward
plt.figure(2)
plt.subplot(2, 2, 1)  # 2 righe, 1 colonna, primo subplot

avg_ist_rewards_ucb = ucb_rewards_per_experiment.mean(axis=0)
avg_ist_rewards_ucb_active = ucb_active_rewards_per_experiment.mean(axis=0)
avg_ist_rewards_ucb_passive = ucb_passive_rewards_per_experiment.mean(axis=0)


plt.plot(avg_ist_rewards_ucb, "-", label="ist avg ucb", color="g")
plt.plot(avg_ist_rewards_ucb_active, "-", label="ist avg ucb active", color="r")
plt.plot(avg_ist_rewards_ucb_passive, "-", label="ist avg passive", color="y")

plt.legend()
plt.title("Average of Instantaneous Reward")
plt.subplot(2, 2, 2)  # 2 righe, 1 colonna, primo subplot

std_ist_rewards_ucb = np.std(ucb_rewards_per_experiment, axis=0)
std_ist_rewards_ucb_active = np.std(ucb_active_rewards_per_experiment, axis=0)
std_ist_rewards_ucb_passive = np.std(ucb_passive_rewards_per_experiment, axis=0)


plt.plot(std_ist_rewards_ucb, "-", label="ist std ucb", color="g")
plt.plot(std_ist_rewards_ucb_active, "-", label="ist std ucb active", color="r")
plt.plot(std_ist_rewards_ucb_passive, "-", label="ist std ucb passive", color="y")
plt.legend()
plt.title("Standard Deviation Instantaneous Reward")
# average instantaneous regret
plt.subplot(2, 2, 3)  # 2 righe, 1 colonna, primo subplot

avg_ist_regregpts_ucb = ucb_regregpts_per_experiment.mean(axis=0)
avg_ist_regregpts_ucb_active = ucb_active_regregpts_per_experiment.mean(axis=0)
avg_ist_regregpts_ucb_passive = ucb_passive_regregpts_per_experiment.mean(axis=0)


plt.plot(avg_ist_regregpts_ucb, "-", label="ist avg regret ucb", color="g")
plt.plot(avg_ist_regregpts_ucb_active, "-", label="ist avg regret ucb active", color="r")
plt.plot(avg_ist_regregpts_ucb_passive, "-", label="ist avg regret ucb passive", color="y")

plt.legend()
plt.title("Average of Instantaneous Regregpts")
plt.subplot(2, 2, 4)  # 2 righe, 1 colonna, primo subplot

std_ist_regregpts_ucb = np.std(ucb_regregpts_per_experiment, axis=0)
std_ist_regregpts_ucb_active = np.std(ucb_active_regregpts_per_experiment, axis=0)
std_ist_regregpts_ucb_passive = np.std(ucb_passive_regregpts_per_experiment, axis=0)

plt.plot(std_ist_regregpts_ucb, "-", label="ist std regregpts ucb", color="g")
plt.plot(std_ist_regregpts_ucb_active, "-", label="ist std regregpts ucb active", color="r")
plt.plot(std_ist_regregpts_ucb_passive, "-", label="ist std regregpts ucb passive", color="y")

plt.legend()
plt.title("Standard Deviation Instantaneous Regregpts")

ucb_rewards_per_experiment_cumsum = ucb_rewards_per_experiment.cumsum(axis=1)
ucb_regregpts_per_experiment_cumsum = ucb_regregpts_per_experiment.cumsum(axis=1)
ucb_active_rewards_per_experiment_cumsum = ucb_active_rewards_per_experiment.cumsum(axis=1)
ucb_active_regregpts_per_experiment_cumsum = ucb_active_regregpts_per_experiment.cumsum(axis=1)
ucb_passive_rewards_per_experiment_cumsum = ucb_passive_rewards_per_experiment.cumsum(axis=1)
ucb_passive_regregpts_per_experiment_cumsum = ucb_passive_regregpts_per_experiment.cumsum(axis=1)

plt.figure(3)
plt.subplot(2, 2, 1)  # 2 righe, 1 colonna, primo subplot
avg_cumsum_rewards_ucb = ucb_rewards_per_experiment_cumsum.mean(axis=0)
avg_cumsum_rewards_ucb_active = ucb_active_rewards_per_experiment_cumsum.mean(axis=0)
avg_cumsum_rewards_ucb_passive = ucb_passive_rewards_per_experiment_cumsum.mean(axis=0)

plt.plot(avg_cumsum_rewards_ucb, "-", label="cumsum avg ucb", color="g")
plt.plot(avg_cumsum_rewards_ucb_active, "-", label="cumsum avg ucb active", color="r")
plt.plot(avg_cumsum_rewards_ucb_passive, "-", label="cumsum avg ucb passive", color="y")

plt.legend()
plt.title("Average of Cumulative Reward")
plt.subplot(2, 2, 2)  # 2 righe, 1 colonna, primo subplot
std_cumsum_rewards_ucb = np.std(ucb_rewards_per_experiment_cumsum, axis=0)
std_cumsum_rewards_ucb_active = np.std(ucb_active_rewards_per_experiment_cumsum, axis=0)
std_cumsum_rewards_ucb_passive = np.std(ucb_passive_rewards_per_experiment_cumsum, axis=0)

plt.plot(std_cumsum_rewards_ucb, "-", label="cumsum std ucb", color="g")
plt.plot(std_cumsum_rewards_ucb_active, "-", label="cumsum std ucb active", color="r")
plt.plot(std_cumsum_rewards_ucb_passive, "-", label="cumsum std ucb passive", color="y")

plt.legend()
plt.title("Standard Deviation of Cumulative Reward")
plt.subplot(2, 2, 3)  # 2 righe, 1 colonna, primo subplot
avg_cumsum_regregpts_ucb = ucb_regregpts_per_experiment_cumsum.mean(axis=0)
avg_cumsum_regregpts_ucb_active = ucb_active_regregpts_per_experiment_cumsum.mean(axis=0)
avg_cumsum_regregpts_ucb_passive = ucb_passive_regregpts_per_experiment_cumsum.mean(axis=0)

plt.plot(avg_cumsum_regregpts_ucb, "-", label="cumsum avg regregpts ucb", color="g")
plt.plot(avg_cumsum_regregpts_ucb_active, "-", label="cumsum avg regregpts ucb active", color="r")
plt.plot(avg_cumsum_regregpts_ucb_passive, "-", label="cumsum avg regregpts ucb passive", color="y")

plt.legend()
plt.title("Average of Cumulative Regret")
plt.subplot(2, 2, 4)  # 2 righe, 1 colonna, primo subplot
std_cumsum_regregpts_ucb = np.std(ucb_regregpts_per_experiment_cumsum, axis=0)
std_cumsum_regregpts_ucb_active = np.std(ucb_active_regregpts_per_experiment_cumsum, axis=0)
std_cumsum_regregpts_ucb_passive = np.std(ucb_passive_regregpts_per_experiment_cumsum, axis=0)

plt.plot(std_cumsum_regregpts_ucb, "-", label="cumsum std regregpts ucb", color="g")
plt.plot(std_cumsum_regregpts_ucb_active, "-", label="cumsum std regregpts ucb active", color="r")
plt.plot(std_cumsum_regregpts_ucb_passive, "-", label="cumsum std regregpts ucb", color="y")

plt.legend()
plt.title("Standard Deviation of Cumulative Regret")


plt.show()




print("-------------------------")
print(gpucb_learner.pulled_arms)
print(opt_bid)