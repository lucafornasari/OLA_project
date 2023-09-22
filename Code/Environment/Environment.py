import numpy as np
import random
import matplotlib.pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
import joblib


class Environment:
    def __init__(self):
        self.users = []
        self.prices = [150, 200, 250, 300, 350]
        self.prod_cost = 80
        self.bids = np.linspace(0.0, 100, 100, dtype = int)
        self.classes = ["C1", "C2", "C3"]

    def get_class_from_features(self, f1, f2):
        if f1 == 0 and f2 == 0:
            return "C1"
        elif f1 == 0 and f2 == 1:
            return "C2"
        elif f1 == 1:
            return "C3"
        else:
            return None

    def get_clicks(self, bid, _user_class, min_bid=10):
        # Define the function for number of clicks for a specific class
        # Return the number of clicks based on the bid and user class
        configs = {"C1": {"max_clicks": 40, "steepness": 0.07, "noise": 1.0},
                   "C2": {"max_clicks": 80, "steepness": 0.25, "noise": 1.0},
                   "C3": {"max_clicks": 50, "steepness": 0.42, "noise": 1.0}}
        max_clicks = configs[_user_class]["max_clicks"]
        steepness = configs[_user_class]["steepness"]
        noise = configs[_user_class]["noise"]

        if bid < min_bid:
            return 0
        ret_val = max_clicks * (1 - np.exp(-steepness * (bid)))

        return ret_val

    def sample_clicks(self, bid, _user_class, min_bid=10):
        configs = {"C1": {"max_clicks": 40, "steepness": 0.07, "noise": 1.0},
                   "C2": {"max_clicks": 80, "steepness": 0.25, "noise": 1.0},
                   "C3": {"max_clicks": 50, "steepness": 0.42, "noise": 1.0}}
        max_clicks = configs[_user_class]["max_clicks"]
        steepness = configs[_user_class]["steepness"]
        noise = configs[_user_class]["noise"]
        return self.get_clicks(bid, _user_class)+np.random.normal(0, noise)


    def get_costs(self, bid, _user_class, scale_factor = 1, min_bid=10):
        configs = {"C1": {"max_clicks": 40, "steepness": 0.07, "noise": 1.0},
                   "C2": {"max_clicks": 80, "steepness": 0.25, "noise": 1.0},
                   "C3": {"max_clicks": 50, "steepness": 0.42, "noise": 1.0}}
        max_clicks = configs[_user_class]["max_clicks"]
        steepness = configs[_user_class]["steepness"]

        if bid < min_bid:
            return 0
        ret_val = max_clicks * (1 - np.exp(-steepness *bid)+np.sqrt(bid))

        return ret_val
        # return bid * scale_factor * self.get_clicks(bid, _user_class)

    def sample_costs(self, bid, _user_class,scale_factor =1 ):
        configs = {"C1": {"max_clicks": 40, "steepness": 0.07, "noise": 1.6},
                   "C2": {"max_clicks": 80, "steepness": 0.25, "noise": 1.6},
                   "C3": {"max_clicks": 50, "steepness": 0.42, "noise": 1.6}}
        max_clicks = configs[_user_class]["max_clicks"]
        steepness = configs[_user_class]["steepness"]
        noise = configs[_user_class]["noise"]
        return self.get_costs(bid, _user_class)+np.random.normal(0, noise)

    def get_conversion_prob(self, price, _user_class):
        # Define the function for conversion probability for a specific class
        # Return the conversion probability based on the price and user class
        if _user_class == "C1":
            prob = None
            if price == self.prices[0]:
                prob = 0.15
            elif price == self.prices[1]:
                prob = 0.35
            elif price == self.prices[2]:
                prob = 0.2
            elif price == self.prices[3]:
                prob = 0.1
            elif price == self.prices[4]:
                prob = 0.05
            return prob
        elif _user_class == "C2":
            prob = None
            if price == self.prices[0]:
                prob = 0.15
            elif price == self.prices[1]:
                prob = 0.2
            elif price == self.prices[2]:
                prob = 0.35
            elif price == self.prices[3]:
                prob = 0.15
            elif price == self.prices[4]:
                prob = 0.05
            return prob
        elif _user_class == "C3":
            prob = None
            if price == self.prices[0]:
                prob = 0.05
            elif price == self.prices[1]:
                prob = 0.1
            elif price == self.prices[2]:
                prob = 0.2
            elif price == self.prices[3]:
                prob = 0.30
            elif price == self.prices[4]:
                prob = 0.05
            return prob
        else:
            return None

    def purchase_decision(self, price, _user_class):
        probability = self.get_conversion_prob(price, _user_class)
        return np.random.binomial(1, probability)  # Bernoulli distribution

    def plot_clicks_functions(self):
        bids = np.linspace(0, 100, 100)
        clicks_C1 = [self.get_clicks(bid, "C1") for bid in bids]
        clicks_C2 = [self.get_clicks(bid, "C2") for bid in bids]
        clicks_C3 = [self.get_clicks(bid, "C3") for bid in bids]

        plt.plot(bids, clicks_C1, label='C1')
        plt.plot(bids, clicks_C2, label='C2')
        plt.plot(bids, clicks_C3, label='C3')
        plt.xlabel('Bid')
        plt.ylabel('Number of Clicks')
        plt.title('Number of Clicks as Bid Varies')
        plt.legend()
        plt.show()

    def plot_costs_functions(self):
        bids = np.linspace(0, 100, 100)
        costs_C1 = [self.get_costs(bid, "C1") for bid in bids]
        costs_C2 = [self.get_costs(bid, "C2") for bid in bids]
        costs_C3 = [self.get_costs(bid, "C3") for bid in bids]

        plt.plot(bids, costs_C1, label='C1')
        plt.plot(bids, costs_C2, label='C2')
        plt.plot(bids, costs_C3, label='C3')
        plt.xlabel('Bid')
        plt.ylabel('Cumulative Costs')
        plt.title('Cumulative Costs as Bid Varies')
        plt.legend()
        plt.show()

    def round(self, _user_class, pulled_arm, optimal_bid):
        tests=self.get_clicks(optimal_bid, _user_class)
        positives = np.random.binomial(1, self.get_conversion_prob(self.prices[pulled_arm], _user_class), np.round(tests).astype(int))
        reward = np.sum(positives) * (self.prices[pulled_arm] - self.prod_cost) - self.get_costs(optimal_bid, _user_class)
        return reward, tests, np.sum(positives)

    def part3_round(self, _user_class, pulled_arm_price, pulled_arm_bid):
        tests= np.round(max(0,self.sample_clicks(pulled_arm_bid, _user_class))).astype(int)
        positives = np.random.binomial(1, self.get_conversion_prob(self.prices[pulled_arm_price], _user_class), tests)
        reward = np.sum(positives) *(self.prices[pulled_arm_price] - self.prod_cost) - max(0,self.sample_costs(pulled_arm_bid, _user_class))
        return reward, tests, np.sum(positives)

    def part4_round(self, context_classes, pulled_arm, pulled_bid, t):
        d = {'f_1': [], 'f_2': [], 'pos_conv': [], 'n_clicks': [], 'costs': [], 'price': [], 'bid': [], 't': [], 'price_arm': [], 'bid_arm': []}
        tot_result = 0
        tot_clicks = 0
        tot_reward = 0
        for c in context_classes:
            d['f_1'].append(c[0])
            d['f_2'].append(c[1])
            d['price'].append(self.prices[pulled_arm])
            d['bid'].append(pulled_bid)
            user_class = self.get_class_from_features(int(c[0]), int(c[1]))
            clicks = max(0, self.sample_clicks(pulled_bid, user_class))
            d['n_clicks'].append(clicks)
            positive = np.sum(np.random.binomial(1, self.get_conversion_prob(self.prices[pulled_arm], user_class),
                                               np.round(clicks).astype(int)))
            d['pos_conv'].append(positive)
            costs = self.sample_costs(pulled_bid, user_class)
            reward = positive * (self.prices[pulled_arm] - self.prod_cost) - costs
            d['costs'].append(costs)
            tot_result += positive
            tot_clicks += clicks
            tot_reward += reward
            d['t'].append(t)
            d['price_arm'].append(pulled_arm)
            d['bid_arm'].append(pulled_bid)

        return tot_reward, tot_clicks, tot_result, d

    def seasonal_prob(self,phase,price):
        configs={"phase_1":{150 : 0.15, 200: 0.25, 250: 0.35, 300: 0.1, 350: 0.05},
                 "phase_2":{150 : 0.15, 200: 0.55, 250: 0.15, 300: 0.1, 350: 0.05},
                 "phase_3":{150 : 0.15, 200: 0.15, 250: 0.2, 300: 0.40, 350: 0.05}}
        return configs[phase][price]

    def seasonal_prob_round(self, _user_class,phase, pulled_arm, optimal_bid):
        tests= np.round(self.get_clicks(optimal_bid, _user_class)).astype(int)
        result = np.random.binomial(1, self.seasonal_prob(phase,self.prices[pulled_arm]), tests)
        tested_conv_prob=np.sum(result)/tests
        reward = tests*tested_conv_prob * (self.prices[pulled_arm] - self.prod_cost) - self.get_costs(optimal_bid, _user_class)
        return reward, tests
# env = Environment()
# env.clicks_learning("C1")
