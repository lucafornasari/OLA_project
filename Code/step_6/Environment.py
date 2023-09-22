import numpy as np
import random
import matplotlib.pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from Customer import Customer
import joblib


class Environment:
    def __init__(self):
        self.users = []
        self.prices = [150, 200, 250, 300, 350]
        self.prod_cost = 80
        self.bids = np.linspace(0, 100, 100, dtype=int)

    def generate_users(self, num_users):
        # Generate a list of random users with random features
        for _ in range(num_users):
            f1 = random.randint(0, 1)
            f2 = random.randint(0, 1)
            user = Customer(f1, f2)
            self.users.append(user)

    def get_clicks(self, bid, _user_class, min_bid=10):
        # Define the function for number of clicks for a specific class
        # Return the number of clicks based on the bid and user class
        configs = {"C1": {"max_clicks": 40, "steepness": 0.07, "noise": 1.0},
                   "C2": {"max_clicks": 80, "steepness": 0.95, "noise": 2.0},
                   "C3": {"max_clicks": 50, "steepness": 1.2, "noise": 4}}
        max_clicks = configs[_user_class]["max_clicks"]
        steepness = configs[_user_class]["steepness"]
        noise = configs[_user_class]["noise"]

        if bid < min_bid:
            return 0
        ret_val = max_clicks * (1 - np.exp(-steepness * (bid - min_bid)))

        return ret_val

    def sample_clicks(self, bid, _user_class, min_bid=10):
        configs = {"C1": {"max_clicks": 40, "steepness": 0.05, "noise": 1.0},
                   "C2": {"max_clicks": 80, "steepness": 0.17, "noise": 2.0},
                   "C3": {"max_clicks": 50, "steepness": 0.1, "noise": 4}}
        max_clicks = configs[_user_class]["max_clicks"]
        steepness = configs[_user_class]["steepness"]
        noise = configs[_user_class]["noise"]
        return self.get_clicks(bid, _user_class)+np.random.normal(0, noise)


    def get_costs(self, bid, _user_class, scale_factor = 1, min_bid=10):
        configs = {"C1": {"max_clicks": 40, "steepness": 0.55, "noise": 1.0},
                   "C2": {"max_clicks": 80, "steepness": 0.95, "noise": 2.0},
                   "C3": {"max_clicks": 50, "steepness": 1.2, "noise": 4}}
        max_clicks = configs[_user_class]["max_clicks"]
        steepness = configs[_user_class]["steepness"]

        if bid < min_bid:
            return 0
        ret_val = max_clicks * (1 - np.exp(-steepness * (bid - min_bid)))*2

        return ret_val
        # return bid * scale_factor * self.get_clicks(bid, _user_class)

    def sample_costs(self, bid, _user_class,scale_factor =1 ):
        configs = {"C1": {"max_clicks": 40, "steepness": 0.05, "noise": 1.0},
                   "C2": {"max_clicks": 80, "steepness": 0.17, "noise": 2.0},
                   "C3": {"max_clicks": 50, "steepness": 0.1, "noise": 4}}
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
                prob = 0.25
            elif price == self.prices[2]:
                prob = 0.35
            elif price == self.prices[3]:
                prob = 0.1
            elif price == self.prices[4]:
                prob = 0.05
            return prob
            # return 1 / (1 + np.exp(-0.1 * price + 1))
        elif _user_class == "C2":
            return 1 / (1 + np.exp(-0.05 * price + 0.5))
        elif _user_class == "C3":
            return 1 / (1 + np.exp(-0.2 * price + 2))
        else:
            return None  # Handle the case if features do not match any class

    def purchase_decision(self, price, _user_class):
        probability = self.get_conversion_prob(price, _user_class)
        return np.random.binomial(1, probability)  # Bernoulli distribution

    def generate_observations(x, _user_class, noise_std):
        return None  # n(x) + np.random.normal(0, noise_std, size=n(x).shape)

    def clicks_learning(self, _user_class):
        n_obs = 365
        # for the 3 classes need to change the parameters a bit
        x_obs = np.array([])
        y_obs = np.array([])
        noise_std = 5.0

        for i in range(0, n_obs):
            new_x_obs = np.random.choice(self.bids, 1)
            new_y_obs = self.get_clicks(new_x_obs, _user_class)

            x_obs = np.append(x_obs, new_x_obs)
            y_obs = np.append(y_obs, new_y_obs)

            X = np.atleast_2d(x_obs).T
            Y = y_obs.ravel()

            theta = 1.0
            l = 1.0
            kernel = C(theta, (1e-3, 1e3)) * RBF(l, (1e-3, 1e3))
            gp = GaussianProcessRegressor(kernel=kernel, alpha=noise_std ** 2, normalize_y=True,
                                          n_restarts_optimizer=10)  # alpha=noise_std**2

            gp.fit(X, Y)

            x_pred = np.atleast_2d(self.bids).T
            y_pred, sigma = gp.predict(x_pred, return_std=True)

            print(i)

            self.plot_gp(x_pred, y_pred, X, Y, sigma, _user_class)
        joblib.dump(gp, "model.joblib")

    def plot_gp(self, _x_pred, _y_pred, _X, _Y, _sigma, _userclass):
        plt.figure(364)
        plt.plot(_x_pred, self.get_clicks(_x_pred, _userclass), 'r:', label=r'$n(x)$')
        plt.plot(_X.ravel(), _Y, 'ro', label=u'Observed Clicks')
        plt.plot(_x_pred, _y_pred, 'b-', label=u'Predicted Clicks')
        plt.fill(np.concatenate([_x_pred, _x_pred[::-1]]),
                 np.concatenate([_y_pred - 1.96 * _sigma, (_y_pred + 1.96 * _sigma)[::-1]]),
                 alpha=.5, fc='b', ec='None', label='95% conf interval')
        plt.xlabel('$x$')
        plt.ylabel('$n(x)$')
        plt.legend(loc='lower right')
        plt.show()


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

    def plot_conversion_functions(self):
        prices = np.linspace(0, 50, 100)
        conversion_probs_C1 = [self.get_conversion_prob_for_class(price, "C1") for price in prices]
        conversion_probs_C2 = [self.get_conversion_prob_for_class(price, "C2") for price in prices]
        conversion_probs_C3 = [self.get_conversion_prob_for_class(price, "C3") for price in prices]

        plt.plot(prices, conversion_probs_C1, label='C1')
        plt.plot(prices, conversion_probs_C2, label='C2')
        plt.plot(prices, conversion_probs_C3, label='C3')
        plt.xlabel('Price')
        plt.ylabel('Conversion Probability')
        plt.title('Conversion Probability as Price Varies')
        plt.legend()
        plt.show()

    def part3_round(self, _user_class, pulled_arm, optimal_bid):
        clicks=max(0,self.sample_clicks(optimal_bid, _user_class))
        result = np.random.binomial(1, self.get_conversion_prob(self.prices[pulled_arm], _user_class), np.round(clicks).astype(int))
        reward = np.sum(result) * (self.prices[pulled_arm] - self.prod_cost) - self.sample_costs(optimal_bid, _user_class)
        return np.sum(result), clicks - np.sum(result), reward

    def seasonal_prob(self,phase,price):
        configs={"phase_1":{150 : 0.15, 200: 0.25, 250: 0.35, 300: 0.1, 350: 0.05},
                 "phase_2":{150 : 0.15, 200: 0.55, 250: 0.15, 300: 0.1, 350: 0.05},
                 "phase_3":{150 : 0.15, 200: 0.15, 250: 0.2, 300: 0.40, 350: 0.05},
                 "phase_4":{150 : 0.15, 200: 0.45, 250: 0.1, 300: 0.15, 350: 0.05},
                 "phase_5":{150 : 0.15, 200: 0.15, 250: 0.2, 300: 0.1, 350: 0.35}}
        return configs[phase][price]
    def sample_seasonal_prob(self,phase,price):
        configs={"phase_1":{150 : 0.15, 200: 0.25, 250: 0.35, 300: 0.1, 350: 0.05},
                 "phase_2":{150 : 0.15, 200: 0.45, 250: 0.25, 300: 0.1, 350: 0.05},
                 "phase_3":{150 : 0.15, 200: 0.25, 250: 0.1, 300: 0.40, 350: 0.05},
                 "phase_4":{150 : 0.15, 200: 0.45, 250: 0.1, 300: 0.15, 350: 0.05},
                 "phase_5":{150 : 0.15, 200: 0.15, 250: 0.2, 300: 0.1, 350: 0.35}}
        prob=configs[phase][price]


    def seasonal_prob_round(self, _user_class,phase, pulled_arm, optimal_bid):
        tests= np.round(self.get_clicks(optimal_bid, _user_class)).astype(int)
        result = np.random.binomial(1, self.seasonal_prob(phase,self.prices[pulled_arm]), tests)
        tested_conv_prob=np.sum(result)/tests
        reward = self.get_clicks(optimal_bid, _user_class)*tested_conv_prob * (self.prices[pulled_arm] - self.prod_cost) - self.get_costs(optimal_bid, _user_class)
        return reward, tests
    

env = Environment()
#env.plot_clicks_functions()
#env.plot_costs_functions()