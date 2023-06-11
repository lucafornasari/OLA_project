import numpy as np
import random
import matplotlib.pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from Code.Environment.Customer import Customer
import joblib


class Environment:
    def __init__(self):
        self.users = []
        self.prices = [150, 175, 190, 210, 225]
        self.prod_cost = 80
        self.bids = np.linspace(0.0, 1.0, 100)

    def generate_users(self, num_users):
        # Generate a list of random users with random features
        for _ in range(num_users):
            f1 = random.randint(0, 1)
            f2 = random.randint(0, 1)
            user = Customer(f1, f2)
            self.users.append(user)

    def get_clicks(self, bid, _user_class):
        # Define the function for number of clicks for a specific class
        # Return the number of clicks based on the bid and user class
        if _user_class == "C1":
            _clicks = (1.0 - np.exp(-5.0*bid)) * 100
            sigma = 5.0
        elif _user_class == "C2":
            _clicks = 0.5 * bid + 5
            sigma = 2
        elif _user_class == "C3":
            _clicks = 0.3 * bid + 2
            sigma = 3
        else:
            return None  # Handle the case if features do not match any class

        return _clicks + np.random.normal(0, sigma)

    def get_costs(self, bid, _user_class):
        # Define the function for cumulative cost for a specific class
        # Return the cumulative cost based on the bid and user class
        if _user_class == "C1":
            costs = 0.005 * bid**2 + 0.2 * bid + 10
            sigma = 1
        elif _user_class == "C2":
            costs = 0.002 * bid**2 + 0.1 * bid + 5
            sigma = 2
        elif _user_class == "C3":
            costs = 0.001 * bid**2 + 0.05 * bid + 2
            sigma = 3
        else:
            return None  # Handle the case if features do not match any class

        return max(0, costs + np.random.normal(0, sigma))

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
            gp = GaussianProcessRegressor(kernel=kernel, alpha=noise_std, normalize_y=True,
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
        clicks_C1 = [self.get_clicks_for_class(bid, "C1") for bid in bids]
        clicks_C2 = [self.get_clicks_for_class(bid, "C2") for bid in bids]
        clicks_C3 = [self.get_clicks_for_class(bid, "C3") for bid in bids]

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
        costs_C1 = [self.get_costs_for_class(bid, "C1") for bid in bids]
        costs_C2 = [self.get_costs_for_class(bid, "C2") for bid in bids]
        costs_C3 = [self.get_costs_for_class(bid, "C3") for bid in bids]

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


#env = Environment()
#env.clicks_learning("C1")