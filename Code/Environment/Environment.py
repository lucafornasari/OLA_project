import numpy as np
import random
import matplotlib.pyplot as plt
from Code.Environment.Customer import Customer


class Environment:
    def __init__(self):
        self.users = []
        self.prices = [150, 175, 190, 210, 225]
        self.margin = 0.4

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
            _clicks = 0.8 * bid + 10
            sigma = 1
        elif _user_class == "C2":
            _clicks = 0.5 * bid + 5
            sigma = 2
        elif _user_class == "C3":
            _clicks = 0.3 * bid + 2
            sigma = 3
        else:
            return None  # Handle the case if features do not match any class

        return max(0, _clicks + np.random.normal(0, sigma))

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
            return 1 / (1 + np.exp(-0.1 * price + 1))
        elif _user_class == "C2":
            return 1 / (1 + np.exp(-0.05 * price + 0.5))
        elif _user_class == "C3":
            return 1 / (1 + np.exp(-0.2 * price + 2))
        else:
            return None  # Handle the case if features do not match any class

    def purchase_decision(self, price, _user_class):
        probability = self.get_conversion_prob(price, _user_class)
        return np.random.binomial(1, probability)  # Bernoulli distribution

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
