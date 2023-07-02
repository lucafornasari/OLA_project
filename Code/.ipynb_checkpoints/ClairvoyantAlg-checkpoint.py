import numpy as np
import random
from Code.Environment.Environment import Environment


def optimize(_env):

    classes = ["C1", "C2", "C3"]
    prices = env.prices
    bids = env.bids

    opt_prices = dict([
        ("C1", 0),
        ("C2", 0),
        ("C3", 0)
    ])

    opt_bids = dict([
        ("C1", 0),
        ("C2", 0),
        ("C3", 0)
    ])

    for c in classes:
        opt_prob = 0
        for p in prices:
            conv_prob = env.get_conversion_prob(p, c)
            opt_p = 0
            if conv_prob > opt_prob:
                opt_prob = conv_prob
                opt_p = p
        opt_prices[c] = opt_p

    for c in classes:
        reward = 0
        for b in bids:
            reward_1 = env.get_clicks(b, c)*env.get_conversion_prob(opt_prices[c], c)*(opt_prices[c] - env.prod_cost) \
                       - env.get_costs(b, c)
            if reward_1 > reward:
                opt_bids[c] = b

    print(opt_prices)
    print(opt_bids)
    return opt_bids, opt_prices


    # define a 3 5x100 matrix containing all possible rewards
    # for each user class find the price with highest conversion probability
    # for each bid:
    # compute the reward



env = Environment()
optimize(env)
