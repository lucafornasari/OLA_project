import numpy as np
import random
from Environment import Environment


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
        opt_p = 0
        for p in prices:
            conv_prob = env.get_conversion_prob(p, c) * p
            if conv_prob > opt_prob:
                opt_prob = conv_prob
                opt_p = p
        opt_prices[c] = opt_p

    for c in classes:
        reward = 0
        for b in bids:
            _clicks = env.get_clicks(b, c)
            _conv = env.get_conversion_prob(opt_prices[c], c)
            _margin = opt_prices[c] - env.prod_cost
            _cost = env.get_costs(b, c)
            reward_1 = _clicks * _conv * _margin - _cost
            if reward_1 > reward:
                opt_bids[c] = b
                reward = reward_1
        print(reward)

    print(opt_prices)
    print(opt_bids)
    return opt_prices,opt_bids


env = Environment()
optimize(env)
