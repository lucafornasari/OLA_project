from Code.Environment.Learner import *
from Code.Environment.GPUCB1_Learner import GPUCB1_Learner
from Code.Environment.TS_Learner import TS_Learner


class GPUCB1_Learner_3(Learner):
    def __init__(self, bids, prices):
        self.bid = GPUCB1_Learner(bids)
        self.prices = TS_Learner(prices)

    def update(self, pulled_arm_price, pulled_arm_bid, reward):
        self.bid.update(pulled_arm_bid, reward[0])
        self.prices.update(pulled_arm_price, reward)

    def pull_arm(self):
        idx_bid = self.bid.pull_arm()
        idx_prices=self.prices.pull_arms()
        return idx_bid, idx_prices
