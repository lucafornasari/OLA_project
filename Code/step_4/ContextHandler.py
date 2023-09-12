import numpy as np
import pandas as pd
from Code.Environment.Environment import Environment
from Code.step_3.GPTS_Learner_3 import GPTS_Learner_3
from Code.step_3.GPUCB1_Learner_3 import GPUCB1_Learner_3
from Code.step_2.Clairvoyant import *

env = Environment()


class ContextHandler:

    def __init__(self):
        self.context_ts = [GPTS_Learner_3(env.bids, env.prices)]
        self.context_ucb = [GPUCB1_Learner_3(env.bids, env.prices)]
        self.context_classes_ts = [["00", "01", "10", "11"]]
        self.context_classes_ucb = [["00", "01", "10", "11"]]
        d = {'f_1': [], 'f_2': [], 'pos_conv': [], 'n_clicks': [], 'costs': [], 'price': [], 'bid': []}
        self.dataset_ts = pd.DataFrame(data=d)
        self.dataset_ucb = pd.DataFrame(data=d)

    def update_dataset_ts(self, d):
        ds = pd.DataFrame(data=d)
        self.dataset_ts = pd.concat([self.dataset_ts, ds], ignore_index=True)

    def update_dataset_ucb(self, d):
        ds = pd.DataFrame(data=d)
        self.dataset_ucb = pd.concat([self.dataset_ucb, ds], ignore_index=True)
