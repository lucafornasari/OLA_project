import numpy as np
import matplotlib.pyplot as plt
from Environment import Environment
from GPTS_Learner import GPTS_Learner
from GPUCB1_Learner import GPUCB1_Learner
from GPUCB_Learner import GPUCB_Learner
from Clairvoyant import*

env = Environment()
customer_class = "C1"

T = 250

for t in range(T):
    print(env.sample_clicks(env.bids[4], customer_class))