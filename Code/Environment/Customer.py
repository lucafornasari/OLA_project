import json
import numpy as np
import os

class Customer:
    def __init__(self, feature_1, feature_2):
        self.feature_1 = feature_1
        self.feature_2 = feature_2

    def get_features(self):
        return self.feature_1, self.feature_2

