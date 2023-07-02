import numpy as np
import random
import matplotlib.pyplot as plt


class Customer:
    def __init__(self, f1, f2):
        self.f1 = f1
        self.f2 = f2
        self.user_class = self.assign_user_class()

    def assign_user_class(self):
        # Assign user class based on features (F1 and F2)
        if self.f1 == 0 and self.f2 == 0:
            return "C1"
        elif self.f1 == 0 and self.f2 == 1:
            return "C2"
        elif self.f1 == 1:
            return "C3"
        else:
            return None  # Handle the case if features do not match any class

    def get_class(self):
        return self.user_class