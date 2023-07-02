import json
import numpy as np
class Settings(dict):
    _instance = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls, *args, **kwargs)
        return cls._instance

    def __init__(self):
        self.load_settings()

    def load_settings(self):
        with open('settings.json', 'r') as file:
            settings = json.load(file)
            settings["prices"] = np.array(settings["prices"], dtype=np.float32)
            self.clear()
            self.update(settings)

    def update_settings(self):
        self.load_settings()