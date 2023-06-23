import numpy as np
import matplotlib.pyplot as plt
import json
import pandas as pd

import json

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

class Payment_characteristics_generator:
    settings = Settings()
    _reasonable_price = {1: settings["c1_reasonable_price"], 2: settings["c2_reasonable_price"], 3: settings["c3_reasonable_price"]}
    _elasticity = {1: settings["c1_elasticity"], 2: settings["c2_elasticity"], 3: settings["c3_elasticity"]}
    _skepticism = {1: settings["c1_skepticism"], 2: settings["c2_skepticism"], 3: settings["c3_skepticism"]}
    _max_conversion_rate = {1: settings["c1_max_conversion_rate"], 2: settings["c2_max_conversion_rate"], 3: settings["c3_max_conversion_rate"]}

    _noise_reasonable_price = {1: settings["c1_noise_reasonable_price"], 2: settings["c2_noise_reasonable_price"], 3: settings["c3_noise_reasonable_price"]}
    _noise_elasticity = {1: settings["c1_noise_elasticity"], 2: settings["c2_noise_elasticity"], 3: settings["c3_noise_elasticity"]}
    _noise_max_conversion_rate = {1: settings["c1_noise_max_conversion_rate"], 2: settings["c2_noise_max_conversion_rate"], 3: settings["c3_noise_max_conversion_rate"]}

class PopulationGenerator:
    
    def generate_payment_characteristics(belonging_class):
        charachteristics = np.array([Payment_characteristics_generator._reasonable_price[belonging_class] + np.random.normal(0, Payment_characteristics_generator._noise_reasonable_price[belonging_class]),
                            Payment_characteristics_generator._elasticity[belonging_class] + np.random.normal(0, Payment_characteristics_generator._noise_elasticity[belonging_class]),
                            Payment_characteristics_generator._skepticism[belonging_class] + np.random.normal(0, Payment_characteristics_generator._noise_elasticity[belonging_class]),
                            Payment_characteristics_generator._max_conversion_rate[belonging_class] + np.random.normal(0, Payment_characteristics_generator._noise_max_conversion_rate[belonging_class])])
        # make all values positive
        charachteristics = np.abs(charachteristics)
        # make sure that the max conversion rate is not greater than 1
        charachteristics[3] = min(charachteristics[3], 1)
        return charachteristics

    def generate_payment_characteristics_for_population(population):
        return np.array([PopulationGenerator.generate_payment_characteristics(x) for x in population])

        

    #generate a population with two binary features
    def generate_population():
        settings = Settings()
        f1 =  np.random.choice([0, 1], size=settings["sample_size"], p=[1 - settings["pf1"], settings["pf1"]])
        f2 =  np.random.choice([0, 1], size=settings["sample_size"], p=[1 - settings["pf2"], settings["pf2"]])
        return np.array(list(zip(f1, f2)))

    #splits the sample into three groups based on the features
    def classify(population):
        classifier = lambda x: 3 if x[0] == 1 else 2 if x[1] == 1 else 1
        return np.array([classifier(x) for x in population])

    #generate pandas dataframe with f1,f2, class, payment characteristics
    def generate_dataframe():
        population = PopulationGenerator.generate_population()
        classes = PopulationGenerator.classify(population)
        payment_characteristics = PopulationGenerator.generate_payment_characteristics_for_population(classes)
        return pd.DataFrame(np.hstack((population, classes.reshape(-1,1), payment_characteristics)), columns=["f1", "f2", "class", "reasonable_price", "elasticity", "skepticism", "max_conversion_rate"])



  
    def generate_conversion_rate_per_price(
        preferred_price,
        elasticity,
        skepticism,
        max_conversion_rate,
        prices = Settings()['prices'],
        fix_seed = True,
        seed = 42,
        noise_level = Settings()["default_noise"]
        ):
        
        if fix_seed:
            np.random.seed(seed)
        
        rates = np.zeros_like(prices)

        above_preferred = prices >= preferred_price
        below_preferred = prices < preferred_price

        diff_above = prices[above_preferred] - preferred_price
        diff_below = preferred_price - prices[below_preferred]

        rates[above_preferred] = np.exp(-elasticity * diff_above)
        rates[below_preferred] = np.exp(-skepticism * diff_below)

        noise = np.random.normal(0, noise_level, prices.shape)
        rates += noise

        # Apply cap and ensure values are within [0, 1] range
        rates = np.clip(rates, 0, 1)

        return rates


    #generate conversion rates per price for every user in the population
    def generate_conversion_rates(population):
        settings = Settings()
        conversion_rates = np.zeros((population.shape[0], settings["prices"].shape[0]))
        for i, row in population.iterrows():
            conversion_rates[i] = PopulationGenerator.generate_conversion_rate_per_price(row["reasonable_price"], row["elasticity"], row["skepticism"], row["max_conversion_rate"])
        return conversion_rates

    #attach conversion rates to the population dataframe, make one column for every price, name is cvr_{idx}
    def attach_conversion_rates(population):
        settings = Settings()
        conversion_rates = PopulationGenerator.generate_conversion_rates(population)
        #return pd.concat([population, pd.DataFrame(conversion_rates, columns=settings["prices"])], axis=1)
        columns = ["cvr_{}".format(i) for i in range(settings["prices"].shape[0])]
        return pd.concat([population, pd.DataFrame(conversion_rates, columns=columns)], axis=1)

    


    #decide whether the user buys the item or not for every price
    def generate_purchases(population):
        settings = Settings()
        columns = ["cvr_{}".format(i) for i in range(settings["prices"].shape[0])]
        purchases = np.zeros((population.shape[0], settings["prices"].shape[0]))
        for i, row in population.iterrows():
            purchases[i] = np.random.binomial(1, row[columns])
        return purchases

    #attach purchases to the population dataframe, make one column for every price
    def attach_purchases(population):
        settings = Settings()
        purchases = PopulationGenerator.generate_purchases(population)
        columns = ["p{}".format(i) for i in range(settings["prices"].shape[0])]
        return pd.concat([population, pd.DataFrame(purchases, columns=columns)], axis=1)

    def create():
        population = PopulationGenerator.generate_dataframe()    
        population = PopulationGenerator.attach_conversion_rates(population)
        population = PopulationGenerator.attach_purchases(population)
        return population
