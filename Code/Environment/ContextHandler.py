import numpy as np
import pandas as pd
from Code.Environment.Environment import Environment
from Code.Environment.GPTS_Learner_3 import GPTS_Learner_3
from Code.Environment.GPUCB1_Learner_3 import GPUCB1_Learner_3
from Code.Environment.Clairvoyant import *

env = Environment()


class ContextHandler:

    def __init__(self):
        self.context_ts = [GPTS_Learner_3(env.bids, env.prices)]
        self.context_ucb = [GPUCB1_Learner_3(env.bids, env.prices)]
        self.context_classes_ts = [["00", "01", "10", "11"]]
        self.context_classes_ucb = [["00", "01", "10", "11"]]
        d = {'f_1': [], 'f_2': [], 'pos_conv': [], 'n_clicks': [], 'costs': [], 'price': [], 'bid': [], 't': [],
             'price_arm': [], 'bid_arm': []}
        self.dataset_ts = pd.DataFrame(data=d)
        self.dataset_ucb = pd.DataFrame(data=d)
        self.confidence = 0.95
        self.generated_contexts = []

    def update_dataset_ts(self, d):
        ds = pd.DataFrame(data=d)
        self.dataset_ts = pd.concat([self.dataset_ts, ds], ignore_index=True)

    def update_dataset_ucb(self, d):
        ds = pd.DataFrame(data=d)
        self.dataset_ucb = pd.concat([self.dataset_ucb, ds], ignore_index=True)

    def generate_context(self, learner):
        if learner == "TS":
            df = self.dataset_ts.copy()
        elif learner == "UCB":
            df = self.dataset_ucb.copy()

        df = df.tail(56)
        df['reward'] = df['pos_conv'] * (df['price'] - env.prod_cost) - df['costs']

        self.generated_contexts = []
        self.generated_contexts.append(df)
        self.gen_context(df, ['f_1', 'f_2'])

        if learner == "TS":
            self.context_classes_ts = self.extract_classes_from_df()
            print("new ts context: ")
            print(self.context_classes_ts)
            self.context_ts = [GPTS_Learner_3(env.bids, env.prices) for c in self.context_classes_ts]
            self.train_new_learners(self.dataset_ts, self.context_classes_ts, learner)
        elif learner == "UCB":
            self.context_classes_ucb = self.extract_classes_from_df()
            print("new ucb context: ")
            print(self.context_classes_ucb)
            self.context_ucb = [GPUCB1_Learner_3(env.bids, env.prices) for c in self.context_classes_ucb]
            self.train_new_learners(self.dataset_ucb, self.context_classes_ts, learner)

    def extract_classes_from_df(self):
        temp_classes = []

        for df in self.generated_contexts:
            temp = []
            for i in range(4):
                row = df.iloc[i]
                temp.append(str(row['f_1']) + str(row['f_2']))
            temp_classes.append(list(set(temp)))

        return temp_classes

    def get_split_value(self, df, splitting_feature):
        df_c1 = df.loc[df[splitting_feature].astype(int) == 0]
        df_c2 = df.loc[df[splitting_feature].astype(int) == 1]

        lb_reward_c1 = self.get_context_reward(df_c1)
        lb_reward_c2 = self.get_context_reward(df_c2)
        # lb_reward_c1 = df_c1['reward'].mean() - np.sqrt(
        #     -np.log(self.confidence) / 2 * df_c1['n_clicks'].sum())  # cambiare df_c1['n_clicks'].sum()?
        # lb_reward_c2 = df_c2['reward'].mean() - np.sqrt(
        #     -np.log(self.confidence) / 2 * df_c2['n_clicks'].sum())  # cambiare df_c2['n_clicks'].sum()?
        lb_prob_c1 = df_c1['n_clicks'].sum() / df['n_clicks'].sum()
        lb_prob_c1 -= np.sqrt(-np.log(self.confidence) / (2 * df_c1['n_clicks'].sum()))
        lb_prob_c2 = df_c2['n_clicks'].sum() / df['n_clicks'].sum() - np.sqrt(
            -np.log(self.confidence) / (2 * df_c2['n_clicks'].sum()))

        return lb_prob_c1 * lb_reward_c1 + lb_prob_c2 * lb_reward_c2

    def split_on_feature(self, df, splitting_feature):
        df_c1 = df.loc[df[splitting_feature].astype(int) == 0]
        df_c2 = df.loc[df[splitting_feature].astype(int) == 1]

        return df_c1, df_c2

    def gen_context(self, df, features):
        if len(features) > 0:
            df['reward'] = df['pos_conv'] * (df['price'] - env.prod_cost) - df['costs']
            not_split_value = self.get_context_reward(df)

            split_values = [self.get_split_value(df, f) for f in features]
            split_feature_index = split_values.index(max(split_values))

            print("not split value with features " + str(features) + ": " + str(not_split_value))
            print("split values:" + str(split_values))

            if (split_values[split_feature_index] > not_split_value) and len(features) > 0:
                for idx, df_in_list in enumerate(self.generated_contexts):
                    if df_in_list.equals(df):
                        df_index = idx
                        break
                # df_index = self.generated_contexts.index(df)
                self.generated_contexts.pop(df_index)
                df1, df2 = self.split_on_feature(df, features[split_feature_index])
                self.generated_contexts.append(df1)
                self.generated_contexts.append(df2)
                features.pop(split_feature_index)
                self.gen_context(df1, features.copy())
                self.gen_context(df2, features.copy())

    def train_new_learners(self, ds, classes, learner):

        if learner == "TS":
            i = 0
            for context_class in self.context_classes_ts:
                df_class = pd.DataFrame()
                for element in context_class:
                    tempdf = ds.loc[
                        (ds['f_1'].astype(int) == int(element[0])) & (ds['f_2'].astype(int) == int(element[1]))]
                    df_class = pd.concat([df_class, tempdf])
                df_class = df_class.sort_values(by='t', ascending=True)
                df_class = df_class.drop(columns=['f_1', 'f_2'])
                df_class = df_class.groupby("t").agg({"costs": "sum", "pos_conv": "sum", "n_clicks": "sum",
                                                      "price": "mean", "bid": "mean", "price_arm": "mean",
                                                      "bid_arm": "mean"})

                for indx, row in df_class.iterrows():
                    pulled_arm_price = int(row['price_arm'])
                    pulled_arm_bid = int(row['bid_arm'])
                    reward = row['pos_conv'] * (row['price'] - env.prod_cost) - row['costs']
                    reward_input = [reward, row['n_clicks'], row['pos_conv']]

                    self.context_ts[i].update(pulled_arm_price, pulled_arm_bid, reward_input)
                i += 1

        if learner == "UCB":
            i = 0
            for context_class in self.context_classes_ucb:
                df_class = pd.DataFrame()
                for element in context_class:
                    tempdf = ds.loc[
                        (ds['f_1'].astype(int) == int(element[0])) & (ds['f_2'].astype(int) == int(element[1]))]
                    df_class = pd.concat([df_class, tempdf])
                    df_class = df_class.sort_values(by='t', ascending=True)
                    df_class = df_class.drop(columns=['f_1', 'f_2'])
                    df_class = df_class.groupby("t").agg({"costs": "sum", "pos_conv": "sum", "n_clicks": "sum",
                                                          "price": "mean", "bid": "mean", "price_arm": "mean",
                                                          "bid_arm": "mean"})

                for indx, row in df_class.iterrows():
                    pulled_arm_price = int(row['price_arm'])
                    pulled_arm_bid = int(row['bid_arm'])
                    reward = row['pos_conv'] * (row['price'] - env.prod_cost) - row['costs']
                    reward_input = [reward, row['n_clicks'], row['pos_conv']]

                    self.context_ucb[i].update(pulled_arm_price, pulled_arm_bid, reward_input)
                i += 1

    def get_regret_sum(self, learner):
        if learner == "TS":
            df = self.dataset_ts
        elif learner == "UCB":
            df = self.dataset_ucb
        else:
            df = None

        opt_prices, opt_bids = optimize(env)
        opt = [env.get_clicks(opt_bids[customer_class], customer_class) * env.get_conversion_prob(
            opt_prices[customer_class], customer_class) * (opt_prices[customer_class] - env.prod_cost) - env.get_costs(
            opt_bids[customer_class], customer_class) for customer_class in env.classes]

        # class 1
        df_1 = df[(df['f_1'].astype(int) == 0) & (df['f_2'].astype(int) == 0)]
        df_1['reward'] = df_1['pos_conv'] * (df_1['price'] - env.prod_cost) - df_1['costs']
        df_1 = df_1[['reward', 't']]
        regret_1 = [opt[0] - r for r in df_1['reward']]

        # class 2
        df_2 = df[(df['f_1'].astype(int) == 0) & (df['f_2'].astype(int) == 1)]
        df_2['reward'] = df_2['pos_conv'] * (df_2['price'] - env.prod_cost) - df_2['costs']
        df_2 = df_2[['reward', 't']]
        regret_2 = [opt[1] - r for r in df_2['reward']]

        # class 3
        df_3 = df[(df['f_1'].astype(int) == 1)]
        df_3['reward'] = df_3['pos_conv'] * (df_3['price'] - env.prod_cost) - df_3['costs']
        df_3 = df_3[['reward', 't']]
        df_3 = df_3.groupby('t')['reward'].sum().reset_index()
        regret_3 = [2*opt[2] - r for r in df_3['reward']]

        return [r1 + r2 + r3 for r1, r2, r3 in zip(regret_1, regret_2, regret_3)]

    def get_context_reward(self, dataset):
        # reward = dataset['reward'].max()
        # reward -= np.sqrt(-(np.log(self.confidence) / (len(dataset) * 2)))

        prices = dataset['price'].unique().tolist()
        conv_rates = []
        for p in prices:
            ds = dataset[(dataset['price'].astype(int) == p)]
            conv_rates.append(ds['pos_conv'].sum() / dataset['n_clicks'].sum())

        earnings = [conv_rates[i] * (prices[i] - env.prod_cost) for i in range(len(prices))]
        opt_earning = np.max(earnings)
        # opt_price_index = earnings.index(opt_earning)
        # opt_earning -= np.sqrt(-(np.log(self.confidence) / (dataset[(dataset['price'].astype(int) == prices[opt_price_index])]['n_clicks'].mean() * 2)))
        #opt_earning -= np.sqrt(-(np.log(self.confidence) / (dataset['n_clicks'].sum() * 2)))

        bids = dataset['bid'].unique().tolist()
        n_clicks = []
        cum_costs = []
        for bid in bids:
            n_clicks.append(dataset[(dataset['bid'] == bid)]['n_clicks'].mean())
            cum_costs.append(dataset[(dataset['bid'] == bid)]['costs'].mean())

        reward = np.max([(n_clicks[i] * opt_earning - cum_costs[i]) for i in range(len(bids))])
        reward -= np.sqrt(-(np.log(self.confidence) / (dataset['n_clicks'].sum() * 2)))  # * reward
        return reward
