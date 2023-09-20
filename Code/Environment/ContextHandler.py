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

        df['reward'] = df['pos_conv'] * (df['price'] - env.prod_cost) - df['costs']

        self.generated_contexts = []
        self.generated_contexts.append(df)
        self.gen_context(df, ['f_1', 'f_2'])

        if learner == "TS":
            self.context_classes_ts = self.extract_classes_from_df()
            self.context_ts = [GPTS_Learner_3(env.bids, env.prices) for c in self.context_classes_ts]
            self.train_new_learners(self.dataset_ts, self.context_classes_ts, learner)
        elif learner == "UCB":
            self.context_classes_ucb = self.extract_classes_from_df()
            self.context_ucb = [GPUCB1_Learner_3(env.bids, env.prices) for c in self.context_classes_ucb]
            self.train_new_learners(self.dataset_ucb, self.context_classes_ts, learner)

        # retrainare i learner

        # if learner == "TS":
        #     self.context_ts = [GPTS_Learner_3(env.bids, env.prices) for c in self.context_classes_ts]
        #     self.train_new_learners(self.dataset_ts, self.context_classes_ts, learner)
        # elif learner == "UCB":
        #     self.context_ucb = [GPUCB1_Learner_3(env.bids, env.prices) for c in self.context_classes_ucb]
        #     self.train_new_learners(self.dataset_ucb, self.context_classes_ts, learner)

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

        lb_reward_c1 = df_c1['reward'].mean() - np.sqrt(
            -np.log(self.confidence) / 2 * df_c1['n_clicks'].sum())  # cambiare df_c1['n_clicks'].sum()?
        lb_reward_c2 = df_c2['reward'].mean() - np.sqrt(
            -np.log(self.confidence) / 2 * df_c2['n_clicks'].sum())  # cambiare df_c2['n_clicks'].sum()?
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
            df.loc[:, 'reward'] = df['pos_conv'] * (df['price'] - env.prod_cost) - df['costs']
            not_split_value = df['reward'].mean() - np.sqrt(
                -np.log(self.confidence) / 2 * df['n_clicks'].sum())  # da rivedere

            split_values = [self.get_split_value(df, f) for f in features]  # da rivedere
            split_feature_index = split_values.index(max(split_values))

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
                self.gen_context(df1, features)
                self.gen_context(df2, features)

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
                    reward_input = [row['pos_conv'], row['n_clicks'] - row['pos_conv'], reward]

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
                    reward_input = [row['pos_conv'], row['n_clicks'] - row['pos_conv'], reward]

                    self.context_ucb[i].update(pulled_arm_price, pulled_arm_bid, reward_input)
                i += 1

    def get_regret_sum(self, learner):
        if learner == "TS":
            df = self.dataset_ts
        elif learner == "UCB":
            df = self.dataset_ucb
        else:
            df = None

        df['reward'] = df['pos_conv'] * (df['price'] - env.prod_cost) - df['costs']

        opt_prices, opt_bids = optimize(env)
        opt = [env.get_clicks(opt_bids[customer_class], customer_class) * env.get_conversion_prob(
            opt_prices[customer_class], customer_class) * (opt_prices[customer_class] - env.prod_cost) - env.get_costs(
            opt_bids[customer_class], customer_class) for customer_class in env.classes]

        # class 1
        df_1 = df[(df['f_1'].astype(int) == 0) & (df['f_2'].astype(int) == 0)]
        df_1 = df_1[['reward', 't']]
        regret_1 = [opt[0] - r for r in df_1['reward']]

        # class 2
        df_2 = df[(df['f_1'].astype(int) == 0) & (df['f_2'].astype(int) == 1)]
        df_2 = df_2[['reward', 't']]
        regret_2 = [opt[1] - r for r in df_2['reward']]

        # class 3
        df_3 = df[(df['f_1'].astype(int) == 1)]
        df_3 = df_3[['reward', 't']]
        df_3 = df_3.groupby('t')['reward'].sum().reset_index()
        regret_3 = [2*opt[2] - r for r in df_3['reward']]

        return [r1 + r2 + r3 for r1, r2, r3 in zip(regret_1, regret_2, regret_3)]
