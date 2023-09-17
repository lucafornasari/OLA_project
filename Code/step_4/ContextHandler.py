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
        d = {'f_1': [], 'f_2': [], 'pos_conv': [], 'n_clicks': [], 'costs': [], 'price': [], 'bid': [], 't': []}
        self.dataset_ts = pd.DataFrame(data=d)
        self.dataset_ucb = pd.DataFrame(data=d)
        self.featuresToSplit = [0, 1]
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
        else:
            df = None

        df['reward'] = df['pos_conv'] * (df['price'] - env.prod_cost) - df['costs']

        self.generated_contexts.append(df)
        self.gen_context(df, ['f_1', 'f_2'])
        self.extract_classes_from_df()

        # retrainare i learner

    def extract_classes_from_df(self):
        temp_classes = []

        for df in self.generated_contexts:
            temp = []
            for i in range(4):
                row = df.iloc[i]
                temp.append(str(row['f_1']) + str(row['f_2']))
            temp_classes.append(list(set(temp)))

    def get_split_value(self, df, splitting_feature):
        df_c1 = df.loc[df[splitting_feature] == 0]
        df_c2 = df.loc[df[splitting_feature] == 1]

        lb_reward_c1 = df_c1['reward'].mean() - np.sqrt(
            -np.log(self.confidence) / 2 * df_c1['n_clicks'].sum())  # cambiare df_c1['n_clicks'].sum()?
        lb_reward_c2 = df_c2['reward'].mean() - np.sqrt(
            -np.log(self.confidence) / 2 * df_c2['n_clicks'].sum())  # cambiare df_c2['n_clicks'].sum()?
        lb_prob_c1 = df_c1['n_clicks'].sum() / df['n_clicks'].sum() - np.sqrt(
            -np.log(self.confidence) / 2 * df_c1['n_clicks'].sum())
        lb_prob_c2 = df_c2['n_clicks'].sum() / df['n_clicks'].sum() - np.sqrt(
            -np.log(self.confidence) / 2 * df_c2['n_clicks'].sum())

        return lb_prob_c1 * lb_reward_c1 + lb_prob_c2 * lb_reward_c2

    def split_on_feature(self, df, splitting_feature):
        df_c1 = df.loc[df[splitting_feature] == 0]
        df_c2 = df.loc[df[splitting_feature] == 1]

        return df_c1, df_c2

    def gen_context(self, df, features):
        if len(features) > 0:
            df['reward'] = df['pos_conv'] * (df['price'] - env.prod_cost) - df['costs']
            not_split_value = df['reward'].mean() - np.sqrt(-np.log(self.confidence) / 2 * df['n_clicks'].sum())  # da rivedere

            split_values = [self.get_split_value(df, f) for f in features]  # da rivedere
            split_feature_index = split_values.index(max(split_values))

            if (split_values[split_feature_index] > not_split_value) and len(features) > 0:
                self.generated_contexts.pop(self.generated_contexts.index(df))
                df1, df2 = self.split_on_feature(df, features[split_feature_index])
                self.generated_contexts.append(df1)
                self.generated_contexts.append(df2)
                self.gen_context(df1, features.pop(split_feature_index))
                self.gen_context(df2, features.pop(split_feature_index))
