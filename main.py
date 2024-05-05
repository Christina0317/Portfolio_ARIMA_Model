import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from data_preprocess import processing, split_data
from arima_model import ARIMAPredictor
from investment_strategy import InvestmentStrategy
import warnings
warnings.filterwarnings('ignore')


def price_predicting(train_data, test_data, train_index, test_index, total_index, p, q):
    model = ARIMAPredictor(train_index, test_index, total_index)
    d, train_diff = model.term_d(train_data)

    arima_model = model.arima_fitting(train_diff, p, d, q)

    prediction = model.arima_predicting(arima_model, train_data, test_data, type='predict_total')
    return prediction


def trading_strategy(data, initial_cash, weight):
    trading_strategy = InvestmentStrategy(initial_cash, weight)
    result = trading_strategy.milp_optimize(data)
    return result


class PortfolioManager:
    def __init__(self, coin_path, gold_path, initial_cash, weight, start_date, end_date, split_date):
        self.df_gold = pd.read_csv(gold_path)
        self.df_coin = pd.read_csv(coin_path)
        self.start_date = start_date
        self.end_date = end_date
        self.split_date = split_date
        self.initial_cash = initial_cash
        self.weight = weight

    def run(self):
        df_gold = processing(self.df_gold)
        df_coin = processing(self.df_coin)

        train_data, test_data, train_index, test_index = split_data(df_gold, self.start_date, self.end_date, self.split_date)
        pre_gold = price_predicting(train_data, test_data, train_index, test_index, df_gold.index, 1, 1)

        train_data, test_data, train_index, test_index = split_data(df_coin, self.start_date, self.end_date, self.split_date)
        pre_coin = price_predicting(train_data, test_data, train_index, test_index, df_coin.index, 1, 1)

        data = pd.merge(pre_gold['predict'], pre_coin['predict'], left_index=True, right_index=True, how='outer')
        data.columns = ['gold', 'coin']
        portfolio = trading_strategy(data, self.initial_cash, self.weight)
        return portfolio


if __name__ == "__main__":
    coin_path = 'BCHAIN-MKPRU.csv'
    gold_path = 'LBMA-GOLD.csv'
    initial_cash = 10000
    weight = 0.9
    start_date = '2016-01-01'
    end_date = '2022-01-01'
    split_date = '2021-01-01'
    manager = PortfolioManager(coin_path, gold_path, initial_cash, weight, start_date, end_date, split_date)
    portfolio = manager.run()






