import pandas as pd
import numpy as np


def processing(df):
    df['Date'] = pd.to_datetime(df['Date'], format='%m/%d/%y')
    df['Date'] = df['Date'].dt.strftime('%Y-%m-%d')

    df.set_index('Date', inplace=True)
    df.columns = ['value']

    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df = df.ffill()
    contains_nan_or_inf = df.isna().any().any()
    if contains_nan_or_inf:
        print('There are still nan values')
    return df


def split_data(df, start_date, end_date, split_date):
    # 确保索引是日期时间类型，如果不是，先转换
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)

    # 用日期索引进行切片
    train_data = df[start_date:split_date]
    test_data = df[split_date:end_date]
    train_data.index = train_data.index.date
    test_data.index = test_data.index.date

    if train_data.index[-1] == test_data.index[0]:
        train_data = train_data.iloc[:-1]

    # 获得训练和测试数据的索引范围
    train_index = train_data.index
    test_index = test_data.index

    return train_data, test_data, train_index, test_index


if __name__ == '__main__':
    df_coin = pd.read_csv('BCHAIN-MKPRU.csv')
    # df_gold = pd.read_csv('LBMA-GOLD.csv')
    df = processing(df_coin)

    train_data, test_data, train_index_range, test_index_range = split_data(df, '2016-01-01', '2021-12-31', '2021-01-01')
