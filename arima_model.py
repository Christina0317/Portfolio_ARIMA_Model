import pandas as pd
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima.model import ARIMA
import matplotlib.pyplot as plt
from data_preprocess import processing, split_data
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf


class ARIMAPredictor:
    def __init__(self, train_index, test_index, total_index):
        self.train_index = train_index
        self.test_index = test_index
        self.total_index = total_index

        self.model = None
        self.p = None
        self.q = None
        self.d = None

    def term_d(self, train_data):
        d = 0
        series = train_data['value']

        p_value = adfuller(series)[1]

        # 当p-value大于0.05，数据不平稳，需要继续差分
        while p_value > 0.05:
            d += 1  # 差分次数加1
            series = series.diff().dropna()  # 进行差分
            p_value = adfuller(series)[1]  # 重新检查平稳性
        self.d = d
        return d, pd.DataFrame(series)

    def term_p_q(self, train_data, lags=10):
        """
        根据指定的滞后数绘制ACF和PACF图。
        df: 包含时间序列数据的DataFrame
        lags: 要计算和绘制的滞后数
        """
        fig, ax = plt.subplots(1, 2, figsize=(12, 6))
        plot_acf(train_data['value'], lags=lags, ax=ax[0])
        ax[0].set_title('Autocorrelation Function (ACF)')

        plot_pacf(train_data['value'], lags=lags, ax=ax[1])
        ax[1].set_title('Partial Autocorrelation Function (PACF)')

        plt.show()
        return

    def arima_fitting(self, df, p, d, q):
        self.p = p
        self.q = q
        model = ARIMA(df['value'], order=(p,d,q))   # parameter = (p,d,q)
        model_fit = model.fit()
        self.model = model_fit
        return model_fit

    def arima_predicting(self, model, train_data, test_data, type):
        if type == 'predict_total':
            forecast_index = self.total_index
            actual_value = pd.concat([train_data, test_data]).sort_index()['value']
        if type == 'predict_test':
            forecast_index = self.test_index
            actual_value = test_data['value']

        forecast = model.forecast(steps=len(forecast_index))

        # 预测结果转换为DataFrame并设置索引
        forecast = pd.Series(forecast.values, index=forecast_index, name='forecast')
        forecast.index = forecast.index.date

        # 对于多阶差分，逆差分每一步
        if self.d > 0:
            last_values = train_data['value'].iloc[-self.d:]  # 获取差分前的最后d个值
            for i in range(self.d):
                forecast = forecast.cumsum() + last_values.iloc[-i - 1]

        # 将预测结果和测试数据合并为一个DataFrame
        combined_df = pd.DataFrame({
            'actual': actual_value,
            'predict': forecast
        })
        return combined_df


if __name__ == '__main__':
    # df_coin = pd.read_csv('BCHAIN-MKPRU.csv')
    df_gold = pd.read_csv('LBMA-GOLD.csv')
    df = processing(df_gold)
    train_data, test_data, train_index, test_index = split_data(df, '2016-01-01', '2021-12-31', '2021-01-01')
    total_index = df.index

    model = ARIMAPredictor(train_index, test_index, total_index)

    d, train_diff = model.term_d(train_data)

    model.term_p_q(train_data)

    arima_model = model.arima_fitting(train_diff, 1, d,1)

    prediction = model.arima_predicting(arima_model, train_data, test_data, type='predict_total')