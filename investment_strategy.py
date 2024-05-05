from pulp import LpProblem, LpMaximize, LpVariable, value, PULP_CBC_CMD
import pandas as pd
from tqdm import tqdm


class InvestmentStrategy:
    def __init__(self, initial_cash, weight):
        self.initial_cash = initial_cash
        self.weight = weight

    def data_cleaned(self, df):
        df['gold_filled'] = df['gold'].ffill()
        df['x_c'] = 0 * len(df)
        df['x_c'].iloc[0] = self.initial_cash
        df['x_c'].iloc[1] = df['x_c'].iloc[0]
        df['x_g'] = 0 * len(df)
        df['x_b'] = 0 * len(df)
        df['expected_return'] = 0 * len(df)
        df['risk'] = 0 * len(df)
        df['portfolio_value'] = 0 * len(df)
        df['portfolio_value'].iloc[0] = df['x_c'].iloc[0]
        df['portfolio_value'].iloc[1] = df['x_c'].iloc[1]
        return df

    def milp_optimize(self, df):
        df = self.data_cleaned(df)

        for i in tqdm(range(1, len(df.index) - 1)):
            if i == 43:
                print(1)
            # 定义问题
            model = LpProblem("Portfolio_Optimization", LpMaximize)

            value_g_t = df['gold_filled'].iloc[i]
            value_b_t = df['coin'].iloc[i]
            value_g_tp1 = df['gold_filled'].iloc[i+1]
            value_b_tp1 = df['coin'].iloc[i+1]
            S_g = df['gold'].iloc[:i].std()
            S_b = df['coin'].iloc[:i].std()
            M_g = df['gold'].iloc[:i].mean()
            M_b = df['coin'].iloc[:i].mean()
            if not pd.notna(S_g):
                S_g = 0
            if not pd.notna(S_b):
                S_b = 0

            # 定义变量
            x_g_tp1 = LpVariable('x_g_tp1', lowBound=0, cat='Integer')
            x_b_tp1 = LpVariable('x_b_tp1', lowBound=0, cat='Integer')

            # 收益和风险计算
            # 若 gold 可交易
            if pd.notna(df['gold'].iloc[i+1]):
                P = (value_g_tp1 - value_g_t) * x_g_tp1 + (value_b_tp1 - value_b_t) * x_b_tp1
                R = x_g_tp1 * S_g/M_g + x_b_tp1 * S_b/M_b
            # 若 gold 不可交易
            else:
                P = (value_b_tp1 - value_b_t) * x_b_tp1
                R = x_b_tp1 * S_b/M_b

            # 目标函数
            model += self.weight * P - (1 - self.weight) * R
            # 约束方程
            model += x_g_tp1 * value_g_t + x_b_tp1 * value_b_t <= df['portfolio_value'].iloc[i]

            if not pd.notna(df['gold'].iloc[i+1]):
                model += x_g_tp1 == df['x_g'].iloc[i]

            # 解决问题
            model.solve(PULP_CBC_CMD(msg=False))

            df['x_c'].iloc[i+1] = df['portfolio_value'].iloc[i] - x_g_tp1.varValue*value_g_tp1 - x_b_tp1.varValue*value_b_tp1
            df['x_g'].iloc[i+1] = x_g_tp1.varValue
            df['x_b'].iloc[i+1] = x_b_tp1.varValue
            df['expected_return'].iloc[i+1] = value(P)
            df['risk'].iloc[i+1] = value(R)
            df['portfolio_value'].iloc[i+1] = df['expected_return'].iloc[i+1] + df['portfolio_value'].iloc[i]

        return df


if __name__ == '__main__':
    df_gold = pd.read_csv('LBMA-GOLD.csv')
    df_coin = pd.read_csv('BCHAIN-MKPRU.csv')

    df_gold = df_gold.set_index('Date')
    df_coin = df_coin.set_index('Date')

    trading_strategy = InvestmentStrategy(10000, 1)

    df1 = pd.merge(df_gold, df_coin, left_index=True, right_index=True, how='outer')
    df1.columns = ['gold', 'coin']
    df = trading_strategy.milp_optimize(df1)