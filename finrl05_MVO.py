# -*- encoding: utf-8 -*-
"""
@File: finrl_MVO.py
@Modify Time: 2025/4/6 19:50       
@Author: Kevin-Chen
@Descriptions: 
"""
import pandas as pd
import numpy as np
from pypfopt.efficient_frontier import EfficientFrontier

train = pd.read_csv('train_data.csv')
trade = pd.read_csv('trade_data.csv')

train = train.set_index(train.columns[0])
train.index.names = ['']
trade = trade.set_index(trade.columns[0])
trade.index.names = ['']


# 将数据框处理成适用于均值-方差优化（MVO）的形式
def process_df_for_mvo(df):
    """
    将数据框df根据日期和股票代码进行重塑，以便于后续的均值-方差优化计算。

    参数:
    df (DataFrame): 包含日期（date）、股票代码（tic）和收盘价（close）等列的数据框。

    返回:
    DataFrame: 重塑后的数据框，其中日期为索引，股票代码为列，收盘价为值。
    """
    return df.pivot(index="date", columns="tic", values="close")


# 计算股票的每日收益率
def StockReturnsComputing(StockPrice, Rows, Columns):
    """
    计算给定股票价格矩阵的每日收益率。

    参数:
    StockPrice (numpy.ndarray): 股票价格的二维数组，行表示不同日期的股票价格，列表示不同股票的价格。
    Rows (int): 股票价格数据的行数，即日期数。
    Columns (int): 股票价格数据的列数，即股票数。

    返回:
    numpy.ndarray: 股票每日收益率的二维数组，行数为(Rows-1)，列数为Columns。
    """
    StockReturn = np.zeros([Rows - 1, Columns])
    for j in range(Columns):  # j: Assets
        for i in range(Rows - 1):  # i: Daily Prices
            # 计算每日股票收益率，并以百分比形式表示
            StockReturn[i, j] = ((StockPrice[i + 1, j] - StockPrice[i, j]) / StockPrice[i, j]) * 100
    return StockReturn


# 对训练数据进行处理，以适应均值-方差优化（MVO）的需求
StockData = process_df_for_mvo(train)
# 对交易数据进行处理，以适应均值-方差优化（MVO）的需求
TradeData = process_df_for_mvo(trade)

# 将处理后的股票数据转换为NumPy数组，以便进行数值计算
arStockPrices = np.asarray(StockData)
# 获取股票价格数组的形状，用于后续计算
[Rows, Cols] = arStockPrices.shape
# 计算股票的收益率矩阵
arReturns = StockReturnsComputing(arStockPrices, Rows, Cols)

# 计算每只股票收益率的均值
meanReturns = np.mean(arReturns, axis=0)

# 计算股票收益率的协方差矩阵
covReturns = np.cov(arReturns, rowvar=False)

# 设置NumPy打印选项，限制小数点后精度为3位，并抑制科学计数法输出
np.set_printoptions(precision=3, suppress=True)

# 创建EfficientFrontier对象，使用给定的资产期望收益和协方差矩阵，设置权重界限为0到0.1
ef_mean = EfficientFrontier(meanReturns, covReturns, weight_bounds=(0, 1))

# 计算并获取最大夏普比率对应的投资组合权重
raw_weights_mean = ef_mean.max_sharpe()

# 对获取的权重进行精简处理，以去除接近于零的权重值
cleaned_weights_mean = ef_mean.clean_weights()

# # 将处理后的权重与10000相乘，转换为以万元为单位的投资金额
# mvo_weights = np.array([10000 * cleaned_weights_mean[i] for i in range(len(cleaned_weights_mean))])
# # 计算每只股票的最新价格的倒数，用于后续计算初始投资组合
# LastPrice = np.array([1 / p for p in StockData.tail(1).to_numpy()[0]])
# # 通过最新价格和MVO权重计算初始投资组合的价值
# Initial_Portfolio = np.multiply(mvo_weights, LastPrice)
# # 计算投资组合中每个资产的实际投资金额
# Portfolio_Assets = TradeData @ Initial_Portfolio

# 初始金额1万元，转为每个资产的金额
mvo_weights = 100_0000 * np.array(list(cleaned_weights_mean.values()))

# 使用trade数据的第一天作为买入日
StartPrice = TradeData.iloc[0].to_numpy()
Initial_Shares = mvo_weights / StartPrice  # 每个资产买多少股

# 每天组合价值 = 当天价格 × 股数
Portfolio_Assets = TradeData @ Initial_Shares

# 将投资组合资产数据封装到DataFrame中，列名为"Mean Var"
MVO_result = pd.DataFrame(Portfolio_Assets, columns=["Mean Var"])

from finrl.meta.preprocessor.yahoodownloader import YahooDownloader

TRAIN_START_DATE = '2009-01-01'
TRAIN_END_DATE = '2015-01-01'
TRADE_START_DATE = '2016-01-01'
TRADE_END_DATE = '2018-01-01'

df_dji = YahooDownloader(
    start_date=TRADE_START_DATE, end_date=TRADE_END_DATE, ticker_list=["dji"]
).fetch_data()
df_dji = df_dji[["date", "close"]]
fst_day = df_dji["close"][0]
dji = pd.merge(
    df_dji["date"],
    df_dji["close"].div(fst_day).mul(1000_000),
    how="outer",
    left_index=True,
    right_index=True,
).set_index("date")

from finrl04_prediction import df_result_ddpg, df_result_sac, df_result_ppo, df_result_a2c, df_result_td3

result = pd.DataFrame(
    {
        "ddpg": df_result_ddpg["account_value"],
        "sac": df_result_sac["account_value"],
        "ppo": df_result_ppo["account_value"],
        "a2c": df_result_a2c["account_value"],
        "td3": df_result_td3["account_value"],
        "mvo": MVO_result["Mean Var"],
        "dji": dji["close"],
    }
)

# import matplotlib
#
# matplotlib.use("MacOSX")  # 或 "Qt5Agg", "MacOSX", 取决于你的系统和环境
# import matplotlib.pyplot as plt
#
# plt.rcParams["figure.figsize"] = (15, 5)
# plt.figure()
# result.plot()
# plt.title("MVO vs. DJI")
# plt.xlabel("Date")
# plt.ylabel("Portfolio Value")
# plt.grid(True)
# plt.show()


import plotly.graph_objects as go

# 创建图表
fig = go.Figure()

# 添加每条策略线
for column in result.columns:
    fig.add_trace(
        go.Scatter(
            x=result.index,
            y=result[column],
            mode="lines",
            name=column  # 图例名称，点击可控制显示
        )
    )

# 设置布局
fig.update_layout(
    title="MVO vs. DJI",
    xaxis_title="Date",
    yaxis_title="Portfolio Value",
    legend_title="Strategies",
    hovermode="x unified",
    template="plotly_white",
    autosize=True,
    height=500,
    width=1200,
)

# 显示图表
fig.show()
