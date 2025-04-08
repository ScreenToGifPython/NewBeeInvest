# -*- encoding: utf-8 -*-
"""
@File: finrl_prediction.py
@Modify Time: 2025/4/6 15:27       
@Author: Kevin-Chen
@Descriptions: 
"""
import yfinance as yf
import pandas as pd
import itertools
import matplotlib.pyplot as plt
import numpy as np

from stable_baselines3 import SAC, A2C, DDPG, PPO, TD3
from finrl.meta.preprocessor.yahoodownloader import YahooDownloader
from finrl.meta.preprocessor.preprocessors import FeatureEngineer, data_split
from finrl import config_tickers
from stable_baselines3.common.logger import configure
from finrl.agents.stablebaselines3.models import DRLAgent
from finrl.config import INDICATORS, TRAINED_MODEL_DIR, RESULTS_DIR
from finrl.main import check_and_make_directories
from finrl.meta.env_stock_trading.env_stocktrading import StockTradingEnv
from finrl02_create_env import env_kwargs

# 检查并创建所需的目录，如果目录不存在
check_and_make_directories([TRAINED_MODEL_DIR])
train = pd.read_csv('train_data.csv')
trade = pd.read_csv('trade_data.csv')

train = train.set_index(train.columns[0])
train.index.names = ['']
trade = trade.set_index(trade.columns[0])
trade.index.names = ['']

"""
StockTradingEnv是一个股票交易环境类，用于模拟股票交易的过程。
它根据给定的股票数据和交易参数，提供一个强化学习代理可以交互的环境。
参数:
    - df: pd.DataFrame, 包含股票信息的数据框，用于模拟交易环境。
    - stock_dim: int, 股票的维度，表示环境中股票的数量。
    - hmax: int, 交易时允许的最大股数，限制单次交易的规模。
    - initial_amount: int, 初始投资金额，表示交易开始时的总资本。
    - num_stock_shares: list[int], 每只股票的初始持有数量列表。
    - buy_cost_pct: list[float], 购买股票时的成本比例列表，表示每只股票的买入费用。
    - sell_cost_pct: list[float], 卖出股票时的成本比例列表，表示每只股票的卖出费用。
    - reward_scaling: float, 奖励缩放因子，用于调整奖励的大小，以便于学习。
    - state_space: int, 状态空间的维度，表示描述环境状态的变量数量。
    - action_space: int, 动作空间的维度，表示代理可以采取的动作数量。
    - tech_indicator_list: list[str], 技术指标列表，用于提供给代理作为状态的一部分。
    - turbulence_threshold=None, 湍流阈值，用于判断市场是否处于高波动状态。
    - risk_indicator_col="turbulence", 风险指标列的名称，默认为"turbulence"。
    - make_plots: bool = False, 是否绘制交易结果图表，默认不绘制。
    - print_verbosity=10, 打印日志的频率，表示每隔多少次迭代打印一次信息。
    - day=0, 当前交易日的索引，初始值为0。
    - initial=True, 是否为初始状态的标志，默认为True。
    - previous_state=[], 前一个状态的列表，用于跟踪状态变化。
    - model_name="", 使用的模型名称，默认为空字符串。
    - mode="", 运行模式，默认为空字符串。
    - iteration="", 迭代次数或标识，默认为空字符串。
"""


def prediction_ddpg():
    trained_ddpg = DDPG.load(TRAINED_MODEL_DIR + "/agent_ddpg")

    e_trade_gym = StockTradingEnv(df=trade,
                                  risk_indicator_col='vix',
                                  **env_kwargs
                                  )

    # 使用训练好的DDPG模型在交易环境中进行预测，获取账户价值和动作数据
    df_account_value_ddpg, df_actions_ddpg = DRLAgent.DRL_prediction(
        model=trained_ddpg,
        environment=e_trade_gym
    )
    df_result_ddpg = (
        df_account_value_ddpg.set_index(df_account_value_ddpg.columns[0])
    )
    print(df_result_ddpg.head())
    return df_result_ddpg, df_actions_ddpg


def prediction_sac():
    trained_sac = SAC.load(TRAINED_MODEL_DIR + "/agent_sac")
    e_trade_gym = StockTradingEnv(df=trade,
                                  risk_indicator_col='vix',
                                  **env_kwargs
                                  )
    df_account_value_sac, df_actions_sac = DRLAgent.DRL_prediction(
        model=trained_sac,
        environment=e_trade_gym
    )
    df_result_sac = (
        df_account_value_sac.set_index(df_account_value_sac.columns[0])
    )
    return df_result_sac, df_actions_sac


def prediction_ppo():
    trained_ppo = PPO.load(TRAINED_MODEL_DIR + "/agent_ppo")
    e_trade_gym = StockTradingEnv(df=trade,
                                  risk_indicator_col='vix',
                                  **env_kwargs
                                  )
    df_account_value_ppo, df_actions_ppo = DRLAgent.DRL_prediction(
        model=trained_ppo,
        environment=e_trade_gym
    )
    df_result_ppo = (
        df_account_value_ppo.set_index(df_account_value_ppo.columns[0])
    )
    return df_result_ppo, df_actions_ppo


def prediction_a2c():
    trained_a2c = A2C.load(TRAINED_MODEL_DIR + "/agent_a2c")
    e_trade_gym = StockTradingEnv(df=trade,
                                  risk_indicator_col='vix',
                                  **env_kwargs
                                  )
    df_account_value_a2c, df_actions_a2c = DRLAgent.DRL_prediction(
        model=trained_a2c,
        environment=e_trade_gym
    )
    df_result_a2c = (
        df_account_value_a2c.set_index(df_account_value_a2c.columns[0])
    )
    return df_result_a2c, df_actions_a2c


def prediction_td3():
    trained_td3 = TD3.load(TRAINED_MODEL_DIR + "/agent_td3")
    e_trade_gym = StockTradingEnv(df=trade,
                                  risk_indicator_col='vix',
                                  **env_kwargs
                                  )
    df_account_value_td3, df_actions_td3 = DRLAgent.DRL_prediction(
        model=trained_td3,
        environment=e_trade_gym
    )
    df_result_td3 = (
        df_account_value_td3.set_index(df_account_value_td3.columns[0])
    )
    return df_result_td3, df_actions_td3


df_result_ddpg, _ = prediction_ddpg()
df_result_sac, _ = prediction_sac()
df_result_ppo, _ = prediction_ppo()
df_result_a2c, _ = prediction_a2c()
df_result_td3, _ = prediction_td3()
