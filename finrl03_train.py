# -*- encoding: utf-8 -*-
"""
@File: finrl_train.py
@Modify Time: 2025/4/6 11:22       
@Author: Kevin-Chen
@Descriptions:

The core element in reinforcement learning are agent and environment. You can understand RL as the following process:
- The agent is active in a world, which is the environment. It observe its current condition as a state, and is allowed
  to do certain actions. After the agent execute an action, it will arrive at a new state. At the same time, the
  environment will have feedback to the agent called reward, a numerical signal that tells how good or bad the new
  state is.
- The goal of agent is to get as much cumulative reward as possible. Reinforcement learning is the method that agent
  learns to improve its behavior and achieve that goal.
- To achieve this in Python, we follow the OpenAI gym style to build the stock data into environment.
- state-action-reward are specified as follows:
  - State s: The state space represents an agent’s perception of the market environment. Just like a human trader
    analyzing various information, here our agent passively observes the price data and technical indicators based on
    the past data. It will learn by interacting with the market environment (usually by replaying historical data).
  - Action a: The action space includes allowed actions that an agent can take at each state. For example,
    a ∈ {−1, 0, 1}, where −1, 0, 1 represent selling, holding, and buying. When an action operates multiple shares,
    a ∈{−k, …, −1, 0, 1, …, k}, e.g.. “Buy 10 shares of AAPL” or “Sell 10 shares of AAPL” are 10 or −10, respectively
  - Reward function r(s, a, s′): Reward is an incentive for an agent to learn a better policy. For example, it can be
    the change of the portfolio value when taking a at state s and arriving at new state s’, i.e., r(s, a, s′) = v′ − v,
    where v′ and v represent the portfolio values at state s′ and s, respectively
  - Market environment: 30 constituent stocks of Dow Jones Industrial Average (DJIA) index. Accessed at the starting
    date of the testing period.

"""
import pandas as pd

from finrl02_create_env import env_kwargs
from stable_baselines3.common.logger import configure
from finrl.agents.stablebaselines3.models import DRLAgent
from finrl.config import INDICATORS, TRAINED_MODEL_DIR, RESULTS_DIR
from finrl.main import check_and_make_directories
from stable_baselines3 import SAC
from stable_baselines3 import DDPG
from finrl.meta.env_stock_trading.env_stocktrading import StockTradingEnv

# 检查并创建所需的目录，如果目录不存在
check_and_make_directories([TRAINED_MODEL_DIR])

# 设置pandas选项，以显示更多的列，便于查看数据
pd.set_option('display.max_columns', 1000)  # 显示字段的数量
pd.set_option('display.width', 1000)  # 表格不分段显示

# 读取训练数据CSV文件，将其加载到一个pandas DataFrame中
train = pd.read_csv('train_data.csv')
# 将数据框的第一列设置为索引列，这有助于后续的数据处理和查询
train = train.set_index(train.columns[0])
# 重命名索引列的名称为默认值（空字符串），以保持数据框列名称的简洁和通用性
train.index.names = ['']

"""
训练模型
- RL agents are from Stable Baselines 3. We use SAC as an example below.
- Stable Baselines3（简称 SB3） 是一个基于 PyTorch 的强化学习库，它提供了标准化、易用且高性能的强化学习算法实现。
"""


def train_ddpg():
    # 创建股票交易环境实例
    e_train_gym = StockTradingEnv(df=train, **env_kwargs)

    # 获取适用于Stable Baselines3的训练环境
    env_train, _ = e_train_gym.get_sb_env()

    # 创建一个DRLAgent实例，用于在给定的训练环境中应用DDPG算法, https://arxiv.org/abs/1509.02971
    agent = DRLAgent(env=env_train)

    # 获取DDPG模型
    model_ddpg = agent.get_model("ddpg")

    # 设置日志记录器: 创建一个临时路径用于存储DDPG相关的日志
    tmp_path = RESULTS_DIR + '/ddpg'
    # 配置日志记录设置，包括标准输出、CSV文件和TensorBoard格式
    new_logger_ddpg = configure(tmp_path, ["stdout", "csv", "tensorboard"])

    # 设置新的日志记录器到DDPG模型
    model_ddpg.set_logger(new_logger_ddpg)

    # 训练DDPG模型 使用指定的模型进行训练，训练过程中在TensorBoard中的名称为'ddpg'，总训练步数为50000步
    trained_ddpg = agent.train_model(model=model_ddpg,
                                     tb_log_name='ddpg',
                                     total_timesteps=5000
                                     )

    # 保存训练好的DDPG模型
    trained_ddpg.save(TRAINED_MODEL_DIR + "/agent_ddpg")


def train_sac():
    e_train_gym = StockTradingEnv(df=train, **env_kwargs)
    env_train, _ = e_train_gym.get_sb_env()
    agent = DRLAgent(env=env_train)
    SAC_PARAMS = {
        "batch_size": 128,
        "buffer_size": 100000,
        "learning_rate": 0.0001,
        "learning_starts": 100,
        "ent_coef": "auto_0.1",
    }

    model_sac = agent.get_model("sac", model_kwargs=SAC_PARAMS)
    # set up logger
    tmp_path = RESULTS_DIR + '/sac'
    new_logger_sac = configure(tmp_path, ["stdout", "csv", "tensorboard"])
    # Set new logger
    model_sac.set_logger(new_logger_sac)

    trained_sac = agent.train_model(model=model_sac,
                                    tb_log_name='sac',
                                    total_timesteps=5000)
    # save model
    trained_sac.save(TRAINED_MODEL_DIR + "/agent_sac")


def train_ppo():
    # PPO https://arxiv.org/abs/1707.06347
    e_train_gym = StockTradingEnv(df=train, **env_kwargs)
    env_train, _ = e_train_gym.get_sb_env()
    agent = DRLAgent(env=env_train)
    PPO_PARAMS = {
        "n_steps": 2048,
        "ent_coef": 0.01,
        "learning_rate": 0.00025,
        "batch_size": 128,
    }
    model_ppo = agent.get_model("ppo", model_kwargs=PPO_PARAMS)
    # set up logger
    tmp_path = RESULTS_DIR + '/ppo'
    new_logger_ppo = configure(tmp_path, ["stdout", "csv", "tensorboard"])
    # Set new logger
    model_ppo.set_logger(new_logger_ppo)
    trained_ppo = agent.train_model(model=model_ppo,
                                    tb_log_name='ppo',
                                    total_timesteps=200000)
    trained_ppo.save(TRAINED_MODEL_DIR + "/agent_ppo")


def train_a2c():
    # A2C https://arxiv.org/abs/1602.01783
    e_train_gym = StockTradingEnv(df=train, **env_kwargs)
    env_train, _ = e_train_gym.get_sb_env()
    agent = DRLAgent(env=env_train)
    model_a2c = agent.get_model("a2c")
    # set up logger
    tmp_path = RESULTS_DIR + '/a2c'
    new_logger_a2c = configure(tmp_path, ["stdout", "csv", "tensorboard"])
    # Set new logger
    model_a2c.set_logger(new_logger_a2c)
    trained_a2c = agent.train_model(model=model_a2c,
                                    tb_log_name='a2c',
                                    total_timesteps=50000)
    trained_a2c.save(TRAINED_MODEL_DIR + "/agent_a2c")


def train_td3():
    # TD3 https://arxiv.org/pdf/1802.09477
    e_train_gym = StockTradingEnv(df=train, **env_kwargs)
    env_train, _ = e_train_gym.get_sb_env()
    agent = DRLAgent(env=env_train)
    TD3_PARAMS = {"batch_size": 100,
                  "buffer_size": 1000000,
                  "learning_rate": 0.001}

    model_td3 = agent.get_model("td3", model_kwargs=TD3_PARAMS)
    # set up logger
    tmp_path = RESULTS_DIR + '/td3'
    new_logger_td3 = configure(tmp_path, ["stdout", "csv", "tensorboard"])
    # Set new logger
    model_td3.set_logger(new_logger_td3)
    trained_td3 = agent.train_model(model=model_td3,
                                    tb_log_name='td3',
                                    total_timesteps=50000)
    trained_td3.save(TRAINED_MODEL_DIR + "/agent_td3")


if __name__ == '__main__':
    # train_ddpg()
    # train_sac()
    # train_ppo()
    # train_a2c()
    train_td3()
