# -*- encoding: utf-8 -*-
"""
@File: finrl_create_env.py
@Modify Time: 2025/4/6 15:31       
@Author: Kevin-Chen
@Descriptions: 
"""
import pandas as pd
from finrl.main import check_and_make_directories
from finrl.config import INDICATORS, TRAINED_MODEL_DIR

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

# 计算股票种类数量
stock_dimension = len(train.tic.unique())

# 计算状态空间大小，包括股票价格、持有数量和技术指标
state_space = 1 + 2 * stock_dimension + len(INDICATORS) * stock_dimension

# 设置买卖股票的成本比例
buy_cost_list = sell_cost_list = [0.001] * stock_dimension

# 初始化持有股票的数量为0
num_stock_shares = [0] * stock_dimension

# 定义环境参数字典
env_kwargs = {
    # 定义环境参数
    "hmax": 100,  # 单次交易的最大股票数量
    "initial_amount": 100_0000,  # 初始资金量
    "num_stock_shares": num_stock_shares,  # 每只股票的初始股数
    "buy_cost_pct": buy_cost_list,  # 买入成本百分比列表，每只股票的买入成本不同
    "sell_cost_pct": sell_cost_list,  # 卖出成本百分比列表，每只股票的卖出成本不同
    "state_space": state_space,  # 状态空间大小，即状态的特征数量
    "stock_dim": stock_dimension,  # 股票维度，即股票种类数量
    "tech_indicator_list": INDICATORS,  # 技术指标列表，用于提供给模型的技术分析数据
    "action_space": stock_dimension,  # 动作空间大小，即可能的交易动作数量，通常与股票种类数量一致
    "reward_scaling": 1e-4  # 奖励缩放因子，用于调整奖励的大小，避免过大的数值导致的学习不稳定
}

"""
StockTradingEnv的参数为:
    df: pd.DataFrame, 用于交易的数据集，包含股票价格和其他技术指标。
    stock_dim: int, 股票的维度，即投资组合中股票的种类数量。
    hmax: int, 每个交易日允许的最大交易量。
    initial_amount: int, 初始投资金额。
    num_stock_shares: list[int], 每只股票的初始持有量。
    buy_cost_pct: list[float], 购买每只股票的成本比例。
    sell_cost_pct: list[float], 出售每只股票的成本比例。
    reward_scaling: float, 奖励缩放因子，用于调整奖励的大小。
    state_space: int, 状态空间的维度。
    action_space: int, 动作空间的维度。
    tech_indicator_list: list[str], 技术指标列表，用于提供给智能体的观察值。
    turbulence_threshold=None, 湍流阈值，超过此阈值则市场被认为是湍流的。
    risk_indicator_col="turbulence", 风险指标列的名称，默认为“turbulence”。
    make_plots: bool = False, 是否在运行时生成和保存图表。
    print_verbosity=10, 打印频率，表示每隔多少个步骤打印一次信息。
    day=0, 当前的交易日索引。
    initial=True, 是否为初始化状态。
    previous_state=[], 前一个状态的记录。
    model_name="", 使用的模型名称。
    mode="", 运行模式，例如“train”或“test”。
    iteration="", 迭代次数或标识。
"""


