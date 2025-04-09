# -*- encoding: utf-8 -*-
"""
@File: factor_factory.py
@Modify Time: 2025/4/8 19:25       
@Author: Kevin-Chen
@Descriptions: 因子加工厂
"""
import gc
import numpy as np
import pandas as pd
from Tools.tushare_get_ETF_data import get_etf_info

pd.set_option('display.max_columns', 1000)  # 显示字段的数量
pd.set_option('display.width', 1000)  # 表格不分段显示


# 数据预处理, 生成一系列宽表数据帧
def data_prepare(min_data_req=500):
    """
    数据预处理函数，用于对ETF基金数据进行清洗、筛选和转换，最终生成宽表数据帧。

    该函数的主要步骤包括：
    1. 读取原始数据并进行基本清洗（去重、处理缺失值、转换日期格式）。
    2. 剔除货币类基金。
    3. 剔除数据量不足的基金。
    4. 剔除一段时间内平均成交额在后25%的基金。
    5. 计算收益率。
    6. 建立宽表数据帧，方便后续分析。

    :param min_data_req: int, 可选参数，默认值为500。表示每只基金所需的最小数据量，低于此值的基金将被剔除。
    :return: 返回一个包含多个宽表数据帧的元组，每个数据帧对应一个字段（如开盘价、收盘价、收益率等）。
    """
    ''' 数据预处理 '''
    # 读取数据并进行基本清洗
    etf_df = pd.read_parquet("../data/etf_daily.parquet")
    etf_df = etf_df.dropna(subset=["trade_date", "ts_code"])
    etf_df["trade_date"] = pd.to_datetime(etf_df["trade_date"])
    etf_df = etf_df.drop_duplicates(subset=["trade_date", "ts_code"], keep="last")

    ''' 剔除货币类基金 '''
    # 获取货币型基金的 ts_code 并剔除
    etf_info = get_etf_info()
    mm_etf_df = etf_info[etf_info["invest_type"] == "货币型"]['ts_code'].tolist()
    etf_df = etf_df[~etf_df["ts_code"].isin(mm_etf_df)]
    del mm_etf_df
    gc.collect()

    ''' 剔除数据量不足的基金 '''
    # 统计每个 ts_code 的数据量并剔除数据量不足的基金
    ts_code_counts = etf_df["ts_code"].value_counts()
    etf_df = etf_df[etf_df["ts_code"].isin(ts_code_counts[ts_code_counts >= min_data_req].index)].reset_index(drop=True)
    del ts_code_counts
    gc.collect()

    ''' 剔除一段时间内平均成交额在后25%的基金 '''
    # 筛选近N天内的数据并计算平均成交额，剔除后25%的基金
    df_last_year = etf_df[etf_df["trade_date"] >= etf_df["trade_date"].max() - pd.Timedelta(days=min_data_req)]
    mean_amount = df_last_year.groupby("ts_code")["amount"].mean()
    valid_days = df_last_year.groupby("ts_code")["amount"].count()
    ts_codes_with_enough_data = valid_days[valid_days >= (valid_days.max() * 0.9)].index
    mean_amount = mean_amount.loc[ts_codes_with_enough_data]
    selected_ts_codes = mean_amount[mean_amount > mean_amount.quantile(0.25)].index
    etf_df = etf_df[etf_df["ts_code"].isin(selected_ts_codes)]
    del df_last_year, mean_amount, valid_days, ts_codes_with_enough_data, selected_ts_codes
    gc.collect()

    ''' 计算收益率 '''
    # 计算对数收益率和简单收益率
    etf_df["log_return"] = np.log(etf_df["close"] / etf_df["pre_close"])
    del etf_df['pct_chg']
    etf_df['pct_chg'] = etf_df['close'] / etf_df['pre_close'] - 1

    ''' 建立宽表数据帧 '''
    # 设置索引并生成宽表数据帧
    etf_df.set_index(["trade_date", "ts_code"], inplace=True)
    fields = ["open", "high", "low", "close", "change", "pct_chg", "vol", "amount", "log_return"]
    pivot_dfs = {
        field + "_df": etf_df[field].unstack(level="ts_code") for field in fields
    }
    pivot_dfs['etf_info'] = etf_info

    return pivot_dfs.values()


# 计算基金之间的相关性矩阵
def compute_correlation_matrix(df: pd.DataFrame, method_name: str = "pearson") -> pd.DataFrame:
    """
    计算基金之间的相关性矩阵

    参数:
        df (pd.DataFrame): 对数收益率矩阵，行是日期，列是基金代码
        method_name (str): 相关性计算方法，可选 "pearson"（默认）、"spearman"、"kendall"

    返回:
        pd.DataFrame: 基金两两之间的相关性矩阵
    """
    # 保证数据按列为基金，按行对齐日期
    corr_matrix = df.corr(method=method_name)
    return corr_matrix


(open_df,  # 包含ETF基金的开盘价数据的数据框，行索引为日期，列索引为基金代码
 high_df,  # 包含ETF基金的最高价数据的数据框，行索引为日期，列索引为基金代码
 low_df,  # 包含ETF基金的最低价数据的数据框，行索引为日期，列索引为基金代码
 close_df,  # 包含ETF基金的收盘价数据的数据框，行索引为日期，列索引为基金代码
 change_df,  # 包含ETF基金的价格变动数据的数据框，行索引为日期，列索引为基金代码
 pct_chg_df,  # 包含ETF基金的价格百分比变动数据的数据框，行索引为日期，列索引为基金代码
 vol_df,  # 包含ETF基金的交易量数据的数据框，行索引为日期，列索引为基金代码
 amount_df,  # 包含ETF基金的交易金额数据的数据框，行索引为日期，列索引为基金代码
 log_return_df,  # 包含ETF基金的对数收益率数据的数据框，行索引为日期，列索引为基金代码
 etf_info_df  # 包含ETF基金的基本信息的数据框，如基金代码、基金名称、投资类型等
 ) = data_prepare()

# 将索引日期扩充到自然日
open_df = open_df.resample('D').asfreq()
high_df = high_df.resample('D').asfreq()
low_df = low_df.resample('D').asfreq()
close_df = close_df.resample('D').asfreq()
change_df = change_df.resample('D').asfreq()
pct_chg_df = pct_chg_df.resample('D').asfreq()
vol_df = vol_df.resample('D').asfreq()
amount_df = amount_df.resample('D').asfreq()
log_return_df = log_return_df.resample('D').asfreq()

print("open_df:", open_df.head())
