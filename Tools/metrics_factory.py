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
from dateutil.relativedelta import relativedelta
from Tools.tushare_get_ETF_data import get_etf_info
from Tools.metrics_cal import CalMetrics
from Tools.metrics_cal_config import period_list, create_period_metrics_map

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


# 根据给定区间 period 和结束日期 last_day, 返回该区间的开始日期
def get_start_date(last_day, period):
    """
    根据给定区间 period 和结束日期 last_day, 返回该区间的开始日期(“包含”逻辑可以配合 > start_date, <= last_day使用).
    返回一个 pd.Timestamp (如需 date, 可自行调用 .date()).

    参数:
    last_day (str or datetime): 区间的结束日期，可以是字符串或 datetime 对象。
    period (str): 区间类型，支持以下格式 (全部是不包含开始日期的模式)：
        - 'mtd': 从上个月尾开始
        - 'qtd': 从上个季度末开始
        - 'ytd': 从去年年末开始
        - 'Nd': 过去N天
        - 'Nw': 过去N周
        - 'Nm': 过去N个月
        - 'Ny': 过去N年

    返回:
    pd.Timestamp: 区间的开始日期。
    """
    # 将 last_day 转换为 pd.Timestamp 并只保留日期部分
    ld = pd.to_datetime(last_day).normalize()

    # 根据 period 类型计算开始日期
    if period.lower() == 'mtd':
        # Month-To-Date: 从当月1号开始，但为了满足 > start_date 的过滤条件，start_date 需要比真正区间早1天
        real_start = ld.replace(day=1)
        start = real_start - pd.Timedelta(days=1)

    elif period.lower() == 'qtd':
        # Quarter-To-Date: 从当季度1号开始，同样为了满足过滤条件，start_date 需要比真正区间早1天
        current_month = ld.month
        quarter_start_month = ((current_month - 1) // 3) * 3 + 1
        real_start = ld.replace(month=quarter_start_month, day=1)
        start = real_start - pd.Timedelta(days=1)

    elif period.lower() == 'ytd':
        # Year-To-Date: 从当年1号开始，为了满足过滤条件，start_date 需要比真正区间早1天
        real_start = ld.replace(month=1, day=1)
        start = real_start - pd.Timedelta(days=1)

    elif period.endswith('d'):
        # 过去N天: 从 last_day 减去 N 天
        num_days = int(period[:-1])
        start = ld - pd.Timedelta(days=num_days)

    elif period.endswith('w'):
        # 过去N周: 从 last_day 减去 N 周
        num_weeks = int(period[:-1])
        start = ld - pd.Timedelta(weeks=num_weeks)

    elif period.endswith('m'):
        # 过去N个月: 从 last_day 减去 N 个月
        num_months = int(period[:-1])
        start = ld - relativedelta(months=num_months)

    elif period.endswith('y'):
        # 过去N年: 从 last_day 减去 N 年
        num_years = int(period[:-1])
        start = ld - relativedelta(years=num_years)

    else:
        # 如果 period 类型不支持，抛出异常
        raise ValueError(f"不支持的区间类型: {period}")

    return pd.to_datetime(start)  # 返回开始日期，如果需要 date 类型, 可以 return start.date()


def find_date_range_indices(nature_days_array, start_date, end_date):
    start_idx = np.searchsorted(nature_days_array, start_date, side='left')
    end_idx = np.searchsorted(nature_days_array, end_date, side='right') - 1
    return start_idx, end_idx


# 遍历要计算的区间
def compute_metrics_for_period_initialize(log_return_df):
    # 获取交易日序列
    trading_days_array = pd.to_datetime(np.array(log_return_df.index))
    # 将索引日期扩充到自然日
    log_return_df = log_return_df.resample('D').asfreq()
    # 获取自然日序列
    nature_days_array = pd.to_datetime(np.array(log_return_df.index))
    # 将 log_return_df 转换为 numpy 数组
    log_return_array = log_return_df.values
    # 得到区间与该区间要计算指标的映射
    period_metrics_map = create_period_metrics_map()
    # 释放内存
    del log_return_df
    gc.collect()

    for period in ['2d']:
    # for period in period_list:
        for end_date in reversed(trading_days_array):
            # 计算开始日期
            start_date = get_start_date(end_date, period)
            # 找到 开始日期 & 解释日期在 nature_days_array 的索引位置
            start_idx, end_idx = find_date_range_indices(nature_days_array, start_date, end_date)
            # 如果 start_idx + 1 >= end_idx，表示 start_date + 1 超出或无数据，跳出循环
            if start_idx + 1 > end_idx:
                break
            # 截取在区间内的 log_return_array 数据
            in_p_log_return_array = log_return_array[start_idx + 1: end_idx + 1]
            # 计算区间内有多少个自然日
            days_in_p = end_idx - start_idx

            # # 遍历该区间要计算的指标
            # for metric in period_metrics_map[period]:
            c_m = CalMetrics(in_p_log_return_array, days_in_p)
            sub_res = [c_m.cal_metric(metric_name) for metric_name in period_metrics_map[period]]
            print(sub_res)


if __name__ == '__main__':
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

    close_df.to_parquet('close_df.parquet')
    the_close_df = pd.read_parquet('close_df.parquet')

    log_return_df.to_parquet('log_return_df.parquet')
    the_log_return_df = pd.read_parquet('log_return_df.parquet')

# # 获取交易日序列
# trading_days = np.array(open_df.index)
#
# # 将索引日期扩充到自然日
# open_df = open_df.resample('D').asfreq()
# high_df = high_df.resample('D').asfreq()
# low_df = low_df.resample('D').asfreq()
# close_df = close_df.resample('D').asfreq()
# change_df = change_df.resample('D').asfreq()
# pct_chg_df = pct_chg_df.resample('D').asfreq()
# vol_df = vol_df.resample('D').asfreq()
# amount_df = amount_df.resample('D').asfreq()
# log_return_df = log_return_df.resample('D').asfreq()
#
# # 获取自然日序列
# nature_days = np.array(open_df.index)
