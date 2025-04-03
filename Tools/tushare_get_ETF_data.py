# -*- encoding: utf-8 -*-
"""
@File: tushare_getdata.py
@Modify Time: 2025/4/3 17:21       
@Author: Kevin-Chen
@Descriptions: 使用 Tushare 接口获取 ETF 数据
"""

from set_tushare import pro
import pandas as pd

pd.set_option('display.max_columns', 1000)  # 显示字段的数量
pd.set_option('display.width', 1000)  # 表格不分段显示


# 获取ETF的基本信息
def get_etf_info(drop_dlist=True, save_parquet=True):
    # 获取ETF基本信息
    etf_info_df = pro.fund_basic(market='E')
    # 将ETF按list_date升序排列
    etf_info_df = etf_info_df.sort_values(by='list_date', ascending=True)
    # 剔除已经清盘的ETF
    if drop_dlist:
        etf_info_df = etf_info_df[etf_info_df['delist_date'].isnull()]
    etf_info_df = etf_info_df.reset_index(drop=True)
    # 将数据保存为parquet格式
    if save_parquet:
        etf_info_df.to_parquet('etf_info.parquet')
    return etf_info_df



# df = pro.fund_daily(ts_code='160105.SZ', start_date='20041201', end_date='20050101')
# print(df)
