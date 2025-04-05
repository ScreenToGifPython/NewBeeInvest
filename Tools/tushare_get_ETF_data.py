# -*- encoding: utf-8 -*-
"""
@File: tushare_getdata.py
@Modify Time: 2025/4/3 17:21       
@Author: Kevin-Chen
@Descriptions: 使用 Tushare 接口获取 ETF 数据
"""
from datetime import datetime, timedelta
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


# 按年份分割日期, 得到每年的起始和结束日期二维列表
def split_dates_by_year(start_date, end_date):
    # 将字符串转换为datetime对象
    start = datetime.strptime(start_date, '%Y%m%d')
    end = datetime.strptime(end_date, '%Y%m%d')

    result = []
    current_year = start.year

    while current_year <= end.year:
        # 计算当前年份的第一天和最后一天
        year_start = datetime(current_year, 1, 1)
        year_end = datetime(current_year, 12, 31)

        # 如果是第一年，使用实际的起始日期
        if current_year == start.year:
            current_start = start
        else:
            current_start = year_start

        # 如果是最后一年，使用实际的结束日期
        if current_year == end.year:
            current_end = end
        else:
            current_end = year_end

        # 将日期格式化为字符串
        result.append([
            current_start.strftime('%Y%m%d'),
            current_end.strftime('%Y%m%d')
        ])

        current_year += 1

    return result


# 获取ETF的日线数据, 根据指定的开始和结束日期
def get_etf_daily_data(etf_code, start_date='20040101', end_date=datetime.now().strftime('%Y%m%d'), save_parquet=True):
    # 得到每年的起始和结束日期二维列表
    dates_list = split_dates_by_year(start_date, end_date)
    final_df = list()

    # 遍历每年的起始和结束日期, 获取日线数据
    for s_t, e_t in dates_list:
        print(f'正在获取{etf_code}的日线数据，起始日期：{s_t}，结束日期：{e_t}')
        sub_df = pro.fund_daily(ts_code=etf_code, start_date=s_t, end_date=e_t)
        final_df.append(sub_df)

    # 将所有数据合并为一个DataFrame, 并按日期升序排列
    final_df = pd.concat(final_df, ignore_index=True)
    final_df = final_df.sort_values(by='trade_date', ascending=True)
    final_df = final_df.reset_index(drop=True)
    print(f"{etf_code}的日线数据获取完成！")
    print(final_df.info())

    # 将数据保存为parquet格式
    if save_parquet:
        final_df.to_parquet(f'../data/{etf_code}_daily.parquet')


if __name__ == '__main__':
    # df = pro.fund_daily(ts_code='510050.SH', start_date='20050223', end_date='20060101')
    # print(df)

    get_etf_daily_data('510050.SH')
    # df = pd.read_parquet('510050.SH_daily.parquet')
    # print(df.info())
