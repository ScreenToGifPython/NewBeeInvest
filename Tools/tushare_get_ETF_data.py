# -*- encoding: utf-8 -*-
"""
@File: tushare_getdata.py
@Modify Time: 2025/4/3 17:21       
@Author: Kevin-Chen
@Descriptions: 使用 Tushare 接口获取 ETF 数据
"""

from set_tushare import pro


df = pro.fund_daily(ts_code='511520.SZ', start_date='20250101', end_date='20250403')
print(df)
