# -*- encoding: utf-8 -*-
"""
@File: DRL_Portfolio.py
@Modify Time: 2025/4/3 15:48       
@Author: Kevin-Chen
@Descriptions: <基于深度强化学习的投资组合资产配置动态优化研究> 论文复现
"""

import pandas as pd
import random
from datetime import datetime, timedelta


# 生成虚拟的基金历史交易数据
def generate_fund_data(n_funds: int, start_date: str, n_rows: int) -> pd.DataFrame:
    """
    生成包含多个基金历史交易数据的DataFrame

    参数：
    n_funds  - 需要生成的基金数量
    start_date - 数据开始日期 (格式：'YYYYMMDD')
    n_rows   - 每个基金生成的数据条数

    返回：
    pandas.DataFrame - 包含以下字段：
        ts_code    基金代码
        trade_date 交易日期
        open       开盘价
        close      收盘价
        high       最高价
        low        最低价
        vol        成交量
        amount     成交额(千元)
    """
    data = []

    for fund_id in range(n_funds):
        # 生成基金代码 (格式：FUND001)
        ts_code = f"FUND{fund_id + 1:03d}"

        # 生成日期序列
        base_date = datetime.strptime(start_date, "%Y%m%d")
        dates = [(base_date + timedelta(days=i)).strftime("%Y%m%d")
                 for i in range(n_rows)]

        # 初始化价格
        prev_close = None

        for date in dates:
            # 生成开盘价
            if prev_close is None:  # 首日开盘价
                open_price = round(random.uniform(10, 100), 2)
            else:  # 后续开盘价基于前日收盘价波动
                open_price = round(prev_close * (1 + random.uniform(-0.005, 0.005)), 2)

            # 生成收盘价（当日涨跌幅在±5%之间）
            close_price = round(open_price * (1 + random.uniform(-0.05, 0.05)), 2)

            # 生成最高价和最低价（基于当日价格波动）
            max_price = max(open_price, close_price)
            min_price = min(open_price, close_price)

            # 生成最高价（比当日最高价高0-3%）
            high_price = round(max_price * (1 + random.uniform(0, 0.03)), 2)

            # 生成最低价（比当日最低价低0-3%）
            low_price = round(min_price * (1 - random.uniform(0, 0.03)), 2)

            # 生成成交量（10万-50万手）
            vol = random.randint(100000, 500000)

            # 计算成交额（千元）
            avg_price = (open_price + close_price + high_price + low_price) / 4
            amount = round((vol * avg_price) / 1000, 2)  # 转换为千元

            # 记录数据
            data.append([
                ts_code,
                date,
                open_price,
                close_price,
                high_price,
                low_price,
                vol,
                amount
            ])

            # 保存收盘价供下日使用
            prev_close = close_price

    return pd.DataFrame(
        data,
        columns=['ts_code', 'trade_date', 'open', 'close', 'high', 'low', 'vol', 'amount']
    )


if __name__ == '__main__':
    ddf = generate_fund_data(5, '20200101', 600)
    print(ddf)
