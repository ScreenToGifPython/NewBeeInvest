# -*- encoding: utf-8 -*-
"""
@File: metrics_cal.py
@Modify Time: 2025/4/9 08:37       
@Author: Kevin-Chen
@Descriptions: 金融指标计算器
"""
import traceback
import numpy as np

cal_dict = {
    'TotalReturn': 1
}


class CalMetrics:
    def __init__(self, in_p_log_return_array, days_in_p):
        self.return_array = in_p_log_return_array
        self.days_in_p = days_in_p

    # 计算给定回报数组的总回报或累积收益率
    def cal_TotalReturn(self, trans_to_cumulative_return=False):
        """
        计算给定回报数组的总回报或累积收益率。

        参数:
        return_array (numpy.ndarray): 包含回报数据的数组，为每日或每期的对数回报率。
        trans_to_cumulative_return (bool, 可选): 是否将回报转换为普通累积收益率。默认为 False。

        返回值:
        numpy.ndarray: 如果 trans_to_cumulative_return 为 False，返回累积和数组；
                       如果为 True，返回累积收益率数组，即 (e^累积和 - 1)。
        """
        if not trans_to_cumulative_return:
            # 计算回报数组的累积和
            return np.cumsum(self.return_array, axis=0)
        else:
            # 将累积和转换为累积回报
            return np.exp(np.cumsum(self.return_array, axis=0)) - 1

    # 计算并返回每日回报率的平均值
    def cal_AverageDailyReturn(self):
        """
        计算并返回每日回报率的平均值。

        该函数通过计算 `self.return_array` 中每日回报率的平均值来得到平均每日回报率。
        `self.return_array` 是一个包含每日回报率的数组，通常是一个二维数组，其中每一行代表一个资产，每一列代表一个交易日。

        返回值:
        float or numpy.ndarray
            返回每日回报率的平均值。如果 `self.return_array` 是二维数组，则返回一个一维数组，其中每个元素对应一个资产的平均每日回报率。
        """

        return np.mean(self.return_array, axis=0)

    def cal_metric(self, metric_name):
        """
        根据指标名 计算相应的指标值。

        参数:
        metric_name (str): 指标名称，支持 'TotalReturn'。

        返回值:
        numpy.ndarray: 计算得到的指标值。
        """
        try:
            return getattr(self, f'cal_{metric_name}')()
        except Exception as e:
            print(f"Error occurred: {e}")
            print(traceback.format_exc())
            return None
