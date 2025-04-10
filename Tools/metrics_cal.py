# -*- encoding: utf-8 -*-
"""
@File: metrics_cal.py
@Modify Time: 2025/4/9 08:37       
@Author: Kevin-Chen
@Descriptions: 金融指标计算器
"""
import warnings
import traceback
import numpy as np
import pandas as pd
from functools import wraps

from Tools.metrics_cal_config import return_ann_factor, risk_ann_factor

warnings.filterwarnings("ignore")
pd.set_option('display.width', 1000)  # 表格不分段显示
pd.set_option('display.max_columns', 1000)  # 显示字段的数量


# 装饰器函数，用于缓存被装饰函数的计算结果
def cache_metric(func):
    """
    装饰器函数，用于缓存被装饰函数的计算结果。

    该装饰器会将被装饰函数的计算结果缓存到 `self.res_dict` 中，以避免重复计算。
    如果缓存中已经存在该函数的计算结果，则直接返回缓存中的值。

    Args:
        func (function): 需要被装饰的函数，通常是一个计算指标的函数。

    Returns:
        function: 返回一个包装函数 `wrapper`，该函数会执行缓存逻辑。
    """

    @wraps(func)
    def wrapper(self, *args, **kwargs):
        """
        包装函数，用于执行缓存逻辑。

        该函数会从被装饰函数的名称中提取指标名称，并检查该指标是否已经存在于 `self.res_dict` 中。
        如果存在，则直接返回缓存中的值；如果不存在，则调用被装饰函数计算结果，并将结果缓存到 `self.res_dict` 中。

        Args:
            self: 类的实例对象。
            *args: 传递给被装饰函数的位置参数。
            **kwargs: 传递给被装饰函数的关键字参数。

        Returns:
            Any: 返回被装饰函数的计算结果或缓存中的值。
        """
        # 从函数名中提取指标名称，去掉前缀 "cal_"
        metric_name = func.__name__.replace("cal_", "")

        # 如果指标已经存在于缓存中，则直接返回缓存中的值
        if metric_name in self.res_dict:
            return self.res_dict[metric_name]

        # 调用被装饰函数计算结果，并将结果缓存到 `self.res_dict` 中
        result = func(self, *args, **kwargs)
        self.res_dict[metric_name] = result

        return result

    return wrapper


class CalMetrics:
    def __init__(self, log_return_array, close_price_array, days_in_p, trans_to_cumulative_return=False):
        self.return_array = log_return_array
        self.price_array = close_price_array
        self.days_in_p = days_in_p
        self.cum_rtn = trans_to_cumulative_return
        self.res_dict = dict()
        self.n_days, self.n_funds = log_return_array.shape

    @cache_metric  # 累积收益率
    def cal_TotalReturn(self, **kwargs):
        if not self.cum_rtn:  # 判断累计收益率的计算逻辑
            return np.nansum(self.return_array, axis=0)  # 计算对数收益率的累计值
        else:
            np.exp(np.nansum(self.return_array, axis=0)) - 1  # 计算普通收益率的累计值

    @cache_metric  # 年化收益率
    def cal_AnnualizedReturn(self, **kwargs):
        # 计算累计收益率
        total_rtn = self.cal_TotalReturn()

        if not self.cum_rtn:  # 判断累计收益率的计算逻辑
            return (total_rtn / self.days_in_p) * return_ann_factor  # 使用对数收益率
        else:
            return (total_rtn + 1) ** (return_ann_factor / self.days_in_p) - 1  # 使用简单收益率

    @cache_metric  # 每日回报率的平均值
    def cal_AverageDailyReturn(self, **kwargs):
        return np.nanmean(self.return_array, axis=0)

    @cache_metric  # 每日回报率的中位数
    def cal_MedianDailyReturn(self, **kwargs):
        return np.nanmedian(self.return_array, axis=0)

    @cache_metric  # 每日回报率波动率
    def cal_Volatility(self, **kwargs):
        return np.nanstd(self.return_array, axis=0, ddof=1)

    @cache_metric  # 计算所有最大回撤相关的指标
    def cal_max_draw_down_for_all(self, metric_name: str, **kwargs) -> float:
        """
        计算所有基金的最大回撤及相关衍生指标

        :param metric_name: 需要返回的指标名称（如"MaxDrawDown", "MaxDrawDownDays" 等）
        :param kwargs: 其他可选参数（当前未被使用）
        :return: 指定指标对应的结果值
        """

        ''' (1) 计算 最大回撤：通过逐日跟踪历史最高净值，计算当前回撤率并取最小值 '''
        # 计算历史最大净值（逐日累计最大值）
        max_price = np.fmax.accumulate(np.where(np.isnan(self.price_array), -np.inf, self.price_array), axis=0)
        # 计算回撤序列（当前价格 / 历史最大值 - 1）
        draw_down = self.price_array / max_price - 1
        # 处理NaN值（无价格数据时回撤为0）
        draw_down = np.where(np.isnan(draw_down), 0, draw_down)
        # 取每只基金的最大回撤（最小回撤率即最大跌幅）
        max_dd = np.nanmin(draw_down, axis=0)

        ''' (2) 确定最大回撤结束位置：通过反转数组找到最后出现的最小值索引 '''
        # 反转数组后寻找各列最小值首次出现位置（对应原始数组的最后出现位置）
        reversed_argmin = np.argmin(draw_down[::-1, :], axis=0)
        # 转换为原始数组坐标系下的索引
        max_dd_idx = (self.n_days - 1) - reversed_argmin

        ''' (3) 计算最大回撤持续天数：通过构造有效掩码定位回撤起始与结束点 '''
        # 构建有效回撤区间掩码（满足回撤期间且未超过结束索引）
        mask_valid = (draw_down >= 0) & (np.arange(self.n_days)[:, None] <= max_dd_idx[None, :])
        # 反转掩码后寻找峰值索引（最后一次满足条件的位置）
        reversed_mask_valid = mask_valid[::-1, :]
        reversed_idx_peak = np.argmax(reversed_mask_valid, axis=0)
        # 转换为原始坐标系并计算持续天数
        peak_idx = (self.n_days - 1) - reversed_idx_peak
        self.res_dict['MaxDrawDownDays'] = max_dd_idx - peak_idx

        ''' (4) 计算回撤斜率：最大回撤绝对值与持续天数的比率 '''
        dd_slope = np.abs(max_dd) / self.res_dict['MaxDrawDownDays']
        self.res_dict['DrawDownSlope'] = np.where(np.isnan(dd_slope), 0, dd_slope)

        ''' (5) 计算收益回撤比及年化版本：包含普通和年化两种计算方式 '''
        # 转换对数累计收益率为普通收益率
        total_return = np.exp(self.cal_TotalReturn()) - 1
        # 计算收益回撤比（普通和年化版本）
        r_dd = total_return / np.abs(max_dd)
        ann_r_dd = (((total_return + 1) ** (return_ann_factor / self.days_in_p) - 1) / np.abs(max_dd))
        # 处理无穷大情况
        self.res_dict['ReturnDrawDownRatio'] = np.where(r_dd == -np.inf, 0, r_dd)
        self.res_dict['AnnReturnDrawDownRatio'] = np.where(ann_r_dd == -np.inf, 0, ann_r_dd)
        self.res_dict["MaxDrawDown"] = max_dd

        ''' (6) 计算溃疡指数：衡量回撤波动性的指标（回撤平方均值的平方根） '''
        mean_dd_sq = np.mean(draw_down ** 2, axis=0)
        self.res_dict["UlcerIndex"] = np.sqrt(mean_dd_sq)

        ''' (7) 返回指定指标结果 '''
        return self.res_dict[metric_name]

    @cache_metric  # 年化波动率
    def cal_AnnualizedVolatility(self):
        return self.cal_Volatility() * np.sqrt(risk_ann_factor)

    @cache_metric  # 平均绝对偏差
    def cal_MeanAbsoluteDeviation(self, **kwargs):
        return np.nanmean(np.abs(self.return_array - self.cal_AverageDailyReturn()), axis=0)

    @cache_metric  # 最大单日收益
    def cal_MaxGain(self, **kwargs):
        return np.nanmax(self.return_array, axis=0)

    @cache_metric  # 最大单日亏损
    def cal_MaxLoss(self, **kwargs):
        return np.nanmin(self.return_array, axis=0)

    @cache_metric  # 收益率范围
    def cal_ReturnRange(self, **kwargs):
        return self.cal_MaxGain() - self.cal_MaxLoss()

    # 调用不同指标计算的函数
    def cal_metric(self, metric_name, **kwargs):
        """
        根据指标名 计算相应的指标值。

        参数:
        metric_name (str): 指标名称，如 'TotalReturn'

        返回值:
        numpy.ndarray: 计算得到的指标值。
        """

        # 最大回撤指标的统一处理
        if metric_name in ['MaxDrawDown', 'MaxDrawDownDays', 'ReturnDrawDownRatio',
                           'AnnReturnDrawDownRatio', 'DrawDownSlope', 'UlcerIndex']:
            return self.cal_max_draw_down_for_all(metric_name)

        method_name = f'cal_{metric_name}'
        if hasattr(self, method_name):
            try:
                return getattr(self, method_name)()
            except Exception as e:
                print(f"Error when calling '{method_name}': {e}")
                print(traceback.format_exc())
        else:
            print(f"Method '{method_name}' does not exist. 该方法不存在! 请确认是否有该指标!")


if __name__ == '__main__':
    the_close_price_array = pd.read_parquet('close_df.parquet')
    the_log_return_df = pd.read_parquet('log_return_df.parquet')

    the_close_price_array = the_close_price_array.resample('D').asfreq()
    the_log_return_df = the_log_return_df.resample('D').asfreq()

    the_close_price_array = the_close_price_array.values[-20:, :10]
    the_log_return_df = the_log_return_df.values[-20:, :10]

    # 删除所有全为 NaN 的行
    the_close_price_array = the_close_price_array[~np.all(np.isnan(the_close_price_array), axis=1)]
    the_log_return_df = the_log_return_df[~np.all(np.isnan(the_log_return_df), axis=1)]
    the_close_price_array[:, 3] = 1
    the_close_price_array[:, 3] = 1

    c_m = CalMetrics(the_log_return_df, the_close_price_array[::-1, :], 50)
    res = c_m.cal_metric('MaxDrawDown')
    res_q = c_m.cal_metric('MaxDrawDownDays')
    res_u = c_m.cal_metric('UlcerIndex')
    print(c_m.res_dict)
