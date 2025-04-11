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
from scipy.stats import norm, kurtosis, skew

from Tools.metrics_cal_config import return_ann_factor, risk_ann_factor, log_ann_return, log_daily_return

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

        # VaR / CVaR 指标特殊处理
        if metric_name.startswith('VaR') or metric_name.startswith('CVaR'):
            metric_name = metric_name + '-' + str(int(kwargs["confidence_level"] * 100))
        # 分位数指标特殊处理
        elif metric_name.startswith('Percentile') or metric_name.startswith('TailRatio'):
            metric_name = metric_name + '-' + str(int(kwargs["tile"]))

        # 调用被装饰函数计算结果，并将结果缓存到 `self.res_dict` 中
        result = func(self, *args, **kwargs)
        self.res_dict[metric_name] = result

        return result

    return wrapper


# 金融指标计算器类
class CalMetrics:
    def __init__(self, log_return_array, close_price_array, nature_days_in_p, trading_days_in_p,
                 trans_to_cumulative_return=False):
        self.return_array = log_return_array
        self.price_array = close_price_array
        self.nature_days = nature_days_in_p
        self.trading_days = trading_days_in_p
        self.cum_rtn = trans_to_cumulative_return
        self.n_days, self.n_funds = log_return_array.shape
        self.res_dict = dict()

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
            return (total_rtn / self.nature_days) * return_ann_factor  # 使用对数收益率
        else:
            return (total_rtn + 1) ** (return_ann_factor / self.nature_days) - 1  # 使用简单收益率

    @cache_metric  # 每日回报率的平均值
    def cal_AverageDailyReturn(self, **kwargs):
        return np.nanmean(self.return_array, axis=0)

    @cache_metric  # 平均正收益率
    def cal_AvgPositiveReturn(self, **kwargs):
        return np.nanmean(np.where(self.return_array > 0, self.return_array, 0), axis=0)

    @cache_metric  # 平均负收益率
    def cal_AvgNegativeReturn(self, **kwargs):
        return np.nanmean(np.where(self.return_array < 0, self.return_array, 0), axis=0)

    @cache_metric  # 平均盈亏比
    def cal_AvgReturnRatio(self, **kwargs):
        ratio = self.cal_AvgPositiveReturn / self.cal_AvgNegativeReturn
        return np.where(np.isnan(ratio), 0, ratio)

    @cache_metric  # 总累计盈利 = 所有正收益的总和
    def cal_TotalPositiveReturn(self, **kwargs):
        return np.nansum(np.where(self.return_array > 0, self.return_array, 0), axis=0)

    @cache_metric  # 总累计亏损 = 所有负收益的总和
    def cal_TotalNegativeReturn(self, **kwargs):
        return np.nansum(np.where(self.return_array < 0, self.return_array, 0), axis=0)

    @cache_metric  # 盈利总和 / 亏损总和
    def cal_TotalReturnRatio(self, **kwargs):
        ratio = self.cal_TotalPositiveReturn() / np.abs(self.cal_TotalNegativeReturn())
        return np.where(np.isnan(ratio), 0, ratio)

    @cache_metric  # 每日回报率的中位数
    def cal_MedianDailyReturn(self, **kwargs):
        return np.nanmedian(self.return_array, axis=0)

    @cache_metric  # 每日回报率波动率
    def cal_Volatility(self, **kwargs):
        return np.nanstd(self.return_array, axis=0, ddof=1)

    @cache_metric  # 收益率范围
    def cal_ReturnRange(self, **kwargs):
        return self.cal_MaxGain() - self.cal_MaxLoss()

    @cache_metric  # 最大单日收益
    def cal_MaxGain(self, **kwargs):
        return np.nanmax(self.return_array, axis=0)

    @cache_metric  # 最大单日亏损
    def cal_MaxLoss(self, **kwargs):
        return np.nanmin(self.return_array, axis=0)

    @cache_metric  # 年化波动率
    def cal_AnnualizedVolatility(self):
        return self.cal_Volatility() * np.sqrt(risk_ann_factor)

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
        ann_r_dd = (((total_return + 1) ** (return_ann_factor / self.nature_days) - 1) / np.abs(max_dd))
        # 处理无穷大情况
        self.res_dict['ReturnDrawDownRatio'] = np.where(r_dd == -np.inf, 0, r_dd)
        self.res_dict['AnnReturnDrawDownRatio'] = np.where(ann_r_dd == -np.inf, 0, ann_r_dd)
        self.res_dict["MaxDrawDown"] = max_dd

        ''' (6) 计算溃疡指数：衡量回撤波动性的指标（回撤平方均值的平方根） '''
        mean_dd_sq = np.mean(draw_down ** 2, axis=0)
        self.res_dict["UlcerIndex"] = np.sqrt(mean_dd_sq)

        ''' (7) 返回指定指标结果 '''
        return self.res_dict[metric_name]

    @cache_metric  # 年化夏普比率
    def cal_AnnualizedSharpeRatio(self, **kwargs):
        ratio = (self.cal_AnnualizedReturn() - log_ann_return) / self.cal_AnnualizedVolatility()
        return np.where(np.isnan(ratio), 0, ratio)

    @cache_metric  # 夏普比率
    def cal_SharpeRatio(self, **kwargs):
        ratio = (self.cal_TotalReturn() - (log_daily_return * self.nature_days)) / self.cal_Volatility()
        return np.where(np.isnan(ratio), 0, ratio)

    @cache_metric  # 收益率波动率比
    def cal_ReturnVolatilityRatio(self, **kwargs):
        ratio = self.cal_TotalReturn() / self.cal_Volatility()
        return np.where(np.isnan(ratio), 0, ratio)

    @cache_metric  # 下行波动率
    def cal_DownsideVolatility(self, mar=0, **kwargs):
        # 计算低于 MAR 的差值，否则为 0
        down_diff = np.where(self.return_array < mar, self.return_array - mar, 0.0)
        # 计算方差（1个自由度）
        downside_var = np.nanvar(down_diff, axis=0, ddof=1)
        # 返回标准差（即下行波动率）
        ratio = np.sqrt(downside_var)
        return np.where(np.isnan(ratio), 0, ratio)

    @cache_metric  # 上行波动率
    def cal_UpsideVolatility(self, mar=0, **kwargs):
        # 对于超过 MAR 的收益部分，计算偏差；其余为 0
        up_diff = np.where(self.return_array > mar, self.return_array - mar, 0.0)
        # 计算上行方差（忽略 nan，支持自由度）
        upside_var = np.nanvar(up_diff, axis=0, ddof=1)
        # 返回上行标准差
        ratio = np.sqrt(upside_var)
        return np.where(np.isnan(ratio), 0, ratio)

    @cache_metric  # 波动率偏度
    def cal_VolatilitySkew(self, **kwargs):
        ratio = (self.cal_UpsideVolatility() - self.cal_DownsideVolatility()) / self.cal_Volatility()
        return np.where(np.isnan(ratio), 0, ratio)

    @cache_metric  # 波动率比率
    def cal_VolatilityRatio(self, **kwargs):
        ratio = self.cal_UpsideVolatility() / self.cal_DownsideVolatility()
        return np.where(np.isnan(ratio), 0, ratio)

    @cache_metric  # 索提诺比率
    def cal_SortinoRatio(self, **kwargs):
        ratio = (self.cal_TotalReturn() - (log_daily_return * self.nature_days)) / self.cal_DownsideVolatility()
        return np.where(np.isnan(ratio), 0, ratio)

    @cache_metric  # 收益趋势一致性
    def cal_GainConsistency(self, **kwargs):
        # 筛选正收益
        gain_only = np.where(self.return_array > 0, self.return_array, np.nan)

        # 计算均值和标准差（忽略 nan）
        gain_mean = np.nanmean(gain_only, axis=0)  # shape = (N,)
        gain_std = np.nanstd(gain_only, axis=0)  # shape = (N,)

        # 避免除以 0 或 nan
        gain_consistency = gain_std / gain_mean

        return np.where(np.isnan(gain_consistency), 0, gain_consistency)

    @cache_metric  # 损失趋势一致性
    def cal_LossConsistency(self, **kwargs):
        # 筛选负收益
        loss_only = np.where(self.return_array < 0, self.return_array, np.nan)

        # 计算负收益的平均值和标准差
        loss_mean = np.nanmean(loss_only, axis=0)  # 为负值
        loss_std = np.nanstd(loss_only, axis=0)

        # 用 abs(loss_mean) 避免除以负值
        loss_consistency = loss_std / np.abs(loss_mean)

        return np.where(np.isnan(loss_consistency), 0, loss_consistency)

    @cache_metric  # 计算盈利率
    def cal_WinningRatio(self, **kwargs):
        # 计算胜率
        return (np.sum((self.return_array > 0) & ~np.isnan(self.return_array), axis=0)
                / np.sum(~np.isnan(self.return_array), axis=0))

    @cache_metric  # 计算亏损率
    def cal_LosingRatio(self, **kwargs):
        # 计算亏损率
        return (np.sum((self.return_array < 0) & ~np.isnan(self.return_array), axis=0)
                / np.sum(~np.isnan(self.return_array), axis=0))

    @cache_metric  # 平均绝对偏差
    def cal_MeanAbsoluteDeviation(self, **kwargs):
        return np.nanmean(np.abs(self.return_array - self.cal_AverageDailyReturn()), axis=0)

    @cache_metric  # 收益率偏度
    def cal_ReturnSkewness(self, **kwargs):
        return skew(self.return_array, axis=0, bias=False, nan_policy='omit')

    @cache_metric  # 收益率峰度
    def cal_ReturnKurtosis(self, excess=True, **kwargs):
        excess_kurtosis = kurtosis(self.return_array, axis=0, bias=False, nan_policy='omit')  # 返回的是超峰度
        if excess:
            return excess_kurtosis  # 正态分布的超峰度是 0
        else:
            return excess_kurtosis + 3  # 正态分布的总峰度是 3

    @cache_metric  # 计算在险价值 (参数法)
    def cal_VaR(self, confidence_level=0.99, **kwargs):
        mean_return = self.cal_AverageDailyReturn()
        std_return = self.cal_Volatility()
        alpha = 1.0 - confidence_level
        z_left = norm.ppf(alpha)
        var = - (mean_return + z_left * std_return)
        return np.maximum(var, 0.0)  # VaR 是损失，不小于0

    @cache_metric  # 基于 VaR 计算夏普比率
    def cal_VaRSharpe(self, confidence_level=0.99, **kwargs):
        var_name = 'VaR' + '-' + str(int(confidence_level * 100))
        ratio = (self.cal_TotalReturn() - (log_daily_return * self.nature_days)) / self.cal_metric(var_name)
        # 将 inf 转为 0
        ratio = np.where(np.isinf(ratio), 0, ratio)
        return np.where(np.isnan(ratio), 0, ratio)

    @cache_metric  # 计算 Cornish-Fisher修正后的在险价值 (参数法)
    def cal_VaRModified(self, confidence_level=0.99, **kwargs):
        mu = self.cal_AverageDailyReturn()
        sigma = self.cal_Volatility()
        skewness = self.cal_ReturnSkewness()
        k_excess = self.cal_ReturnKurtosis()

        alpha = 1.0 - confidence_level
        z = norm.ppf(alpha)

        z_mod = (z
                 + (skewness / 6.0) * (z ** 2 - 1.0)
                 + (k_excess / 24.0) * (z ** 3 - 3.0 * z)
                 - ((skewness ** 2) / 36.0) * (2.0 * z ** 3 - 5.0 * z)
                 )

        var_modified = - (mu + z_mod * sigma)
        return np.maximum(var_modified, 0.0)

    @cache_metric  # 基于 修正后的VaR 计算夏普比率
    def cal_VaRModifiedSharpe(self, confidence_level=0.99, **kwargs):
        var_name = 'VaRModified' + '-' + str(int(confidence_level * 100))
        ratio = (self.cal_TotalReturn() - (log_daily_return * self.nature_days)) / self.cal_metric(var_name)
        # 将 inf 转为 0
        ratio = np.where(np.isinf(ratio), 0, ratio)
        return np.where(np.isnan(ratio), 0, ratio)

    @cache_metric  # 计算期望损失ES (参数法)
    def cal_CVaR(self, confidence_level=0.99, **kwargs):
        mu = self.cal_AverageDailyReturn()
        sigma = self.cal_Volatility()
        alpha = 1.0 - confidence_level
        z = norm.ppf(alpha)
        phi_z = norm.pdf(z)
        cvar = - (mu - (sigma * phi_z) / alpha)
        return np.maximum(cvar, 0.0)  # CVaR 是损失，不应为负

    @cache_metric  # 基于 CVaR 计算夏普比率
    def cal_CVaRSharpe(self, confidence_level=0.99, **kwargs):
        cvar_name = 'CVaR' + '-' + str(int(confidence_level * 100))
        ratio = (self.cal_TotalReturn() - (log_daily_return * self.nature_days)) / self.cal_metric(cvar_name)
        # 将 inf 转为 0
        ratio = np.where(np.isinf(ratio), 0, ratio)
        return np.where(np.isnan(ratio), 0, ratio)

    @cache_metric  # 计算 Cornish-Fisher修正后的期望损失ES (参数法)
    def cal_CVaRModified(self, confidence_level=0.99, **kwargs):
        mu = self.cal_AverageDailyReturn()
        sigma = self.cal_Volatility()
        alpha = 1.0 - confidence_level
        skewness = self.cal_ReturnSkewness()
        k_excess = self.cal_ReturnKurtosis()

        alpha = 1.0 - confidence_level
        z = norm.ppf(alpha)

        z_mod = (z
                 + (skewness / 6.0) * (z ** 2 - 1.0)
                 + (k_excess / 24.0) * (z ** 3 - 3.0 * z)
                 - ((skewness ** 2) / 36.0) * (2.0 * z ** 3 - 5.0 * z))

        phi_z_mod = norm.pdf(z_mod)
        cvar_mod = - (mu - (sigma * phi_z_mod) / alpha)
        return np.maximum(cvar_mod, 0.0)

    @cache_metric  # 基于 修正后的CVaR 计算夏普比率
    def cal_CVaRModifiedSharpe(self, confidence_level=0.99, **kwargs):
        cvar_name = 'CVaRModified' + '-' + str(int(confidence_level * 100))
        ratio = (self.cal_TotalReturn() - (log_daily_return * self.nature_days)) / self.cal_metric(cvar_name)
        # 将 inf 转为 0
        ratio = np.where(np.isinf(ratio), 0, ratio)
        return np.where(np.isnan(ratio), 0, ratio)

    @cache_metric  # 计算收益率的分位数 (考虑所有收益率)
    def cal_Percentile(self, tile=5, **kwargs):
        perc_tile = np.nanpercentile(self.return_array, tile, axis=0)
        return perc_tile

    @cache_metric  # 计算收益率的分位数 (仅仅考虑正收益率)
    def cal_PercentileWin(self, tile=95, **kwargs):
        # 替换负值为 NaN，只保留正收益
        positive_only = np.where(self.return_array > 0, self.return_array, np.nan)
        # 计算分位（即最差的前5%盈利）
        perc_tile = np.nanpercentile(positive_only, tile, axis=0)
        # 任何负值或 nan 都设为0
        return np.where(np.isnan(perc_tile), 0.0, perc_tile)

    @cache_metric  # 计算收益率的分位数 (仅仅考虑负收益率, 并且负收益取绝对值)
    def cal_PercentileLoss(self, tile=95, **kwargs):
        # 替换正值为 NaN，只保留负收益
        negative_only = np.where(self.return_array < 0, self.return_array, np.nan)
        # 负收益率取绝对值
        negative_only = np.abs(negative_only)
        # 计算分位（即最严重的亏损）
        perc_tile = np.nanpercentile(negative_only, tile, axis=0)
        # 任何正值或 nan 都设为0
        return np.where(np.isnan(perc_tile), 0.0, perc_tile)

    @cache_metric
    def cal_TailRatio(self, tile=95, **kwargs):
        win_name = 'PercentileWin' + '-' + str(int(tile))
        loss_name = 'PercentileLoss' + '-' + str(int(tile))
        # 计算盈利和亏损的分位数
        win = self.cal_metric(win_name)
        loss = self.cal_metric(loss_name)

        # 设置极小值防止除以 0 或非常接近 0 导致爆炸
        eps = 1e-6
        safe_loss = np.where(np.abs(loss) < eps, eps, loss)

        return win / safe_loss

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
        # VaR 指标的处理
        elif metric_name.startswith('VaR') or metric_name.startswith('CVaR'):
            kwargs["confidence_level"] = float(metric_name.split("-")[1]) / 100
            metric_name = metric_name.split("-")[0]
        # 分位数指标的处理
        elif metric_name.startswith('Percentile') or metric_name.startswith('TailRatio'):
            kwargs["tile"] = float(metric_name.split("-")[1])
            metric_name = metric_name.split("-")[0]

        method_name = f'cal_{metric_name}'
        try:
            return getattr(self, method_name)(**kwargs)
        except Exception as e:
            print(f"Error when calling '{method_name}': {e}")
            print(traceback.format_exc())


if __name__ == '__main__':
    the_close_price_array = pd.read_parquet('close_df.parquet')
    the_log_return_df = pd.read_parquet('log_return_df.parquet')

    the_close_price_array = the_close_price_array.resample('D').asfreq()
    the_log_return_df = the_log_return_df.resample('D').asfreq()

    the_close_price_array = the_close_price_array.values[:, :10]
    the_log_return_df = the_log_return_df.values[:, :10]

    # 删除所有全为 NaN 的行
    the_close_price_array = the_close_price_array[~np.all(np.isnan(the_close_price_array), axis=1)]
    the_log_return_df = the_log_return_df[~np.all(np.isnan(the_log_return_df), axis=1)]
    # the_close_price_array[:, 3] = 1
    # the_log_return_df[:, 3] = -0.001
    # the_log_return_df[:, 1] = 0.001

    c_m = CalMetrics(the_log_return_df, the_close_price_array, 61, 50)
    res = c_m.cal_metric('TailRatio-90')
    print(res)
    res = c_m.cal_metric('TailRatio-95')
    print(res)
    print(c_m.res_dict)
