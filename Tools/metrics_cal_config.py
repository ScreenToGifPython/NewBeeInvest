# -*- encoding: utf-8 -*-
"""
@File: metrics_cal_config.py
@Modify Time: 2025/4/9 10:52       
@Author: Kevin-Chen
@Descriptions: 金融指标计算参数文件
"""
import numpy as np
from collections import defaultdict

# 年化收益率计算的天数
return_ann_factor = 365
# 年化波动率计算的天数
risk_ann_factor = 252

# 无风险普通年化收益率
no_risk_ann_return = 0.015
# 无风险普通每日收益率 (使用 自然日or交易日 ? 请自由调整)
daily_return = (1 + no_risk_ann_return) ** (1 / return_ann_factor) - 1
# 无风险对数每日收益率
log_daily_return = np.log(1 + daily_return)
# 无风险对数年化收益率
log_ann_return = np.log(1 + no_risk_ann_return)

# 支持的区间代码
period_list = [
    '2d',  # 2天
    '3d',  # 3天
    '1w',  # 1周
    '2w',  # 2周
    '1m',  # 1个月
    '2m',  # 2个月
    '3m',  # 3个月
    '5m',  # 5个月
    '6m',  # 6个月
    '12m',  # 12个月（1年）
    '2y',  # 2年
    '3y',  # 3年
    '5y',  # 5年
    '10y',  # 10年
    'mtd',  # 本月至今（Month-to-Date）
    'qtd',  # 本季度至今（Quarter-to-Date）
    'ytd',  # 本年至今（Year-to-Date）
]

# 基于对数收益率计算的指标
log_return_metrics_dict = {
    'TotalReturn':
        ['总收益率',  # 指标名称
         '总收益率 = (最终价值 - 初始价值) / 初始价值',  # 指标的简要计算方法说明
         # 支持计算的区间
         ['2d', '3d', '1w', '2w', '1m', '2m', '3m', '5m', '6m', '12m', '2y', '3y', '5y', 'mtd', 'qtd', 'ytd'],
         ],
    'AnnualizedReturn':
        ['年化收益率',
         '年化收益率 = (1 + 总收益率)^(return_ann_factor/天数) - 1',
         ['1w', '2w', '1m', '2m', '3m', '5m', '6m', '12m', '2y', '3y', '5y', 'mtd', 'qtd', 'ytd'],
         ],
    'AverageDailyReturn':
        ['日均收益率',
         '日均收益率 = 总收益率 / 交易天数',
         ['2d', '3d', '1w', '2w', '1m', '2m', '3m', '5m', '6m', '12m', '2y', '3y', '5y', 'mtd', 'qtd', 'ytd'],
         ],
    'AvgPositiveReturn':
        ['平均正收益率',
         '平均正收益率 = 所有正收益的平均值',
         ['1w', '2w', '1m', '2m', '3m', '5m', '6m', '12m', '2y', '3y', '5y', 'mtd', 'qtd', 'ytd'],
         ],
    'AvgNegativeReturn':
        ['平均负收益率',
         '平均负收益率 = 所有负收益的平均值',
         ['1w', '2w', '1m', '2m', '3m', '5m', '6m', '12m', '2y', '3y', '5y', 'mtd', 'qtd', 'ytd'],
         ],
    'AvgReturnRatio':
        ['平均盈亏比',
         '平均盈亏比 = 平均正收益率 / 平均负收益率',
         ['1w', '2w', '1m', '2m', '3m', '5m', '6m', '12m', '2y', '3y', '5y', 'mtd', 'qtd', 'ytd'],
         ],
    'TotalPositiveReturn':
        ['总累计盈利',
         '总累计盈利 = 所有正收益的总和',
         ['1w', '2w', '1m', '2m', '3m', '5m', '6m', '12m', '2y', '3y', '5y', 'mtd', 'qtd', 'ytd'],
         ],
    'TotalNegativeReturn':
        ['总累计亏损',
         '总累计亏损 = 所有负收益的总和',
         ['1w', '2w', '1m', '2m', '3m', '5m', '6m', '12m', '2y', '3y', '5y', 'mtd', 'qtd', 'ytd'],
         ],
    'TotalReturnRatio':
        ['累计盈亏比',
         '累计盈亏比 = 总累计盈利 / 总累计亏损',
         ['1w', '2w', '1m', '2m', '3m', '5m', '6m', '12m', '2y', '3y', '5y', 'mtd', 'qtd', 'ytd'],
         ],
    'MedianDailyReturn':
        ['日中位收益率',
         '日中位收益率 = 所有日收益率的中位数',
         ['1w', '2w', '1m', '2m', '3m', '5m', '6m', '12m', '2y', '3y', '5y', 'mtd', 'qtd', 'ytd'],
         ],
    'Volatility':
        ['波动率',
         '波动率 = 收益率的标准差',
         ['1w', '2w', '1m', '2m', '3m', '5m', '6m', '12m', '2y', '3y', '5y', 'mtd', 'qtd', 'ytd'],
         ],
    'MeanAbsoluteDeviation':
        ['平均绝对偏差',
         '平均绝对偏差 = 平均每期收益率距离均值的绝对值, 比标准差更不受极端值影响',
         ['3d', '1w', '2w', '1m', '2m', '3m', '5m', '6m', '12m', '2y', '3y', '5y', 'mtd', 'qtd', 'ytd'],
         ],
    'ReturnRange':
        ['收益率范围',
         '收益率范围 = 收益率的最大值 - 收益率的最小值',
         ['3d', '1w', '2w', '1m', '2m', '3m', '5m', '6m', '12m', '2y', '3y', '5y', 'mtd', 'qtd', 'ytd'],
         ],
    'MaxGain':
        ['最大单日收益',
         '最大收益 = 收益率的最大值',
         ['3d', '1w', '2w', '1m', '2m', '3m', '5m', '6m', '12m', '2y', '3y', '5y', 'mtd', 'qtd', 'ytd'],
         ],
    'MaxLoss':
        ['最大单日亏损',
         '最大亏损 = 收益率的最小值',
         ['3d', '1w', '2w', '1m', '2m', '3m', '5m', '6m', '12m', '2y', '3y', '5y', 'mtd', 'qtd', 'ytd'],
         ],
    'AnnualizedVolatility':
        ['年化波动率',
         '年化波动率 = 波动率 * sqrt(risk_ann_factor)',
         ['1w', '2w', '1m', '2m', '3m', '5m', '6m', '12m', '2y', '3y', '5y', 'mtd', 'qtd', 'ytd'],
         ],
    'MaxDrawDown':  # 最大回撤用的是普通回撤率,而非对数回撤率
        ['最大回撤',
         '最大回撤 = (峰值 - 谷值) / 峰值',
         ['1w', '2w', '1m', '2m', '3m', '5m', '6m', '12m', '2y', '3y', '5y', 'mtd', 'qtd', 'ytd'],
         ],
    'MaxDrawDownDays':  # 最大回撤用的是普通回撤率,而非对数回撤率
        ['最大回撤天数',
         '最大回撤天数 = 回撤开始到谷底的交易日天数',
         ['1w', '2w', '1m', '2m', '3m', '5m', '6m', '12m', '2y', '3y', '5y', 'mtd', 'qtd', 'ytd'],
         ],
    # 'MaxDrawDownPaybackDays': # 该指标不适用于强化学习
    #     ['最大回撤恢复天数',
    #      '最大回撤恢复天数 = 从最大回撤底部恢复到历史最高点所需的天数',
    #      ['1w', '2w', '1m', '2m', '3m', '5m', '6m', '12m', '2y', '3y', '5y', 'mtd', 'qtd', 'ytd'],
    #      ],
    'ReturnDrawDownRatio':  # 最大回撤用的是普通回撤率,而非对数回撤率
        ['收益率回撤比',
         '收益率回撤比 = 总收益率 / 最大回撤',
         ['1w', '2w', '1m', '2m', '3m', '5m', '6m', '12m', '2y', '3y', '5y', 'mtd', 'qtd', 'ytd'],
         ],
    'AnnReturnDrawDownRatio':  # 最大回撤用的是普通回撤率,而非对数回撤率
        ['年化收益率回撤比, 即 卡尔马比率',
         '年化收益率回撤比/卡尔马比率 = 年化收益率 / 最大回撤',
         ['1w', '2w', '1m', '2m', '3m', '5m', '6m', '12m', '2y', '3y', '5y', 'mtd', 'qtd', 'ytd'],
         ],
    # 'RecoveryFactor': # 该指标不适用于强化学习
    #     ['恢复因子',
    #      '恢复因子 = 从最大回撤底部恢复回历史最高点所获得的收益 / 最大回撤',
    #      ['1w', '2w', '1m', '2m', '3m', '5m', '6m', '12m', '2y', '3y', '5y', 'mtd', 'qtd', 'ytd'],
    #      ],
    'DrawDownSlope':  # 最大回撤用的是普通回撤率,而非对数回撤率
        ['回撤斜率',
         '回撤斜率 = 最大回撤幅度 / 回撤所用时间',
         ['1w', '2w', '1m', '2m', '3m', '5m', '6m', '12m', '2y', '3y', '5y', 'mtd', 'qtd', 'ytd'],
         ],
    'UlcerIndex':  # 最大回撤用的是普通回撤率,而非对数回撤率
        ['溃疡指数 (衡量最大回撤深度和持续时间的指标)',
         '溃疡指数 = 所有 draw down 的均方根（RMS）',
         ['1m', '2m', '3m', '5m', '6m', '12m', '2y', '3y', '5y', 'mtd', 'qtd', 'ytd'],
         ],
    'SharpeRatio':
        ['夏普比率',
         '夏普比率 = (累计收益率 - 无风险收益) / 投资组合波动率',
         ['1w', '2w', '1m', '2m', '3m', '5m', '6m', '12m', '2y', '3y', '5y', 'mtd', 'qtd', 'ytd'],
         ],
    'AnnualizedSharpeRatio':
        ['年化夏普比率',
         '年化夏普比率 = 夏普比率 * sqrt(年交易天数)',
         ['1w', '2w', '1m', '2m', '3m', '5m', '6m', '12m', '2y', '3y', '5y', 'mtd', 'qtd', 'ytd'],
         ],
    'ReturnVolatilityRatio':
        ['收益率波动率比',
         '收益率波动率比 = 总收益率 / 波动率',
         ['1w', '2w', '1m', '2m', '3m', '5m', '6m', '12m', '2y', '3y', '5y', 'mtd', 'qtd', 'ytd'],
         ],
    'DownsideVolatility':
        ['下行波动率',
         '下行波动率 = 负收益率的标准差',
         ['1w', '2w', '1m', '2m', '3m', '5m', '6m', '12m', '2y', '3y', '5y', 'mtd', 'qtd', 'ytd'],
         ],
    'UpsideVolatility':
        ['上行波动率',
         '上行波动率 = 正收益率的标准差',
         ['1w', '2w', '1m', '2m', '3m', '5m', '6m', '12m', '2y', '3y', '5y', 'mtd', 'qtd', 'ytd'],
         ],
    'VolatilitySkew':
        ['波动率偏度',
         '波动率偏度 = (上行波动率 - 下行波动率) / 总波动率',
         ['1w', '2w', '1m', '2m', '3m', '5m', '6m', '12m', '2y', '3y', '5y', 'mtd', 'qtd', 'ytd'],
         ],
    'VolatilityRatio':
        ['波动率比',
         '波动率比 = 上行波动率 / 下行波动率',
         ['1w', '2w', '1m', '2m', '3m', '5m', '6m', '12m', '2y', '3y', '5y', 'mtd', 'qtd', 'ytd'],
         ],
    'SortinoRatio':
        ['索提诺比率',
         '索提诺比率 = (累计收益率 - 无风险收益率) / 下行波动率',
         ['1w', '2w', '1m', '2m', '3m', '5m', '6m', '12m', '2y', '3y', '5y', 'mtd', 'qtd', 'ytd'],
         ],
    'GainConsistency':
        ['收益趋势一致性',
         '收益趋势一致性 = 盈利日的标准差(上行波动率) / 盈利日的平均收益(平均正收益率)',
         ['1w', '2w', '1m', '2m', '3m', '5m', '6m', '12m', '2y', '3y', '5y', 'mtd', 'qtd', 'ytd'],
         ],
    'LossConsistency':
        ['亏损趋势一致性',
         '亏损趋势一致性 = 亏损日的标准差(下行波动率) / 亏损日的平均亏损(平均负收益率)',
         ['1w', '2w', '1m', '2m', '3m', '5m', '6m', '12m', '2y', '3y', '5y', 'mtd', 'qtd', 'ytd'],
         ],
    'WinningRatio':
        ['胜率',
         '胜率 = 盈利交易次数(不含0) / 总交易次数',
         ['1w', '2w', '1m', '2m', '3m', '5m', '6m', '12m', '2y', '3y', '5y', 'mtd', 'qtd', 'ytd'],
         ],
    'LosingRatio':
        ['亏损率',
         '亏损率 = 亏损交易次数(不含0) / 总交易次数',
         ['1w', '2w', '1m', '2m', '3m', '5m', '6m', '12m', '2y', '3y', '5y', 'mtd', 'qtd', 'ytd'],
         ],
    'ReturnSkewness':
        ['收益率偏度',
         '收益率偏度 = 收益率分布的偏度',
         ['1w', '2w', '1m', '2m', '3m', '5m', '6m', '12m', '2y', '3y', '5y', 'mtd', 'qtd', 'ytd'],
         ],
    'ReturnKurtosis':
        ['收益率峰度',
         '收益率峰度 = 收益率分布的峰度',
         ['1w', '2w', '1m', '2m', '3m', '5m', '6m', '12m', '2y', '3y', '5y', 'mtd', 'qtd', 'ytd'],
         ],
    # 'MaxConsecutiveWinsDays': # 不适合强化学习使用
    #     ['最长连续胜利天数',
    #      '最长连续胜利天数 = 连续盈利交易的最长天数',
    #      ['1w', '2w', '1m', '2m', '3m', '5m', '6m', '12m', '2y', '3y', '5y', 'mtd', 'qtd', 'ytd'],
    #      ],
    # 'MaxConsecutiveLossDays': # 不适合强化学习使用
    #     ['最长连续失败天数',
    #      '最长连续失败天数 = 连续亏损交易的最长天数',
    #      ['1w', '2w', '1m', '2m', '3m', '5m', '6m', '12m', '2y', '3y', '5y', 'mtd', 'qtd', 'ytd'],
    #      ],
    # 'MaxConsecutiveRate': # 不适合强化学习使用
    #     ['最长连续上涨下跌日比率',
    #      '最长连续上涨下跌日比率 = 最长连续胜利天数 / 最长连续失败天数',
    #      ['1w', '2w', '1m', '2m', '3m', '5m', '6m', '12m', '2y', '3y', '5y', 'mtd', 'qtd', 'ytd'],
    #      ],
    'VaR-99':
        ['99% VaR',
         '99% VaR = 在99%的置信水平下，投资组合可能的最大损失',
         ['2w', '1m', '2m', '3m', '5m', '6m', '12m', '2y', '3y', '5y', 'mtd', 'qtd', 'ytd'],
         ],
    'VaR-95':
        ['95% VaR',
         '95% VaR = 在95%的置信水平下，投资组合可能的最大损失',
         ['2w', '1m', '2m', '3m', '5m', '6m', '12m', '2y', '3y', '5y', 'mtd', 'qtd', 'ytd'],
         ],
    'VaR-90':
        ['90% VaR',
         '90% VaR = 在90%的置信水平下，投资组合可能的最大损失',
         ['2w', '1m', '2m', '3m', '5m', '6m', '12m', '2y', '3y', '5y', 'mtd', 'qtd', 'ytd'],
         ],
    'ModifiedVaR-99':
        ['99% 修正VaR',
         '99% VaR = 在传统正态 VaR 的基础上，使用 Cornish-Fisher 展开修正尾部概率',
         ['2w', '1m', '2m', '3m', '5m', '6m', '12m', '2y', '3y', '5y', 'mtd', 'qtd', 'ytd'],
         ],
    'ModifiedVaR-95':
        ['95% VaR',
         '95% VaR = 在传统正态 VaR 的基础上，使用 Cornish-Fisher 展开修正尾部概率',
         ['2w', '1m', '2m', '3m', '5m', '6m', '12m', '2y', '3y', '5y', 'mtd', 'qtd', 'ytd'],
         ],
    'ModifiedVaR-90':
        ['90% VaR',
         '90% VaR = 在传统正态 VaR 的基础上，使用 Cornish-Fisher 展开修正尾部概率',
         ['2w', '1m', '2m', '3m', '5m', '6m', '12m', '2y', '3y', '5y', 'mtd', 'qtd', 'ytd'],
         ],
    'VaRSharpe-95':
        ['基于 95% VaR 计算夏普比率',
         '基于 95% VaR 计算夏普比率 = (投资组合收益率 - 无风险收益率) / 95% VaR',
         ['2w', '1m', '2m', '3m', '5m', '6m', '12m', '2y', '3y', '5y', 'mtd', 'qtd', 'ytd'],
         ],
    'ModifiedVaRSharpe-95':
        ['基于 95% 修正VaR 计算夏普比率',
         '基于 95% 修正VaR 计算夏普比率 = (投资组合收益率 - 无风险收益率) / 95% 修正VaR',
         ['2w', '1m', '2m', '3m', '5m', '6m', '12m', '2y', '3y', '5y', 'mtd', 'qtd', 'ytd'],
         ],
    'CVaR-99':
        ['99% CVaR',
         '99% CVaR = 在99%的置信水平下，投资组合可能的平均损失',
         ['2w', '1m', '2m', '3m', '5m', '6m', '12m', '2y', '3y', '5y', 'mtd', 'qtd', 'ytd'],
         ],
    'CVaR-95':
        ['95% CVaR',
         '95% CVaR = 在95%的置信水平下，投资组合可能的平均损失',
         ['2w', '1m', '2m', '3m', '5m', '6m', '12m', '2y', '3y', '5y', 'mtd', 'qtd', 'ytd'],
         ],
    'CVaR-90':
        ['90% CVaR',
         '90% CVaR = 在90%的置信水平下，投资组合可能的平均损失',
         ['2w', '1m', '2m', '3m', '5m', '6m', '12m', '2y', '3y', '5y', 'mtd', 'qtd', 'ytd'],
         ],
    'ModifiedCVaR-99':
        ['99% 修正CVaR',
         '99% 修正CVaR = 在传统正态 CVaR 的基础上，使用 Cornish-Fisher 展开修正尾部概率',
         ['2w', '1m', '2m', '3m', '5m', '6m', '12m', '2y', '3y', '5y', 'mtd', 'qtd', 'ytd'],
         ],
    'ModifiedCVaR-95':
        ['95% 修正CVaR',
         '95% 修正CVaR = 在传统正态 CVaR 的基础上，使用 Cornish-Fisher 展开修正尾部概率',
         ['2w', '1m', '2m', '3m', '5m', '6m', '12m', '2y', '3y', '5y', 'mtd', 'qtd', 'ytd'],
         ],
    'ModifiedCVaR-90':
        ['90% 修正CVaR',
         '90% 修正CVaR = 在传统正态 CVaR 的基础上，使用 Cornish-Fisher 展开修正尾部概率',
         ['2w', '1m', '2m', '3m', '5m', '6m', '12m', '2y', '3y', '5y', 'mtd', 'qtd', 'ytd'],
         ],
    'CVaRSharpe-95':
        ['基于 95% CVaR 计算夏普比率',
         '基于 95% CVaR 计算夏普比率 = (投资组合收益率 - 无风险收益率) / 95% CVaR',
         ['2w', '1m', '2m', '3m', '5m', '6m', '12m', '2y', '3y', '5y', 'mtd', 'qtd', 'ytd'],
         ],
    'ModifiedCVaRSharpe-95':
        ['基于 95% 修正CVaR 计算夏普比率',
         '基于 95% 修正CVaR 计算夏普比率 = (投资组合收益率 - 无风险收益率) / 95% 修正CVaR',
         ['2w', '1m', '2m', '3m', '5m', '6m', '12m', '2y', '3y', '5y', 'mtd', 'qtd', 'ytd'],
         ],
    'PercentileWin-5':
        ['5%分位数的正收益率',
         '分位数收益率 = 收益率的分位数',
         ['1m', '2m', '3m', '5m', '6m', '12m', '2y', '3y', '5y', 'mtd', 'qtd', 'ytd'],
         ],
    'PercentileLoss-5':
        ['5%分位数的负收益率',
         '分位数收益率 = 收益率的分位数',
         ['1m', '2m', '3m', '5m', '6m', '12m', '2y', '3y', '5y', 'mtd', 'qtd', 'ytd'],
         ],
    'PercentileWin-10':
        ['10%分位数的正收益率',
         '分位数收益率 = 收益率的分位数',
         ['1m', '2m', '3m', '5m', '6m', '12m', '2y', '3y', '5y', 'mtd', 'qtd', 'ytd'],
         ],
    'PercentileLoss-10':
        ['10%分位数的负收益率',
         '分位数收益率 = 收益率的分位数',
         ['1m', '2m', '3m', '5m', '6m', '12m', '2y', '3y', '5y', 'mtd', 'qtd', 'ytd'],
         ],
    'TailRatio-10':
        ['尾部比率, 极端正收益与极端负收益的比值',
         '尾部比率 = 极端正收益(90分位) / 极端负收益(10分位)',
         ['1m', '2m', '3m', '5m', '6m', '12m', '2y', '3y', '5y', 'mtd', 'qtd', 'ytd'],
         ],
    'TailRatio-5':
        ['尾部比率, 极端正收益与极端负收益的比值',
         '尾部比率 = 极端正收益(95分位) / 极端负收益(5分位)',
         ['1m', '2m', '3m', '5m', '6m', '12m', '2y', '3y', '5y', 'mtd', 'qtd', 'ytd'],
         ],
    'TailRatio-1':
        ['尾部比率, 极端正收益与极端负收益的比值',
         '尾部比率 = 极端正收益(99分位) / 极端负收益(1分位)',
         ['1m', '2m', '3m', '5m', '6m', '12m', '2y', '3y', '5y', 'mtd', 'qtd', 'ytd'],
         ],
    'NewHighRatio':
        ['净值新高比率',
         '净值新高比率 = 净值创新高的交易日数 / 总交易日数',
         ['2w', '1m', '2m', '3m', '5m', '6m', '12m', '2y', '3y', '5y', 'mtd', 'qtd', 'ytd'],
         ],
    'CrossProductRatio':
        ['交叉乘积比率',
         '交叉乘积比率 = (WW * LL) / (WL * LW)',
         ['1m', '2m', '3m', '5m', '6m', '12m', '2y', '3y', '5y', 'mtd', 'qtd', 'ytd'],
         ],
    'HurstExponent':
        ['赫斯特指数',
         '赫斯特指数 = 对净值构造时间序列，进行 R/S 分析',
         ['2m', '3m', '5m', '6m', '12m', '2y', '3y', '5y', 'qtd', 'ytd'],
         ],
    'OmegaRatio':
        ['欧米茄比率',
         '欧米茄比率 = 更全面的夏普比率扩展版本（考虑全部分布）',
         ['1m', '2m', '3m', '5m', '6m', '12m', '2y', '3y', '5y', 'mtd', 'qtd', 'ytd'],
         ],
    'ReturnDistributionIntegral':
        ['收益分布积分',
         '收益分布积分 = Omega Ratio 的分子部分',
         ['1m', '2m', '3m', '5m', '6m', '12m', '2y', '3y', '5y', 'mtd', 'qtd', 'ytd'],
         ],
    'UpsidePotentialRatio':
        ['上行潜在比率 (反映赚多少钱 vs 承担的亏损风险)',
         '上行潜在比率 = 上行平均目标差 / 下行标准差',
         ['1m', '2m', '3m', '5m', '6m', '12m', '2y', '3y', '5y', 'mtd', 'qtd', 'ytd'],
         ],
    'KRatio':
        ['K比率 (绩效趋势比率), 衡量趋势的可复制性',
         'K比率 = 净值的 log 值拟合直线的 slope / 标准差',
         ['1m', '2m', '3m', '5m', '6m', '12m', '2y', '3y', '5y', 'mtd', 'qtd', 'ytd'],
         ],
    'MartinRatio':
        ['马丁比率 (绩效趋势比率), 衡量趋势的可复制性',
         '马丁比率 = 年化收益率 / 溃疡指数',
         ['1m', '2m', '3m', '5m', '6m', '12m', '2y', '3y', '5y', 'mtd', 'qtd', 'ytd'],
         ],
    'SortinoSkewness':
        ['索提诺偏度',
         '索提诺偏度 = 类似 Skewness，但只考虑负收益部分的偏度（衡量尾部亏损分布）',
         ['1m', '2m', '3m', '5m', '6m', '12m', '2y', '3y', '5y', 'mtd', 'qtd', 'ytd'],
         ],
    'NetEquitySlope':
        ['净值增长斜率 (代表了净值随时间的平均上升速度)',
         '净值增长斜率 = 净值的 log 值拟合直线的 slope',
         ['1m', '2m', '3m', '5m', '6m', '12m', '2y', '3y', '5y', 'mtd', 'qtd', 'ytd']
         ],
    'EquitySmoothness':
        ['净值平滑度 (衡量策略净值序列与其线性拟合值之间的拟合程度)',
         '净值平滑度 = 净值的 log 值拟合曲线的 R^2',
         ['1m', '2m', '3m', '5m', '6m', '12m', '2y', '3y', '5y', 'mtd', 'qtd', 'ytd']
         ],
}

# 历史相对指标 (这些指标会基于上面已经计算好的指标来进行计算)
log_return_relative_metrics_dict = {

    ''' 相对历史收益指标 '''
    'ReturnZScore_2m':
        ['当前平均收益相对近2个月收益的 Z 分数',
         '收益率Z分数 = (当前平均收益 - 近2个月平均收益) / 近2个月标准差',
         ['1d', '2d', '3d', '1w', '2w', '1m'],
         ],
    'ReturnZScore_6m':
        ['当前平均收益相对近半年收益的 Z 分数',
         '收益率Z分数 = (当前平均收益 - 近半年平均收益) / 近半年标准差',
         ['1d', '2d', '3d', '1w', '2w', '1m', '2m', '3m'],
         ],
    'ReturnZScore_1y':
        ['当前平均收益相对近1年收益的 Z 分数',
         '收益率Z分数 = (当前平均收益 - 近1年平均收益) / 近1年标准差',
         ['1d', '2d', '3d', '1w', '2w', '1m', '2m', '3m', '5m', '6m'],
         ],
    'ReturnZScore_2y':
        ['当前平均收益相对近2年收益的 Z 分数',
         '收益率Z分数 = (当前平均收益 - 近2年平均收益) / 近2年标准差',
         ['1d', '2d', '3d', '1w', '2w', '1m', '2m', '3m', '5m', '6m', '1y', 'mtd', 'qtd'],
         ],
    'ReturnPercentile_2m':
        ['当前收益率相对近2个月收益的分位数',
         '收益率分位数 = 当前收益率在近2个月的同类收益率中的分位数',
         ['1d', '2d', '3d', '1w', '2w', '1m'],
         ],
    'ReturnPercentile_6m':
        ['当前收益率相对近6个月收益的分位数',
         '收益率分位数 = 当前收益率在近2个月的同类收益率中的分位数',
         ['1d', '2d', '3d', '1w', '2w', '1m', '2m', '3m'],
         ],
    'ReturnPercentile_1y':
        ['当前收益率相对近1年收益的分位数',
         '收益率分位数 = 当前收益率在近1年的同类收益率中的分位数',
         ['1d', '2d', '3d', '1w', '2w', '1m', '2m', '3m', '5m', '6m'],
         ],
    'ReturnPercentile_2y':
        ['当前收益率相对近2年收益的分位数',
         '收益率分位数 = 当前收益率在近2年的同类收益率中的分位数',
         ['1d', '2d', '3d', '1w' '2w', '1m', '2m', '3m', '5m', '6m', '1y', 'mtd', 'qtd'],
         ],
    'AverageDeviationReturn_2m':
        ['相较近2月收益均线偏离率',
         '均线偏离率 = (当前收益率均值 - 近2月收益均值) / 近2月收益均值',
         ['1d', '2d', '3d', '1w', '2w', '1m'],
         ],
    'AverageDeviationReturn_6m':
        ['相较近6月收益均线偏离率',
         '均线偏离率 = (当前收益率均值 - 近6月收益均值) / 近6月收益均值',
         ['1d', '2d', '3d', '1w', '2w', '1m', '2m', '3m'],
         ],
    'AverageDeviationReturn_1y':
        ['相较近1年收益均线偏离率',
         '均线偏离率 = (当前收益率均值 - 近1年收益均值) / 近1年收益均值',
         ['1d', '2d', '3d', '1w', '2w', '1m', '2m', '3m', '5m', '6m'],
         ],
    'AverageDeviationReturn_2y':
        ['相较近2年收益均线偏离率',
         '均线偏离率 = (当前收益率均值 - 近2年收益均值) / 近2年收益均值',
         ['1d', '2d', '3d', '1w' '2w', '1m', '2m', '3m', '5m', '6m', '1y', 'mtd', 'qtd'],
         ],

    ''' 相对历史风险指标 '''
    'VolatilityRollingRatio_2m':
        ['当前波动率相较近2个月波动率的滚动比率',
         '波动率滚动比率 = 当前波动率 / 近2个月波动率 (值 > 1 表示近期强于长期，< 1 则反之)',
         ['1w', '2w', '1m'],
         ],
    'VolatilityRollingRatio_6m':
        ['当前波动率相较近6个月波动率的滚动比率',
         '波动率滚动比率 = 当前波动率 / 近6个月波动率 (值 > 1 表示近期强于长期，< 1 则反之)',
         ['1w', '2w', '1m', '2m', '3m'],
         ],
    'VolatilityRollingRatio_1y':
        ['当前波动率相较近1年波动率的滚动比率',
         '波动率滚动比率 = 当前波动率 / 近1年波动率 (值 > 1 表示近期强于长期，< 1 则反之)',
         ['1w', '2w', '1m', '2m', '3m', '5m', '6m'],
         ],
    'VolatilityRollingRatio_2y':
        ['当前波动率相较近2年波动率的滚动比率',
         '波动率滚动比率 = 当前波动率 / 近2年波动率 (值 > 1 表示近期强于长期，< 1 则反之)',
         ['1w', '2w', '1m', '2m', '3m', '5m', '6m', '1y', 'mtd', 'qtd'],
         ],
    'VolatilityPercentile_2m':
        ['当前波动率相对近2个月波动率的分位数',
         '波动率分位数 = 当前收益率在近2个月的同类波动率中的分位数',
         ['1d', '2d', '3d', '1w', '2w', '1m'],
         ],
    'VolatilityPercentile_6m':
        ['当前波动率相对近6个月波动率的分位数',
         '波动率分位数 = 当前收益率在近2个月的同类波动率中的分位数',
         ['1d', '2d', '3d', '1w', '2w', '1m', '2m', '3m'],
         ],
    'VolatilityPercentile_1y':
        ['当前波动率相对近1年收波动率分位数',
         '波动率分位数 = 当前收益率在近1年的同类收波动中的分位数',
         ['1d', '2d', '3d', '1w', '2w', '1m', '2m', '3m', '5m', '6m'],
         ],
    'VolatilityPercentile_2y':
        ['当前波动率相对近2年收波动率分位数',
         '波动率分位数 = 当前收益率在近2年的同类收波动中的分位数',
         ['1d', '2d', '3d', '1w' '2w', '1m', '2m', '3m', '5m', '6m', '1y', 'mtd', 'qtd'],
         ],
}


# 创建一个按周期分组的指标映射表
def create_period_metrics_map():
    """
    创建一个按周期分组的指标映射表。

    该函数遍历 `log_return_metrics_dict` 中的每个指标，根据指标的周期将其分组，
    最终返回一个字典，其中键为周期，值为该周期下的所有指标键列表。

    返回值:
        dict: 一个字典，键为周期，值为该周期下的所有指标键列表。
    """
    # 初始化一个字典，默认值为列表
    period_metrics_map = defaultdict(list)

    # 遍历 log_return_metrics_dict 中的每个指标
    for metric_key, (name, formula, periods) in log_return_metrics_dict.items():
        # 遍历该指标的所有周期，并将指标键添加到对应周期的列表中
        for period in periods:
            period_metrics_map[period].append(metric_key)

    # 转换为普通字典并返回
    return dict(period_metrics_map)
