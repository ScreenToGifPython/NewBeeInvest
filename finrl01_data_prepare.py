import yfinance as yf
import pandas as pd
import itertools

from finrl.meta.preprocessor.yahoodownloader import YahooDownloader
from finrl.meta.preprocessor.preprocessors import FeatureEngineer, data_split
from finrl import config_tickers
from finrl.config import INDICATORS

pd.set_option('display.max_columns', 1000)  # 显示字段的数量
pd.set_option('display.width', 1000)  # 表格不分段显示


def get_prepared_data(TRAIN_START_DATE='2009-01-01',
                      TRAIN_END_DATE='2015-01-01',
                      TRADE_START_DATE='2016-01-01',
                      TRADE_END_DATE='2018-01-01'):
    """
In FinRL's YahooDownloader, we modified the data frame to the form that convenient for further data processing process.
We use adjusted close price instead of close price, and add a column representing the day of a week (0-4 corresponding to Monday-Friday).
    """
    df_raw = YahooDownloader(start_date=TRAIN_START_DATE,
                             end_date=TRADE_END_DATE,
                             ticker_list=config_tickers.DOW_30_TICKER).fetch_data()

    df_raw.to_parquet('data/yahoo_data.parquet')
    # df_raw = pd.read_parquet('data/yahoo_data.parquet')
    # print(df_raw.head())
    # print(df_raw.tail())

    '''
    We need to check for missing data and do feature engineering to convert the data point into a state.
    - Adding technical indicators. 
      In practical trading, various information needs to be taken into account, such as historical prices, current holding 
      shares, technical indicators, etc. Here, we demonstrate two trend-following technical indicators: MACD and RSI. 
      Moving average convergence/divergence (MACD) is one of the most commonly used indicator showing bull and bear market. 
      Its calculation is based on EMA (Exponential Moving Average indicator, measuring trend direction over a period of 
      time.)
    - Adding turbulence index. 
      Risk-aversion reflects whether an investor prefers to protect the capital. It also influences one's trading strategy 
      when facing different market volatility level. To control the risk in a worst-case scenario, such as financial crisis 
      of 2007–2008, we may consider the turbulence index that measures extreme fluctuation of asset price.
    '''

    # 初始化FeatureEngineer对象，配置特征工程参数
    fe = FeatureEngineer(use_technical_indicator=True,  # 使用技术指标
                         tech_indicator_list=INDICATORS,  # 技术指标列表
                         use_vix=True,  # 使用VIX指数
                         use_turbulence=True,  # 使用动荡指数
                         user_defined_feature=False  # 不使用用户自定义特征
                         )

    # 调用preprocess_data方法对原始数据进行预处理
    processed = fe.preprocess_data(df_raw)

    # 获取处理后数据中的所有股票代码并生成列表
    list_ticker = processed["tic"].unique().tolist()
    # 生成从数据最早日期到最晚日期的所有日期列表
    list_date = list(pd.date_range(processed['date'].min(), processed['date'].max()).astype(str))
    # 生成所有日期和股票代码的笛卡尔积，确保每个日期都有所有股票的数据
    combination = list(itertools.product(list_date, list_ticker))

    # 创建一个包含所有日期和股票代码组合的DataFrame，并与处理后的数据合并, 以左连接的方式合并，确保保留所有日期和股票代码的组合
    processed_full = pd.DataFrame(combination, columns=["date", "tic"]).merge(processed, on=["date", "tic"], how="left")
    # 筛选出原始数据中存在的日期，去除因笛卡尔积引入的额外日期
    processed_full = processed_full[processed_full['date'].isin(processed['date'])]
    # 按日期和股票代码排序，为后续处理做准备
    processed_full = processed_full.sort_values(['date', 'tic'])
    # 将缺失值填充为0，假设缺失数据对股票价格无影响
    processed_full = processed_full.fillna(0)

    # Split the data for training and trading
    train = data_split(processed_full, TRAIN_START_DATE, TRAIN_END_DATE)
    trade = data_split(processed_full, TRADE_START_DATE, TRADE_END_DATE)
    # Save data to csv file
    train.to_csv('train_data.csv')
    trade.to_csv('trade_data.csv')


if __name__ == '__main__':
    train_data = pd.read_csv('train_data.csv')
    train_data = train_data[train_data['tic'] == 'AAPL']
    train_data = train_data.set_index(train_data.columns[0]).reset_index(drop=True)
    print(train_data.head())
