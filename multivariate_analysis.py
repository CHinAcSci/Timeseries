import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import adfuller, coint
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.vector_ar.vecm import coint_johansen
from statsmodels.stats.diagnostic import acorr_ljungbox
import matplotlib.pyplot as plt
# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']  # 设置中文字体
plt.rcParams['axes.unicode_minus'] = False    # 解决负号显示问题
import seaborn as sns

class BTCMultivariateAnalysis:
    def __init__(self, df, events):
        """
        初始化多元时间序列分析类

        参数:
        df: 包含'Close'和'Volume'的DataFrame
        events: 重大事件字典
        """
        self.df = df.copy()
        self.events = events
        self.df_processed = None
        self.intervention_vars = None

    def preprocess_data(self):
        """数据预处理"""
        # 1. 对数变换
        self.df_processed = self.df.copy()
        self.df_processed['log_price'] = np.log(self.df['Close'])
        self.df_processed['log_volume'] = np.log(self.df['Volume'])

        # 2. 创建干预变量
        self._create_intervention_variables()

        # 3. 检查并处理缺失值
        self._handle_missing_values()

        return self.df_processed

    def _create_intervention_variables(self):
        """创建干预变量"""
        for date, (event_name, _) in self.events.items():
            # 脉冲效应
            self.df_processed[f'pulse_{event_name}'] = (
                    self.df_processed.index == date).astype(int)

            # 阶跃效应
            self.df_processed[f'step_{event_name}'] = (
                    self.df_processed.index >= date).astype(int)

            # 衰减效应（指数衰减）
            decay_rate = 0.95
            days_since = (self.df_processed.index - pd.to_datetime(date)).days
            self.df_processed[f'decay_{event_name}'] = np.where(
                days_since >= 0,
                decay_rate ** days_since,
                0
            )

    def _handle_missing_values(self):
        """处理缺失值"""
        # 检查缺失值
        missing_values = self.df_processed.isnull().sum()
        if missing_values.any():
            print("发现缺失值:\n", missing_values[missing_values > 0])
            # 使用前向填充处理缺失值
            self.df_processed.fillna(method='ffill', inplace=True)

    def check_stationarity(self):
        """检查平稳性"""
        results = {}
        for column in ['log_price', 'log_volume']:
            adf_result = adfuller(self.df_processed[column])
            results[column] = {
                'adf_statistic': adf_result[0],
                'p_value': adf_result[1],
                'critical_values': adf_result[4]
            }

        return results

    def plot_initial_analysis(self):
        """绘制初始分析图"""
        fig, axes = plt.subplots(3, 1, figsize=(15, 15))

        # 价格和成交量的时间序列图
        axes[0].plot(self.df_processed.index, self.df_processed['Close'])
        axes[0].set_title('BTC价格时间序列')
        axes[0].set_ylabel('价格 (USD)')

        axes[1].plot(self.df_processed.index, self.df_processed['Volume'])
        axes[1].set_title('BTC成交量时间序列')
        axes[1].set_ylabel('成交量')

        # 标记重大事件
        for date, (event_name, desc) in self.events.items():
            for ax in axes[:2]:
                ax.axvline(x=pd.to_datetime(date), color='r', alpha=0.3)

        # 价格-成交量散点图
        axes[2].scatter(self.df_processed['log_volume'],
                        self.df_processed['log_price'],
                        alpha=0.5)
        axes[2].set_title('价格-成交量散点图（对数尺度）')
        axes[2].set_xlabel('log(成交量)')
        axes[2].set_ylabel('log(价格)')

        plt.tight_layout()
        plt.show()