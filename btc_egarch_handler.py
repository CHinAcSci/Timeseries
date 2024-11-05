import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from arch import arch_model
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from statsmodels.graphics.tsaplots import plot_acf
from typing import Tuple, Dict, Optional
from dataclasses import dataclass
from tqdm import tqdm
from btc_lstm_handler import BTCLSTMHandler


class BTCEGARCHHandler:
    """EGARCH模型处理器"""
    def __init__(self):
        """初始化EGARCH处理器"""
        self.model = None
        self.results = None
        self.residuals = None
        self.std_residuals = None
        self.conditional_volatility = None

    def prepare_data(self, df: pd.DataFrame) -> pd.Series:
        """准备数据，计算收益率"""
        # 使用收盘价计算对数收益率
        returns = 100 * np.log(df['Close'] / df['Close'].shift(1))
        return returns.dropna()

    def fit_egarch(self, returns: pd.Series, p: int = 1, o: int = 1, q: int = 1) -> Dict:
        """拟合EGARCH模型

        Args:
            returns: 收益率序列
            p: ARCH项阶数
            o: 杠杆效应项阶数
            q: GARCH项阶数

        Returns:
            Dict: 包含模型结果的字典
        """
        # 设置EGARCH模型
        self.model = arch_model(
            returns,
            vol='EGARCH', p=p, o=o, q=q,
            dist='t'  # 使用t分布以更好地拟合尾部
        )

        # 拟合模型
        self.results = self.model.fit(disp='off')

        # 获取标准化残差和条件波动率
        self.residuals = self.results.resid
        self.std_residuals = self.results.std_resid
        self.conditional_volatility = self.results.conditional_volatility

        # 返回模型结果摘要
        return {
            'params': self.results.params,
            'aic': self.results.aic,
            'bic': self.results.bic,
            'log_likelihood': self.results.loglikelihood,
        }

    def diagnostic_tests(self) -> Dict:
        """进行模型诊断检验"""
        # Ljung-Box检验（自相关性）
        lb_test = sm.stats.diagnostic.acorr_ljungbox(
            self.std_residuals,
            lags=[10, 15, 20],
            return_df=True
        )

        # Jarque-Bera正态性检验
        jb_stat, jb_pval = stats.jarque_bera(self.std_residuals)

        # ARCH效应检验
        arch_test = sm.stats.diagnostic.het_arch(
            self.std_residuals,
            nlags=12
        )

        return {
            'ljung_box': lb_test,
            'jarque_bera': {'statistic': jb_stat, 'pvalue': jb_pval},
            'arch_effect': {'statistic': arch_test[0], 'pvalue': arch_test[1]}
        }

    def plot_diagnostics(self, save_path: str = 'EGARCH模型诊断.png'):
        """绘制诊断图"""
        # 设置中文字体
        plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
        plt.rcParams['axes.unicode_minus'] = False    # 用来正常显示负号

        fig = plt.figure(figsize=(15, 12))

        # 1. 标准化残差时间序列图
        ax1 = plt.subplot(321)
        plt.plot(self.std_residuals)
        plt.title('标准化残差')

        # 2. 条件波动率
        ax2 = plt.subplot(322)
        plt.plot(self.conditional_volatility)
        plt.title('条件波动率')

        # 3. Q-Q图
        ax3 = plt.subplot(323)
        stats.probplot(self.std_residuals, dist="norm", plot=plt)
        plt.title('Q-Q图')

        # 4. 残差自相关图
        ax4 = plt.subplot(324)
        plot_acf(self.std_residuals, lags=40, ax=ax4)
        plt.title('标准化残差的自相关函数')

        # 5. 标准化残差直方图
        ax5 = plt.subplot(325)
        sns.histplot(self.std_residuals, stat='density', alpha=0.5)
        x = np.linspace(-4, 4, 100)
        plt.plot(x, stats.norm.pdf(x, 0, 1), 'r-', lw=2)
        plt.title('标准化残差直方图')

        # 6. 波动率聚类效应
        ax6 = plt.subplot(326)
        plt.plot(np.abs(self.std_residuals))
        plt.title('绝对标准化残差')

        plt.tight_layout()

        plt.savefig('img/EGARCH诊断图.png')
        plt.show()

    def forecast(self, horizon: int = 30) -> Tuple[np.ndarray, np.ndarray]:
        """使用Monte Carlo模拟预测未来波动率

        Args:
            horizon: 预测期数

        Returns:
            Tuple[np.ndarray, np.ndarray]: (均值预测, 波动率预测)
        """
        # 设置模拟次数
        n_sims = 1000

        # 进行模拟预测
        sim_results = []

        # 获取最新的波动率（使用iloc避免警告）
        last_vol = self.conditional_volatility.iloc[-1]

        for _ in range(n_sims):
            # 模拟未来horizon天的波动率
            sim_vols = [last_vol]
            for _ in range(horizon-1):
                # 使用EGARCH方程生成下一期波动率
                omega = float(self.results.params['omega'])
                alpha = float(self.results.params['alpha[1]'])
                gamma = float(self.results.params['gamma[1]'])
                beta = float(self.results.params['beta[1]'])

                z = np.random.standard_t(df=float(self.results.params['nu']))
                log_var = omega + alpha * (abs(z) - np.sqrt(2/np.pi)) + \
                      gamma * z + beta * np.log(sim_vols[-1]**2)
                next_vol = np.sqrt(np.exp(log_var))
                sim_vols.append(next_vol)
            sim_results.append(sim_vols)

        # 计算均值和标准差，确保返回numpy数组
        sim_results = np.array(sim_results)
        mean_forecast = np.mean(sim_results, axis=0)
        std_forecast = np.std(sim_results, axis=0)

        # 确保预测结果是正确的形状
        mean_forecast = np.array(mean_forecast).reshape(-1, 1)
        std_forecast = np.array(std_forecast).reshape(-1, 1)

        return mean_forecast, std_forecast

    def plot_forecast(self, returns: pd.Series, mean_forecast: pd.Series,
                      volatility_forecast: pd.Series, save_path: str = None):
        """绘制预测结果"""
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

        # 绘制收益率和预测区间
        ax1.plot(returns.index[-100:], returns[-100:], label='Actual Returns')
        ax1.plot(mean_forecast.index, mean_forecast, 'r--', label='Forecasted Mean')

        # 添加预测区间
        confidence_intervals = np.array([0.95])
        for ci in confidence_intervals:
            z_score = stats.norm.ppf((1 + ci) / 2)
            upper = mean_forecast + z_score * volatility_forecast
            lower = mean_forecast - z_score * volatility_forecast
            ax1.fill_between(mean_forecast.index, lower, upper, alpha=0.2,
                             label=f'{int(ci*100)}% Confidence Interval')

        ax1.set_title('Returns Forecast')
        ax1.legend()

        # 绘制波动率预测
        ax2.plot(returns.index[-100:],
                 self.conditional_volatility[-100:],
                 label='Historical Volatility')
        ax2.plot(volatility_forecast.index,
                 volatility_forecast,
                 'r--',
                 label='Forecasted Volatility')
        ax2.set_title('Volatility Forecast')
        ax2.legend()

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path)
            plt.close()
        else:
            plt.show()

@dataclass
class HybridPrediction:
    """混合预测结果数据类"""
    mean: np.ndarray
    volatility: np.ndarray
    lower_bound: np.ndarray
    upper_bound: np.ndarray
    attention_weights: Optional[np.ndarray] = None

class BTCHybridHandler:
    """LSTM与EGARCH混合模型处理器"""
    def __init__(self, lstm_handler, sequence_length=60):
        self.lstm_handler = lstm_handler
        self.sequence_length = sequence_length
        self.egarch_handler = BTCEGARCHHandler()

    def prepare_hybrid_data(self, df: pd.DataFrame) -> Tuple[pd.Series, pd.DataFrame]:
        """准备混合模型所需数据"""
        # 计算对数收益率用于EGARCH
        returns = self.egarch_handler.prepare_data(df)

        # 准备LSTM数据
        train_loader, test_loader, test_data = self.lstm_handler.prepare_data(df)

        return returns, test_data

    def fit_models(self, df: pd.DataFrame, train_loader: torch.utils.data.DataLoader) -> Dict:
        """拟合LSTM和EGARCH模型"""
        # 训练LSTM
        print("训练LSTM模型...")
        lstm_losses = self.lstm_handler.train(train_loader)

        # 准备并拟合EGARCH
        print("\n拟合EGARCH模型...")
        returns = self.egarch_handler.prepare_data(df)
        egarch_results = self.egarch_handler.fit_egarch(returns)

        return {
            'lstm_losses': lstm_losses,
            'egarch_results': egarch_results
        }

    def predict_hybrid(self, df: pd.DataFrame, days: int = 30,
                       confidence_level: float = 0.95) -> HybridPrediction:
        """混合模型预测"""
        # LSTM预测
        future_dates, lstm_predictions, attention_weights = \
            self.lstm_handler.predict_future(df, days)

        # EGARCH预测
        mean_forecast, volatility_forecast = self.egarch_handler.forecast(days)

        # 确保形状匹配
        pred_std = volatility_forecast
        if len(pred_std.shape) == 1:
            pred_std = pred_std.reshape(-1, 1)

        # 计算置信区间
        z_score = stats.norm.ppf((1 + confidence_level) / 2)

        lower_bound = lstm_predictions - z_score * pred_std
        upper_bound = lstm_predictions + z_score * pred_std

        return HybridPrediction(
            mean=lstm_predictions,
            volatility=volatility_forecast,
            lower_bound=lower_bound,
            upper_bound=upper_bound,
            attention_weights=attention_weights
        )

    def evaluate_hybrid_model(self, test_loader: torch.utils.data.DataLoader,
                              test_data: pd.DataFrame) -> Dict:
        """评估混合模型"""
        # LSTM预测评估
        predictions, actuals, attention_weights, lstm_metrics = \
            self.lstm_handler.evaluate(test_loader)

        # EGARCH评估
        egarch_diagnostics = self.egarch_handler.diagnostic_tests()

        # 合并评估指标
        metrics = {
            'lstm_metrics': lstm_metrics,
            'egarch_diagnostics': egarch_diagnostics
        }

        return metrics

def train_and_evaluate_hybrid_model(df: pd.DataFrame, sequence_length: int = 60):
    """训练并评估混合模型"""
    # 初始化LSTM处理器
    lstm_handler = BTCLSTMHandler(sequence_length=sequence_length)

    # 初始化混合模型处理器
    hybrid_handler = BTCHybridHandler(lstm_handler, sequence_length)

    # 准备数据
    train_loader, test_loader, test_data = lstm_handler.prepare_data(df)
    returns, _ = hybrid_handler.prepare_hybrid_data(df)

    # 训练模型
    training_results = hybrid_handler.fit_models(df, train_loader)

    # 评估模型
    evaluation_results = hybrid_handler.evaluate_hybrid_model(test_loader, test_data)

    # 预测未来30天
    predictions = hybrid_handler.predict_hybrid(df, days=30)


    # 绘制EGARCH诊断图
    hybrid_handler.egarch_handler.plot_diagnostics()

    return hybrid_handler, evaluation_results, predictions