import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import adfuller, coint
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.vector_ar.vecm import coint_johansen
from statsmodels.stats.diagnostic import acorr_ljungbox
import statsmodels.api as sm
from statsmodels.tsa.stattools import grangercausalitytests
from itertools import product
from scipy import stats
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
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

    def preprocess_data(self):
        """数据预处理"""
        self.df_processed = self.df.copy()

        # 1. 对价格和交易量取对数前检查是否有0或负值
        if (self.df_processed['Close'] <= 0).any() or (self.df_processed['Volume'] <= 0).any():
            print("警告：数据中存在0或负值，将被移除")
            self.df_processed = self.df_processed[
                (self.df_processed['Close'] > 0) &
                (self.df_processed['Volume'] > 0)
                ]

        # 2. 取对数
        self.df_processed['log_price'] = np.log(self.df_processed['Close'])
        self.df_processed['log_volume'] = np.log(self.df_processed['Volume'])

        # 3. 计算差分
        self.df_processed['d_log_price'] = self.df_processed['log_price'].diff()
        self.df_processed['d_log_volume'] = self.df_processed['log_volume'].diff()

        # 4. 创建干预变量
        self._create_intervention_variables()

        # 5. 处理缺失值
        self._handle_missing_values()

        # 6. 处理所有NaN和inf值
        self.df_processed = self.df_processed.replace([np.inf, -np.inf], np.nan)
        self.df_processed = self.df_processed.dropna()

        return self.df_processed

    def _create_intervention_variables(self):
        """创建干预变量"""
        for date, (event_name, _) in self.events.items():
            event_date = pd.to_datetime(date)

            # 脉冲效应
            self.df_processed[f'pulse_{event_name}'] = (
                    self.df_processed.index == event_date).astype(float)

            # 阶跃效应
            self.df_processed[f'step_{event_name}'] = (
                    self.df_processed.index >= event_date).astype(float)

            # 衰减效应
            days_since = (self.df_processed.index - event_date).days
            max_days = 30  # 30天衰减期
            self.df_processed[f'decay_{event_name}'] = np.where(
                days_since >= 0,
                np.maximum(0, 1 - days_since/max_days),
                0
            ).astype(float)

    def _handle_missing_values(self):
        """处理缺失值"""
        missing_values = self.df_processed.isnull().sum()
        if missing_values.any():
            print("发现缺失值:\n", missing_values[missing_values > 0])
            self.df_processed.fillna(method='ffill', inplace=True)

    def check_stationarity(self):
        """检查平稳性"""
        results = {}
        for column in ['Close', 'Volume']:
            adf_result = adfuller(self.df_processed[column])
            results[column] = {
                'adf_statistic': adf_result[0],
                'p_value': adf_result[1],
                'critical_values': adf_result[4]
            }
        return results

    def cointegration_test(self):
        """Johansen协整检验"""
        try:
            # 准备数据
            data = pd.DataFrame({
                'price': self.df_processed['log_price'],
                'volume': self.df_processed['log_volume']
            }).dropna()

            # 执行Johansen检验
            result = coint_johansen(data.values, det_order=0, k_ar_diff=1)

            # 整理结果
            coint_results = {
                'trace_stat': result.lr1,
                'crit_vals': result.cvt,
                'eigenvalues': result.eig
            }

            return coint_results
        except Exception as e:
            print(f"协整检验出现错误: {str(e)}")
            return None

    def granger_causality_test(self, maxlag=12):
        """Granger因果检验"""
        try:
            # 准备数据
            data = pd.DataFrame({
                'price': self.df_processed['d_log_price'].dropna(),
                'volume': self.df_processed['d_log_volume'].dropna()
            })

            # 价格对交易量的因果检验
            price_to_volume = grangercausalitytests(
                data[['volume', 'price']],
                maxlag=maxlag,
                verbose=False
            )

            # 交易量对价格的因果检验
            volume_to_price = grangercausalitytests(
                data[['price', 'volume']],
                maxlag=maxlag,
                verbose=False
            )

            # 整理结果
            causality_results = {
                'price_to_volume': {lag+1: test[0]['ssr_chi2test'][1]
                                    for lag, test in price_to_volume.items()},
                'volume_to_price': {lag+1: test[0]['ssr_chi2test'][1]
                                    for lag, test in volume_to_price.items()}
            }

            return causality_results
        except Exception as e:
            print(f"Granger因果检验出现错误: {str(e)}")
            return None

    def estimate_ecm(self):
        """估计误差修正模型"""
        try:
            # 准备数据
            log_price = self.df_processed['log_price']
            log_volume = self.df_processed['log_volume']

            # 估计协整方程
            coint_eq = sm.OLS(log_price, sm.add_constant(log_volume)).fit()

            # 计算误差修正项
            ect = log_price - (coint_eq.params[0] + coint_eq.params[1] * log_volume)

            # 构建ECM
            X = sm.add_constant(np.column_stack([
                ect[:-1],  # 误差修正项
                self.df_processed['d_log_price'][:-1],  # 价格差分滞后1期
                self.df_processed['d_log_volume'][:-1]  # 交易量差分滞后1期
            ]))

            # 估计ECM
            ecm = sm.OLS(self.df_processed['d_log_price'][1:], X).fit()

            return {
                'coint_eq': coint_eq,
                'ecm': ecm
            }
        except Exception as e:
            print(f"ECM估计出现错误: {str(e)}")
            return None

    def intervention_analysis_with_arimax(self):
        """使用ARIMAX进行干预分析"""
        try:
            # 1. 准备数据
            endog = self.df_processed['Close']
            exog_base = self.df_processed['Volume']

            # 获取所有干预变量
            intervention_cols = [col for col in self.df_processed.columns
                                 if col.startswith(('pulse_', 'step_', 'decay_'))]
            intervention_vars = self.df_processed[intervention_cols]

            # 合并所有外生变量
            exog = pd.concat([exog_base, intervention_vars], axis=1)

            # 2. 选择最优模型阶数
            best_order = self._select_best_order(endog, exog)

            # 3. 拟合ARIMAX模型
            model = SARIMAX(
                endog=endog,
                exog=exog,
                order=best_order,
                enforce_stationarity=False,
                enforce_invertibility=False
            )
            results = model.fit(disp=False)

            # 4. 分析干预效果
            intervention_effects = self._analyze_intervention_effects(results)

            # 5. 显著性检验
            intervention_tests = self._test_intervention_significance(
                endog, exog_base, intervention_vars, best_order
            )

            return {
                'model': results,
                'intervention_effects': intervention_effects,
                'intervention_tests': intervention_tests
            }
        except Exception as e:
            print(f"ARIMAX分析出现错误: {str(e)}")
            return None

    def _select_best_order(self, endog, exog, max_order=3):
        """选择最优的ARIMAX模型阶数"""
        best_aic = np.inf
        best_order = None

        for p, d, q in product(range(max_order), [0, 1], range(max_order)):
            try:
                model = SARIMAX(
                    endog=endog,
                    exog=exog,
                    order=(p, d, q)
                )
                results = model.fit(disp=False)
                if results.aic < best_aic:
                    best_aic = results.aic
                    best_order = (p, d, q)
            except:
                continue

        return best_order or (1,1,1)  # 如果没有找到最优阶数，返回默认值

    def _analyze_intervention_effects(self, model_results):
        """分析每个干预变量的效果"""
        effects = {}
        params = model_results.params
        conf = model_results.conf_int()

        for col in self.df_processed.columns:
            if col.startswith(('pulse_', 'step_', 'decay_')):
                if col in params:
                    effects[col] = {
                        'coefficient': params[col],
                        'p_value': model_results.pvalues[col],
                        'conf_int_lower': conf.loc[col, 0],
                        'conf_int_upper': conf.loc[col, 1]
                    }

        return effects

    def _test_intervention_significance(self, endog, exog_base, intervention_vars, order):
        """测试干预效果的显著性"""
        try:
            # 完整模型
            full_model = SARIMAX(
                endog=endog,
                exog=pd.concat([exog_base, intervention_vars], axis=1),
                order=order
            ).fit(disp=False)

            # 受限模型
            restricted_model = SARIMAX(
                endog=endog,
                exog=exog_base,
                order=order
            ).fit(disp=False)

            # 似然比检验
            lr_stat = -2 * (restricted_model.llf - full_model.llf)
            df = len(intervention_vars.columns)
            p_value = 1 - stats.chi2.cdf(lr_stat, df)

            return {
                'lr_statistic': lr_stat,
                'p_value': p_value,
                'df': df
            }
        except Exception as e:
            print(f"干预显著性检验出现错误: {str(e)}")
            return None

    def plot_intervention_effects(self, results):
        """可视化干预效果"""
        if results is None:
            print("无法绘制图形：分析结果为空")
            return None

        fig, axes = plt.subplots(2, 1, figsize=(15, 10))

        # 1. 原始序列和拟合值对比
        actual = self.df_processed['Close']
        fitted = results['model'].fittedvalues

        axes[0].plot(actual.index, actual, label='实际值')
        axes[0].plot(fitted.index, fitted, label='拟合值', alpha=0.7)
        axes[0].set_title('价格序列与拟合值对比')
        axes[0].legend()
        axes[0].ticklabel_format(style='plain', axis='y')

        # 2. 干预效果柱状图
        effects = results['intervention_effects']
        event_names = [name.split('_', 1)[1] for name in effects.keys()]
        coefficients = [effect['coefficient'] for effect in effects.values()]

        axes[1].bar(event_names, coefficients)
        axes[1].set_xticklabels(event_names, rotation=45, ha='right')
        axes[1].set_title('各干预事件的效果系数')
        axes[1].ticklabel_format(style='plain', axis='y')

        plt.tight_layout()
        plt.show()

        return fig

    def plot_initial_analysis(self):
        """绘制初始分析图"""
        fig, axes = plt.subplots(3, 1, figsize=(15, 15))

        # 价格时间序列
        axes[0].plot(self.df_processed.index, self.df_processed['Close'])
        axes[0].set_title('BTC价格时间序列')
        axes[0].set_ylabel('价格 (USD)')
        axes[0].ticklabel_format(style='plain', axis='y')

        # 成交量时间序列
        axes[1].plot(self.df_processed.index, self.df_processed['Volume'])
        axes[1].set_title('BTC成交量时间序列')
        axes[1].set_ylabel('成交量')
        axes[1].ticklabel_format(style='plain', axis='y')

        # 标记重大事件
        max_price = self.df_processed['Close'].max()
        for date, (event_name, _) in self.events.items():
            event_date = pd.to_datetime(date)
            event_date = pd.to_datetime(date)
            # 在价格图上标记
            axes[0].axvline(x=event_date, color='r', alpha=0.3)
            axes[0].text(event_date, max_price, event_name,
                         rotation=45, ha='right', fontsize=8)

            # 在成交量图上标记
            axes[1].axvline(x=event_date, color='r', alpha=0.3)

        # 价格-成交量散点图
        axes[2].scatter(self.df_processed['Volume'],
                        self.df_processed['Close'],
                        alpha=0.5)
        axes[2].set_title('价格-成交量散点图')
        axes[2].set_xlabel('成交量')
        axes[2].set_ylabel('价格 (USD)')
        axes[2].ticklabel_format(style='plain', axis='both')

        plt.tight_layout()
        plt.show()