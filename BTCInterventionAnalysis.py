from itertools import product

import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX


class BTCInterventionAnalysis:
    def __init__(self, df, events):
        self.df = df.copy()
        self.events = events
        self.df_processed = None
        self.best_model = None

    def intervention_analysis_with_arimax(self):
        """使用ARIMAX进行干预分析"""
        # 1. 准备数据
        # 分离内生变量、外生变量和干预变量
        endog = self.df_processed['Close']
        exog_base = self.df_processed['Volume']  # 基础外生变量

        # 获取所有干预变量
        intervention_cols = [col for col in self.df_processed.columns
                             if col.startswith(('pulse_', 'step_', 'decay_'))]
        intervention_vars = self.df_processed[intervention_cols]

        # 合并所有外生变量
        exog = pd.concat([exog_base, intervention_vars], axis=1)

        # 2. 模型选择
        best_order = self._select_best_order(endog, exog)

        # 3. 拟合完整ARIMAX模型
        model = SARIMAX(
            endog=endog,
            exog=exog,
            order=best_order
        )
        results = model.fit()

        # 4. 分析每个干预变量的效果
        intervention_effects = self._analyze_intervention_effects(results)

        # 5. 进行干预效果检验
        intervention_tests = self._test_intervention_significance(
            endog, exog_base, intervention_vars, best_order
        )

        return {
            'model': results,
            'intervention_effects': intervention_effects,
            'intervention_tests': intervention_tests
        }

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
                results = model.fit()
                if results.aic < best_aic:
                    best_aic = results.aic
                    best_order = (p, d, q)
            except:
                continue

        return best_order

    def _analyze_intervention_effects(self, model_results):
        """分析每个干预变量的效果"""
        effects = {}
        params = model_results.params
        conf = model_results.conf_int()

        # 提取干预变量的系数和置信区间
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
        # 1. 拟合包含干预的完整模型
        full_model = SARIMAX(
            endog=endog,
            exog=pd.concat([exog_base, intervention_vars], axis=1),
            order=order
        ).fit()

        # 2. 拟合仅包含基础外生变量的受限模型
        restricted_model = SARIMAX(
            endog=endog,
            exog=exog_base,
            order=order
        ).fit()

        # 3. 进行似然比检验
        lr_stat = -2 * (restricted_model.llf - full_model.llf)
        df = len(intervention_vars.columns)  # 自由度为干预变量数量
        p_value = 1 - stats.chi2.cdf(lr_stat, df)

        return {
            'lr_statistic': lr_stat,
            'p_value': p_value,
            'df': df
        }

    def plot_intervention_effects(self, results):
        """可视化干预效果"""
        fig, axes = plt.subplots(2, 1, figsize=(15, 10))

        # 1. 绘制原始序列和拟合值
        actual = self.df_processed['Close']
        fitted = results['model'].fittedvalues

        axes[0].plot(actual.index, actual, label='实际值')
        axes[0].plot(fitted.index, fitted, label='拟合值', alpha=0.7)
        axes[0].set_title('价格序列与拟合值对比')
        axes[0].legend()

        # 2. 绘制干预效果
        effects = results['intervention_effects']
        event_names = list(effects.keys())
        coefficients = [effect['coefficient'] for effect in effects.values()]

        axes[1].bar(event_names, coefficients)
        axes[1].set_xticklabels(event_names, rotation=45, ha='right')
        axes[1].set_title('各干预事件的效果系数')

        plt.tight_layout()
        plt.show()

        return fig