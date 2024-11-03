import numpy as np
import matplotlib.pyplot as plt
from statsmodels.stats.diagnostic import het_white
from statsmodels.graphics.tsaplots import plot_acf
import scipy.stats as stats
from arch import arch_model
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

class ResidualAnalyzer:
    def __init__(self, model, data):
        self.model = model
        self.data = data
        self.resid = model.resid

    def plot_residual_diagnostics(self):
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))

        axes[0,0].plot(self.resid)
        axes[0,0].set_title('残差时序图')
        axes[0,0].set_xlabel('时间')
        axes[0,0].set_ylabel('残差')

        sns.histplot(self.resid, kde=True, ax=axes[0,1])
        axes[0,1].set_title('残差分布')

        stats.probplot(self.resid, dist="norm", plot=axes[1,0])
        axes[1,0].set_title('残差Q-Q图')

        plot_acf(self.resid, ax=axes[1,1], lags=40)
        axes[1,1].set_title('残差ACF图')

        plt.tight_layout()
        plt.show()

    def check_heteroskedasticity(self):
        print("\n异方差性检验:")
        white_test = het_white(self.resid**2, np.ones((len(self.resid), 2)))
        print(f"White检验p值: {white_test[1]:.4f}")

        arch_model_test = arch_model(self.resid)
        arch_result = arch_model_test.fit(disp='off')
        print("\nARCH效应检验:")
        print(arch_result.summary().tables[1])

        return white_test[1] < 0.05

    def check_normality(self):
        print("\n正态性检验:")
        jb_test = stats.jarque_bera(self.resid)
        print(f"Jarque-Bera检验p值: {jb_test[1]:.4f}")

        if len(self.resid) <= 5000:
            sw_test = stats.shapiro(self.resid)
            print(f"Shapiro-Wilk检验p值: {sw_test[1]:.4f}")

        return jb_test[1] < 0.05

    def apply_garch(self, p=1, q=1):
        print("\n使用GARCH模型处理异方差:")
        garch_model = arch_model(self.resid, vol='Garch', p=p, q=q)
        garch_fit = garch_model.fit(disp='off')
        standardized_resid = garch_fit.resid / garch_fit.conditional_volatility
        print(garch_fit.summary())
        return standardized_resid

    def box_cox_transform(self, lambda_param=None):
        min_val = min(self.resid)
        if min_val <= 0:
            shifted_data = self.resid - min_val + 1
        else:
            shifted_data = self.resid

        if lambda_param is None:
            transformed_data, lambda_param = stats.boxcox(shifted_data)
        else:
            transformed_data = stats.boxcox(shifted_data, lambda_param)

        print(f"\nBox-Cox转换使用的lambda值: {lambda_param:.4f}")
        return transformed_data