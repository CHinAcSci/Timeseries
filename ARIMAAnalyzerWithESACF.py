import numpy as np
from statsmodels.tsa.stattools import acf
import warnings
warnings.filterwarnings('ignore')

class ARIMAAnalyzerWithESACF:
    def __init__(self):
        self.best_model = None
        self.best_params = None
        self.best_aic = float('inf')

    def compute_ESACF(self, data, ar_max=7, ma_max=13):
        """计算ESACF表"""
        print("计算ESACF表...")
        ESACF = np.zeros((ar_max + 1, ma_max))
        n = len(data)
        data = (data - np.mean(data)) / np.std(data)

        for i in range(ar_max + 1):
            if i == 0:
                acf_values = acf(data, nlags=ma_max+1, fft=False)[1:ma_max+1]
                ESACF[i, :] = acf_values
            else:
                y = data[i:]
                X = np.zeros((len(y), i))
                for j in range(i):
                    X[:, j] = data[i-j-1:-j-1]

                try:
                    ar_coeffs = np.linalg.lstsq(X, y, rcond=None)[0]
                    resid = y - X @ ar_coeffs
                    acf_values = acf(resid, nlags=ma_max+1, fft=False)[1:ma_max+1]
                    ESACF[i, :] = acf_values
                except:
                    ESACF[i, :] = np.nan

        symbol_matrix = np.empty((ar_max + 1, ma_max), dtype=str)
        critical_value = 1.96 / np.sqrt(n)

        for i in range(ar_max + 1):
            for j in range(ma_max):
                if abs(ESACF[i, j]) > critical_value:
                    symbol_matrix[i, j] = 'x'
                else:
                    symbol_matrix[i, j] = 'o'

        return symbol_matrix

    def print_ESACF_table(self, ESACF_matrix):
        print("\nESACF表 (x: 显著相关, o: 不显著相关)")
        print("   MA阶数")
        print("AR  " + "".join(f"{i:4}" for i in range(ESACF_matrix.shape[1])))
        print("阶  " + "----" * ESACF_matrix.shape[1])

        for i in range(ESACF_matrix.shape[0]):
            print(f"{i:2} |" + "".join(f"{x:4}" for x in ESACF_matrix[i, :]))

    def suggest_orders(self, ESACF_matrix):
        ar_max, ma_max = ESACF_matrix.shape
        suggested_orders = []

        for ar in range(ar_max):
            for ma in range(ma_max):
                if ESACF_matrix[ar, ma] == 'o':
                    is_valid = True
                    for i in range(ar, min(ar + 2, ar_max)):
                        for j in range(ma, min(ma + 2, ma_max)):
                            if ESACF_matrix[i, j] == 'x':
                                is_valid = False
                                break
                        if not is_valid:
                            break

                    if is_valid:
                        suggested_orders.append((ar, 1, ma))

        return suggested_orders[:5]