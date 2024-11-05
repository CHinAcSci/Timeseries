import matplotlib.pyplot as plt
import numpy as np


def plot_complete_prediction(df, future_dates, future_predictions, metrics=None,
                             confidence_intervals=None, volatility=None, last_n_days=None):
    """绘制完整的历史数据和预测数据，包括波动率"""
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 设置中文字体
    plt.rcParams['axes.unicode_minus'] = False    # 解决负号显示问题

    # 创建两个子图
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(20, 15))

    # 价格预测图
    if last_n_days:
        historical_data = df['Close'][-last_n_days:]
        ax1.plot(df.index[-last_n_days:], historical_data,
                 label='历史数据', color='blue', linewidth=2)
    else:
        ax1.plot(df.index, df['Close'],
                 label='历史数据', color='blue', linewidth=2)

    # 绘制预测数据
    ax1.plot(future_dates, future_predictions,
             label='预测数据', color='red', linewidth=2, linestyle='--')

    # 如果有置信区间，则绘制
    if confidence_intervals is not None:
        lower_bound, upper_bound = confidence_intervals
        ax1.fill_between(future_dates,
                         lower_bound.flatten(),
                         upper_bound.flatten(),
                         color='red', alpha=0.2,
                         label='95%置信区间')

    # 添加垂直线分隔
    ax1.axvline(x=df.index[-1], color='gray', linestyle='--', alpha=0.5)
    ax1.text(df.index[-1], ax1.get_ylim()[1], '预测开始',
             rotation=90, verticalalignment='top')

    # 设置价格图属性
    ax1.set_title('比特币价格历史数据和未来预测', fontsize=16, pad=20)
    ax1.set_xlabel('日期', fontsize=12)
    ax1.set_ylabel('价格 (USD)', fontsize=12)
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=12)
    plt.setp(ax1.get_xticklabels(), rotation=45)

    # 添加评估指标
    if metrics:
        metrics_text = f'模型评估指标:\nMSE: {metrics["mse"]:.2f}\nRMSE: {metrics["rmse"]:.2f}\nMAE: {metrics["mae"]:.2f}\nMAPE: {metrics["mape"]:.2f}%'
        ax1.text(0.02, 0.98, metrics_text, transform=ax1.transAxes,
                 bbox=dict(facecolor='white', alpha=0.8),
                 verticalalignment='top', fontsize=10)

    # 波动率预测图
    if volatility is not None:
        ax2.plot(future_dates, volatility,
                 label='波动率预测', color='green', linewidth=2)
        ax2.set_title('波动率预测', fontsize=16, pad=20)
        ax2.set_xlabel('日期', fontsize=12)
        ax2.set_ylabel('波动率', fontsize=12)
        ax2.grid(True, alpha=0.3)
        ax2.legend(fontsize=12)
        plt.setp(ax2.get_xticklabels(), rotation=45)

    plt.tight_layout()
    plt.savefig('img/complete_prediction.png', dpi=300)
    plt.show()

def plot_long_term_prediction(df, future_dates, predictions, confidence_bounds, last_n_days=None):
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 设置中文字体
    plt.rcParams['axes.unicode_minus'] = False    # 解决负号显示问题
    """
    绘制长期预测结果及置信区间

    参数:
    - df: 历史数据DataFrame
    - future_dates: 预测日期
    - predictions: 预测值
    - confidence_bounds: 置信区间(lower_bound, upper_bound)
    - last_n_days: 显示最近n天的历史数据
    """
    plt.figure(figsize=(20, 10))

    # 绘制历史数据
    if last_n_days:
        historical_data = df['Close'][-last_n_days:]
        plt.plot(df.index[-last_n_days:], historical_data,
                 label='历史数据', color='blue', linewidth=2)
    else:
        plt.plot(df.index, df['Close'],
                 label='历史数据', color='blue', linewidth=2)

    # 绘制预测值
    plt.plot(future_dates, predictions,
             label='预测值', color='red', linewidth=2, linestyle='--')

    # 绘制置信区间
    lower_bound, upper_bound = confidence_bounds
    plt.fill_between(future_dates,
                     lower_bound.flatten(),
                     upper_bound.flatten(),
                     color='red', alpha=0.2,
                     label='95%置信区间')

    # 添加垂直线分隔历史和预测
    plt.axvline(x=df.index[-1], color='gray', linestyle='--', alpha=0.5)

    # 添加关键信息标注
    last_actual = df['Close'].iloc[-1]
    final_prediction = predictions[-1][0]
    max_prediction = np.max(predictions)
    min_prediction = np.min(predictions)

    # 设置图表属性
    plt.title('比特币价格长期预测（一年）', fontsize=16, pad=20)
    plt.xlabel('日期', fontsize=12)
    plt.ylabel('价格 (USD)', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=12)
    plt.xticks(rotation=45)

    # 添加预测信息文本框
    info_text = (
        f'预测期间: {future_dates[0].strftime("%Y-%m-%d")} 至 {future_dates[-1].strftime("%Y-%m-%d")}\n'
        f'当前价格: ${last_actual:,.0f}\n'
        f'预测价格: ${final_prediction:,.0f}\n'
        f'预测范围: ${min_prediction:,.0f} - ${max_prediction:,.0f}\n'
        f'预测变化: {((final_prediction-last_actual)/last_actual*100):,.1f}%'
    )

    plt.text(0.02, 0.98, info_text,
             transform=plt.gca().transAxes,
             bbox=dict(facecolor='white', alpha=0.8),
             verticalalignment='top',
             fontsize=10)

    plt.tight_layout()
    plt.savefig('img/future365.png', dpi=300)
    plt.show()