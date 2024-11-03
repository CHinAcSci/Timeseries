import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime

class BitcoinPlotter:
    def __init__(self):
        # 定义重大事件
        self.events = {
            "2013-10-02": ("首个丝绸之路被查封", "FBI关闭丝绸之路，缴获26000个比特币"),
            "2014-02-24": ("Mt.Gox破产", "最大交易所Mt.Gox宣布破产，损失85万个比特币"),
            "2016-07-09": ("比特币减半", "第二次减半，区块奖励从25降至12.5 BTC"),
            "2017-09-04": ("中国禁止交易", "中国禁止ICO"),
            "2018-11-15": ("BCH硬分叉", "比特币现金分叉为BCHABC和BCHSV"),
            "2020-03-12": ("黑色星期四", "比特币暴跌50%，跌至3800美元"),
            "2020-05-11": ("第三次减半", "区块奖励从12.5降至6.25 BTC"),
            "2021-02-08": ("特斯拉购买", "特斯拉宣布购买15亿美元比特币"),
            "2021-09-07": ("萨尔瓦多", "萨尔瓦多将比特币列为法定货币"),
            "2022-05-12": ("Luna崩溃", "UST/Luna生态系统崩溃"),
            "2022-11-11": ("FTX破产", "全球第二大加密货币交易所FTX申请破产"),
            "2023-03-10": ("硅谷银行", "硅谷银行破产引发金融动荡"),
            "2023-06-10": ("BlackRock", "贝莱德申请比特币现货ETF")
        }

    def read_data(self, csv_path='data/BTC-USD.csv'):
        """从CSV文件读取数据"""
        try:
            df = pd.read_csv(csv_path)
            df['Date'] = pd.to_datetime(df['Date'])
            df.set_index('Date', inplace=True)
            return df
        except Exception as e:
            print(f"读取数据时出错: {e}")
            return None

    def plot_data(self, csv_path='data/BTC-USD.csv'):
        """绘制比特币价格、交易量和重大事件"""
        try:
            # 读取数据
            print("正在读取数据...")
            df = self.read_data(csv_path)
            if df is None:
                return

            print("正在生成图表...")
            # 创建子图
            fig = make_subplots(
                rows=2, cols=1,
                shared_xaxes=True,
                vertical_spacing=0.03,
                subplot_titles=('比特币价格 (USD)', '交易量'),
                row_heights=[0.7, 0.3]
            )

            # 添加价格线
            fig.add_trace(
                go.Scatter(
                    x=df.index,
                    y=df['Close'],
                    name='价格',
                    line=dict(color='#2962FF', width=1.5)
                ),
                row=1, col=1
            )

            # 添加成交量柱状图
            fig.add_trace(
                go.Bar(
                    x=df.index,
                    y=df['Volume'],
                    name='成交量',
                    marker_color='rgba(41, 98, 255, 1)'
                ),
                row=2, col=1
            )

            # 添加事件标记
            event_x = []
            event_y = []
            event_texts = []

            for date, (title, description) in self.events.items():
                event_date = pd.to_datetime(date)
                if event_date in df.index:
                    event_x.append(event_date)
                    event_y.append(df.loc[event_date, 'Close'])
                    event_texts.append(f"{title}<br>{description}")

            fig.add_trace(
                go.Scatter(
                    x=event_x,
                    y=event_y,
                    mode='markers+text',
                    name='重大事件',
                    marker=dict(
                        symbol='star',
                        size=8,
                        color='red'
                    ),
                    text=event_texts,
                    textposition="top center",
                    textfont=dict(size=8),
                    hoverinfo='text'
                ),
                row=1, col=1
            )

            # 更新布局
            fig.update_layout(
                title='比特币价格走势与重大事件（2013-2023）',
                height=800,  # 增加高度以容纳事件标签
                template='plotly_white',
                showlegend=True,
                legend=dict(
                    yanchor="top",
                    y=0.99,
                    xanchor="left",
                    x=0.01
                ),
                # 添加悬停模式配置
                hovermode='x unified'
            )

            # 更新价格Y轴格式
            fig.update_yaxes(
                title_text="价格 (USD)",
                tickprefix="$",
                tickformat=",.0f",
                row=1, col=1
            )

            # 更新成交量Y轴格式
            fig.update_yaxes(
                title_text="成交量",
                tickformat=",.0f",
                row=2, col=1
            )

            # 更新X轴标签
            fig.update_xaxes(
                title_text="日期",
                row=2, col=1
            )

            # 显示图表
            fig.show()

        except Exception as e:
            print(f"生成图表时出错: {e}")

if __name__ == "__main__":
    plotter = BitcoinPlotter()
    plotter.plot_data()