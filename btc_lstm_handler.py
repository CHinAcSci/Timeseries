import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler
from typing import Tuple


class SimpleLSTM(nn.Module):
    def __init__(self, input_size=1, hidden_size=50, num_layers=2):
        super(SimpleLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.2
        )

        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        # x shape: (batch_size, sequence_length, input_size)
        lstm_out, _ = self.lstm(x)
        # 只使用最后一个时间步的输出
        last_output = lstm_out[:, -1, :]
        prediction = self.fc(last_output)
        return prediction


class BTCLSTMHandler:
    def __init__(self, sequence_length=60):
        self.sequence_length = sequence_length
        self.model = None
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")

    def create_sequences(self, data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """创建时间序列序列"""
        X, y = [], []
        for i in range(len(data) - self.sequence_length):
            X.append(data[i:(i + self.sequence_length)])
            y.append(data[i + self.sequence_length])
        return np.array(X), np.array(y)

    def prepare_data(self, df: pd.DataFrame, batch_size: int = 32) -> Tuple[DataLoader, DataLoader, pd.DataFrame]:
        """准备训练和测试数据"""
        # 获取收盘价数据并归一化
        data = self.scaler.fit_transform(df[['Close']].values)

        # 创建序列
        X, y = self.create_sequences(data)

        # 分割训练集和测试集
        train_size = int(len(X) * 0.8)
        X_train, X_test = X[:train_size], X[train_size:]
        y_train, y_test = y[:train_size], y[train_size:]

        # 转换为PyTorch张量
        X_train = torch.FloatTensor(X_train).to(self.device)
        y_train = torch.FloatTensor(y_train).to(self.device)
        X_test = torch.FloatTensor(X_test).to(self.device)
        y_test = torch.FloatTensor(y_test).to(self.device)

        # 创建数据加载器
        train_data = TensorDataset(X_train, y_train)
        test_data = TensorDataset(X_test, y_test)

        train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

        return train_loader, test_loader, df[-len(X_test):]

    def train(self, train_loader: DataLoader, epochs: int = 100) -> list:
        """训练模型"""
        # 初始化模型
        self.model = SimpleLSTM().to(self.device)
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)

        train_losses = []
        for epoch in range(epochs):
            self.model.train()
            total_loss = 0
            for X_batch, y_batch in train_loader:
                optimizer.zero_grad()
                y_pred = self.model(X_batch)
                loss = criterion(y_pred, y_batch)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            avg_loss = total_loss / len(train_loader)
            train_losses.append(avg_loss)
            if (epoch + 1) % 10 == 0:
                print(f'Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.6f}')

        return train_losses

    def evaluate(self, test_loader: DataLoader) -> Tuple[np.ndarray, np.ndarray, None, dict]:
        """评估模型"""
        self.model.eval()
        predictions = []
        actuals = []

        with torch.no_grad():
            for X_batch, y_batch in test_loader:
                y_pred = self.model(X_batch)
                predictions.extend(y_pred.cpu().numpy())
                actuals.extend(y_batch.cpu().numpy())

        predictions = np.array(predictions)
        actuals = np.array(actuals)

        # 反归一化
        predictions = self.scaler.inverse_transform(predictions)
        actuals = self.scaler.inverse_transform(actuals)

        # 计算评估指标
        mse = np.mean((predictions - actuals) ** 2)
        rmse = np.sqrt(mse)
        mae = np.mean(np.abs(predictions - actuals))
        mape = np.mean(np.abs((predictions - actuals) / actuals)) * 100

        metrics = {
            'mse': mse,
            'rmse': rmse,
            'mae': mae,
            'mape': mape
        }

        return predictions, actuals, None, metrics

    def predict_future(self, df: pd.DataFrame, days: int = 30) -> Tuple[pd.DatetimeIndex, np.ndarray, None]:
        """预测未来价格"""
        self.model.eval()

        # 获取最后一个序列
        last_sequence = self.scaler.transform(df[['Close']].values)[-self.sequence_length:]
        current_sequence = torch.FloatTensor(last_sequence).unsqueeze(0).to(self.device)

        predictions = []
        with torch.no_grad():
            for _ in range(days):
                prediction = self.model(current_sequence)
                predictions.append(prediction.cpu().numpy()[0])

                # 更新序列
                current_sequence = torch.roll(current_sequence, -1, dims=1)
                current_sequence[0, -1] = prediction

        predictions = np.array(predictions).reshape(-1, 1)
        predictions = self.scaler.inverse_transform(predictions)

        # 生成未来日期
        last_date = df.index[-1]
        future_dates = pd.date_range(
            start=last_date + pd.Timedelta(days=1),
            periods=days,
            freq='D'
        )

        return future_dates, predictions, None

    def save_model(self, path: str = 'model/btc_lstm_model.pth'):
        """保存模型"""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'scaler': self.scaler,
            'sequence_length': self.sequence_length
        }, path)
        print(f'Model saved to {path}')

    def load_model(self, path: str = 'model/btc_lstm_model.pth'):
        """加载模型"""
        checkpoint = torch.load(path)
        self.sequence_length = checkpoint['sequence_length']
        self.model = SimpleLSTM().to(self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.scaler = checkpoint['scaler']
        print(f'Model loaded from {path}')