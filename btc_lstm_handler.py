# btc_lstm_handler.py
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy import stats
from models import BTCPredictor
from data_utils import prepare_data
from visualization import plot_complete_prediction,plot_long_term_prediction

class BTCLSTMHandler:
    def __init__(self, sequence_length=60):
        self.sequence_length = sequence_length
        self.model = None
        self.scaler = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def prepare_data(self, df, batch_size=32):
        train_loader, test_loader, test_data, scaler = prepare_data(
            df, self.sequence_length, batch_size
        )
        self.scaler = scaler
        return train_loader, test_loader, test_data

    def train(self, train_loader, epochs=100, learning_rate=0.001):
        """训练模型"""
        self.model = BTCPredictor(sequence_length=self.sequence_length).to(self.device)
        criterion = nn.HuberLoss()
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=learning_rate)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=10, verbose=True
        )

        train_losses = []
        best_loss = float('inf')

        print("开始训练...")
        for epoch in range(epochs):
            self.model.train()
            total_loss = 0

            with tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs}') as pbar:
                for X_batch, y_batch in pbar:
                    X_batch = X_batch.to(self.device)
                    y_batch = y_batch.to(self.device)

                    optimizer.zero_grad()
                    outputs, _ = self.model(X_batch)
                    loss = criterion(outputs, y_batch)

                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    optimizer.step()

                    total_loss += loss.item()
                    pbar.set_postfix({'loss': f'{loss.item():.6f}'})

            avg_loss = total_loss / len(train_loader)
            train_losses.append(avg_loss)
            scheduler.step(avg_loss)

            if avg_loss < best_loss:
                best_loss = avg_loss
                torch.save(self.model.state_dict(), 'best_model.pth')

            print(f'Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.6f}')

        self.model.load_state_dict(torch.load('best_model.pth'))
        return train_losses

    def evaluate(self, test_loader):
        """评估模型"""
        self.model.eval()
        predictions = []
        actuals = []
        attention_weights_list = []

        with torch.no_grad():
            for X_batch, y_batch in test_loader:
                X_batch = X_batch.to(self.device)
                outputs, attention_weights = self.model(X_batch)
                predictions.extend(outputs.cpu().numpy())
                actuals.extend(y_batch.numpy())
                attention_weights_list.extend(attention_weights.cpu().numpy())

        predictions = np.array(predictions)
        actuals = np.array(actuals)

        # 反归一化
        predictions = self.scaler.inverse_transform(predictions)
        actuals = self.scaler.inverse_transform(actuals)

        # 计算误差
        mse = np.mean((predictions - actuals) ** 2)
        rmse = np.sqrt(mse)
        mae = np.mean(np.abs(predictions - actuals))
        mape = np.mean(np.abs((predictions - actuals) / actuals)) * 100

        print('\n模型评估结果:')
        print(f'MSE: {mse:.2f}')
        print(f'RMSE: {rmse:.2f}')
        print(f'MAE: {mae:.2f}')
        print(f'MAPE: {mape:.2f}%')

        metrics = {
            'mse': mse,
            'rmse': rmse,
            'mae': mae,
            'mape': mape
        }

        return predictions, actuals, np.array(attention_weights_list), metrics

    def predict_future(self, df, days=30):
        """预测未来价格"""
        self.model.eval()
        last_sequence = self.scaler.transform(df[['Close']].values[-self.sequence_length:])

        future_predictions = []
        attention_weights_list = []
        current_sequence = torch.FloatTensor(last_sequence).unsqueeze(0).to(self.device)

        with torch.no_grad():
            for _ in range(days):
                prediction, attention_weights = self.model(current_sequence)
                future_predictions.append(prediction.cpu().numpy()[0])
                attention_weights_list.append(attention_weights.cpu().numpy())

                # 更新序列
                current_sequence = torch.roll(current_sequence, -1, dims=1)
                current_sequence[0, -1] = prediction

        future_predictions = np.array(future_predictions).reshape(-1, 1)
        future_predictions = self.scaler.inverse_transform(future_predictions)

        last_date = df.index[-1]
        future_dates = pd.date_range(
            start=last_date + pd.Timedelta(days=1),
            periods=days,
            freq='D'
        )

        return future_dates, future_predictions, np.array(attention_weights_list)

    def predict_long_term(self, df, days=365, confidence_level=0.95):
        """长期预测"""
        self.model.eval()
        last_sequence = self.scaler.transform(df[['Close']].values[-self.sequence_length:])

        # 存储预测结果
        all_simulations = np.zeros((100, days))  # 100次模拟

        print("正在进行长期预测...")
        for sim in tqdm(range(100)):
            current_sequence = torch.FloatTensor(last_sequence).unsqueeze(0).to(self.device)
            sim_predictions = []

            with torch.no_grad():
                for _ in range(days):
                    prediction, _ = self.model(current_sequence)
                    # 添加随机噪声模拟不确定性
                    noise = torch.normal(0, 0.01, size=prediction.shape).to(self.device)
                    prediction = prediction + noise

                    sim_predictions.append(prediction.cpu().numpy()[0][0])

                    # 更新序列
                    current_sequence = torch.roll(current_sequence, -1, dims=1)
                    current_sequence[0, -1] = prediction

            all_simulations[sim, :] = sim_predictions

        # 计算统计量
        mean_predictions = np.mean(all_simulations, axis=0)
        std_predictions = np.std(all_simulations, axis=0)

        # 计算置信区间
        z_score = stats.norm.ppf((1 + confidence_level) / 2)
        upper_bound = mean_predictions + z_score * std_predictions
        lower_bound = mean_predictions - z_score * std_predictions

        # 反归一化
        mean_predictions = self.scaler.inverse_transform(mean_predictions.reshape(-1, 1))
        upper_bound = self.scaler.inverse_transform(upper_bound.reshape(-1, 1))
        lower_bound = self.scaler.inverse_transform(lower_bound.reshape(-1, 1))

        # 生成未来日期
        last_date = df.index[-1]
        future_dates = pd.date_range(
            start=last_date + pd.Timedelta(days=1),
            periods=days,
            freq='D'
        )

        return future_dates, mean_predictions, (lower_bound, upper_bound)

    def save_model(self, path='model/btc_lstm_model.pth'):
        """保存模型"""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'sequence_length': self.sequence_length,
            'scaler': self.scaler
        }, path)
        print(f'模型已保存至 {path}')

    def load_model(self, path='model/btc_lstm_model.pth'):
        """加载模型"""
        checkpoint = torch.load(path)
        self.sequence_length = checkpoint['sequence_length']
        self.model = BTCPredictor(sequence_length=self.sequence_length).to(self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.scaler = checkpoint['scaler']
        print(f'模型已从 {path} 加载')