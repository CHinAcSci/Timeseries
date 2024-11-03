import torch
import torch.nn as nn

class BTCPredictor(nn.Module):
    def __init__(self, input_size=1, hidden_size=128, num_layers=3, dropout=0.3, sequence_length=60):
        super(BTCPredictor, self).__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.sequence_length = sequence_length

        # 特征提取层
        self.feature_extractor = nn.Sequential(
            nn.Linear(input_size, hidden_size // 2),
            nn.ReLU(),
            nn.BatchNorm1d(sequence_length),
            nn.Linear(hidden_size // 2, hidden_size),
            nn.ReLU(),
            nn.BatchNorm1d(sequence_length)
        )

        # 双向LSTM
        self.lstm = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            batch_first=True,
            bidirectional=True
        )

        # 注意力层
        self.attention = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1),
            nn.Softmax(dim=1)
        )

        # 输出层
        self.fc = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, 1)
        )

    def forward(self, x):
        batch_size = x.size(0)

        # 特征提取
        x = self.feature_extractor(x)

        # LSTM层
        device = next(self.parameters()).device
        h0 = torch.zeros(self.num_layers * 2, batch_size, self.hidden_size).to(device)
        c0 = torch.zeros(self.num_layers * 2, batch_size, self.hidden_size).to(device)
        lstm_out, _ = self.lstm(x, (h0, c0))

        # 注意力机制
        attention_weights = self.attention(lstm_out)
        context_vector = torch.sum(lstm_out * attention_weights, dim=1)

        # 输出层
        out = self.fc(context_vector)
        return out, attention_weights