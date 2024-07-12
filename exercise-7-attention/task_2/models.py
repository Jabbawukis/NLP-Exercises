import math

import torch
from torch import nn

class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        _, (hidden, _) = self.lstm(x)
        out = self.fc(hidden.squeeze())
        return out

class SelfAttention(nn.Module):
    def __init__(self, hidden_dim):
        super(SelfAttention, self).__init__()
        self.hidden_dim = hidden_dim
        self.query = nn.Linear(hidden_dim, hidden_dim)
        self.key = nn.Linear(hidden_dim, hidden_dim)
        self.value = nn.Linear(hidden_dim, hidden_dim)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, lstm_output):
        queries = self.query(lstm_output)
        keys = self.key(lstm_output)
        values = self.value(lstm_output)
        attention_scores = torch.matmul(queries, keys.transpose(1, 2)) / math.sqrt(self.hidden_dim)
        attention_scores = self.softmax(attention_scores)
        attention_output = torch.matmul(attention_scores, values)
        return attention_output

class LSTMSelfAttentionModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(LSTMSelfAttentionModel, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.attention = SelfAttention(hidden_dim)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        attn_out = self.attention(lstm_out)
        mean_attn_out = attn_out.mean(dim=1)
        out = self.fc(mean_attn_out)
        return out
