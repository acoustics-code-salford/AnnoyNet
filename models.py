import torch
from torch import nn
import pytorch_lightning as pl


class SimpleDenseLSTM(nn.Module):
    def __init__(self, input_size=20, hidden_size=124, num_layers=1, out_size=3):
        super().__init__()
        self.norm = nn.BatchNorm1d(input_size)
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers)
        self.dense = nn.Linear(hidden_size, out_size)

    def forward(self, x):
        x = x.swapaxes(1, 2)
        x = self.norm(x)
        x = x.swapaxes(1, 2)
        x, _ = self.lstm(x)
        y = self.dense(x)
        return y