"""
PyTorch Sequence Models for Time-Series Forecasting
Implements configurable LSTM and GRU architectures.
"""

import torch
import torch.nn as nn
from typing import Optional


class LSTMForecaster(nn.Module):
    """
    Recurrent Neural Network Forecaster using Long Short-Term Memory (LSTM) cells.
    Maps input sequences of length L with D features to forecast horizon H.
    """

    def __init__(
        self,
        input_size: int = 1,
        hidden_dim: int = 32,
        num_layers: int = 2,
        output_horizon: int = 7,
        dropout: float = 0.1
    ):
        super().__init__()
        self.input_size = input_size
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.output_horizon = output_horizon

        effective_dropout = dropout if num_layers > 1 else 0.0

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=effective_dropout
        )
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, output_horizon)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x shape: [batch_size, seq_len, input_size]
        output shape: [batch_size, output_horizon]
        """
        lstm_out, _ = self.lstm(x)
        # Use representation of the last time step
        last_step_repr = lstm_out[:, -1, :]
        out = self.fc(last_step_repr)
        return out


class GRUForecaster(nn.Module):
    """
    Recurrent Neural Network Forecaster using Gated Recurrent Units (GRU).
    Provides computationally efficient alternative to LSTM with comparable capacity.
    """

    def __init__(
        self,
        input_size: int = 1,
        hidden_dim: int = 32,
        num_layers: int = 2,
        output_horizon: int = 7,
        dropout: float = 0.1
    ):
        super().__init__()
        self.input_size = input_size
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.output_horizon = output_horizon

        effective_dropout = dropout if num_layers > 1 else 0.0

        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=effective_dropout
        )
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, output_horizon)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x shape: [batch_size, seq_len, input_size]
        output shape: [batch_size, output_horizon]
        """
        gru_out, _ = self.gru(x)
        last_step_repr = gru_out[:, -1, :]
        out = self.fc(last_step_repr)
        return out
