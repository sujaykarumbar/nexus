"""
PyTorch Asynchronous Deep Learning Training Pipeline
Handles sliding window dataset generation, CPU/CUDA hardware management,
early stopping, loss tracking, and multi-step sequence forecasting.
"""

from typing import Dict, Any, List, Optional, Tuple, Callable
import os
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from .models import LSTMForecaster, GRUForecaster


class DeepLearningTrainer:
    """
    Trains and validates PyTorch sequence forecasters with safety limits and early stopping.
    """

    def __init__(
        self,
        architecture: str = "lstm", # 'lstm' or 'gru'
        seq_length: int = 14,
        horizon: int = 7,
        hidden_dim: int = 32,
        num_layers: int = 2,
        dropout: float = 0.1,
        learning_rate: float = 0.005,
        batch_size: int = 16,
        max_epochs: int = 40,
        patience: int = 6,
        device: Optional[str] = None
    ):
        self.architecture = architecture.lower()
        self.seq_length = seq_length
        self.horizon = horizon
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self.patience = patience

        if device:
            self.device = torch.device(device)
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model: Optional[nn.Module] = None
        self.scaler_mean: float = 0.0
        self.scaler_std: float = 1.0
        self.train_losses: List[float] = []
        self.val_losses: List[float] = []
        self.last_sequence: Optional[np.ndarray] = None

    def _create_sliding_windows(
        self,
        series: np.ndarray,
        seq_length: int,
        horizon: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Builds input sequences and target horizons:
        X: [N, seq_length, 1], y: [N, horizon]
        """
        X, y = [], []
        total_len = len(series)
        for i in range(total_len - seq_length - horizon + 1):
            x_window = series[i:i + seq_length]
            y_window = series[i + seq_length:i + seq_length + horizon]
            X.append(x_window)
            y.append(y_window)
        return np.array(X, dtype=np.float32)[:, :, np.newaxis], np.array(y, dtype=np.float32)

    def fit(
        self,
        series: pd.Series,
        progress_callback: Optional[Callable[[Dict[str, Any]], None]] = None
    ) -> Dict[str, Any]:
        """
        Trains the sequence model using chronological holdout validation.
        """
        raw_vals = pd.to_numeric(series, errors="coerce").dropna().values
        n_obs = len(raw_vals)

        # Minimum observations needed
        min_required = self.seq_length + self.horizon + 3
        if n_obs < min_required:
            # Adjust sequence length dynamically if dataset is compact
            self.seq_length = max(3, (n_obs - self.horizon) // 2)
            min_required = self.seq_length + self.horizon + 2
            if n_obs < min_required:
                raise ValueError(f"Insufficient time-series observations ({n_obs}) for DL windowing.")

        # Standardize strictly on training split
        val_split_idx = int(n_obs * 0.8)
        train_raw = raw_vals[:val_split_idx]
        self.scaler_mean = float(np.mean(train_raw))
        self.scaler_std = float(np.std(train_raw)) if np.std(train_raw) > 1e-6 else 1.0

        norm_series = (raw_vals - self.scaler_mean) / self.scaler_std
        self.last_sequence = norm_series[-self.seq_length:]

        # Create windows
        X_all, y_all = self._create_sliding_windows(norm_series, self.seq_length, self.horizon)
        if len(X_all) < 2:
            raise ValueError("Time series too short to generate validation windows.")

        split = max(1, int(len(X_all) * 0.8))
        X_train, y_train = X_all[:split], y_all[:split]
        X_val, y_val = X_all[split:], y_all[split:]
        if len(X_val) == 0:
            X_val, y_val = X_train[-1:], y_train[-1:]

        # DataLoaders
        train_dataset = TensorDataset(torch.tensor(X_train), torch.tensor(y_train))
        val_dataset = TensorDataset(torch.tensor(X_val), torch.tensor(y_val))
        train_loader = DataLoader(train_dataset, batch_size=min(self.batch_size, len(train_dataset)), shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=min(self.batch_size, len(val_dataset)), shuffle=False)

        # Instantiate model
        input_size = 1
        if self.architecture == "gru":
            self.model = GRUForecaster(
                input_size=input_size,
                hidden_dim=self.hidden_dim,
                num_layers=self.num_layers,
                output_horizon=self.horizon,
                dropout=self.dropout
            ).to(self.device)
        else:
            self.model = LSTMForecaster(
                input_size=input_size,
                hidden_dim=self.hidden_dim,
                num_layers=self.num_layers,
                output_horizon=self.horizon,
                dropout=self.dropout
            ).to(self.device)

        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate, weight_decay=1e-5)

        best_val_loss = float("inf")
        best_state = None
        patience_counter = 0
        self.train_losses = []
        self.val_losses = []

        start_time = time.time()

        for epoch in range(1, self.max_epochs + 1):
            # Training phase
            self.model.train()
            train_batch_losses = []
            for b_x, b_y in train_loader:
                b_x, b_y = b_x.to(self.device), b_y.to(self.device)
                optimizer.zero_grad()
                preds = self.model(b_x)
                loss = criterion(preds, b_y)
                loss.backward()
                optimizer.step()
                train_batch_losses.append(loss.item())

            avg_train_loss = float(np.mean(train_batch_losses))
            self.train_losses.append(round(avg_train_loss, 5))

            # Validation phase
            self.model.eval()
            val_batch_losses = []
            with torch.no_grad():
                for b_x, b_y in val_loader:
                    b_x, b_y = b_x.to(self.device), b_y.to(self.device)
                    v_preds = self.model(b_x)
                    v_loss = criterion(v_preds, b_y)
                    val_batch_losses.append(v_loss.item())

            avg_val_loss = float(np.mean(val_batch_losses)) if val_batch_losses else avg_train_loss
            self.val_losses.append(round(avg_val_loss, 5))

            # Early stopping check
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                best_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
                patience_counter = 0
            else:
                patience_counter += 1

            # Progress callback
            if progress_callback:
                progress_pct = int((epoch / self.max_epochs) * 100)
                progress_callback({
                    "status": "TRAINING",
                    "epoch": epoch,
                    "total_epochs": self.max_epochs,
                    "train_loss": avg_train_loss,
                    "val_loss": avg_val_loss,
                    "progress": progress_pct
                })

            if patience_counter >= self.patience:
                break

        # Restore best weights
        if best_state is not None:
            self.model.load_state_dict(best_state)

        duration = time.time() - start_time

        return {
            "architecture": self.architecture,
            "device": str(self.device),
            "epochs_trained": len(self.train_losses),
            "final_train_loss": self.train_losses[-1],
            "best_val_loss": round(best_val_loss, 5),
            "train_loss_history": self.train_losses,
            "val_loss_history": self.val_losses,
            "training_duration_seconds": round(duration, 2)
        }

    def predict(self) -> np.ndarray:
        """
        Produces multi-step horizon forecast from the most recent historical sequence.
        """
        if self.model is None or self.last_sequence is None:
            raise ValueError("Deep learning model has not been fitted.")

        self.model.eval()
        with torch.no_grad():
            x_in = torch.tensor(self.last_sequence, dtype=torch.float32).unsqueeze(0).unsqueeze(-1).to(self.device)
            out_norm = self.model(x_in).cpu().numpy().flatten()

        # Invert normalization back to original scale
        raw_forecast = (out_norm * self.scaler_std) + self.scaler_mean
        return np.asarray(raw_forecast, dtype=float)

    def save(self, filepath: str) -> None:
        """
        Saves trained model state and normalizer parameters.
        """
        if self.model is None:
            raise ValueError("No trained model to save.")
        payload = {
            "architecture": self.architecture,
            "seq_length": self.seq_length,
            "horizon": self.horizon,
            "hidden_dim": self.hidden_dim,
            "num_layers": self.num_layers,
            "scaler_mean": self.scaler_mean,
            "scaler_std": self.scaler_std,
            "last_sequence": self.last_sequence,
            "state_dict": self.model.state_dict()
        }
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        torch.save(payload, filepath)

    @classmethod
    def load(cls, filepath: str, device: Optional[str] = None) -> "DeepLearningTrainer":
        """
        Loads saved model checkpoint.
        """
        payload = torch.load(filepath, map_location="cpu", weights_only=False)
        trainer = cls(
            architecture=payload["architecture"],
            seq_length=payload["seq_length"],
            horizon=payload["horizon"],
            hidden_dim=payload["hidden_dim"],
            num_layers=payload["num_layers"],
            device=device
        )
        trainer.scaler_mean = payload["scaler_mean"]
        trainer.scaler_std = payload["scaler_std"]
        trainer.last_sequence = payload["last_sequence"]

        if trainer.architecture == "gru":
            trainer.model = GRUForecaster(
                input_size=1,
                hidden_dim=trainer.hidden_dim,
                num_layers=trainer.num_layers,
                output_horizon=trainer.horizon
            ).to(trainer.device)
        else:
            trainer.model = LSTMForecaster(
                input_size=1,
                hidden_dim=trainer.hidden_dim,
                num_layers=trainer.num_layers,
                output_horizon=trainer.horizon
            ).to(trainer.device)

        trainer.model.load_state_dict(payload["state_dict"])
        trainer.model.eval()
        return trainer
