"""
PyTorch Autoencoder for Deep Learning Anomaly Detection
Uses reconstruction error across multivariate features to identify out-of-distribution anomalies.
"""

from typing import Dict, Any, List, Optional, Tuple
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler


class AutoencoderNet(nn.Module):
    """
    Multi-layer neural autoencoder mapping input to latent bottleneck and reconstructing output.
    """

    def __init__(self, input_dim: int, latent_dim: int = 4):
        super().__init__()
        hidden_dim = max(latent_dim * 2, input_dim // 2)

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        latent = self.encoder(x)
        reconstructed = self.decoder(latent)
        return reconstructed


class AutoencoderAnomalyDetector:
    """
    Trains PyTorch Autoencoder on regular data patterns and flags instances with high reconstruction error.
    """

    def __init__(self, latent_dim: int = 4, epochs: int = 30, lr: float = 0.01):
        self.latent_dim = latent_dim
        self.epochs = epochs
        self.lr = lr
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.scaler = StandardScaler()
        self.model: Optional[AutoencoderNet] = None
        self.threshold: float = 0.0

    def fit_detect(
        self,
        df: pd.DataFrame,
        feature_cols: List[str],
        percentile_threshold: float = 95.0
    ) -> pd.DataFrame:
        """
        Trains autoencoder and computes sample reconstruction error.
        """
        X = df[feature_cols].copy().apply(pd.to_numeric, errors="coerce").bfill().ffill().fillna(0.0)
        X_scaled = self.scaler.fit_transform(X)
        n_samples, input_dim = X_scaled.shape

        effective_latent = min(self.latent_dim, max(1, input_dim // 2))
        self.model = AutoencoderNet(input_dim=input_dim, latent_dim=effective_latent).to(self.device)

        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=1e-5)

        x_tensor = torch.tensor(X_scaled, dtype=torch.float32).to(self.device)

        # Train loop
        self.model.train()
        for epoch in range(self.epochs):
            optimizer.zero_grad()
            recon = self.model(x_tensor)
            loss = criterion(recon, x_tensor)
            loss.backward()
            optimizer.step()

        # Evaluation & Reconstruction MSE
        self.model.eval()
        with torch.no_grad():
            reconstructed = self.model(x_tensor).cpu().numpy()

        sample_mse = np.mean((X_scaled - reconstructed) ** 2, axis=1)
        self.threshold = float(np.percentile(sample_mse, percentile_threshold))

        is_anomaly = sample_mse > self.threshold
        # Normalize score 0.0 to 1.0
        max_mse = float(sample_mse.max()) if sample_mse.max() > 0 else 1.0
        norm_scores = np.clip(sample_mse / (max_mse + 1e-6), 0.0, 1.0)

        return pd.DataFrame({
            "reconstruction_error": np.round(sample_mse, 5),
            "threshold": round(self.threshold, 5),
            "anomaly_score": np.round(norm_scores, 4),
            "is_anomaly": is_anomaly,
            "method": "deep_autoencoder"
        }, index=df.index)
