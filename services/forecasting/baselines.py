"""
NEXUS Baseline Forecasting Models
Implements Naive, Seasonal Naive, and Moving Average benchmarks.
"""

from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd


class BaselineForecaster:
    """
    Computes standard time-series baselines to establish benchmark metrics.
    """

    def __init__(self, method: str = "naive", seasonal_period: int = 7, window_size: int = 7):
        self.method = method.lower()
        self.seasonal_period = seasonal_period
        self.window_size = window_size
        self.history: np.ndarray = np.array([])

    def fit(self, y: pd.Series) -> "BaselineForecaster":
        """
        Fits baseline model on historical series.
        """
        self.history = pd.to_numeric(y, errors="coerce").dropna().values
        if len(self.history) == 0:
            raise ValueError("Empty series provided to BaselineForecaster.")
        return self

    def predict(self, horizon: int = 7) -> np.ndarray:
        """
        Generates forecast for the specified horizon.
        """
        if len(self.history) == 0:
            raise ValueError("BaselineForecaster has not been fitted.")

        if self.method == "naive":
            # Forecast is constant equal to last observed value
            last_val = self.history[-1]
            return np.full(horizon, last_val, dtype=float)

        elif self.method == "seasonal_naive":
            # Forecast repeats values from the last seasonal cycle
            sp = min(self.seasonal_period, len(self.history))
            cycle = self.history[-sp:]
            preds = []
            for h in range(horizon):
                preds.append(cycle[h % sp])
            return np.array(preds, dtype=float)

        elif self.method == "moving_average":
            # Forecast is the average of the last k observations
            w = min(self.window_size, len(self.history))
            ma_val = float(np.mean(self.history[-w:]))
            return np.full(horizon, ma_val, dtype=float)

        else:
            raise ValueError(f"Unknown baseline method: {self.method}. Choose 'naive', 'seasonal_naive', or 'moving_average'.")
