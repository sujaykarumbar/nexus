"""
NEXUS Residual Anomaly Detector
Identifies time-series anomalies based on deviation between actual values and expected model forecasts.
"""

from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd


class ResidualAnomalyDetector:
    """
    Computes time-series anomalies from forecast residuals and rolling deviation envelopes.
    """

    def __init__(self, threshold_std: float = 3.0, window_size: int = 14):
        self.threshold_std = threshold_std
        self.window_size = window_size

    def detect_residuals(
        self,
        actual: pd.Series,
        expected: Optional[pd.Series] = None
    ) -> pd.DataFrame:
        """
        Calculates forecast residuals: deviation = actual - expected.
        Flags observations where residual magnitude exceeds rolling standard deviation envelope.
        """
        y_act = pd.to_numeric(actual, errors="coerce")

        if expected is not None:
            y_exp = pd.to_numeric(expected, errors="coerce")
        else:
            # Derive expected using robust centered rolling median
            y_exp = y_act.rolling(window=self.window_size, min_periods=3, center=True).median().bfill().ffill()

        residuals = y_act - y_exp
        rolling_res_std = residuals.rolling(window=self.window_size, min_periods=3).std().bfill().ffill().replace(0.0, 1e-6)

        z_residuals = residuals / rolling_res_std
        is_anomaly = z_residuals.abs() > self.threshold_std
        norm_score = (z_residuals.abs() / (self.threshold_std * 2.0)).clip(0.0, 1.0)

        return pd.DataFrame({
            "actual": np.round(y_act, 2),
            "expected": np.round(y_exp, 2),
            "deviation": np.round(residuals, 2),
            "residual_z": np.round(z_residuals, 3),
            "anomaly_score": np.round(norm_score, 4),
            "is_anomaly": is_anomaly,
            "method": "forecast_residual"
        })
