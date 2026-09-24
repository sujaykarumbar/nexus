"""
NEXUS Statistical Anomaly Detection
Implements standard Z-Score, Modified Z-Score (MAD), IQR, and Rolling Z-Score anomaly detectors.
"""

from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd


class StatisticalAnomalyDetector:
    """
    Computes statistical anomaly indicators and scores across univariate or multi-column data.
    """

    @staticmethod
    def z_score_detect(
        series: pd.Series,
        threshold: float = 3.0
    ) -> pd.DataFrame:
        """
        Detects anomalies via standard Z-score: |(x - mean) / std| > threshold.
        """
        vals = pd.to_numeric(series, errors="coerce")
        mean = vals.mean()
        std = vals.std()
        if std == 0 or np.isnan(std):
            z_scores = pd.Series(0.0, index=vals.index)
        else:
            z_scores = (vals - mean) / std

        is_anomaly = z_scores.abs() > threshold
        normalized_score = (z_scores.abs() / (threshold * 2)).clip(0.0, 1.0)

        return pd.DataFrame({
            "value": vals,
            "expected": mean,
            "deviation": vals - mean,
            "metric_score": z_scores,
            "anomaly_score": normalized_score,
            "is_anomaly": is_anomaly,
            "method": "z_score"
        })

    @staticmethod
    def modified_z_score_detect(
        series: pd.Series,
        threshold: float = 3.5
    ) -> pd.DataFrame:
        """
        Detects anomalies using Median Absolute Deviation (MAD) which is resilient to extreme outliers.
        M_i = 0.6745 * |x_i - median| / MAD
        """
        vals = pd.to_numeric(series, errors="coerce")
        median = vals.median()
        mad = (vals - median).abs().median()

        if mad == 0 or np.isnan(mad):
            mod_z = pd.Series(0.0, index=vals.index)
        else:
            mod_z = 0.6745 * (vals - median) / mad

        is_anomaly = mod_z.abs() > threshold
        normalized_score = (mod_z.abs() / (threshold * 2)).clip(0.0, 1.0)

        return pd.DataFrame({
            "value": vals,
            "expected": median,
            "deviation": vals - median,
            "metric_score": mod_z,
            "anomaly_score": normalized_score,
            "is_anomaly": is_anomaly,
            "method": "modified_z_score"
        })

    @staticmethod
    def iqr_detect(
        series: pd.Series,
        factor: float = 1.5
    ) -> pd.DataFrame:
        """
        Interquartile range (Tukey's method) anomaly detection.
        Lower: Q1 - factor*IQR, Upper: Q3 + factor*IQR
        """
        vals = pd.to_numeric(series, errors="coerce")
        q1 = vals.quantile(0.25)
        q3 = vals.quantile(0.75)
        iqr = q3 - q1
        median = vals.median()

        lower_bound = q1 - (factor * iqr)
        upper_bound = q3 + (factor * iqr)

        is_anomaly = (vals < lower_bound) | (vals > upper_bound)
        # Distance from nearest bound normalized by IQR
        distance = np.maximum(0.0, np.maximum(lower_bound - vals, vals - upper_bound))
        norm_score = (distance / (iqr + 1e-6)).clip(0.0, 1.0)

        return pd.DataFrame({
            "value": vals,
            "expected": median,
            "deviation": vals - median,
            "metric_score": distance,
            "anomaly_score": norm_score,
            "is_anomaly": is_anomaly,
            "method": "iqr"
        })

    @staticmethod
    def rolling_z_score_detect(
        series: pd.Series,
        window: int = 14,
        threshold: float = 2.8
    ) -> pd.DataFrame:
        """
        Detects local / contextual anomalies using a rolling statistical window.
        """
        vals = pd.to_numeric(series, errors="coerce")
        rolling_mean = vals.rolling(window=window, min_periods=3).mean().bfill()
        rolling_std = vals.rolling(window=window, min_periods=3).std().bfill().replace(0.0, 1e-6)

        rolling_z = (vals - rolling_mean) / rolling_std
        is_anomaly = rolling_z.abs() > threshold
        norm_score = (rolling_z.abs() / (threshold * 2)).clip(0.0, 1.0)

        return pd.DataFrame({
            "value": vals,
            "expected": rolling_mean,
            "deviation": vals - rolling_mean,
            "metric_score": rolling_z,
            "anomaly_score": norm_score,
            "is_anomaly": is_anomaly,
            "method": "rolling_z_score"
        })
