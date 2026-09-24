"""
NEXUS Machine Learning Anomaly Detectors
Implements Isolation Forest and Local Outlier Factor (LOF) for multivariate anomaly detection.
"""

from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler


class MLAnomalyDetector:
    """
    Multivariate non-linear anomaly detectors using Isolation Forest and LOF.
    """

    def __init__(self, contamination: float = 0.05):
        self.contamination = contamination
        self.scaler = StandardScaler()

    def isolation_forest_detect(
        self,
        df: pd.DataFrame,
        feature_cols: List[str]
    ) -> pd.DataFrame:
        """
        Detects collective and point anomalies using Isolation Forest trees.
        """
        X = df[feature_cols].copy().apply(pd.to_numeric, errors="coerce").bfill().ffill().fillna(0.0)
        X_scaled = self.scaler.fit_transform(X)

        iso = IsolationForest(
            contamination=self.contamination,
            random_state=42,
            n_estimators=100
        )
        preds = iso.fit_predict(X_scaled) # -1 is anomaly, 1 is normal
        # Decision function: lower values mean more anomalous
        scores = iso.decision_function(X_scaled)

        # Normalize score to 0.0 (normal) - 1.0 (most anomalous)
        min_s, max_s = float(scores.min()), float(scores.max())
        if max_s - min_s > 1e-6:
            norm_scores = 1.0 - ((scores - min_s) / (max_s - min_s))
        else:
            norm_scores = np.zeros(len(scores))

        is_anomaly = preds == -1

        return pd.DataFrame({
            "anomaly_score": np.round(norm_scores, 4),
            "is_anomaly": is_anomaly,
            "raw_score": np.round(scores, 4),
            "method": "isolation_forest"
        }, index=df.index)

    def lof_detect(
        self,
        df: pd.DataFrame,
        feature_cols: List[str],
        n_neighbors: int = 15
    ) -> pd.DataFrame:
        """
        Detects density-based local anomalies using Local Outlier Factor.
        """
        X = df[feature_cols].copy().apply(pd.to_numeric, errors="coerce").bfill().ffill().fillna(0.0)
        X_scaled = self.scaler.fit_transform(X)

        n_neighbors = min(n_neighbors, max(2, len(df) - 1))
        lof = LocalOutlierFactor(
            n_neighbors=n_neighbors,
            contamination=self.contamination
        )
        preds = lof.fit_predict(X_scaled) # -1 is outlier, 1 is inlier
        negative_outlier_factor = lof.negative_outlier_factor_

        # LOF scores: closer to -1 is normal, large negative numbers are outliers
        norm_scores = np.maximum(0.0, -negative_outlier_factor - 1.0)
        max_norm = norm_scores.max() if norm_scores.max() > 0 else 1.0
        norm_scores = norm_scores / max_norm

        is_anomaly = preds == -1

        return pd.DataFrame({
            "anomaly_score": np.round(norm_scores, 4),
            "is_anomaly": is_anomaly,
            "raw_score": np.round(negative_outlier_factor, 4),
            "method": "local_outlier_factor"
        }, index=df.index)
