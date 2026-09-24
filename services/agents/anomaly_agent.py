"""
NEXUS Anomaly Detection Agent
Orchestrates outlier detection, multivariate isolation trees, deep autoencoders,
change-point detection, deterministic severity scoring, and evidence explanations.
"""

from typing import Dict, Any, List, Optional
import pandas as pd

from services.anomaly_detection.engine import AnomalyDetectionEngine
from services.anomaly_detection.change_point import ChangePointDetector


class AnomalyAgent:
    """
    Autonomous Anomaly Agent equipped with deterministic detection algorithms
    for business and IoT telemetry anomaly diagnosis.
    """

    def __init__(self):
        self.engine = AnomalyDetectionEngine()
        self.change_point_detector = ChangePointDetector()

    def detect_anomalies(
        self,
        df: pd.DataFrame,
        time_col: Optional[str] = None,
        target_col: Optional[str] = None,
        feature_cols: Optional[List[str]] = None,
        include_deep_learning: bool = True
    ) -> Dict[str, Any]:
        """Tool: Run consensus anomaly detection, return flagged events, severity, and timeline."""
        return self.engine.detect(
            df=df,
            time_col=time_col,
            target_col=target_col,
            feature_cols=feature_cols,
            include_deep_learning=include_deep_learning
        )

    def detect_change_points(
        self,
        series: pd.Series,
        timestamps: Optional[pd.Series] = None
    ) -> List[Dict[str, Any]]:
        """Tool: Scan series for regime shifts and step-change points."""
        return self.change_point_detector.detect_change_points(
            series=series,
            timestamps=timestamps
        )
