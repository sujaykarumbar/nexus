"""
NEXUS Forecasting Agent
Orchestrates time-series detection, temporal feature engineering, multi-model benchmarks,
model evaluation, multi-step future forecasting, and uncertainty estimation.
"""

from typing import Dict, Any, List, Optional
import pandas as pd

from services.forecasting.time_series_detector import TimeSeriesDetector
from services.forecasting.feature_engineer import TimeSeriesFeatureEngineer
from services.forecasting.forecaster import ForecastingEngine
from services.forecasting.evaluator import ForecastEvaluator
from services.forecasting.uncertainty import UncertaintyEstimator


class ForecastingAgent:
    """
    Autonomous Forecasting Agent equipped with concrete Python/ML tools
    for time-series analysis and forecasting.
    """

    def __init__(self):
        self.detector = TimeSeriesDetector()
        self.feature_engineer = TimeSeriesFeatureEngineer()
        self.engine = ForecastingEngine()
        self.evaluator = ForecastEvaluator()
        self.uncertainty = UncertaintyEstimator()

    def detect_time_series(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """Tool: Identify candidate temporal columns with confidence scores."""
        return self.detector.detect_candidate_time_columns(df)

    def analyze_temporal_profile(self, df: pd.DataFrame, time_col: str, target_col: Optional[str] = None) -> Dict[str, Any]:
        """Tool: Extract frequency, seasonality period, trend, and regularity."""
        return self.detector.analyze_temporal_profile(df, time_col=time_col, target_col=target_col)

    def engineer_features(self, df: pd.DataFrame, time_col: str, target_col: str) -> pd.DataFrame:
        """Tool: Build lag, rolling, calendar, and expanding window features without leakage."""
        return self.feature_engineer.transform(df, time_col=time_col, target_col=target_col)

    def train_and_forecast(
        self,
        df: pd.DataFrame,
        time_col: str,
        target_col: str,
        horizon: int = 7,
        candidate_models: Optional[List[str]] = None,
        covariates: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Tool: Train candidate forecasting models, evaluate leaderboard, and return future forecasts."""
        return self.engine.train_and_evaluate(
            df=df,
            time_col=time_col,
            target_col=target_col,
            horizon=horizon,
            candidate_models=candidate_models,
            covariates=covariates
        )

    def estimate_uncertainty(
        self,
        point_forecasts: List[float],
        residuals: Optional[List[float]] = None
    ) -> List[Dict[str, Any]]:
        """Tool: Generate 80% and 95% prediction intervals."""
        return self.uncertainty.estimate_intervals(
            point_forecasts=point_forecasts,
            residuals=residuals
        )
