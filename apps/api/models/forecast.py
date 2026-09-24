"""
NEXUS Forecasting Database Models
Stores trained forecast models, evaluation leaderboards, uncertainty intervals, and future predictions.
"""

from sqlalchemy import Column, String, Integer, Float, Text, ForeignKey, JSON
from apps.api.core.database import Base
from apps.api.models.base import TimestampMixin


class ForecastModel(Base, TimestampMixin):
    __tablename__ = "forecast_models"

    project_id = Column(String(36), ForeignKey("projects.id"), nullable=False, index=True)
    dataset_id = Column(String(36), ForeignKey("datasets.id"), nullable=False, index=True)
    time_column = Column(String(100), nullable=False)
    target_column = Column(String(100), nullable=False)
    horizon = Column(Integer, nullable=False, default=7)
    best_model_name = Column(String(100), nullable=False)
    lifecycle_stage = Column(String(50), nullable=False, default="TRAINED") # TRAINED, VALIDATED, STAGING, PRODUCTION, ARCHIVED

    metrics = Column(JSON, nullable=False) # MAE, RMSE, MAPE, sMAPE, R2
    leaderboard = Column(JSON, nullable=False) # Full candidate tournament ranking
    temporal_profile = Column(JSON, nullable=True) # Freq, trend, seasonality
    future_forecast = Column(JSON, nullable=False) # Array of {timestamp, prediction, lower_80, upper_80, lower_95, upper_95}
    recent_history = Column(JSON, nullable=True) # Historical overlay points
    validation_comparison = Column(JSON, nullable=True) # Actual vs Predicted holdout points
    feature_importances = Column(JSON, nullable=True)

    artifact_path = Column(String(255), nullable=True)
    training_duration_seconds = Column(Float, nullable=True)


class AnomalyReport(Base, TimestampMixin):
    __tablename__ = "anomaly_reports"

    project_id = Column(String(36), ForeignKey("projects.id"), nullable=False, index=True)
    dataset_id = Column(String(36), ForeignKey("datasets.id"), nullable=False, index=True)
    analyzed_metric = Column(String(100), nullable=False)

    summary = Column(JSON, nullable=False) # total_anomalies, critical_count, high_count, etc.
    anomalies = Column(JSON, nullable=False) # List of detected anomaly records
    change_points = Column(JSON, nullable=True) # List of detected regime shifts
    timeline = Column(JSON, nullable=True) # Timeline data points for chart rendering
